// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "aie/AIEDialect.h"
#include "aie/AIEXDialect.h"
#include "aievec/AIEVecDialect.h"
#include "aievec/Passes.h"
#include "aievec/XLLVMDialect.h"
#include "iree-amd-aie/Target/XCLBinGen.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

using namespace mlir;
using namespace mlir::iree_compiler::AMDAIE;

void registerDialects(DialectRegistry &registry) {
  registry.insert<xilinx::AIE::AIEDialect>();
  registry.insert<xilinx::AIEX::AIEXDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<cf::ControlFlowDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<mlir::vector::VectorDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<LLVM::LLVMDialect>();
  registry.insert<iree_compiler::IREE::HAL::HALDialect>();
  registry.insert<iree_compiler::aievec::xllvm::XLLVMDialect>();
  registry.insert<iree_compiler::aievec::AIEVecDialect>();
}

// The following code parses an MLIR file containing a `xilinx.aie.device` op,
// generates the elf files using `aie2xclbin`, and writes the elf files to the
// working directory.
//
// Usage:
//   aie_elf_files_gen_test <input_file> <working_directory> <emit_ctrlpkt>
//
// It is used as a helper for testing the `AMDAIEConvertDeviceToControlPackets`
// pass.
int main(int argc, char **argv) {
  llvm::StringRef sourceMlirPath(argv[1]);
  llvm::SmallString<128> workDir(argv[2]);
  llvm::SmallString<128> artifactPath(workDir);
  llvm::sys::path::append(artifactPath, "artifact.pdi");
  if (std::error_code ecode = llvm::sys::fs::create_directories(workDir)) {
    llvm::errs() << "Error: failed to create working directory: "
                 << ecode.message() << "\n";
    return 1;
  }
  bool emitCtrlPkt = false;
  if (argc > 3 && std::string(argv[3]) == "true") emitCtrlPkt = true;

  DialectRegistry registry;
  registerDialects(registry);
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  mlir::iree_compiler::aievec::registerXLLVMDialectTranslation(registry);
  MLIRContext context(registry);

  // Parse the input MLIR file.
  mlir::ParserConfig parserConfig(&context);
  auto moduleOp = llvm::cast<ModuleOp>(
      mlir::parseSourceFile(sourceMlirPath, parserConfig).release());

  // Find the `xilinx.aie.device` op.
  SmallVector<xilinx::AIE::DeviceOp> deviceOps(
      moduleOp.getOps<xilinx::AIE::DeviceOp>());
  if (deviceOps.size() != 1) {
    llvm::errs() << "Error: Expected exactly one xilinx.aie.device op\n";
    return 1;
  }
  xilinx::AIE::DeviceOp deviceOp = deviceOps[0];

  // Get the `npuVersion` and `targetArch` strings necessary for elf generation.
  AMDAIEDeviceModel deviceModel = getDeviceModel(deviceOp.getDevice());
  std::optional<std::string> npuVersion = deviceModel.getNPUVersionString();
  std::optional<std::string> targetArch = deviceModel.getTargetArchString();
  if (!npuVersion.has_value() || !targetArch.has_value()) {
    llvm::errs() << "Error: unhandled NPU partitioning.\n";
    return 1;
  }

#ifndef NDEBUG
  // Enable the `iree-amdaie-ert` debug flag to print program size for
  // verification purposes.
  llvm::DebugFlag = true;
  llvm::setCurrentDebugType("iree-amdaie-ert");
#endif

  const char *peanoDir = std::getenv("PEANO_INSTALL_DIR");
  if (!peanoDir) {
    llvm::errs()
        << "Error: PEANO_INSTALL_DIR environment variable not set. A path to "
           "an llvm-aie directory is needed to run aie2xclbin.";
    return 1;
  }
  std::string peanoDirStr = peanoDir;

  // Use `aie2xclbin` to generate the elf files.
  if (failed(aie2xclbin(
          /*ctx=*/&context,
          /*deviceOp=*/deviceOp,
          /*outputNPU=*/std::nullopt,
          /*emitCtrlPkt=*/emitCtrlPkt,
          /*artifactPath=*/artifactPath.str().str(),
          /*printIRBeforeAll=*/false,
          /*printIRAfterAll=*/false,
          /*printIRModuleScope=*/false,
          /*timing=*/false,
          /*tempDir=*/workDir.str().str(),
          /*useChess=*/false,
          /*useChessForUKernel=*/false,
          /*verbose=*/false,
          /*vitisDir=*/std::nullopt,
          /*targetArch=*/targetArch.value(),
          /*npuVersion=*/npuVersion.value(),
          /*peanoDir=*/peanoDirStr,
          /*deviceHal=*/AMDAIEOptions::DeviceHAL::XRT_LITE,
          /*xclBinKernelID=*/"",
          /*xclBinKernelName=*/"",
          /*xclBinInstanceName=*/"",
          /*amdAIEInstallDir=*/"",
          /*InputXCLBin=*/std::nullopt,
          /*ukernel=*/std::nullopt,
          /*additionalPeanoOptFlags=*/""))) {
    llvm::errs() << "Error: failed to generate xclbin\n";
    return 1;
  }
  return 0;
}
