// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <string>

#include "aie/AIEDialect.h"
#include "aie/AIEXDialect.h"
#include "iree-amd-aie/Target/AMDAIEControlPacket.h"
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
}

int main(int argc, char **argv) {
  llvm::StringRef sourceMlirPath(argv[1]);
  llvm::SmallString<128> workDir(argv[2]);

  llvm::SmallString<128> ctrlPktMlirPath(workDir);
  llvm::sys::path::append(ctrlPktMlirPath, "control_packet.mlir");

  llvm::SmallString<128> artifactPath(workDir);
  llvm::sys::path::append(artifactPath, "artifact.pdi");

  if (auto ecode = llvm::sys::fs::create_directories(workDir)) {
    std::cerr << "Error: failed to create working directory: "
              << ecode.message() << "\n";
    return 1;
  }

  DialectRegistry registry;
  registerDialects(registry);
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  MLIRContext context(registry);

  // Parse the input MLIR file.
  mlir::ParserConfig parserConfig(&context);
  auto moduleOp = llvm::cast<ModuleOp>(
      mlir::parseSourceFile(sourceMlirPath, parserConfig).release());

  // Find the `xilinx.aie.device` op.
  SmallVector<xilinx::AIE::DeviceOp> deviceOps(
      moduleOp.getOps<xilinx::AIE::DeviceOp>());
  if (deviceOps.size() != 1) {
    std::cerr << "Error: Expected exactly one xilinx.aie.device op\n";
    return 1;
  }
  xilinx::AIE::DeviceOp deviceOp = deviceOps[0];

  // Get the `npuVersion` and `targetArch` strings necessary for elf generation.
  AMDAIEDeviceModel deviceModel = getDeviceModel(deviceOp.getDevice());
  std::optional<std::string> npuVersion = deviceModel.getNPUVersionString();
  std::optional<std::string> targetArch = deviceModel.getTargetArchString();
  if (!npuVersion.has_value() || !targetArch.has_value()) {
    std::cerr << "Error: unhandled NPU partitioning.\n";
    return 1;
  }

  // Generate the elf files, which are required for generating control
  // packets.
  if (failed(aie2xclbin(
          /*ctx=*/&context,
          /*deviceOp=*/deviceOp,
          /*outputNPU=*/std::nullopt,
          /*artifactPath=*/artifactPath.str().str(),
          /*printIRBeforeAll=*/false,
          /*printIRAfterAll=*/false,
          /*printIRModuleScope=*/false,
          /*timing=*/false,
          /*tempDir=*/workDir.str().str(),
          /*useChess=*/false,
          /*verbose=*/false,
          /*vitisDir=*/std::nullopt,
          /*targetArch=*/targetArch.value(),
          /*npuVersion=*/npuVersion.value(),
          /*peanoDir=*/std::getenv("PEANO_INSTALL_DIR"),
          /*deviceHal=*/AMDAIEOptions::DeviceHAL::XRT_LITE,
          /*xclBinKernelID=*/"",
          /*xclBinKernelName=*/"",
          /*xclBinInstanceName=*/"",
          /*amdAIEInstallDir=*/"",
          /*InputXCLBin=*/std::nullopt,
          /*ukernel=*/std::nullopt,
          /*additionalPeanoOptFlags=*/""))) {
    std::cerr << "Error: failed to generate xclbin\n";
    return 1;
  }

  // Convert the AIE device to control packet operations.
  if (failed(convertAieToControlPacket(moduleOp, deviceOp,
                                       ctrlPktMlirPath.str().str(),
                                       workDir.str().str()))) {
    std::cerr << "Error: failed to convert AIE to control packet\n";
    return 1;
  }
}
