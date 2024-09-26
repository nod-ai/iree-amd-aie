// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <filesystem>
#include <iostream>
#include <string>

#include "aie/AIEDialect.h"
#include "aie/AIEXDialect.h"
#include "aie/Passes.h"
#include "iree-amd-aie/Target/XCLBinGen.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::iree_compiler::AMDAIE;
using Path = std::filesystem::path;

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
}

int main(int argc, char **argv) {
  Path mlirFilePath(argv[1]);
  mlirFilePath = absolute(mlirFilePath);
  Path workDir(argv[2]);
  workDir = absolute(workDir);

  DialectRegistry registry;
  registerDialects(registry);
  MLIRContext context(registry);

  mlir::ParserConfig parserConfig(&context);
  auto moduleOp = llvm::cast<ModuleOp>(
      mlir::parseSourceFile(mlirFilePath.string(), parserConfig).release());

  auto deviceOps = moduleOp.getOps<xilinx::AIE::DeviceOp>();
  auto nDeviceOps = std::distance(deviceOps.begin(), deviceOps.end());
  if (nDeviceOps != 1) {
    std::cerr << "Error: Expected exactly one xilinx.aie.device op\n";
    return -1;
  }
  auto deviceOp = *deviceOps.begin();
  PassManager pm(&context, xilinx::AIE::DeviceOp::getOperationName());
  pm.addPass(createAMDAIEDmaToNpuPass());
  if (failed(pm.run(deviceOp))) {
    std::cerr << "Failed to run AMDAIEDmaToNpuPass";
    return -1;
  }

  std::string npuInstFilePath = (workDir / mlirFilePath.filename()).string();
  npuInstFilePath += ".npu.txt";
  LogicalResult status = emitNpuInstructions(deviceOp, npuInstFilePath);
  std::vector<std::string> diagnostics;
  ScopedDiagnosticHandler handler(moduleOp.getContext(), [&](Diagnostic &d) {
    llvm::raw_string_ostream(diagnostics.emplace_back())
        << d.getLocation() << ": " << d;
  });
  if (failed(status))
    for (const auto &diagnostic : diagnostics) std::cerr << diagnostic << "\n";
  return 0;
}
