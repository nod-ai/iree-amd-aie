// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <string>

#include "aie/AIEDialect.h"
#include "aie/AIEXDialect.h"
#include "iree-amd-aie/Target/AMDAIETargets.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Parser/Parser.h"

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
}

int main(int argc, char **argv) {
  llvm::StringRef mlirAbsPath(argv[1]);
  llvm::StringRef workDir(argv[2]);

  DialectRegistry registry;
  registerDialects(registry);
  MLIRContext context(registry);

  mlir::ParserConfig parserConfig(&context);
  auto moduleOp = llvm::cast<ModuleOp>(
      mlir::parseSourceFile(mlirAbsPath, parserConfig).release());

  auto deviceOps = moduleOp.getOps<xilinx::AIE::DeviceOp>();
  auto nDeviceOps = std::distance(deviceOps.begin(), deviceOps.end());
  if (nDeviceOps != 1) {
    std::cerr << "Error: Expected exactly one xilinx.aie.device op\n";
    return 1;
  }
  auto deviceOp = *deviceOps.begin();
  llvm::DebugFlag = true;
  const char *debugTypes[3] = {"aie-generate-cdo", "iree-aie-runtime",
                               "iree-aie-cdo-emitter"};
#ifndef NDEBUG
  llvm::setCurrentDebugTypes(debugTypes, 3);
#endif
  auto status = AIETranslateToCDODirect(deviceOp, workDir, false, false, false);
  std::vector<std::string> diagnostics;
  ScopedDiagnosticHandler handler(moduleOp.getContext(), [&](Diagnostic &d) {
    llvm::raw_string_ostream(diagnostics.emplace_back())
        << d.getLocation() << ": " << d;
  });

  if (failed(status))
    for (const auto &diagnostic : diagnostics) std::cerr << diagnostic << "\n";

  llvm::DebugFlag = false;
#ifndef NDEBUG
  llvm::setCurrentDebugType("aie-cdo-driver-debug");
#endif
  status = AIETranslateToCDODirect(deviceOp, workDir, false, false, true);
  if (failed(status))
    for (const auto &diagnostic : diagnostics) std::cerr << diagnostic << "\n";
}
