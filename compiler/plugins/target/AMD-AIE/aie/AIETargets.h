//===- AIETargets.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_TARGETS_AIETARGETS_H
#define AIE_TARGETS_AIETARGETS_H

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace xilinx {
namespace AIE {

mlir::LogicalResult AIETranslateToNPU(mlir::ModuleOp module,
                                      llvm::raw_ostream &output);
std::vector<uint32_t> AIETranslateToNPU(mlir::ModuleOp);
mlir::LogicalResult AIETranslateToLdScript(mlir::ModuleOp module,
                                           llvm::raw_ostream &output,
                                           int tileCol, int tileRow);
mlir::LogicalResult AIETranslateToBCF(mlir::ModuleOp module,
                                      llvm::raw_ostream &output, int tileCol,
                                      int tileRow);
mlir::LogicalResult AIETranslateToCDODirect(
    mlir::ModuleOp m, llvm::StringRef workDirPath, bool bigEndian = false,
    bool emitUnified = false, bool cdoDebug = false, bool aieSim = false,
    bool xaieDebug = false, bool enableCores = true);
}  // namespace AIE

}  // namespace xilinx

#endif
