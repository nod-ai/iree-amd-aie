// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AIE_TARGETS_AIETARGETS_H
#define AIE_TARGETS_AIETARGETS_H

#include "aie/AIEDialect.h"
#include "aie/Passes.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::AMDAIE {
std::vector<uint32_t> AIETranslateToNPU(mlir::ModuleOp);

mlir::LogicalResult AIETranslateToLdScript(xilinx::AIE::DeviceOp,
                                           llvm::raw_ostream &output,
                                           int tileCol, int tileRow,
                                           int stackSize);

mlir::LogicalResult AIETranslateToBCF(xilinx::AIE::DeviceOp,
                                      llvm::raw_ostream &output, int tileCol,
                                      int tileRow, int stackSize);

mlir::LogicalResult AIETranslateToCDODirect(
    xilinx::AIE::DeviceOp, llvm::StringRef workDirPath, int stackSize,
    bool bigEndian = false, bool emitUnified = false, bool cdoDebug = false,
    bool aieSim = false, bool enableCores = true);
}  // namespace mlir::iree_compiler::AMDAIE

#endif
