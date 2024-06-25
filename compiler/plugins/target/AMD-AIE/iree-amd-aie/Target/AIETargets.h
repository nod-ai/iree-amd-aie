// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AIE_TARGETS_AIETARGETS_H
#define AIE_TARGETS_AIETARGETS_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Passes.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::AMDAIE {
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

inline void collectTiles(
    xilinx::AIE::DeviceOp &device,
    DenseMap<mlir::iree_compiler::AMDAIE::TileID, Operation *> &tiles) {
  for (auto tile : device.getOps<xilinx::AIE::TileOp>()) {
    int colIndex = tile.colIndex();
    int rowIndex = tile.rowIndex();
    tiles[{colIndex, rowIndex}] = tile;
  }
}

inline void collectBuffers(
    xilinx::AIE::DeviceOp &device,
    DenseMap<Operation *, SmallVector<xilinx::AIE::BufferOp, 4>> &buffers) {
  for (xilinx::AIE::BufferOp buffer : device.getOps<xilinx::AIE::BufferOp>()) {
    Operation *tileOp = buffer.getTile().getDefiningOp();
    buffers[tileOp].push_back(buffer);
  }
}

}  // namespace mlir::iree_compiler::AMDAIE

#endif
