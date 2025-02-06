// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_AMDAIEOPUTILS_H_
#define IREE_AMD_AIE_TRANSFORMS_AMDAIEOPUTILS_H_

#include <limits>

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"

namespace mlir::iree_compiler::AMDAIE {

/// Utility struct to describe a region of core tiles on the array.
struct CoreRegionInfo {
  int64_t startCol{0};
  int64_t numCols{0};
  int64_t startRow{0};
  int64_t numRows{0};
};

/// Utility to return the core region info based on the provided op. This will
/// look for a parent Function op and for the tile region in which all core
/// tiles are found.
template <typename Op>
FailureOr<CoreRegionInfo> getCoreRegionInfo(Op op) {
  int64_t firstCol{std::numeric_limits<int64_t>::max()};
  int64_t lastCol{0};
  int64_t firstRow{std::numeric_limits<int64_t>::max()};
  int64_t lastRow{0};
  WalkResult res = op->walk([&](AMDAIE::CoreOp coreOp) {
    AMDAIE::TileOp tileOp = coreOp.getTileOp();
    std::optional<int64_t> maybeCol = getConstantIntValue(tileOp.getCol());
    std::optional<int64_t> maybeRow = getConstantIntValue(tileOp.getRow());
    if (!maybeCol || !maybeCol) {
      coreOp.emitOpError() << "has non-constant tile location";
      return WalkResult::interrupt();
    }
    int64_t col = maybeCol.value();
    int64_t row = maybeRow.value();
    if (col < firstCol) firstCol = col;
    if (col > lastCol) lastCol = col;
    if (row < firstRow) firstRow = row;
    if (row > lastRow) lastRow = row;
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return failure();
  return CoreRegionInfo(
      {firstCol, lastCol - firstCol + 1, firstRow, lastRow - firstRow + 1});
}

/// Return a vector of the parent operations that are of type 'OpTy', including
/// this op if it has type 'OpTy'
template <typename OpTy>
SmallVector<OpTy> getInclusiveParentsOfType(Operation *op) {
  SmallVector<OpTy> res;
  auto *current = op;
  do {
    if (auto typedParent = dyn_cast<OpTy>(current)) {
      res.push_back(typedParent);
    }
  } while ((current = current->getParentOp()));
  return res;
}

template <typename T>
FailureOr<AMDAIE::LogicalObjectFifoFromBuffersOp> getLogicalObjFifoOperatedOn(
    T op) {
  auto copyOp =
      dyn_cast_if_present<CopyOpInterface>(op.getDma().getDefiningOp());
  if (!copyOp)
    return op.emitOpError() << "should operate on a copy-like operation";
  auto logicalObjFifo =
      op.getPort() == LogicalObjectFifoPort::Consume
          ? dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromBuffersOp>(
                copyOp.getTarget().getDefiningOp())
          : dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromBuffersOp>(
                copyOp.getSource().getDefiningOp());
  if (!logicalObjFifo) {
    return copyOp.emitOpError()
           << "should operate on an `amdaie.logicalobjectfifo.from_buffers` op";
  }
  return logicalObjFifo;
}

}  // namespace mlir::iree_compiler::AMDAIE

#endif
