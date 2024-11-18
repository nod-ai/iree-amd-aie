// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_AMDAIEOPUTILS_H_
#define IREE_AMD_AIE_TRANSFORMS_AMDAIEOPUTILS_H_

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"

namespace mlir::iree_compiler::AMDAIE {

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
