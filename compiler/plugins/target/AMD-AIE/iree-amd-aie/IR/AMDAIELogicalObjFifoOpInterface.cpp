// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIELogicalObjFifoOpInterface.h"

/// Include the definitions of the logical-objFifo-like interfaces.
#include "iree-amd-aie/IR/AMDAIELogicalObjFifoOpInterface.cpp.inc"

namespace mlir::iree_compiler::AMDAIE {

namespace detail {

SmallVector<mlir::CopyOpInterface> getCopyLikeConsumers(
    LogicalObjFifoOpInterface op) {
  SmallVector<mlir::CopyOpInterface> copyLikOps;
  for (Operation *userOp : op->getUsers()) {
    if (auto copyOp = dyn_cast<CopyOpInterface>(userOp);
        dyn_cast_if_present<LogicalObjFifoOpInterface>(
            copyOp.getSource().getDefiningOp()) == op) {
      copyLikOps.push_back(copyOp);
    }
  }
  return copyLikOps;
}

SmallVector<mlir::CopyOpInterface> getCopyLikeProducers(
    LogicalObjFifoOpInterface op) {
  SmallVector<mlir::CopyOpInterface> copyLikOps;
  for (Operation *userOp : op->getUsers()) {
    if (auto copyOp = dyn_cast<CopyOpInterface>(userOp);
        dyn_cast_if_present<LogicalObjFifoOpInterface>(
            copyOp.getTarget().getDefiningOp()) == op) {
      copyLikOps.push_back(copyOp);
    }
  }
  return copyLikOps;
}

}  // namespace detail

}  // namespace mlir::iree_compiler::AMDAIE
