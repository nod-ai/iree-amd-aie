// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_AMDAIELOGICALOBJFIFOSPLITTINGUTILS_H_
#define IREE_AMD_AIE_TRANSFORMS_AMDAIELOGICALOBJFIFOSPLITTINGUTILS_H_

#include "iree-amd-aie/IR/AMDAIEOps.h"

namespace mlir::iree_compiler::AMDAIE {

/// Utility to help fetch those input DmaCpyNd Ops which needs to be split.
SmallVector<AMDAIE::DmaCpyNdOp> fetchDmaCpyNdOpsToSplitOrCombine(Operation *op);

/// Utility to split logicalobjectfifos given a vector of L2->L1 dma ops.
LogicalResult splitLogicalObjectFifos(
    IRRewriter &rewriter, SmallVector<AMDAIE::DmaCpyNdOp> &l2ToL1DmaOps,
    MLIRContext *context);

LogicalResult splitObjFifo(IRRewriter &rewriter,
                           AMDAIE::LogicalObjectFifoFromMemrefOp op,
                           size_t splitDim = 0, int64_t splitFactor = -1);

LogicalResult splitDoublyStridedOp(IRRewriter &rewriter,
                                   AMDAIE::DoublyStridedOpInterface op,
                                   size_t splitDim = 0, int64_t splitFactor = -1);

}  // namespace mlir::iree_compiler::AMDAIE

#endif
