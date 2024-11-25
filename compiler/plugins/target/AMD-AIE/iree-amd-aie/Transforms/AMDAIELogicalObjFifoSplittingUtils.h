// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_AMDAIELOGICALOBJFIFOSPLITTINGUTILS_H_
#define IREE_AMD_AIE_TRANSFORMS_AMDAIELOGICALOBJFIFOSPLITTINGUTILS_H_

#include "iree-amd-aie/IR/AMDAIEOps.h"

namespace mlir::iree_compiler::AMDAIE {

/// Utility to split logicalobjectfifos given a vector of L2->L1 dma ops.
LogicalResult splitLogicalObjectFifoForElementwiseOp(
    IRRewriter &rewriter, SmallVector<AMDAIE::DmaCpyNdOp> &l2ToL1DmaOps,
    MLIRContext *context);

/// Split a logical objectFifo on the provided split dimension with the
/// specified splitting factor. If no split factor is provided, the logical
/// objectFifo will be split on the size of the dimension being split.
LogicalResult splitLogicalObjectFifo(
    IRRewriter &rewriter, AMDAIE::LogicalObjectFifoFromMemrefOp op,
    size_t splitDim = 0, std::optional<size_t> splitFactor = std::nullopt);

/// Split doubly strided operations on a source and target split dimension with
/// the provided split factor. If no split factor is provided, the doubly
/// strided operation will be split on the size of the dimension being split.
LogicalResult splitDoublyStridedOp(
    IRRewriter &rewriter, AMDAIE::DoublyStridedOpInterface op,
    size_t sourceSplitDim = 0, size_t targetSplitDim = 0,
    std::optional<size_t> splitFactor = std::nullopt);

}  // namespace mlir::iree_compiler::AMDAIE

#endif
