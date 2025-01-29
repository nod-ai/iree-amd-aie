// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_AMDAIELOGICALOBJFIFOSPLITTINGUTILS_H_
#define IREE_AMD_AIE_TRANSFORMS_AMDAIELOGICALOBJFIFOSPLITTINGUTILS_H_

#include "iree-amd-aie/IR/AMDAIEOps.h"

namespace mlir::iree_compiler::AMDAIE {

/// Utility to get the `DmaCpyNdOp` producers and consumers of a given
/// objectFifo op.
LogicalResult getDmaCpyNdOpProducersAndConsumers(
    AMDAIE::LogicalObjectFifoFromMemrefOp op,
    SmallVector<AMDAIE::DmaCpyNdOp> &producers,
    SmallVector<AMDAIE::DmaCpyNdOp> &consumers);

/// Utility to return the indices of the dimensions with stride equal to the
/// expected stride and with dynamic or non-zero offsets.
SmallVector<size_t> getStrideIndicesWithDynamicOrNonZeroOffset(
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> strides,
    size_t expectedStride);

/// Utility to split logicalobjectfifos given a vector of L2->L1 dma ops.
LogicalResult splitLogicalObjectFifoForElementwiseOp(
    IRRewriter &rewriter, SmallVector<AMDAIE::DmaCpyNdOp> &l2ToL1DmaOps,
    MLIRContext *context);

/// Split a logical objectFifo on the provided split dimension with the
/// specified splitting factor. If no split factor is provided, the logical
/// objectFifo will be split on the size of the dimension being split.
LogicalResult splitLogicalObjectFifo(
    IRRewriter &rewriter, AMDAIE::LogicalObjectFifoFromMemrefOp op,
    size_t splitDim = 0, std::optional<size_t> splitFactor = std::nullopt,
    int64_t splitStride = 1);

/// Split doubly strided operations on a source and target split dimension with
/// the provided split factor.
LogicalResult splitDoublyStridedOp(
    IRRewriter &rewriter, AMDAIE::DoublyStridedOpInterface op,
    size_t sourceSplitDim = 0, size_t targetSplitDim = 0, int64_t splitFactor,
    int64_t sourceSplitStride = 1, int64_t targetSplitStride = 1);

}  // namespace mlir::iree_compiler::AMDAIE

#endif
