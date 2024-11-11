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

/// Split L2 space input and output logical objectFifos.
LogicalResult splitLogicalObjectFifo(IRRewriter &rewriter,
                                     AMDAIE::LogicalObjectFifoFromMemrefOp op);

/// Split DmaCpyNd ops between L2 and L3 memory spaces.
LogicalResult splitDoublyStridedOp(IRRewriter &rewriter, AMDAIE::DmaCpyNdOp op);

}  // namespace mlir::iree_compiler::AMDAIE

#endif
