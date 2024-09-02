// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_AMDAIELOGICALOBJFIFOSPLITTINGUTILS_H_
#define IREE_AMD_AIE_TRANSFORMS_AMDAIELOGICALOBJFIFOSPLITTINGUTILS_H_

#include "iree-amd-aie/IR/AMDAIEOps.h"

namespace mlir::iree_compiler::AMDAIE {

/// A struct utility to encapsulate all the data required to perform splitting
/// of logicalobjectfifos.
struct SplittingLogicalObjectFifoData {
  SmallVector<AMDAIE::DmaCpyNdOp> l2ToL1DmaOps;
  SmallVector<size_t> splitDimsForL2;
  SmallVector<size_t> nonSplitDimsForL2;
  AMDAIE::DmaCpyNdOp l3ToL2DmaOp;
};

/// Utility to check whether splitting of logicalobjectfifos can be performed.
/// If yes, it populates the struct `SplittingLogicalObjectFifoData` with the
/// data required to perform the actual splitting.
LogicalResult checkWhetherSplitIsPossible(
    SplittingLogicalObjectFifoData &splittingLogicalObjectFifoData);

/// Utility to split logicalobjectfifos given a struct
/// `SplittingLogicalObjectFifoData` which contains all the required data to
/// perform the splitting.
LogicalResult splitLogicalObjectFifos(
    IRRewriter &rewriter,
    const SplittingLogicalObjectFifoData &splittingLogicalObjectFifoData,
    MLIRContext *context);

}  // namespace mlir::iree_compiler::AMDAIE

#endif
