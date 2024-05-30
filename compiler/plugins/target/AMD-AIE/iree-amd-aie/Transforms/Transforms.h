// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_AMDAIETRANSFORMS_H_
#define IREE_AMD_AIE_TRANSFORMS_AMDAIETRANSFORMS_H_

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::iree_compiler::AMDAIE {

/// Unroll the loops within the control code regions.
LogicalResult controlCodeLoopUnroll(RewriterBase &rewriter,
                                    AMDAIE::ControlCodeOp controlCodeOp);

/// Utility to create explicit logical objectfifo link operations, linking input
/// and output copy operations. This is useful for conversion to the AIE dialect
/// as that one relies on explicit link operations.
LogicalResult createLogicalObjectFifoLink(
    RewriterBase &rewriter,
    AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo);

/// Hoist an affine apply op on a scf.for op's induction variable
/// TODO(jornt): Can we generalize this to go into upstream?
LogicalResult hoistForAffineApplyOp(RewriterBase &rewriter,
                                    affine::AffineApplyOp applyOp);

/// Normalize the loop bounds of the `scf.for` operation to lowerbound == 0 and
/// step == 1.
LogicalResult normalizeLoopBounds(RewriterBase &rewriter, scf::ForOp forOp);

/// Normalize the loop bounds of the `scf.forall` operation to lowerbound == 0
/// and step == 1.
LogicalResult normalizeLoopBounds(RewriterBase &rewriter,
                                  scf::ForallOp forallOp);

}  // namespace mlir::iree_compiler::AMDAIE

#endif
