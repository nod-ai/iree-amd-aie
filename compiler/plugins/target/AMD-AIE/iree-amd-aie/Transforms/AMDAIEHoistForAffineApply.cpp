// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "iree-amdaie-hoist-for-affine-apply"

namespace mlir::iree_compiler::AMDAIE {

/// Hoist an affine.apply op on a scf.for induction variable into the beginning
/// of that scf.for's body.
LogicalResult hoistForAffineApplyOp(RewriterBase &rewriter,
                                    affine::AffineApplyOp applyOp) {
  // Only handle affine apply with one operand.
  if (applyOp.getNumOperands() != 1) {
    return failure();
  }

  // Check whether the parent op is a for op and the map operand is already the
  // induction variable of the parent for op.
  Value operand = applyOp.getMapOperands()[0];
  if (scf::ForOp forOp = dyn_cast<scf::ForOp>(applyOp->getParentOp());
      forOp == scf::getForInductionVarOwner(operand)) {
    return failure();
  }

  // Rewrite if operand is an induction variable.
  if (scf::ForOp forOp = scf::getForInductionVarOwner(operand)) {
    rewriter.moveOpBefore(applyOp, forOp.getBody(), forOp.getBody()->begin());
    return success();
  }
  return failure();
}

namespace {
struct AMDAIEHoistForLoopAffineApply
    : public impl::AMDAIEHoistForLoopAffineApplyBase<
          AMDAIEHoistForLoopAffineApply> {
  void runOnOperation() override {
    Operation *parentOp = getOperation();
    IRRewriter rewriter(parentOp->getContext());

    parentOp->walk([&](affine::AffineApplyOp applyOp) {
      (void)hoistForAffineApplyOp(rewriter, applyOp);
    });
  }
};
}  // namespace

std::unique_ptr<Pass> createAMDAIEHoistForLoopAffineApplyPass() {
  return std::make_unique<AMDAIEHoistForLoopAffineApply>();
}

}  // namespace mlir::iree_compiler::AMDAIE
