// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the conversion of `scf.forall` within `amdaie.controlcode`
// ops into `scf.for` operations. This can help discover new control code
// optimization opportunities.
//
//===----------------------------------------------------------------------===//

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"

#define DEBUG_TYPE "iree-amdaie-controlcode-forall-to-for"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Converts `scf.forall` operations found within the provided op into nested
/// `scf.for` operations.
LogicalResult forallToFor(RewriterBase &rewriter, Operation *op) {
  WalkResult res = op->walk([&](scf::ForallOp forallOp) {
    rewriter.setInsertionPoint(forallOp);
    if (succeeded(forallOp.promoteIfSingleIteration(rewriter))) {
      return WalkResult::advance();
    }
    if (failed(scf::forallToForLoop(rewriter, forallOp))) {
      forallOp.emitOpError()
          << "was not transformed from `scf.forall` to `scf.for`";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return failure();
  return success();
}

class AMDAIEControlCodeForallToForPass
    : public impl::AMDAIEControlCodeForallToForBase<
          AMDAIEControlCodeForallToForPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, affine::AffineDialect>();
  }
  void runOnOperation() override;
};

void AMDAIEControlCodeForallToForPass::runOnOperation() {
  Operation *parentOp = getOperation();
  IRRewriter rewriter(parentOp->getContext());
  parentOp->walk([&](AMDAIE::ControlCodeOp controlCodeOp) {
    if (failed(forallToFor(rewriter, controlCodeOp.getOperation()))) {
      return signalPassFailure();
    }
    // Make sure to hoist `affine.apply` ops out of the innermost `scf.for` ops
    // if applicable.
    controlCodeOp->walk([&](affine::AffineApplyOp applyOp) {
      (void)hoistForAffineApplyOp(rewriter, applyOp);
    });
  });
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEControlCodeForallToForPass() {
  return std::make_unique<AMDAIEControlCodeForallToForPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
