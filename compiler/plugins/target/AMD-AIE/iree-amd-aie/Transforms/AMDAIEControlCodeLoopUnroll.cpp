// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Support/MathExtras.h"

#define DEBUG_TYPE "iree-amdaie-controlcode-loop-unroll"

namespace mlir::iree_compiler::AMDAIE {

/// Unroll all scf.forall and scf.for loops inside the control code region.
LogicalResult controlCodeLoopUnroll(RewriterBase &rewriter,
                                    AMDAIE::ControlCodeOp controlCodeOp) {
  // Convert all scf.forall's in the control code region to scf.for.
  WalkResult forallRes = controlCodeOp.walk([&](scf::ForallOp forallOp) {
    // TODO(avarma): Remove this after upstream fix.
    rewriter.setInsertionPoint(forallOp);
    if (succeeded(forallOp.promoteIfSingleIteration(rewriter))) {
      return WalkResult::advance();
    }
    if (failed(scf::forallToForLoop(rewriter, forallOp))) {
      forallOp.emitOpError() << "failed to transform scf.forall to scf.for";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (forallRes.wasInterrupted()) return failure();

  // Unroll all scf.for loops in the control code region.
  WalkResult res =
      controlCodeOp.walk([&](scf::ForOp forOp) {
        // TODO(avarma): Remove this after upstream fix.
        rewriter.setInsertionPoint(forOp);
        if (succeeded(forOp.promoteIfSingleIteration(rewriter))) {
          return WalkResult::advance();
        }
        std::optional<int64_t> lbCstOp =
            getConstantIntValue(forOp.getLowerBound());
        std::optional<int64_t> ubCstOp =
            getConstantIntValue(forOp.getUpperBound());
        std::optional<int64_t> stepCstOp = getConstantIntValue(forOp.getStep());
        if (lbCstOp && ubCstOp && stepCstOp) {
          int64_t lbInt = lbCstOp.value();
          int64_t ubInt = ubCstOp.value();
          int64_t stepInt = stepCstOp.value();
          int64_t tripCount = mlir::ceilDiv(ubInt - lbInt, stepInt);
          if (failed(loopUnrollByFactor(forOp, tripCount))) {
            forOp.emitOpError() << "failed to unroll scf.for";
            return WalkResult::interrupt();
          }
        } else {
          forOp.emitOpError()
              << "failed to unroll scf.for with dynamic bounds or step size";
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  if (res.wasInterrupted()) return failure();
  return success();
}

namespace {

struct AMDAIEControlCodeLoopUnrollPass
    : public impl::AMDAIEControlCodeLoopUnrollBase<
          AMDAIEControlCodeLoopUnrollPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  void runOnOperation() override {
    Operation *parentOp = getOperation();
    IRRewriter rewriter(parentOp->getContext());

    WalkResult res = parentOp->walk([&](AMDAIE::ControlCodeOp controlCodeOp) {
      if (failed(controlCodeLoopUnroll(rewriter, controlCodeOp))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (res.wasInterrupted()) return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIEControlCodeLoopUnrollPass() {
  return std::make_unique<AMDAIEControlCodeLoopUnrollPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
