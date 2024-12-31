// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#define DEBUG_TYPE "iree-cleanup"

namespace mlir::iree_compiler::AMDAIE {

namespace {

static void loopIndependentCodeMotion(Operation *funcOp, IRRewriter &rewriter) {
  // This assumes LICM never removes operations so we don't need tracking.
  // TODO: confirm / revisit this assumption and plumb a rewriter through
  // upstream moveLoopInvariantCode if necessary.
  funcOp->walk([](LoopLikeOpInterface loopLike) {
    // Do not hoist from scf.forall ops. These capture isolated computations
    // that will be mapped to a certain level in the GPU hierarchy (e.g.,
    // GPU blocks), so hoisting is not desired.
    if (!isa<scf::ForallOp>(loopLike.getOperation()))
      moveLoopInvariantCode(loopLike);
  });
  // For now, put single loop promotion as part of licm. Underlying
  // implementations perform splice operations which shouldn't need
  // tracking.
  // TODO: confirm / revisit this assumption and plumb a rewriter through
  // upstream moveLoopInvariantCode if necessary.
  funcOp->walk([&](Operation *op) {
    (void)llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case<affine::AffineForOp, scf::ForOp>(
            [&](auto loop) { return loop.promoteIfSingleIteration(rewriter); })
        .Default([](Operation *) { return success(); });
  });
}

static void populateCleanupPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
  linalg::FillOp::getCanonicalizationPatterns(patterns, context);
  // Pulling in upstream scf.for and affine.min canonicalization patterns.
  // They work on tiled (but not distributed) loops.
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  tensor::populateFoldTensorEmptyPatterns(patterns);
}

class AMDAIECleanupPass : public impl::AMDAIECleanupBase<AMDAIECleanupPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect>();
  }

  AMDAIECleanupPass() = default;
  AMDAIECleanupPass(const AMDAIECleanupPass &pass){};

  void runOnOperation() override;
};

void AMDAIECleanupPass::runOnOperation() {
  MLIRContext *context = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();

  RewritePatternSet patterns(context);
  populateCleanupPatterns(patterns);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }
  {
    IRRewriter rewriter(context);
    loopIndependentCodeMotion(funcOp, rewriter);
  }
}
}  // namespace

std::unique_ptr<Pass> createAMDAIECleanupPass() {
  return std::make_unique<AMDAIECleanupPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
