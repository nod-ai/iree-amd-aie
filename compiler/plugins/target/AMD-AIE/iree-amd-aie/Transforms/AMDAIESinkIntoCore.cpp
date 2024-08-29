// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aie/AIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#define DEBUG_TYPE "iree-amdaie-sink-into-core"

namespace mlir::iree_compiler::AMDAIE {

namespace {

bool sinkInto(AMDAIE::CoreOp coreOp, PatternRewriter &rewriter) {
  // Record if any ops are sunk into the core during this iteration.
  bool changed = false;

  // Collect all ops in the amdaie.core op
  SmallVector<Operation *> opsInCore;
  coreOp->walk([&](Operation *op) {
    if (op == coreOp) return WalkResult::advance();
    opsInCore.push_back(op);
    return WalkResult::advance();
  });

  for (auto opInCore : opsInCore) {
    for (Value operand : opInCore->getOperands()) {
      if (!operand || !operand.getDefiningOp()) continue;
      Operation *dependencyOp = operand.getDefiningOp();

      // Skip if the dependency is already in the core.
      if (coreOp->isAncestor(dependencyOp)) {
        continue;
      }

      // Ops in the amdaie dialect are probably related to data movement
      // and should not be sunk into the core. This might need adjustment
      // later.
      if (dependencyOp->getDialect()->getNamespace() ==
          AMDAIE::AMDAIEDialect::getDialectNamespace()) {
        continue;
      }

      // Create a clone of the dependency op in the core region.
      Region &r = coreOp->getRegion(0);
      assert(r.getBlocks().size() == 1 && "expected single block region");
      rewriter.setInsertionPointToStart(&r.front());
      Operation *sunkOp = rewriter.clone(*dependencyOp);

      // Replace uses of the dependency op inside the core.
      dependencyOp->replaceUsesWithIf(sunkOp, [&](OpOperand &use) {
        return coreOp->isAncestor(use.getOwner());
      });
      changed = true;
    }
  }
  return changed;
}

class SinkingPattern : public OpRewritePattern<AMDAIE::CoreOp> {
 public:
  using OpRewritePattern<AMDAIE::CoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AMDAIE::CoreOp coreOp,
                                PatternRewriter &rewriter) const override {
    return success(sinkInto(coreOp, rewriter));
  }
};

class AMDAIESinkIntoCorePass
    : public impl::AMDAIESinkIntoCoreBase<AMDAIESinkIntoCorePass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect,
                    xilinx::AIE::AIEDialect, AMDAIE::AMDAIEDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<SinkingPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIESinkIntoCorePass() {
  return std::make_unique<AMDAIESinkIntoCorePass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
