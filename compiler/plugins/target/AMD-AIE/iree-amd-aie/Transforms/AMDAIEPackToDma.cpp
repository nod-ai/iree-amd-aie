// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "air/Dialect/AIR/AIRDialect.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-propagate-data-layout"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Pattern to rewriter scf.forall -> scf.parallel after bufferization.
class LinalgExtPackToAirDmaMemcpyNd : public OpRewritePattern<IREE::LinalgExt::PackOp> {
  using OpRewritePattern<IREE::LinalgExt::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::LinalgExt::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    return failure();
  }
};

class AMDAIEPackToDmaPass
    : public impl::AMDAIEPackToDmaBase<
          AMDAIEPackToDmaPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect>();
  }

  AMDAIEPackToDmaPass() = default;
  AMDAIEPackToDmaPass(const AMDAIEPackToDmaPass &pass){};
  void runOnOperation() override;
};

void AMDAIEPackToDmaPass::runOnOperation() {
    MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<LinalgExtPackToAirDmaMemcpyNd>(context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEPackToDmaPass() {
  return std::make_unique<AMDAIEPackToDmaPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
