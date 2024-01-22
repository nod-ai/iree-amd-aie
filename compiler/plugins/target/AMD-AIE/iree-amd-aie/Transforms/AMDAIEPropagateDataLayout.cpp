// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/PassDetail.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-propagate-pad"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIEPropagateDataLayoutPass
    : public AMDAIEPropagateDataLayoutBase<AMDAIEPropagateDataLayoutPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect>();
  }

  AMDAIEPropagateDataLayoutPass() = default;
  AMDAIEPropagateDataLayoutPass(const AMDAIEPropagateDataLayoutPass &pass){};
  void runOnOperation() override;
};

void AMDAIEPropagateDataLayoutPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  linalg::populateDataLayoutPropagationPatterns(
      patterns, [](Operation *op) { return true; });
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEPropagateDataLayoutPass() {
  return std::make_unique<AMDAIEPropagateDataLayoutPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
