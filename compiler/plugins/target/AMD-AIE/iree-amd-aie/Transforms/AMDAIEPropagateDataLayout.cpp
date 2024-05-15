// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-propagate-data-layout"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Forces `outs` operands of linalg operations to use `tensor.empty` if the
/// value of the `outs` operand is not used within the op.  This is only
/// implemented for `linalg.generic` operations for now, but should hold for all
/// linalg structured ops.
struct RemoveOutsDependency : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.startOpModification(op);
    bool modifiedOutput = false;
    Location loc = op.getLoc();
    for (OpOperand &opOperand : op.getDpsInitsMutable()) {
      if (!op.payloadUsesValueFromOperand(&opOperand)) {
        Value operandVal = opOperand.get();
        auto operandType = dyn_cast<RankedTensorType>(operandVal.getType());
        if (!operandType) continue;

        // If outs is sparse, leave it to the sparsifier.
        if (sparse_tensor::getSparseTensorEncoding(operandVal.getType()))
          continue;

        // If outs is already an `empty` operation, nothing to do.
        auto definingOp = operandVal.getDefiningOp<tensor::EmptyOp>();
        if (definingOp) continue;
        modifiedOutput = true;
        SmallVector<OpFoldResult> mixedSizes =
            tensor::getMixedSizes(rewriter, loc, operandVal);
        Value emptyTensor = rewriter.create<tensor::EmptyOp>(
            loc, mixedSizes, operandType.getElementType());
        op->setOperand(opOperand.getOperandNumber(), emptyTensor);
      }
    }
    if (!modifiedOutput) {
      rewriter.cancelOpModification(op);
      return failure();
    }
    rewriter.finalizeOpModification(op);
    return success();
  }
};

void populateElementwiseOpsFusionPatterns(RewritePatternSet &patterns) {
  patterns.add<RemoveOutsDependency>(patterns.getContext());
}

class AMDAIEPropagateDataLayoutPass
    : public impl::AMDAIEPropagateDataLayoutBase<
          AMDAIEPropagateDataLayoutPass> {
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
  populateElementwiseOpsFusionPatterns(patterns);

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEPropagateDataLayoutPass() {
  return std::make_unique<AMDAIEPropagateDataLayoutPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
