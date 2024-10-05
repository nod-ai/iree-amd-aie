// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/Transforms/AMDAIEDmaUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-canonicalize-doubly-strided-dma"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Recognize linear accesses across multiple DMA access dimensions and fold
/// them.
struct FoldDmaOpLinearDims
    : public OpInterfaceRewritePattern<AMDAIE::DoublyStridedOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(AMDAIE::DoublyStridedOpInterface op,
                                PatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    SmallVector<OpFoldResult> sourceOffsets = op.getSourceMixedOffsets();
    SmallVector<OpFoldResult> sourceSizes = op.getSourceMixedSizes();
    SmallVector<OpFoldResult> sourceStrides = op.getSourceMixedStrides();
    SmallVector<OpFoldResult> targetOffsets = op.getTargetMixedOffsets();
    SmallVector<OpFoldResult> targetSizes = op.getTargetMixedSizes();
    SmallVector<OpFoldResult> targetStrides = op.getTargetMixedStrides();
    SmallVector<OpFoldResult> newSourceOffsets, newSourceSizes,
        newSourceStrides, newTargetOffsets, newTargetSizes, newTargetStrides;
    LogicalResult sourceRes = foldLinearDims(
        op.getContext(), sourceOffsets, sourceSizes, sourceStrides,
        newSourceOffsets, newSourceSizes, newSourceStrides);
    LogicalResult targetRes = foldLinearDims(
        op.getContext(), targetOffsets, targetSizes, targetStrides,
        newTargetOffsets, newTargetSizes, newTargetStrides);
    if (failed(sourceRes) && failed(targetRes)) {
      return failure();
    }

    rewriter.setInsertionPointAfter(op);
    auto newDoublyStridedOp = op.createDoublyStridedOp(
        rewriter, newTargetOffsets, newTargetSizes, newTargetStrides,
        newSourceOffsets, newSourceSizes, newSourceStrides);
    rewriter.replaceOp(op, newDoublyStridedOp.getOperation());
    return success();
  }
};

/// Fold single dimension linear accesses and make them implicit.
struct FoldDmaOpSingleDims
    : public OpInterfaceRewritePattern<AMDAIE::DoublyStridedOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(AMDAIE::DoublyStridedOpInterface op,
                                PatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    SmallVector<OpFoldResult> sourceOffsets = op.getSourceMixedOffsets();
    SmallVector<OpFoldResult> sourceSizes = op.getSourceMixedSizes();
    SmallVector<OpFoldResult> sourceStrides = op.getSourceMixedStrides();
    SmallVector<OpFoldResult> targetOffsets = op.getTargetMixedOffsets();
    SmallVector<OpFoldResult> targetSizes = op.getTargetMixedSizes();
    SmallVector<OpFoldResult> targetStrides = op.getTargetMixedStrides();
    LogicalResult sourceRes =
        foldSingleDim(sourceOffsets, sourceSizes, sourceStrides);
    LogicalResult targetRes =
        foldSingleDim(targetOffsets, targetSizes, targetStrides);
    if (failed(sourceRes) && failed(targetRes)) {
      return failure();
    }

    rewriter.setInsertionPointAfter(op);
    auto newDoublyStridedOp = op.createDoublyStridedOp(
        rewriter, targetOffsets, targetSizes, targetStrides, sourceOffsets,
        sourceSizes, sourceStrides);
    rewriter.replaceOp(op, newDoublyStridedOp.getOperation());
    return success();
  }
};

/// Fold unit dimensions within a strided access pattern.
struct FoldDmaOpUnitDims
    : public OpInterfaceRewritePattern<AMDAIE::DoublyStridedOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(AMDAIE::DoublyStridedOpInterface op,
                                PatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    SmallVector<OpFoldResult> sourceOffsets = op.getSourceMixedOffsets();
    SmallVector<OpFoldResult> sourceSizes = op.getSourceMixedSizes();
    SmallVector<OpFoldResult> sourceStrides = op.getSourceMixedStrides();
    SmallVector<OpFoldResult> targetOffsets = op.getTargetMixedOffsets();
    SmallVector<OpFoldResult> targetSizes = op.getTargetMixedSizes();
    SmallVector<OpFoldResult> targetStrides = op.getTargetMixedStrides();
    SmallVector<OpFoldResult> newSourceOffsets, newSourceSizes,
        newSourceStrides, newTargetOffsets, newTargetSizes, newTargetStrides;
    LogicalResult sourceRes =
        foldUnitDims(sourceOffsets, sourceSizes, sourceStrides,
                     newSourceOffsets, newSourceSizes, newSourceStrides);
    LogicalResult targetRes =
        foldUnitDims(targetOffsets, targetSizes, targetStrides,
                     newTargetOffsets, newTargetSizes, newTargetStrides);
    if (failed(sourceRes) && failed(targetRes)) {
      return failure();
    }

    rewriter.setInsertionPointAfter(op);
    auto newDoublyStridedOp = op.createDoublyStridedOp(
        rewriter, newTargetOffsets, newTargetSizes, newTargetStrides,
        newSourceOffsets, newSourceSizes, newSourceStrides);
    rewriter.replaceOp(op, newDoublyStridedOp.getOperation());
    return success();
  }
};

class AMDAIECanonicalizeDoublyStridedOpPass
    : public impl::AMDAIECanonicalizeDoublyStridedOpBase<
          AMDAIECanonicalizeDoublyStridedOpPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIECanonicalizeDoublyStridedOpPass() = default;
  AMDAIECanonicalizeDoublyStridedOpPass(
      const AMDAIECanonicalizeDoublyStridedOpPass &pass){};
  AMDAIECanonicalizeDoublyStridedOpPass(
      const AMDAIECanonicalizeDoublyStridedOpOptions &options)
      : AMDAIECanonicalizeDoublyStridedOpBase(options) {}
  void runOnOperation() override;
};

void AMDAIECanonicalizeDoublyStridedOpPass::runOnOperation() {
  Operation *parentOp = getOperation();
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);

  populateCanonicalizeDoublyStridedOpPatterns(patterns, foldSingleDims);
  if (failed(applyPatternsAndFoldGreedily(parentOp, std::move(patterns)))) {
    parentOp->emitOpError(
        "failed to canonicalize doubly strided DMA operations");
    return signalPassFailure();
  }
}

}  // namespace

void populateCanonicalizeDoublyStridedOpPatterns(RewritePatternSet &patterns,
                                                 bool foldSingleDims) {
  patterns.add<FoldDmaOpUnitDims>(patterns.getContext());
  patterns.add<FoldDmaOpLinearDims>(patterns.getContext());
  if (foldSingleDims) {
    patterns.add<FoldDmaOpSingleDims>(patterns.getContext());
  }
}

std::unique_ptr<Pass> createAMDAIECanonicalizeDoublyStridedOpPass(
    AMDAIECanonicalizeDoublyStridedOpOptions options) {
  return std::make_unique<AMDAIECanonicalizeDoublyStridedOpPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
