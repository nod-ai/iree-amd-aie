// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "air/Dialect/AIR/AIRDialect.h"
#include "iree-amd-aie/Transforms/AMDAIEDmaUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-pack-to-dma"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class FoldUnitDimsInDma : public OpRewritePattern<xilinx::air::DmaMemcpyNdOp> {
 public:
  using OpRewritePattern<xilinx::air::DmaMemcpyNdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xilinx::air::DmaMemcpyNdOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    SmallVector<OpFoldResult> srcOffsets =
        getAsOpFoldResult(op.getSrcOffsets());
    SmallVector<OpFoldResult> dstOffsets =
        getAsOpFoldResult(op.getDstOffsets());

    SmallVector<OpFoldResult> srcStrides =
        getAsOpFoldResult(op.getSrcStrides());
    SmallVector<OpFoldResult> dstStrides =
        getAsOpFoldResult(op.getDstStrides());

    SmallVector<OpFoldResult> srcSizes = getAsOpFoldResult(op.getSrcSizes());
    SmallVector<OpFoldResult> dstSizes = getAsOpFoldResult(op.getDstSizes());

    SmallVector<OpFoldResult> newSrcOffsets, newDstOffsets, newSrcStrides,
        newDstStrides, newSrcSizes, newDstSizes;

    // We do not make any assumptions when all offsets are not
    // specified and dont change the op in that case.
    if (srcStrides.size() != srcOffsets.size() ||
        dstStrides.size() != dstOffsets.size()) {
      return rewriter.notifyMatchFailure(
          op, "offset dimensions dont match stride dimensions");
    }
    // Fold source dims.
    LogicalResult foldableUnitDimsFoundInSrc =
        foldUnitDims(srcOffsets, srcSizes, srcStrides, newSrcOffsets,
                     newSrcSizes, newSrcStrides);
    // Fold destination dims.
    LogicalResult foldableUnitDimsFoundInDst =
        foldUnitDims(dstOffsets, dstSizes, dstStrides, newDstOffsets,
                     newDstSizes, newDstStrides);
    if (failed(foldableUnitDimsFoundInSrc) &&
        failed(foldableUnitDimsFoundInDst)) {
      return rewriter.notifyMatchFailure(op, "no foldable unit dims found");
    }

    rewriter.replaceOpWithNewOp<xilinx::air::DmaMemcpyNdOp>(
        op, SmallVector<Type, 1>{}, op.getAsyncDependencies(), op.getDst(),
        getValueOrCreateConstantIndexOp(rewriter, loc, newDstOffsets),
        getValueOrCreateConstantIndexOp(rewriter, loc, newDstSizes),
        getValueOrCreateConstantIndexOp(rewriter, loc, newDstStrides),
        op.getSrc(),
        getValueOrCreateConstantIndexOp(rewriter, loc, newSrcOffsets),
        getValueOrCreateConstantIndexOp(rewriter, loc, newSrcSizes),
        getValueOrCreateConstantIndexOp(rewriter, loc, newSrcStrides));
    return success();
  }
};

class FoldLinearDimsInDma
    : public OpRewritePattern<xilinx::air::DmaMemcpyNdOp> {
 public:
  using OpRewritePattern<xilinx::air::DmaMemcpyNdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xilinx::air::DmaMemcpyNdOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    SmallVector<OpFoldResult> srcOffsets =
        getAsOpFoldResult(op.getSrcOffsets());
    SmallVector<OpFoldResult> dstOffsets =
        getAsOpFoldResult(op.getDstOffsets());

    SmallVector<OpFoldResult> srcStrides =
        getAsOpFoldResult(op.getSrcStrides());
    SmallVector<OpFoldResult> dstStrides =
        getAsOpFoldResult(op.getDstStrides());

    SmallVector<OpFoldResult> srcSizes = getAsOpFoldResult(op.getSrcSizes());
    SmallVector<OpFoldResult> dstSizes = getAsOpFoldResult(op.getDstSizes());

    SmallVector<OpFoldResult> newSrcOffsets, newDstOffsets, newSrcStrides,
        newDstStrides, newSrcSizes, newDstSizes;

    // We do not make any assumptions when all offsets are not
    // specified and dont change the op in that case.
    if (srcStrides.size() != srcOffsets.size() ||
        dstStrides.size() != dstOffsets.size()) {
      return rewriter.notifyMatchFailure(
          op, "offset dimensions dont match stride dimensions");
    }

    // Fold source dims.
    LogicalResult foldableLinearDimsFoundInSrc =
        foldLinearDims(op.getContext(), srcOffsets, srcSizes, srcStrides,
                       newSrcOffsets, newSrcSizes, newSrcStrides);
    // Fold destination dims.
    LogicalResult foldableLinearDimsFoundInDst =
        foldLinearDims(op.getContext(), dstOffsets, dstSizes, dstStrides,
                       newDstOffsets, newDstSizes, newDstStrides);
    if (failed(foldableLinearDimsFoundInSrc) &&
        failed(foldableLinearDimsFoundInDst)) {
      return rewriter.notifyMatchFailure(op, "no foldable linear dims found");
    }

    rewriter.replaceOpWithNewOp<xilinx::air::DmaMemcpyNdOp>(
        op, SmallVector<Type, 1>{}, op.getAsyncDependencies(), op.getDst(),
        getValueOrCreateConstantIndexOp(rewriter, loc, newDstOffsets),
        getValueOrCreateConstantIndexOp(rewriter, loc, newDstSizes),
        getValueOrCreateConstantIndexOp(rewriter, loc, newDstStrides),
        op.getSrc(),
        getValueOrCreateConstantIndexOp(rewriter, loc, newSrcOffsets),
        getValueOrCreateConstantIndexOp(rewriter, loc, newSrcSizes),
        getValueOrCreateConstantIndexOp(rewriter, loc, newSrcStrides));
    return success();
  }
};

class AMDAIECanonicalizeDmaPass
    : public impl::AMDAIECanonicalizeDmaBase<AMDAIECanonicalizeDmaPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect,
                    xilinx::air::airDialect>();
  }

  AMDAIECanonicalizeDmaPass() = default;
  AMDAIECanonicalizeDmaPass(const AMDAIECanonicalizeDmaPass &pass){};
  void runOnOperation() override;
};

void AMDAIECanonicalizeDmaPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<FoldUnitDimsInDma, FoldLinearDimsInDma>(context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIECanonicalizeDmaPass() {
  return std::make_unique<AMDAIECanonicalizeDmaPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
