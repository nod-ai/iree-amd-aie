// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "air/Dialect/AIR/AIRDialect.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-pack-to-dma"

namespace mlir::iree_compiler::AMDAIE {

namespace {

int64_t getConstantIndexOrAssert(OpFoldResult dim) {
  std::optional<int64_t> size = getConstantIntValue(dim);
  assert(size.has_value() && "expect constant index");
  return size.value();
}

bool foldAirDmaUnitDims(const SmallVector<OpFoldResult> &offsets,
                        const SmallVector<OpFoldResult> &strides,
                        const SmallVector<OpFoldResult> &sizes,
                        SmallVector<OpFoldResult> &newOffsets,
                        SmallVector<OpFoldResult> &newStrides,
                        SmallVector<OpFoldResult> &newSizes) {
  bool foldableUnitDimsFound = false;

  for (int i = 0; i < offsets.size(); i++) {
    // Dim can be folded if offset is zero and size is 1
    if (isConstantIntValue(offsets[i], 0) && isConstantIntValue(sizes[i], 1)) {
      foldableUnitDimsFound = true;
      continue;
    }
    newOffsets.push_back(offsets[i]);
    newStrides.push_back(strides[i]);
    newSizes.push_back(sizes[i]);
  }
  return foldableUnitDimsFound;
}

bool foldAirDmaLinearDims(MLIRContext *ctx,
                          const SmallVector<OpFoldResult> &offsets,
                          const SmallVector<OpFoldResult> &strides,
                          const SmallVector<OpFoldResult> &sizes,
                          SmallVector<OpFoldResult> &newOffsets,
                          SmallVector<OpFoldResult> &newStrides,
                          SmallVector<OpFoldResult> &newSizes) {
  bool foldableLinearDimsFound = false;

  newOffsets.push_back(offsets[0]);
  newStrides.push_back(strides[0]);
  newSizes.push_back(sizes[0]);

  for (int i = 1; i < offsets.size(); i++) {
    // Conditions for folding a dim.
    // 1. size(i) x stide(i) == stride(i-1), with this we can have new size(i-1)
    // = size(i-1) * size(i), stride(i-1) = stride(i) and then fold away the i
    // dimension
    // 2. Offset(i-1) = 0. This is required becuase we are dropping the offset
    // of the i-1 dimension and doing offset(i-1) = offset(i)
    int vecSize = newOffsets.size();
    if (isConstantIntValue(newOffsets[vecSize - 1], 0) &&
        getConstantIndexOrAssert(sizes[i]) *
                getConstantIndexOrAssert(strides[i]) ==
            getConstantIndexOrAssert(newStrides[vecSize - 1])) {
      foldableLinearDimsFound = true;
      int vecSize = newOffsets.size();
      newOffsets[vecSize - 1] = offsets[i];
      newStrides[vecSize - 1] = strides[i];
      newSizes[vecSize - 1] = getAsIndexOpFoldResult(
          ctx, getConstantIndexOrAssert(sizes[i]) *
                   getConstantIndexOrAssert(newSizes[vecSize - 1]));

      continue;
    }
    newOffsets.push_back(offsets[i]);
    newStrides.push_back(strides[i]);
    newSizes.push_back(sizes[i]);
  }
  return foldableLinearDimsFound;
}

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
    bool foldableUnitDimsFound = false;

    // We do not make any assumptions when all offsets are not
    // specified and dont change the op in that case.
    if (srcStrides.size() != srcOffsets.size() ||
        dstStrides.size() != dstOffsets.size()) {
      return rewriter.notifyMatchFailure(
          op, "offset dimensions dont match stride dimensions");
    }
    // Fold source dims.
    foldableUnitDimsFound |=
        foldAirDmaUnitDims(srcOffsets, srcStrides, srcSizes, newSrcOffsets,
                           newSrcStrides, newSrcSizes);
    // Fold destination dims.
    foldableUnitDimsFound |=
        foldAirDmaUnitDims(dstOffsets, dstStrides, dstSizes, newDstOffsets,
                           newDstStrides, newDstSizes);
    if (!foldableUnitDimsFound) {
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
    bool foldableLinearDimsFound = false;

    // We do not make any assumptions when all offsets are not
    // specified and dont change the op in that case.
    if (srcStrides.size() != srcOffsets.size() ||
        dstStrides.size() != dstOffsets.size()) {
      return rewriter.notifyMatchFailure(
          op, "offset dimensions dont match stride dimensions");
    }

    // Fold source dims.
    foldableLinearDimsFound |=
        foldAirDmaLinearDims(op.getContext(), srcOffsets, srcStrides, srcSizes,
                             newSrcOffsets, newSrcStrides, newSrcSizes);
    // Fold destination dims.
    foldableLinearDimsFound |=
        foldAirDmaLinearDims(op.getContext(), dstOffsets, dstStrides, dstSizes,
                             newDstOffsets, newDstStrides, newDstSizes);
    if (!foldableLinearDimsFound) {
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
