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
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-pack-to-dma"

namespace mlir::iree_compiler::AMDAIE {

namespace {

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
  patterns.insert<FoldUnitDimsInDma>(context);
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
