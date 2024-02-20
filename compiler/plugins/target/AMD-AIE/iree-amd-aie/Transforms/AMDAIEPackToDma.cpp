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

/// Applies packing to a given input
LogicalResult packDmaInputs(RewriterBase &rewriter,
                            IREE::LinalgExt::PackOp packOp,
                            SmallVector<Value> &offsets,
                            SmallVector<int64_t> &sizes,
                            SmallVector<int64_t> &strides) {
  llvm::ArrayRef<int64_t> permutation = packOp.getOuterDimsPerm();
  llvm::ArrayRef<int64_t> innerTiles = packOp.getStaticInnerTiles();
  SmallVector<int64_t> innerSizes;
  SmallVector<int64_t> innerStrides;
  SmallVector<Value> innerOffsets;
  auto innerDimsPos = packOp.getInnerDimsPos();
  for (int i = 0; i < innerTiles.size(); i++) {
    // Calculate new sizes.
    innerSizes.push_back(innerTiles[i]);
    // Fail if tile doesnt perfectly divide the corresponding outer dim as we do
    // not support the padding semantics yet.
    if (sizes[innerDimsPos[i]] % innerTiles[i] != 0) {
      return failure();
    }
    sizes[innerDimsPos[i]] /= innerTiles[i];
    // The tiled dim inherits the stride from the corresponding outer dim and
    // the outer dims stride gets multiplied by the size of the tile.
    innerStrides.push_back(strides[innerDimsPos[i]]);
    strides[innerDimsPos[i]] *= innerTiles[i];
    // The tiled dim inhertis the offset from the corresponding outer dim and
    // the outer dim offset is set to zero.
    innerOffsets.push_back(offsets[innerDimsPos[i]]);
    offsets[innerDimsPos[i]] =
        rewriter.create<arith::ConstantIndexOp>(packOp.getLoc(), 0);
  }
  // Apply permutations to the outer dims if provided.
  if (!permutation.empty()) {
    applyPermutationToVector(strides, permutation);
    applyPermutationToVector(sizes, permutation);
    applyPermutationToVector(offsets, permutation);
  }
  // Merge the dims.
  sizes.insert(sizes.end(), innerSizes.begin(), innerSizes.end());
  strides.insert(strides.end(), innerStrides.begin(), innerStrides.end());
  offsets.insert(offsets.end(), innerOffsets.begin(), innerOffsets.end());
  return success();
}

/// Applies unpacking to a given input
LogicalResult unPackDmaInputs(RewriterBase &rewriter,
                              IREE::LinalgExt::UnPackOp unPackOp,
                              SmallVector<Value> &offsets,
                              SmallVector<int64_t> &sizes,
                              SmallVector<int64_t> &strides) {
  llvm::ArrayRef<int64_t> permutation = unPackOp.getOuterDimsPerm();
  llvm::ArrayRef<int64_t> innerTiles = unPackOp.getStaticInnerTiles();
  SmallVector<int64_t> innerSizes;
  SmallVector<int64_t> innerStrides;
  SmallVector<Value> innerOffsets;
  auto innerDimsPos = unPackOp.getInnerDimsPos();

  int numOuterDims = sizes.size() - innerTiles.size();
  SmallVector<Value> outerOffsets =
      SmallVector<Value>(offsets.begin(), offsets.begin() + numOuterDims);
  SmallVector<int64_t> outerStrides =
      SmallVector<int64_t>(strides.begin(), strides.begin() + numOuterDims);
  SmallVector<int64_t> outerSizes =
      SmallVector<int64_t>(sizes.begin(), sizes.begin() + numOuterDims);

  // Apply permutations to the outer dims if provided.
  if (!permutation.empty()) {
    applyPermutationToVector(outerStrides, permutation);
    applyPermutationToVector(outerSizes, permutation);
    applyPermutationToVector(outerOffsets, permutation);
  }
  // Do the unpacking on the Outer dims.
  llvm::SmallDenseMap<int64_t, int64_t> outerDimsIndexMap;
  // Intialize the indexing of each outer dim.
  for (int i = 0; i < numOuterDims; i++) {
    outerDimsIndexMap[i] = i;
  }
  for (int i = 0; i < innerTiles.size(); i++) {
    // Insert inner dims adjcant to there corresponding outer dims.
    outerSizes.insert(
        outerSizes.begin() + outerDimsIndexMap[innerDimsPos[i]] + 1,
        innerTiles[i]);
    outerStrides.insert(
        outerStrides.begin() + outerDimsIndexMap[innerDimsPos[i]] + 1,
        strides[numOuterDims + i]);
    outerOffsets.insert(
        outerOffsets.begin() + outerDimsIndexMap[innerDimsPos[i]] + 1,
        offsets[numOuterDims + i]);
    // Update the map as all the dimensions inner to the innerDimsPos[i] are now
    // shifted by 1.
    for (int j = innerDimsPos[i] + 1; j < numOuterDims; j++) {
      outerDimsIndexMap[j]++;
    }
  }
  // Make the outer dims as the final returned dims
  offsets = outerOffsets;
  strides = outerStrides;
  sizes = outerSizes;
  return success();
}

/// Examines an input or an output Value of a pack/unpack op and provides the
/// corresponding offsets, sizes and strides required by the dma op
LogicalResult getDmaInputs(RewriterBase &rewriter, Operation *&operandOp,
                           SmallVector<Value> &offsets,
                           SmallVector<int64_t> &sizes,
                           SmallVector<int64_t> &strides) {
  Location loc = operandOp->getLoc();
  int64_t baseOffset;
  if (auto allocOp = dyn_cast<memref::AllocOp>(operandOp)) {
    std::tie(strides, baseOffset) = getStridesAndOffset(allocOp.getType());
    sizes = SmallVector<int64_t>(allocOp.getType().getShape().begin(),
                                 allocOp.getType().getShape().end());
  } else if (auto subviewOp = dyn_cast<memref::SubViewOp>(operandOp)) {
    // fail if non-unit mixed strides are present as NYI.
    auto mixedStrides = subviewOp.getMixedStrides();
    for (auto mixedStride : mixedStrides) {
      Attribute mixedStrideAttr = dyn_cast_if_present<Attribute>(mixedStride);
      if (!mixedStrideAttr) {
        return failure();
      }
      int64_t mixedStrideValue = cast<IntegerAttr>(mixedStrideAttr).getInt();
      if (mixedStrideValue != 1) {
        return failure();
      }
    }

    offsets = getValueOrCreateConstantIndexOp(rewriter, loc,
                                              subviewOp.getMixedOffsets());
    std::tie(strides, baseOffset) =
        getStridesAndOffset(subviewOp.getSource().getType());
    operandOp = subviewOp.getSource().getDefiningOp();
    sizes = SmallVector<int64_t>(subviewOp.getType().getShape().begin(),
                                 subviewOp.getType().getShape().end());
  } else {
    return failure();
  }
  if (baseOffset != 0) {
    // This is conservative, however need to verify that this can be correctly
    // supported once we see a use case to enable.
    return failure();
  }
  for (int i = offsets.size(); i < sizes.size(); i++) {
    offsets.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
  }
  return success();
}

/// Pattern to rewrite LinalgExt::PackOp -> air::DmaMemcpyNdOp.
class LinalgExtPackToAirDmaMemcpyNd
    : public OpRewritePattern<IREE::LinalgExt::PackOp> {
  using OpRewritePattern<IREE::LinalgExt::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::LinalgExt::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    // 1. Filter out NYI cases.
    llvm::ArrayRef<int64_t> innerTiles = packOp.getStaticInnerTiles();
    if (llvm::any_of(innerTiles, [](int64_t size) {
          return ShapedType::isDynamic(size);
        })) {
      return rewriter.notifyMatchFailure(packOp, "non-static shape NYI");
    }
    Location loc = packOp->getLoc();
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(packOp);

    Value input = packOp.getInput();
    Value output = packOp.getOutput();

    Operation *sourceOp = input.getDefiningOp();
    Operation *dstOp = output.getDefiningOp();

    // prepare source DMA inputs.
    SmallVector<Value> srcBaseStridesValues;
    SmallVector<Value> srcShapeValues;
    SmallVector<Value> srcOffsets;
    SmallVector<int64_t> srcBaseStrides;
    SmallVector<int64_t> srcShape;
    if (!succeeded(getDmaInputs(rewriter, sourceOp, srcOffsets, srcShape,
                                srcBaseStrides))) {
      return rewriter.notifyMatchFailure(
          packOp, "cant infer dma source inputs from op");
    }
    if (!succeeded(packDmaInputs(rewriter, packOp, srcOffsets, srcShape,
                                 srcBaseStrides))) {
      return rewriter.notifyMatchFailure(
          packOp, "could not perform the required packing to create a dma op");
    }
    for (auto stride : srcBaseStrides) {
      srcBaseStridesValues.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, stride));
    }
    for (auto dim : srcShape) {
      srcShapeValues.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, dim));
    }

    // prepare destination DMA inputs.
    SmallVector<Value> dstBaseStridesValues;
    SmallVector<Value> dstShapeValues;
    SmallVector<Value> dstOffsets;
    SmallVector<int64_t> dstBaseStrides;
    SmallVector<int64_t> dstShape;
    if (!succeeded(getDmaInputs(rewriter, dstOp, dstOffsets, dstShape,
                                dstBaseStrides))) {
      return rewriter.notifyMatchFailure(
          packOp, "cant infer dma source inputs from op");
    }
    for (auto stride : dstBaseStrides) {
      dstBaseStridesValues.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, stride));
    }
    for (auto dim : dstShape) {
      dstShapeValues.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, dim));
    }

    SmallVector<Value, 2> emptyVec;
    rewriter.replaceOpWithNewOp<xilinx::air::DmaMemcpyNdOp>(
        packOp, SmallVector<Type, 1>{}, emptyVec, dstOp->getResult(0),
        dstOffsets, dstShapeValues, dstBaseStridesValues,
        sourceOp->getResult(0), srcOffsets, srcShapeValues,
        srcBaseStridesValues);
    return success();
  }
};

/// Pattern to rewrite LinalgExt::UnPackOp -> air::DmaMemcpyNdOp.
class LinalgExtUnPackToAirDmaMemcpyNd
    : public OpRewritePattern<IREE::LinalgExt::UnPackOp> {
  using OpRewritePattern<IREE::LinalgExt::UnPackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::LinalgExt::UnPackOp unPackOp,
                                PatternRewriter &rewriter) const override {
    // 1. Filter out NYI cases.
    llvm::ArrayRef<int64_t> innerTiles = unPackOp.getStaticInnerTiles();
    if (llvm::any_of(innerTiles, [](int64_t size) {
          return ShapedType::isDynamic(size);
        })) {
      return rewriter.notifyMatchFailure(unPackOp, "non-static shape NYI");
    }
    Location loc = unPackOp->getLoc();
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(unPackOp);

    Value input = unPackOp.getInput();
    Value output = unPackOp.getOutput();

    Operation *sourceOp = input.getDefiningOp();
    Operation *dstOp = output.getDefiningOp();

    // prepare source DMA inputs.
    SmallVector<Value> srcBaseStridesValues;
    SmallVector<Value> srcShapeValues;
    SmallVector<Value> srcOffsets;
    SmallVector<int64_t> srcBaseStrides;
    SmallVector<int64_t> srcShape;
    if (!succeeded(getDmaInputs(rewriter, sourceOp, srcOffsets, srcShape,
                                srcBaseStrides))) {
      return rewriter.notifyMatchFailure(
          unPackOp, "cant infer dma source inputs from op");
    }
    if (!succeeded(unPackDmaInputs(rewriter, unPackOp, srcOffsets, srcShape,
                                   srcBaseStrides))) {
      return rewriter.notifyMatchFailure(
          unPackOp,
          "could not perform the required packing to create a dma op");
    }
    for (auto stride : srcBaseStrides) {
      srcBaseStridesValues.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, stride));
    }
    for (auto dim : srcShape) {
      srcShapeValues.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, dim));
    }

    // prepare destination DMA inputs.
    SmallVector<Value> dstBaseStridesValues;
    SmallVector<Value> dstShapeValues;
    SmallVector<Value> dstOffsets;
    SmallVector<int64_t> dstBaseStrides;
    SmallVector<int64_t> dstShape;
    if (!succeeded(getDmaInputs(rewriter, dstOp, dstOffsets, dstShape,
                                dstBaseStrides))) {
      return rewriter.notifyMatchFailure(
          unPackOp, "cant infer dma source inputs from op");
    }
    for (auto stride : dstBaseStrides) {
      dstBaseStridesValues.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, stride));
    }
    for (auto dim : dstShape) {
      dstShapeValues.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, dim));
    }

    SmallVector<Value, 2> emptyVec;
    rewriter.replaceOpWithNewOp<xilinx::air::DmaMemcpyNdOp>(
        unPackOp, SmallVector<Type, 1>{}, emptyVec, dstOp->getResult(0),
        dstOffsets, dstShapeValues, dstBaseStridesValues,
        sourceOp->getResult(0), srcOffsets, srcShapeValues,
        srcBaseStridesValues);
    return success();
  }
};

class AMDAIEPackToDmaPass
    : public impl::AMDAIEPackToDmaBase<AMDAIEPackToDmaPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect,
                    xilinx::air::airDialect>();
  }

  AMDAIEPackToDmaPass() = default;
  AMDAIEPackToDmaPass(const AMDAIEPackToDmaPass &pass){};
  void runOnOperation() override;
};

void AMDAIEPackToDmaPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns
      .insert<LinalgExtPackToAirDmaMemcpyNd, LinalgExtUnPackToAirDmaMemcpyNd>(
          context);
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
