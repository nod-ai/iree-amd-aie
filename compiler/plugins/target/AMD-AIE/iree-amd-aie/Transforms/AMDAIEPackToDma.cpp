// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "air/Dialect/AIR/AIRDialect.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-pack-to-dma"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Applies packing to a given input.
template <typename OpType>
LogicalResult packDmaInputs(RewriterBase &rewriter, OpType packOp,
                            SmallVector<OpFoldResult> &offsets,
                            SmallVector<OpFoldResult> &sizes,
                            SmallVector<OpFoldResult> &strides) {
  MLIRContext *ctx = packOp.getContext();

  llvm::ArrayRef<int64_t> permutation = packOp.getOuterDimsPerm();
  llvm::ArrayRef<int64_t> innerTiles = packOp.getStaticInnerTiles();

  SmallVector<OpFoldResult> innerSizes;
  SmallVector<OpFoldResult> innerStrides;
  SmallVector<OpFoldResult> innerOffsets;

  auto innerDimsPos = packOp.getInnerDimsPos();
  for (int i = 0; i < innerTiles.size(); i++) {
    // Calculate new sizes.
    innerSizes.push_back(getAsIndexOpFoldResult(ctx, innerTiles[i]));
    std::optional<int64_t> size = getConstantIntValue(sizes[innerDimsPos[i]]);
    assert(size.has_value() &&
           "expect constant index here in sizes vector of pack op");
    // Fail if tile doesnt perfectly divide the corresponding outer dim as we do
    // not support the padding semantics yet.
    if (size.value() % innerTiles[i] != 0) {
      return failure();
    }
    sizes[innerDimsPos[i]] =
        getAsIndexOpFoldResult(ctx, size.value() / innerTiles[i]);
    // The tiled dim inherits the stride from the corresponding outer dim and
    // the outer dims stride gets multiplied by the size of the tile.
    innerStrides.push_back(strides[innerDimsPos[i]]);
    std::optional<int64_t> stride =
        getConstantIntValue(strides[innerDimsPos[i]]);
    assert(stride.has_value() &&
           "expect constant index in stride vector of pack op");
    strides[innerDimsPos[i]] =
        getAsIndexOpFoldResult(ctx, stride.value() * innerTiles[i]);
    // The tiled dim inherits the offset from the corresponding outer dim and
    // the outer dim offset is set to zero.
    innerOffsets.push_back(offsets[innerDimsPos[i]]);
    offsets[innerDimsPos[i]] = getAsIndexOpFoldResult(ctx, 0);
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

/// Applies unpacking to a given input.
template <typename OpType>
LogicalResult unPackDmaInputs(RewriterBase &rewriter, OpType unPackOp,
                              SmallVector<OpFoldResult> &offsets,
                              SmallVector<OpFoldResult> &sizes,
                              SmallVector<OpFoldResult> &strides) {
  MLIRContext *ctx = unPackOp.getContext();

  llvm::ArrayRef<int64_t> permutation = unPackOp.getOuterDimsPerm();
  llvm::ArrayRef<int64_t> innerTiles = unPackOp.getStaticInnerTiles();

  SmallVector<OpFoldResult> innerSizes;
  SmallVector<OpFoldResult> innerStrides;
  SmallVector<OpFoldResult> innerOffsets;
  auto innerDimsPos = unPackOp.getInnerDimsPos();

  int numOuterDims = sizes.size() - innerTiles.size();
  SmallVector<OpFoldResult> outerOffsets = SmallVector<OpFoldResult>(
      offsets.begin(), offsets.begin() + numOuterDims);
  SmallVector<OpFoldResult> outerStrides = SmallVector<OpFoldResult>(
      strides.begin(), strides.begin() + numOuterDims);
  SmallVector<OpFoldResult> outerSizes =
      SmallVector<OpFoldResult>(sizes.begin(), sizes.begin() + numOuterDims);

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
    // Insert inner dims adjacent to there corresponding outer dims.
    outerSizes.insert(
        outerSizes.begin() + outerDimsIndexMap[innerDimsPos[i]] + 1,
        getAsIndexOpFoldResult(ctx, innerTiles[i]));
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

/// Examines an input/output of a pack/unpack op and provides the
/// corresponding offsets, sizes and strides required by the dma op
LogicalResult getDmaInputs(RewriterBase &rewriter, Operation *&operandOp,
                           SmallVector<OpFoldResult> &offsets,
                           SmallVector<OpFoldResult> &sizes,
                           SmallVector<OpFoldResult> &strides) {
  MLIRContext *ctx = operandOp->getContext();
  if (auto allocOp = dyn_cast<memref::AllocOp>(operandOp)) {
    auto [stridesI64, baseOffset] = getStridesAndOffset(allocOp.getType());
    if (baseOffset != 0) {
      // This is conservative, however need to verify that this can be correctly
      // supported once we see a use case to enable.
      return failure();
    }
    strides = getAsIndexOpFoldResult(ctx, stridesI64);
    auto sizesI64 = allocOp.getType().getShape();
    // Dynamic shape not supported by air dma op.
    if (llvm::any_of(sizesI64, [](int64_t size) {
          return ShapedType::isDynamic(size);
        })) {
      return failure();
    }
    sizes = getAsIndexOpFoldResult(ctx, sizesI64);
    // Alloc Op has no offsets.
    for (int i = 0; i < sizes.size(); i++) {
      offsets.push_back(getAsIndexOpFoldResult(ctx, 0));
    }
    return success();
  }
  if (auto subviewOp = dyn_cast<memref::SubViewOp>(operandOp)) {
    // fail if non-unit mixed strides are present as NYI.
    auto mixedStrides = subviewOp.getMixedStrides();
    if (llvm::any_of(mixedStrides, [](OpFoldResult ofr) {
          return !isConstantIntValue(ofr, 1);
        })) {
      return failure();
    }
    offsets = subviewOp.getMixedOffsets();
    auto [stridesI64, baseOffset] =
        getStridesAndOffset(subviewOp.getSource().getType());
    if (baseOffset != 0) {
      // This is conservative, however need to verify that this can be correctly
      // supported once we see a use case to enable.
      return failure();
    }
    strides = getAsIndexOpFoldResult(ctx, stridesI64);
    operandOp = subviewOp.getSource().getDefiningOp();
    auto sizesI64 = subviewOp.getType().getShape();
    // Dynamic shape not supported by air dma op.
    if (llvm::any_of(sizesI64, [](int64_t size) {
          return ShapedType::isDynamic(size);
        })) {
      return failure();
    }
    sizes = getAsIndexOpFoldResult(ctx, sizesI64);
    return success();
  }
  return failure();
}

template <typename OpType>
class LinalgExtPackToAirDmaMemcpyNd : public OpRewritePattern<OpType> {
 public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    // 1. Filter out NYI cases.
    llvm::ArrayRef<int64_t> innerTiles = op.getStaticInnerTiles();
    if (llvm::any_of(innerTiles, [](int64_t size) {
          return ShapedType::isDynamic(size);
        })) {
      return rewriter.notifyMatchFailure(op, "non-static shape NYI");
    }
    Location loc = op->getLoc();
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);

    Value input = op.getInput();
    Value output = op.getOutput();

    Operation *sourceOp = input.getDefiningOp();
    Operation *dstOp = output.getDefiningOp();

    // prepare source DMA inputs.
    SmallVector<OpFoldResult> srcOffsets;
    SmallVector<OpFoldResult> srcBaseStrides;
    SmallVector<OpFoldResult> srcShape;
    if (!succeeded(getDmaInputs(rewriter, sourceOp, srcOffsets, srcShape,
                                srcBaseStrides))) {
      return rewriter.notifyMatchFailure(
          op, "cant infer dma source inputs from op");
    }
    if (std::is_same<OpType, IREE::LinalgExt::PackOp>::value) {
      if (!succeeded(packDmaInputs<OpType>(rewriter, op, srcOffsets, srcShape,
                                           srcBaseStrides))) {
        return rewriter.notifyMatchFailure(
            op, "could not perform the required packing to create a dma op");
      }
    } else if (std::is_same<OpType, IREE::LinalgExt::UnPackOp>::value) {
      if (!succeeded(unPackDmaInputs<OpType>(rewriter, op, srcOffsets, srcShape,
                                             srcBaseStrides))) {
        return rewriter.notifyMatchFailure(
            op, "could not perform the required unpacking to create a dma op");
      }
    }
    // prepare destination DMA inputs.
    SmallVector<OpFoldResult> dstOffsets;
    SmallVector<OpFoldResult> dstBaseStrides;
    SmallVector<OpFoldResult> dstShape;
    if (!succeeded(getDmaInputs(rewriter, dstOp, dstOffsets, dstShape,
                                dstBaseStrides))) {
      return rewriter.notifyMatchFailure(
          op, "cant infer dma source inputs from op");
    }
    // Async Tokens are added to the op in later passes.
    SmallVector<Value, 2> asyncTokens;
    rewriter.replaceOpWithNewOp<xilinx::air::DmaMemcpyNdOp>(
        op, SmallVector<Type, 1>{}, asyncTokens, dstOp->getResult(0),
        getValueOrCreateConstantIndexOp(rewriter, loc, dstOffsets),
        getValueOrCreateConstantIndexOp(rewriter, loc, dstShape),
        getValueOrCreateConstantIndexOp(rewriter, loc, dstBaseStrides),
        sourceOp->getResult(0),
        getValueOrCreateConstantIndexOp(rewriter, loc, srcOffsets),
        getValueOrCreateConstantIndexOp(rewriter, loc, srcShape),
        getValueOrCreateConstantIndexOp(rewriter, loc, srcBaseStrides));
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
  patterns.insert<LinalgExtPackToAirDmaMemcpyNd<IREE::LinalgExt::PackOp>,
                  LinalgExtPackToAirDmaMemcpyNd<IREE::LinalgExt::UnPackOp>>(
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
