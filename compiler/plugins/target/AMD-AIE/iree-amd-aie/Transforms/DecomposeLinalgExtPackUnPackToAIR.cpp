// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "air/Dialect/AIR/AIRDialect.h"
#include "iree-amd-aie/Transforms/PassDetail.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Returns the padding value if set. Otherwise, returns the source directly.
/// The method assumes that the `packOp` has static shapes.
static Value getPackOpInputOrPaddedSource(OpBuilder &builder,
                                          IREE::LinalgExt::PackOp packOp) {
  return packOp.getInput();
}

OpFoldResult getMixedSize(OpBuilder &builder, Location loc, Value value,
                          int64_t dim) {
  auto memrefType = llvm::cast<MemRefType>(value.getType());
  SmallVector<OpFoldResult> result;
  // TODO: dynamic shape
  // if (memrefType.isDynamicDim(dim))

  return builder.getIndexAttr(memrefType.getDimSize(dim));
}

SmallVector<OpFoldResult> getMixedSizes(OpBuilder &builder, Location loc,
                                        Value value) {
  auto memrefType = llvm::cast<MemRefType>(value.getType());
  SmallVector<OpFoldResult> result;
  for (int64_t i = 0; i < memrefType.getRank(); ++i)
    result.push_back(getMixedSize(builder, loc, value, i));
  return result;
}

// Normalizes a permutation on a higher rank space to its actual size, e.g.
//   perm = [1, 4, 2]
// becomes
//   norm = [0, 2, 1]
static SmallVector<int64_t> getPackUnpackNormalizedPerm(
    int rank, ArrayRef<int64_t> perm) {
  constexpr int64_t kNonTiledMarker = -1;
  SmallVector<int64_t> vec(rank, kNonTiledMarker);
  for (auto [index, value] : llvm::enumerate(perm)) vec[value] = index;
  SmallVector<int64_t> normalizedPerm = llvm::to_vector(llvm::make_filter_range(
      vec, [&](int64_t v) { return v != kNonTiledMarker; }));
  // This inverts the permutation in addition to normalizing so invert back.
  return invertPermutationVector(normalizedPerm);
}

// Gets the normalized permutation implied by innerDimsPos and outerDimsPerm
// assuming rank reduction of unit outer dims.
static SmallVector<int64_t> getPackUnpackRankReducedPerm(
    ArrayRef<int64_t> shape, ArrayRef<int64_t> innerDimsPos,
    ArrayRef<int64_t> outerDimsPerm) {
  SmallVector<int64_t> rankReducedOuterDimsPerm;
  SmallVector<int64_t> outerDims;
  SmallVector<int64_t> innerDims;
  int64_t dim = 0;
  int64_t unpackedRank = shape.size();
  for (auto i : llvm::seq<unsigned>(0, unpackedRank)) {
    if (llvm::is_contained(innerDimsPos, i)) {
      innerDims.push_back(dim++);
      continue;
    }
    outerDims.push_back(dim++);
    if (!outerDimsPerm.empty())
      rankReducedOuterDimsPerm.push_back(outerDimsPerm[i]);
  }

  // Get the position of the inner dims after permutation.
  SmallVector<int64_t> innerPerm =
      getPackUnpackNormalizedPerm(unpackedRank, innerDimsPos);
  applyPermutationToVector<int64_t>(innerDims, innerPerm);

  // Ditto for the outer dims.
  SmallVector<int64_t> perm = outerDims;

  rankReducedOuterDimsPerm =
      getPackUnpackNormalizedPerm(unpackedRank, rankReducedOuterDimsPerm);
  if (!rankReducedOuterDimsPerm.empty())
    applyPermutationToVector<int64_t>(perm, rankReducedOuterDimsPerm);

  // The tile always ends up as the inner most dims after packing.
  perm.append(innerDims);

  return perm;
}

class GeneralizePackOpPattern
    : public OpRewritePattern<IREE::LinalgExt::PackOp> {
 public:
  using OpRewritePattern<IREE::LinalgExt::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::LinalgExt::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    if (llvm::any_of(packOp.getMixedTiles(),
                     [](OpFoldResult tile) { return tile.is<Value>(); })) {
      return rewriter.notifyMatchFailure(
          packOp, "require inner tile sizes being static");
    }

    auto innerDimsPos = packOp.getInnerDimsPos();
    int64_t srcRank = packOp.getInputRank();
    auto destShape = packOp.getOutputType().getShape();
    if (llvm::any_of(innerDimsPos, [destShape](int64_t index) {
          return destShape[index] != 1;
        })) {
      return rewriter.notifyMatchFailure(
          packOp,
          "require the tiled outer dimensions of the result are all 1s");
    }

    // 1. Use rank-reduced memref.subview op to extract the tile and untiled
    // outer dims.
    Location loc = packOp.getLoc();
    Value input = getPackOpInputOrPaddedSource(rewriter, packOp);
    auto inputShape = packOp.getInputType().getShape();
    DenseMap<int64_t, OpFoldResult> dimAndTileMapping =
        packOp.getDimAndTileMapping();
    Attribute zeroIdxAttr = rewriter.getIndexAttr(0);
    Attribute oneIdxAttr = rewriter.getIndexAttr(1);
    SmallVector<OpFoldResult> readOffsets(srcRank, zeroIdxAttr);
    SmallVector<OpFoldResult> readStrides(srcRank, oneIdxAttr);
    SmallVector<OpFoldResult> readSizes;
    SmallVector<int64_t> readShape;
    for (auto i : llvm::seq<unsigned>(0, srcRank)) {
      if (dimAndTileMapping.count(i)) {
        readShape.push_back(getConstantIntValue(dimAndTileMapping[i])
                                .value_or(ShapedType::kDynamic));
        readSizes.push_back(dimAndTileMapping[i]);
        continue;
      }
      if (ShapedType::isDynamic(inputShape[i])) {
        assert(false);
        // TODO: Dynamic input shape
      } else {
        readSizes.push_back(rewriter.getIndexAttr(inputShape[i]));
      }
      readShape.push_back(inputShape[i]);
    }

    Type elemType = packOp.getInputType().getElementType();
    Attribute memorySpace =
        packOp.getInputType().cast<MemRefType>().getMemorySpace();
    auto readType = MemRefType::get(
        readShape, elemType,
        packOp.getInputType().cast<MemRefType>().getLayout(), memorySpace);

    Value tile = rewriter.create<memref::SubViewOp>(
        loc, readType, input, readOffsets, readSizes, readStrides);

    // 2. Transpose the tile to match the inner tile order.

    SmallVector<int64_t> perm = getPackUnpackRankReducedPerm(
        inputShape, innerDimsPos, packOp.getOuterDimsPerm());

    SmallVector<int64_t> transpShape = readShape;
    applyPermutationToVector<int64_t>(transpShape, perm);

    Value transposed = rewriter.create<memref::TransposeOp>(
        loc, tile,
        AffineMapAttr::get(
            AffineMap::getPermutationMap(perm, packOp->getContext())));

    // 3. Insert the inner tile to the destination.
    int64_t destRank = packOp.getOutputRank();
    SmallVector<OpFoldResult> writeStrides(destRank, oneIdxAttr);
    SmallVector<OpFoldResult> writeOffsets(destRank, zeroIdxAttr);
    SmallVector<OpFoldResult> writeSizes =
        getMixedSizes(rewriter, loc, packOp.getOutput());

    memorySpace = packOp.getOutputType().cast<MemRefType>().getMemorySpace();
    auto writeType = MemRefType::get(destShape, elemType, nullptr, memorySpace);
    auto output_subview = rewriter.create<memref::SubViewOp>(
        loc, writeType, packOp.getOutput(), writeOffsets, writeSizes,
        writeStrides);

    SmallVector<Value, 2> mt;
    rewriter.create<xilinx::air::DmaMemcpyNdOp>(loc, SmallVector<Type, 1>{}, mt,
                                                output_subview, mt, mt, mt,
                                                transposed, mt, mt, mt);

    rewriter.eraseOp(packOp);
    return success();
  }
};

class GeneralizeUnPackOpPattern
    : public OpRewritePattern<IREE::LinalgExt::UnPackOp> {
 public:
  using OpRewritePattern<IREE::LinalgExt::UnPackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::LinalgExt::UnPackOp unpackOp,
                                PatternRewriter &rewriter) const override {
    int64_t srcRank = unpackOp.getInputRank();
    int64_t destRank = unpackOp.getOutputRank();
    auto destShape = unpackOp.getOutputType().getShape();
    ArrayRef<int64_t> srcShape = unpackOp.getInputType().getShape();
    ArrayRef<int64_t> innerDimsPos = unpackOp.getInnerDimsPos();
    if (llvm::any_of(innerDimsPos, [srcShape](int64_t index) {
          return srcShape[index] != 1;
        })) {
      return rewriter.notifyMatchFailure(
          unpackOp,
          "require the tiled outer dimensions of the result are all 1s");
    }

    // 1. Use memref.subview op to extract the tile.
    Location loc = unpackOp.getLoc();
    Value input = unpackOp.getInput();
    DenseMap<int64_t, OpFoldResult> dimAndTileMapping =
        unpackOp.getDimAndTileMapping();
    Attribute zeroIdxAttr = rewriter.getIndexAttr(0);
    Attribute oneIdxAttr = rewriter.getIndexAttr(1);
    SmallVector<OpFoldResult> readOffsets(srcRank, zeroIdxAttr);
    SmallVector<OpFoldResult> readStrides(srcRank, oneIdxAttr);
    SmallVector<OpFoldResult> readSizes;
    SmallVector<int64_t> readShape;
    SmallVector<Value> dynamicDims;
    for (auto i : llvm::seq<unsigned>(0, destRank)) {
      if (dimAndTileMapping.count(i)) {
        readSizes.push_back(oneIdxAttr);
        continue;
      }

      if (ShapedType::isDynamic(srcShape[i])) {
        assert(false);
        // TODO: Dynamic input shape
      } else {
        readSizes.push_back(rewriter.getIndexAttr(srcShape[i]));
      }
      if (srcShape[i] != 1) readShape.push_back(srcShape[i]);
    }

    auto mixedTiles = unpackOp.getMixedTiles();
    readSizes.append(mixedTiles.begin(), mixedTiles.end());

    // Explicitly create the type for subview op because the inner tile
    // size could be 1. We want to represent the whole inner tile in this case.
    auto tileShape = srcShape.drop_front(destRank);
    // Append the inner tile shape to the permuted and rank-reduced outer shape.
    readShape.append(tileShape.begin(), tileShape.end());
    Type elemType = unpackOp.getInputType().getElementType();
    Attribute memorySpace =
        unpackOp.getInputType().cast<MemRefType>().getMemorySpace();
    auto readType = MemRefType::get(readShape, elemType, nullptr, memorySpace);
    Value innerTile = rewriter.create<memref::SubViewOp>(
        loc, readType, input, readOffsets, readSizes, readStrides);

    // 2. Transpose the tile to match the outer corresponding tile order.
    SmallVector<int64_t> perm =
        getPackUnpackRankReducedPerm(srcShape.take_front(destRank),
                                     innerDimsPos, unpackOp.getOuterDimsPerm());
    // Unpack is a transition out of packed space so we invert the permutation.
    perm = invertPermutationVector(perm);

    auto transposed = rewriter.create<memref::TransposeOp>(
        loc, innerTile,
        AffineMapAttr::get(
            AffineMap::getPermutationMap(perm, unpackOp->getContext())));

    // 3. Handle in-complete tiles if needed. It truncates trailing data from
    // the transposed tile.
    // TODO

    // 4. Copy the result to the destination memref.
    SmallVector<OpFoldResult> writeStrides(destRank, oneIdxAttr);
    SmallVector<OpFoldResult> writeOffsets(destRank, zeroIdxAttr);
    SmallVector<OpFoldResult> writeSizes =
        getMixedSizes(rewriter, loc, unpackOp.getOutput());

    memorySpace = unpackOp.getOutputType().cast<MemRefType>().getMemorySpace();
    auto writeType = MemRefType::get(
        destShape, elemType,
        unpackOp.getOutputType().cast<MemRefType>().getLayout(), memorySpace);

    auto output_subview = rewriter.create<memref::SubViewOp>(
        loc, writeType, unpackOp.getOutput(), writeOffsets, writeSizes,
        writeStrides);

    SmallVector<Value, 2> mt;
    rewriter.create<xilinx::air::DmaMemcpyNdOp>(loc, SmallVector<Type, 1>{}, mt,
                                                output_subview, mt, mt, mt,
                                                transposed, mt, mt, mt);

    rewriter.eraseOp(unpackOp);
    return success();
  }
};

struct LowerPackResult {
  memref::ExpandShapeOp expandShapeOp;
  memref::TransposeOp transposeOp;
  xilinx::air::DmaMemcpyNdOp dmaOp;
};

struct LowerUnPackResult {
  memref::TransposeOp transposeOp;
  xilinx::air::DmaMemcpyNdOp dmaOp;
};

FailureOr<LowerPackResult> lowerPack(RewriterBase &rewriter,
                                     IREE::LinalgExt::PackOp packOp) {
  // 1. Filter out NYI cases.
  auto packedMemrefType = packOp.getOutputType();
  if (llvm::any_of(packOp.getStaticInnerTiles(),
                   [](int64_t size) { return ShapedType::isDynamic(size); })) {
    return rewriter.notifyMatchFailure(
        packOp,
        "non-static shape NYI, needs a more powerful memref.expand_shape op");
  }

  Location loc = packOp->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(packOp);

  // 2. Compute the permutation vector to shuffle packed shape into the shape
  // before any outer or inner permutations have been applied. The permutation
  // can be obtained from two permutations:
  //   a) Compute the permutation vector to move the last `numPackedDims` into
  //      the `innerPosDims` of a shape of rank `packedRank`.
  //   b) Compute the permutation vector to move outer dims if the pack op
  //      has outer_dims_perm.
  // Apply (b) permutation on (a) permutation to get the final permutation.
  int64_t numPackedDims = packOp.getInnerDimsPos().size();
  int64_t packedRank = packedMemrefType.getRank();
  auto lastDims = llvm::to_vector(
      llvm::seq<int64_t>(packedRank - numPackedDims, packedRank));
  PackingMetadata packingMetadata = computePackingMetadata(
      packedMemrefType.getRank(), packOp.getInnerDimsPos());
  SmallVector<int64_t> innerPositionsPerm = computePermutationVector(
      packedRank, lastDims, packingMetadata.insertPositions);

  SmallVector<int64_t> outerPos = packingMetadata.outerPositions;
  ArrayRef<int64_t> outerPerm = packOp.getOuterDimsPerm();
  if (!outerPerm.empty()) applyPermutationToVector(outerPos, outerPerm);
  SmallVector<int64_t> outerPositionPerm = computePermutationVector(
      packedRank, packingMetadata.outerPositions, outerPos);

  SmallVector<int64_t> packedToStripMinedShapePerm = innerPositionsPerm;
  applyPermutationToVector(packedToStripMinedShapePerm, outerPositionPerm);

  // 3. Compute the stripMinedShape: this is the packed shape before any outer
  // or inner permutations have been applied.
  SmallVector<int64_t> stripMinedShape(packedMemrefType.getShape());
  applyPermutationToVector(stripMinedShape, packedToStripMinedShapePerm);

  // 4. Expand from the padded result to the stripMinedShape.
  auto reshapeOp = rewriter.create<memref::ExpandShapeOp>(
      loc, stripMinedShape, packOp.getInput(), packingMetadata.reassociations);

  // 5. Transpose stripMinedShape to packedShape.
  SmallVector<int64_t> transpPerm =
      invertPermutationVector(packedToStripMinedShapePerm);
  auto transposeOp = rewriter.create<memref::TransposeOp>(
      loc, reshapeOp.getResult(),
      AffineMapAttr::get(
          AffineMap::getPermutationMap(transpPerm, packOp->getContext())));

  SmallVector<Value, 2> mt;
  auto dmaOp = rewriter.create<xilinx::air::DmaMemcpyNdOp>(
      loc, SmallVector<Type, 1>{}, mt, packOp.getOutput(), mt, mt, mt,
      transposeOp.getResult(), mt, mt, mt);

  // 6. Replace packOp by transposeOp.
  rewriter.eraseOp(packOp);

  return LowerPackResult{reshapeOp, transposeOp, dmaOp};
}

/// A wrapper pattern that calls lowerPack on PackOp. It lowers
/// a pack op to memref.expand_shape + memref.transpose ops.
struct LowerPackPattern : public OpRewritePattern<IREE::LinalgExt::PackOp> {
  using OpRewritePattern<IREE::LinalgExt::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::LinalgExt::PackOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<LowerPackResult> res = lowerPack(rewriter, op);
    if (failed(res)) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower to pad + expand + transpose");
    }
    return success();
  }
};

FailureOr<LowerUnPackResult> lowerUnPack(RewriterBase &rewriter,
                                         IREE::LinalgExt::UnPackOp unPackOp) {
  Location loc = unPackOp->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(unPackOp);

  MemRefType memrefType = unPackOp.getInputType().cast<MemRefType>();
  int64_t packedRank = memrefType.getRank();

  // 1. Compute the permutation vector to move the last `numPackedDims` into
  // the `innerPosDims` of a shape of rank `packedRank`.
  int64_t numPackedDims = unPackOp.getInnerDimsPos().size();
  auto lastDims = llvm::to_vector(
      llvm::seq<int64_t>(packedRank - numPackedDims, packedRank));
  PackingMetadata packingMetadata =
      computePackingMetadata(packedRank, unPackOp.getInnerDimsPos());
  SmallVector<int64_t> lastDimsToInsertPositionsPerm = computePermutationVector(
      packedRank, lastDims, packingMetadata.insertPositions);
  // Apply outer positions permutation.
  SmallVector<int64_t> outerPos = packingMetadata.outerPositions;
  ArrayRef<int64_t> outerPerm = unPackOp.getOuterDimsPerm();
  if (!outerPerm.empty()) applyPermutationToVector(outerPos, outerPerm);
  SmallVector<int64_t> outerPositionPerm = computePermutationVector(
      packedRank, packingMetadata.outerPositions, outerPos);

  SmallVector<int64_t> packedToStripMinedShapePerm =
      lastDimsToInsertPositionsPerm;
  applyPermutationToVector(packedToStripMinedShapePerm, outerPositionPerm);

  // 2. Transpose packedShape to stripMinedShape.
  auto transposeOp = rewriter.create<memref::TransposeOp>(
      loc, unPackOp.getInput(),
      AffineMapAttr::get(AffineMap::getPermutationMap(
          packedToStripMinedShapePerm, unPackOp->getContext())));

  // 3. Inject a copy.
  SmallVector<Value, 2> mt;
  auto dmaOp = rewriter.create<xilinx::air::DmaMemcpyNdOp>(
      loc, SmallVector<Type, 1>{}, mt, unPackOp.getOutput(), mt, mt, mt,
      transposeOp->getResult(0), mt, mt, mt);

  // 4. Erase unPackOp.
  rewriter.eraseOp(unPackOp);

  return LowerUnPackResult{transposeOp, dmaOp};
}

/// A warpper pattern that calls linalg::lowerUnPack on tensor::UnPackOp. It
/// lowers a tensor.unpack op to tensor.empty + linalg.transpose +
/// tensor.collapse_shape + tensor.extract_slice ops.
struct LowerUnPackPattern : public OpRewritePattern<IREE::LinalgExt::UnPackOp> {
  using OpRewritePattern<IREE::LinalgExt::UnPackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::LinalgExt::UnPackOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<LowerUnPackResult> res = lowerUnPack(rewriter, op);
    if (failed(res)) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower to empty + transpose + collapse + subview");
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class AMDAIEDecomposeLinalgExtPackUnPackToAIRPass
    : public AMDAIEDecomposeLinalgExtPackUnPackToAIRBase<
          AMDAIEDecomposeLinalgExtPackUnPackToAIRPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, func::FuncDialect,
                    arith::ArithDialect, scf::SCFDialect, memref::MemRefDialect,
                    IREE::LinalgExt::IREELinalgExtDialect,
                    xilinx::air::airDialect>();
  }

  AMDAIEDecomposeLinalgExtPackUnPackToAIRPass() = default;
  AMDAIEDecomposeLinalgExtPackUnPackToAIRPass(
      const AMDAIEDecomposeLinalgExtPackUnPackToAIRPass &pass){};

  void runOnOperation() override;
};

void AMDAIEDecomposeLinalgExtPackUnPackToAIRPass::runOnOperation() {
  MLIRContext *ctx = &getContext();

  // Generalization patterns for outer unit dims have higher priority because
  // they do not generate reshape ops.
  {
    RewritePatternSet patterns(ctx);
    patterns.add<GeneralizePackOpPattern, GeneralizeUnPackOpPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  // Do not convert pack and unpack ops if outer dims are expected to always be
  // tiled to one.
  RewritePatternSet patterns(ctx);
  patterns.add<LowerPackPattern, LowerUnPackPattern>(ctx);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<>>
createAMDAIEDecomposeLinalgExtPackUnPackToAIRPass() {
  return std::make_unique<AMDAIEDecomposeLinalgExtPackUnPackToAIRPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
