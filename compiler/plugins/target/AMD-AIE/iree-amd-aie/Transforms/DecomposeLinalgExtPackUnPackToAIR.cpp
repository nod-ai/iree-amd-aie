// Copyright 2023 The IREE Authors
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

/// Returns a pad op if padding value is set. Otherwise, returns the
/// source directly. The method assumes that the `packOp` has static shapes.
/// TODO: padding
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
  // TODO: padding
  memref::ExpandShapeOp expandShapeOp;
  memref::TransposeOp transposeOp;
  xilinx::air::DmaMemcpyNdOp dmaOp;
};

struct LowerUnPackResult {
  // TODO: padding
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

  // 4. Pad the source of packOp to a shape we can expand into stripMinedShape.
  // TODO

  // 5. Expand from the padded result to the stripMinedShape.
  auto reshapeOp = rewriter.create<memref::ExpandShapeOp>(
      loc, stripMinedShape, packOp.getInput(), packingMetadata.reassociations);

  // 6. Transpose stripMinedShape to packedShape.
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

  // 7. Replace packOp by transposeOp.
  rewriter.eraseOp(packOp);

  return LowerPackResult{reshapeOp, transposeOp, dmaOp};
}

/// A wrapper pattern that calls lowerPack on PackOp. It lowers
/// a pack op to (TODO: pad) + memref.expand_shape + memref.transpose ops.
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

  // 1. Unpad the source of unPackOp
  // TODO

  // 2. Compute the permutation vector to move the last `numPackedDims` into
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

  // 3. Transpose packedShape to stripMinedShape.
  auto transposeOp = rewriter.create<memref::TransposeOp>(
      loc, unPackOp.getInput(),
      AffineMapAttr::get(AffineMap::getPermutationMap(
          packedToStripMinedShapePerm, unPackOp->getContext())));

  // 4. Inject a copy.
  SmallVector<Value, 2> mt;
  auto dmaOp = rewriter.create<xilinx::air::DmaMemcpyNdOp>(
      loc, SmallVector<Type, 1>{}, mt, unPackOp.getOutput(), mt, mt, mt,
      transposeOp->getResult(0), mt, mt, mt);

  // 5. Erase unPackOp.
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

/// Build a strided memref type by applying `permutationMap` tp `memRefType`.
static MemRefType inferTransposeResultType(MemRefType memRefType,
                                           AffineMap permutationMap) {
  auto rank = memRefType.getRank();
  auto originalSizes = memRefType.getShape();
  auto [originalStrides, offset] = getStridesAndOffset(memRefType);
  assert(originalStrides.size() == static_cast<unsigned>(rank));

  // Compute permuted sizes and strides.
  SmallVector<int64_t> sizes(rank, 0);
  SmallVector<int64_t> strides(rank, 1);
  for (const auto &en : llvm::enumerate(permutationMap.getResults())) {
    unsigned position = cast<AffineDimExpr>(en.value()).getPosition();
    sizes[en.index()] = originalSizes[position];
    strides[en.index()] = originalStrides[position];
  }

  return MemRefType::Builder(memRefType)
      .setShape(sizes)
      .setLayout(
          StridedLayoutAttr::get(memRefType.getContext(), offset, strides));
}

static void extractStridesFromMemrefType(MemRefType memrefTy,
                                         OpBuilder &builder,
                                         SmallVector<Value, 4> &strides) {
  // get the strides and offsets from the memref type
  int64_t offset;
  SmallVector<int64_t, 4> layout_strides;
  auto successStrides = getStridesAndOffset(memrefTy, layout_strides, offset);
  if (failed(successStrides)) {
    llvm::outs() << "Failed to get strides\n";
    return;
  }

  for (auto s : layout_strides)
    strides.push_back(
        builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), s));
}

static void extractSizesFromMemrefType(MemRefType memrefTy, OpBuilder &builder,
                                       SmallVector<Value, 4> &sizes) {
  for (auto s : memrefTy.getShape())
    sizes.push_back(
        builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), s));
}

static void extractOffsetsFromSubview(memref::SubViewOp subview,
                                      OpBuilder &builder,
                                      SmallVector<Value, 4> &offsets) {
  auto subview_offsets = subview.getOffsets().begin();
  auto static_offsets = subview.getStaticOffsets();
  auto loc = subview.getLoc();

  for (auto o : static_offsets) {
    if (o >= 0)
      offsets.push_back(builder.create<arith::ConstantIndexOp>(loc, o));
    else
      offsets.push_back(*subview_offsets++);
  }
}

static LogicalResult canonicalizeAIRDmaOperands(OpBuilder builder,
                                                SmallVector<Value, 4> &offsets,
                                                SmallVector<Value, 4> &sizes,
                                                SmallVector<Value, 4> &strides,
                                                MemRefType memref) {
  // Increase vector sizes up to memref size
  int max_dim_size =
      std::max(std::max(offsets.size(), sizes.size()), strides.size());
  if (max_dim_size && offsets.size() < memref.getRank()) {
    for (unsigned i = offsets.size(); i < memref.getRank(); i++) {
      offsets.insert(offsets.begin(), builder.create<arith::ConstantIndexOp>(
                                          builder.getUnknownLoc(), 0));
    }
  }
  if (max_dim_size && sizes.size() < memref.getRank()) {
    for (unsigned i = sizes.size(); i < memref.getRank(); i++) {
      sizes.insert(sizes.begin(), builder.create<arith::ConstantIndexOp>(
                                      builder.getUnknownLoc(), 1));
    }
  }
  int memref_size = 1;
  for (auto size : memref.getShape()) memref_size *= size;
  if (max_dim_size && strides.size() < memref.getRank()) {
    for (unsigned i = strides.size(); i < memref.getRank(); i++) {
      strides.insert(strides.begin(),
                     builder.create<arith::ConstantIndexOp>(
                         builder.getUnknownLoc(), memref_size));
    }
  }

  // Reduce highest dimensions if more than memref size
  while (strides.size() > memref.getRank() && getConstantIntValue(strides[0]) &&
         *getConstantIntValue(strides[0]) == memref_size) {
    strides.erase(strides.begin());
  }
  while (sizes.size() > memref.getRank() && getConstantIntValue(sizes[0]) &&
         *getConstantIntValue(sizes[0]) == 1) {
    sizes.erase(sizes.begin());
  }
  while (offsets.size() > std::min(sizes.size(), strides.size()) &&
         getConstantIntValue(offsets[0]) &&
         *getConstantIntValue(offsets[0]) == 0) {
    offsets.erase(offsets.begin());
  }

  if (offsets.size() != sizes.size() || sizes.size() != strides.size())
    return failure();

  return success();
}

static LogicalResult CondenseMemrefDataReorderingToAIRDma(
    xilinx::air::DmaMemcpyNdOp dmaOp,
    std::vector<Operation *> src_ancestor_memref_ops,
    std::vector<Operation *> dst_ancestor_memref_ops) {
  OpBuilder rewriter(dmaOp);
  auto src = dmaOp.getSrcMemref();
  auto dst = dmaOp.getDstMemref();
  auto loc = dmaOp->getLoc();

  // It must already be a memref
  auto src_type = src.getType().dyn_cast<MemRefType>();
  auto dst_type = dst.getType().dyn_cast<MemRefType>();
  if (!src_type) return failure();
  if (!(src_type.hasStaticShape() || dst_type.hasStaticShape()))
    return failure();

  // Revert the vector of memref ops, as it was built with push_back.
  std::reverse(src_ancestor_memref_ops.begin(), src_ancestor_memref_ops.end());
  std::reverse(dst_ancestor_memref_ops.begin(), dst_ancestor_memref_ops.end());

  SmallVector<Value, 4> src_offsets, dst_offsets;
  SmallVector<Value, 4> src_strides, dst_strides;
  SmallVector<Value, 4> src_sizes, dst_sizes;
  SmallVector<Value, 4> empty;

  MemRefType src_memref_ty;
  if (!src_ancestor_memref_ops.empty()) {
    if (auto subviewOp =
            dyn_cast<memref::SubViewOp>(src_ancestor_memref_ops[0])) {
      extractOffsetsFromSubview(subviewOp, rewriter, src_offsets);
      src_memref_ty = subviewOp.getSourceType();
      src = subviewOp.getSource();
    } else if (auto transposeOp =
                   dyn_cast<memref::TransposeOp>(src_ancestor_memref_ops[0])) {
      src_memref_ty = transposeOp.getIn().getType().cast<MemRefType>();
      src = transposeOp.getIn();
    }
  }
  MemRefType dst_memref_ty;
  if (!dst_ancestor_memref_ops.empty()) {
    if (auto subviewOp =
            dyn_cast<memref::SubViewOp>(dst_ancestor_memref_ops[0])) {
      extractOffsetsFromSubview(subviewOp, rewriter, dst_offsets);
      dst_memref_ty = subviewOp.getSourceType();
      dst = subviewOp.getSource();
    } else if (auto transposeOp =
                   dyn_cast<memref::TransposeOp>(dst_ancestor_memref_ops[0])) {
      dst_memref_ty = transposeOp.getIn().getType().cast<MemRefType>();
      dst = transposeOp.getIn();
    }
  }

  for (auto memrefOp : src_ancestor_memref_ops) {
    if (auto transposeOp = dyn_cast<memref::TransposeOp>(memrefOp)) {
      src_memref_ty =
          inferTransposeResultType(src_memref_ty, transposeOp.getPermutation());
    } else if (auto expandShapeOp = dyn_cast<memref::ExpandShapeOp>(memrefOp)) {
      FailureOr<MemRefType> compute_expand =
          memref::ExpandShapeOp::computeExpandedType(
              src_memref_ty, expandShapeOp.getResultType().getShape(),
              expandShapeOp.getReassociationIndices());
      if (failed(compute_expand)) {
        assert(false);
      } else {
        src_memref_ty = *compute_expand;
      }
    } else if (auto subviewOp = dyn_cast<memref::SubViewOp>(memrefOp)) {
      // Check if subview is rank reduced
      if (subviewOp.getSourceType().getRank() > subviewOp.getType().getRank())
        src_memref_ty =
            memref::SubViewOp::inferRankReducedResultType(
                subviewOp.getType().getShape(), src_memref_ty,
                subviewOp.getStaticOffsets(), subviewOp.getStaticSizes(),
                subviewOp.getStaticStrides())
                .cast<MemRefType>();
      else
        src_memref_ty =
            memref::SubViewOp::inferResultType(
                src_memref_ty, subviewOp.getStaticOffsets(),
                subviewOp.getStaticSizes(), subviewOp.getStaticStrides())
                .cast<MemRefType>();
    }
  }

  for (auto memrefOp : dst_ancestor_memref_ops) {
    if (auto transposeOp = dyn_cast<memref::TransposeOp>(memrefOp)) {
      dst_memref_ty =
          inferTransposeResultType(dst_memref_ty, transposeOp.getPermutation());
    } else if (auto expandShapeOp = dyn_cast<memref::ExpandShapeOp>(memrefOp)) {
      FailureOr<MemRefType> compute_expand =
          memref::ExpandShapeOp::computeExpandedType(
              dst_memref_ty, expandShapeOp.getResultType().getShape(),
              expandShapeOp.getReassociationIndices());
      if (failed(compute_expand)) {
        assert(false);
      } else {
        dst_memref_ty = *compute_expand;
      }
    } else if (auto subviewOp = dyn_cast<memref::SubViewOp>(memrefOp)) {
      if (subviewOp.getSourceType().getRank() > subviewOp.getType().getRank())
        dst_memref_ty =
            memref::SubViewOp::inferRankReducedResultType(
                subviewOp.getType().getShape(), dst_memref_ty,
                subviewOp.getStaticOffsets(), subviewOp.getStaticSizes(),
                subviewOp.getStaticStrides())
                .cast<MemRefType>();
      else
        dst_memref_ty =
            memref::SubViewOp::inferResultType(
                dst_memref_ty, subviewOp.getStaticOffsets(),
                subviewOp.getStaticSizes(), subviewOp.getStaticStrides())
                .cast<MemRefType>();
    }
  }

  if (src_ancestor_memref_ops.size()) {
    extractStridesFromMemrefType(src_memref_ty, rewriter, src_strides);
    extractSizesFromMemrefType(src_memref_ty, rewriter, src_sizes);
  }
  if (dst_ancestor_memref_ops.size()) {
    extractStridesFromMemrefType(dst_memref_ty, rewriter, dst_strides);
    extractSizesFromMemrefType(dst_memref_ty, rewriter, dst_sizes);
  }

  SmallVector<Value, 4> deps;
  SmallVector<Type, 4> tys;

  if (failed(canonicalizeAIRDmaOperands(rewriter, src_offsets, src_sizes,
                                        src_strides,
                                        src.getType().cast<MemRefType>())) ||
      failed(canonicalizeAIRDmaOperands(rewriter, dst_offsets, dst_sizes,
                                        dst_strides,
                                        dst.getType().cast<MemRefType>()))) {
    assert(false);
  }
  auto new_dma = rewriter.create<xilinx::air::DmaMemcpyNdOp>(
      loc, tys, deps, dst, dst_offsets, dst_sizes, dst_strides, src,
      src_offsets, src_sizes, src_strides);

  assert(!new_dma.getSrcMemref().getDefiningOp<memref::TransposeOp>());
  assert(!new_dma.getDstMemref().getDefiningOp<memref::TransposeOp>());

  dmaOp->erase();

  return success();
}

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

  // Condense memref data pattern reordering ops, including memref.subview,
  // memref.tranpose and memref.expand_shape into air.dma_memcpy_nd op's
  // offsets, sizes and strides fields.
  auto scope = getOperation();
  std::vector<std::tuple<xilinx::air::DmaMemcpyNdOp, std::vector<Operation *>,
                         std::vector<Operation *>>>
      dma_ops;

  scope->walk([&](xilinx::air::DmaMemcpyNdOp dmaOp) {
    bool src_condense = false;
    src_condense |=
        isa<memref::TransposeOp>(dmaOp.getSrcMemref().getDefiningOp());
    src_condense |=
        isa<memref::ExpandShapeOp>(dmaOp.getSrcMemref().getDefiningOp());
    src_condense |=
        isa<memref::SubViewOp>(dmaOp.getSrcMemref().getDefiningOp());
    bool dst_condense = false;
    dst_condense |=
        isa<memref::TransposeOp>(dmaOp.getDstMemref().getDefiningOp());
    dst_condense |=
        isa<memref::ExpandShapeOp>(dmaOp.getDstMemref().getDefiningOp());
    dst_condense |=
        isa<memref::SubViewOp>(dmaOp.getDstMemref().getDefiningOp());
    if (src_condense || dst_condense) {
      std::tuple<xilinx::air::DmaMemcpyNdOp, std::vector<Operation *>,
                 std::vector<Operation *>>
          log_entry;
      std::get<0>(log_entry) = dmaOp;
      if (src_condense) {
        Operation *ancestor = dmaOp.getSrcMemref().getDefiningOp();
        bool exit = false;
        while (ancestor && !exit) {
          if (auto transpose_anc = dyn_cast<memref::TransposeOp>(ancestor)) {
            std::get<1>(log_entry).push_back(ancestor);
            ancestor = transpose_anc.getIn().getDefiningOp();
          } else if (auto expand_anc =
                         dyn_cast<memref::ExpandShapeOp>(ancestor)) {
            std::get<1>(log_entry).push_back(ancestor);
            ancestor = expand_anc.getSrc().getDefiningOp();
          } else if (auto subview_anc = dyn_cast<memref::SubViewOp>(ancestor)) {
            std::get<1>(log_entry).push_back(ancestor);
            ancestor = subview_anc.getSource().getDefiningOp();
          } else
            exit = true;
        }
      }
      if (dst_condense) {
        Operation *ancestor = dmaOp.getDstMemref().getDefiningOp();
        bool exit = false;
        while (ancestor && !exit) {
          if (auto transpose_anc = dyn_cast<memref::TransposeOp>(ancestor)) {
            std::get<2>(log_entry).push_back(ancestor);
            ancestor = transpose_anc.getIn().getDefiningOp();
          } else if (auto expand_anc =
                         dyn_cast<memref::ExpandShapeOp>(ancestor)) {
            std::get<2>(log_entry).push_back(ancestor);
            ancestor = expand_anc.getSrc().getDefiningOp();
          } else if (auto subview_anc = dyn_cast<memref::SubViewOp>(ancestor)) {
            std::get<2>(log_entry).push_back(ancestor);
            ancestor = subview_anc.getSource().getDefiningOp();
          } else
            exit = true;
        }
      }
      dma_ops.push_back(log_entry);
    }
  });
  for (auto dmaOp : dma_ops) {
    if (failed(CondenseMemrefDataReorderingToAIRDma(
            std::get<0>(dmaOp), std::get<1>(dmaOp), std::get<2>(dmaOp)))) {
      return signalPassFailure();
    }
  }
  // Erase condensed memref ops
  for (auto dmaOp : dma_ops) {
    for (auto memref_op : std::get<1>(dmaOp)) memref_op->erase();
    for (auto memref_op : std::get<2>(dmaOp)) memref_op->erase();
  }
}

}  // namespace

std::unique_ptr<OperationPass<>>
createAMDAIEDecomposeLinalgExtPackUnPackToAIRPass() {
  return std::make_unique<AMDAIEDecomposeLinalgExtPackUnPackToAIRPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
