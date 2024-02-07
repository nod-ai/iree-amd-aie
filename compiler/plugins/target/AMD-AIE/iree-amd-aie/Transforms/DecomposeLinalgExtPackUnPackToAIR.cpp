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

namespace mlir::iree_compiler::AMDAIE {

namespace {

// Normalizes a permutation on a higher rank space to its actual size, e.g.
//   perm = [1, 4, 2]
// becomes
//   norm = [0, 2, 1]
static SmallVector<int64_t> getPackUnpackNormalizedPerm(
    int rank, ArrayRef<int64_t> perm) {
  if (rank == perm.size()) {
    SmallVector<int64_t> vec;
    for (auto [index, value] : llvm::enumerate(perm)) vec.push_back(index);
    return vec;
  } else if (rank > perm.size()) {
    // Infer extra dimensions in the incomplete permutation list.
    SmallVector<int64_t> vec;
    for (auto i : llvm::seq<unsigned>(0, rank)) {
      if (llvm::any_of(perm, [i](int64_t elem) { return elem == i; })) continue;
      vec.push_back(i);
    }
    vec.insert(vec.end(), perm.begin(), perm.end());
    return vec;
  } else {
    assert(false &&
           "expected output permutation list's rank must not be less than the "
           "original permutation list");
    return SmallVector<int64_t>{};
  }
}

// Computes the permutation vector to shuffle packed shape into the shape
// before any outer or inner permutations have been applied. The permutation
// can be obtained from two permutations:
//   a) Compute the permutation vector to move the last `numPackedDims` into
//      the `innerPosDims` of a shape of rank `packedRank`.
//   b) Compute the permutation vector to move outer dims if the pack op
//      has outer_dims_perm.
// Apply (b) permutation on (a) permutation to get the final permutation.
static SmallVector<int64_t> getPackUnpackStripMinedPerm(
    ArrayRef<int64_t> shape, ArrayRef<int64_t> innerDimsPos,
    ArrayRef<int64_t> outerDimsPerm) {
  int64_t numPackedDims = innerDimsPos.size();
  int64_t packedRank = shape.size();
  auto lastDims = llvm::to_vector(
      llvm::seq<int64_t>(packedRank - numPackedDims, packedRank));
  PackingMetadata packingMetadata =
      computePackingMetadata(packedRank, innerDimsPos);
  SmallVector<int64_t> innerPositionsPerm = computePermutationVector(
      packedRank, lastDims, packingMetadata.insertPositions);

  SmallVector<int64_t> outerPos = packingMetadata.outerPositions;
  if (!outerDimsPerm.empty()) applyPermutationToVector(outerPos, outerDimsPerm);
  SmallVector<int64_t> outerPositionPerm = computePermutationVector(
      packedRank, packingMetadata.outerPositions, outerPos);

  SmallVector<int64_t> packedToStripMinedShapePerm = innerPositionsPerm;
  applyPermutationToVector(packedToStripMinedShapePerm, outerPositionPerm);

  return packedToStripMinedShapePerm;
}

struct LowerPackUnPackResult {
  memref::TransposeOp transposeOp;
  xilinx::air::DmaMemcpyNdOp dmaOp;
};

FailureOr<LowerPackUnPackResult> lowerPack(RewriterBase &rewriter,
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

  auto innerDimsPos = packOp.getInnerDimsPos();
  auto destShape = packOp.getOutputType().getShape();
  SmallVector<int64_t> transpPerm = {};
  Value tile = nullptr;
  if (llvm::any_of(innerDimsPos, [destShape](int64_t index) {
        return destShape[index] != 1;
      })) {
    // 1. Computes the permutation vector to shuffle packed shape into the shape
    // before any outer or inner permutations have been applied.
    PackingMetadata packingMetadata =
        computePackingMetadata(packedMemrefType.getRank(), innerDimsPos);

    SmallVector<int64_t> packedToStripMinedShapePerm =
        getPackUnpackStripMinedPerm(packedMemrefType.getShape(), innerDimsPos,
                                    packOp.getOuterDimsPerm());

    // 2. Compute the stripMinedShape: this is the packed shape before any outer
    // or inner permutations have been applied.
    SmallVector<int64_t> stripMinedShape(packedMemrefType.getShape());
    applyPermutationToVector(stripMinedShape, packedToStripMinedShapePerm);

    // 3. Expand from the padded result to the stripMinedShape.
    tile = rewriter.create<memref::ExpandShapeOp>(
        loc, stripMinedShape, packOp.getInput(),
        packingMetadata.reassociations);

    // 4. Transpose stripMinedShape to packedShape.
    transpPerm = invertPermutationVector(packedToStripMinedShapePerm);
  } else {
    tile = packOp.getInput();
    // Transpose the tile to match the inner tile order.
    transpPerm = getPackUnpackNormalizedPerm(packOp.getInputType().getRank(),
                                             innerDimsPos);
  }

  memref::TransposeOp transposeOp = rewriter.create<memref::TransposeOp>(
      loc, tile,
      AffineMapAttr::get(
          AffineMap::getPermutationMap(transpPerm, packOp->getContext())));

  SmallVector<Value, 2> emptyVec;
  xilinx::air::DmaMemcpyNdOp dmaOp =
      rewriter.create<xilinx::air::DmaMemcpyNdOp>(
          loc, SmallVector<Type, 1>{}, emptyVec, packOp.getOutput(), emptyVec,
          emptyVec, emptyVec, transposeOp.getResult(), emptyVec, emptyVec,
          emptyVec);

  // Erase packOp.
  rewriter.eraseOp(packOp);
  return LowerPackUnPackResult{transposeOp, dmaOp};
}

/// A wrapper pattern that calls lowerPack on PackOp. It lowers
/// a pack op to memref.expand_shape + memref.transpose ops.
struct LowerPackPattern : public OpRewritePattern<IREE::LinalgExt::PackOp> {
  using OpRewritePattern<IREE::LinalgExt::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::LinalgExt::PackOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<LowerPackUnPackResult> res = lowerPack(rewriter, op);
    if (failed(res)) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower to pad + expand + transpose");
    }
    return success();
  }
};

FailureOr<LowerPackUnPackResult> lowerUnPack(
    RewriterBase &rewriter, IREE::LinalgExt::UnPackOp unPackOp) {
  Location loc = unPackOp->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(unPackOp);

  ArrayRef<int64_t> srcShape = unPackOp.getInputType().getShape();
  ArrayRef<int64_t> innerDimsPos = unPackOp.getInnerDimsPos();

  SmallVector<int64_t> perm = {};
  Value tile = nullptr;

  if (llvm::any_of(innerDimsPos, [srcShape](int64_t index) {
        return srcShape[index] != 1;
      })) {
    MemRefType memrefType = unPackOp.getInputType().cast<MemRefType>();
    int64_t packedRank = memrefType.getRank();

    // Compute the permutation vector to move the last `numPackedDims` into
    // the `innerPosDims` of a shape of rank `packedRank`.
    PackingMetadata packingMetadata =
        computePackingMetadata(packedRank, innerDimsPos);

    perm = getPackUnpackStripMinedPerm(memrefType.getShape(), innerDimsPos,
                                       unPackOp.getOuterDimsPerm());

    tile = unPackOp.getInput();
  } else {
    int64_t srcRank = unPackOp.getInputRank();
    int64_t destRank = unPackOp.getOutputRank();
    if (llvm::any_of(innerDimsPos, [srcShape](int64_t index) {
          return srcShape[index] != 1;
        })) {
      return rewriter.notifyMatchFailure(
          unPackOp,
          "require the tiled outer dimensions of the result are all 1s");
    }

    // Use memref.subview op to extract the tile.
    Location loc = unPackOp.getLoc();
    Value input = unPackOp.getInput();
    DenseMap<int64_t, OpFoldResult> dimAndTileMapping =
        unPackOp.getDimAndTileMapping();
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

    auto mixedTiles = unPackOp.getMixedTiles();
    readSizes.append(mixedTiles.begin(), mixedTiles.end());

    // Explicitly create the type for subview op because the inner tile
    // size could be 1. We want to represent the whole inner tile in this case.
    auto tileShape = srcShape.drop_front(destRank);
    // Append the inner tile shape to the permuted and rank-reduced outer shape.
    readShape.append(tileShape.begin(), tileShape.end());
    Type elemType = unPackOp.getInputType().getElementType();
    Attribute memorySpace =
        unPackOp.getInputType().cast<MemRefType>().getMemorySpace();
    auto readType = MemRefType::get(readShape, elemType, nullptr, memorySpace);
    tile = rewriter.create<memref::SubViewOp>(loc, readType, input, readOffsets,
                                              readSizes, readStrides);
    perm = getPackUnpackNormalizedPerm(readType.getRank(), innerDimsPos);
    perm = invertPermutationVector(perm);
  }

  // Transpose packedShape to stripMinedShape.
  memref::TransposeOp transposeOp = rewriter.create<memref::TransposeOp>(
      loc, tile,
      AffineMapAttr::get(
          AffineMap::getPermutationMap(perm, unPackOp->getContext())));

  // Inject a copy.
  SmallVector<Value, 2> emptyVec;
  xilinx::air::DmaMemcpyNdOp dmaOp =
      rewriter.create<xilinx::air::DmaMemcpyNdOp>(
          loc, SmallVector<Type, 1>{}, emptyVec, unPackOp.getOutput(), emptyVec,
          emptyVec, emptyVec, transposeOp.getResult(), emptyVec, emptyVec,
          emptyVec);

  // Erase unPackOp.
  rewriter.eraseOp(unPackOp);

  return LowerPackUnPackResult{transposeOp, dmaOp};
}

/// A warpper pattern that calls lowerUnPack on IREE::LinalgExt::UnPackOp. It
/// lowers a iree_linalg_ext.unpack op to memref.transpose + memref.subview ops.
struct LowerUnPackPattern : public OpRewritePattern<IREE::LinalgExt::UnPackOp> {
  using OpRewritePattern<IREE::LinalgExt::UnPackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::LinalgExt::UnPackOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<LowerPackUnPackResult> res = lowerUnPack(rewriter, op);
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
    : public impl::AMDAIEDecomposeLinalgExtPackUnPackToAIRBase<
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

  // Second-stage lowering of pack and unpack ops.
  RewritePatternSet patterns(ctx);
  patterns.add<LowerPackPattern, LowerUnPackPattern>(ctx);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEDecomposeLinalgExtPackUnPackToAIRPass() {
  return std::make_unique<AMDAIEDecomposeLinalgExtPackUnPackToAIRPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
