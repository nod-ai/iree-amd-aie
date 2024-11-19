// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-linearize-memref-type"

namespace mlir::iree_compiler::AMDAIE {

namespace {

static Value getIndicesForLoadOrStore(OpBuilder &builder, Location loc,
                                      OpFoldResult linearizedIndex,
                                      int64_t srcBits, int64_t dstBits) {
  AffineExpr s0;
  bindSymbols(builder.getContext(), s0);
  int64_t scaler = dstBits / srcBits;
  OpFoldResult scaledLinearizedIndices = affine::makeComposedFoldedAffineApply(
      builder, loc, s0.floorDiv(scaler), {linearizedIndex});
  return getValueOrCreateConstantIndexOp(builder, loc, scaledLinearizedIndices);
}

static std::pair<memref::LinearizedMemRefInfo, OpFoldResult> getLinearizedMemRefOffsetAndSize(
    OpBuilder &builder, Location loc, int srcBits, int dstBits,
    OpFoldResult offset, ArrayRef<OpFoldResult> sizes,
    ArrayRef<OpFoldResult> strides, ArrayRef<OpFoldResult> indices) {
  unsigned sourceRank = sizes.size();
  assert(sizes.size() == strides.size() &&
         "expected as many sizes as strides for a memref");
  SmallVector<OpFoldResult> indicesVec = llvm::to_vector(indices);
  if (indices.empty())
    indicesVec.resize(sourceRank, builder.getIndexAttr(0));
  assert(indicesVec.size() == strides.size() &&
         "expected as many indices as rank of memref");

  // Create the affine symbols and values for linearization.
  SmallVector<AffineExpr> symbols(2 * sourceRank);
  bindSymbolsList(builder.getContext(), MutableArrayRef{symbols});
  AffineExpr addMulMap = builder.getAffineConstantExpr(0);
  AffineExpr mulMap = builder.getAffineConstantExpr(1);

  SmallVector<OpFoldResult> offsetValues(2 * sourceRank);

  for (unsigned i = 0; i < sourceRank; ++i) {
    unsigned offsetIdx = 2 * i;
    addMulMap = addMulMap + symbols[offsetIdx] * symbols[offsetIdx + 1];
    offsetValues[offsetIdx] = indicesVec[i];
    offsetValues[offsetIdx + 1] = strides[i];

    mulMap = mulMap * symbols[i];
  }

  // Adjust linearizedIndices and size by the scale factor (dstBits / srcBits).
  int64_t scaler = dstBits / srcBits;
  mulMap = mulMap.floorDiv(scaler);

  OpFoldResult linearizedIndices = affine::makeComposedFoldedAffineApply(
      builder, loc, addMulMap.floorDiv(scaler), offsetValues);
  // rewriter.create<affine::AffineLinearizeIndexOp>(
  //       loc, type, newIndices, newBasis, op.getDisjoint());
  OpFoldResult linearizedSize =
      affine::makeComposedFoldedAffineApply(builder, loc, mulMap, sizes);

  // Adjust baseOffset by the scale factor (dstBits / srcBits).
  AffineExpr s0;
  bindSymbols(builder.getContext(), s0);
  OpFoldResult adjustBaseOffset = affine::makeComposedFoldedAffineApply(
      builder, loc, s0.floorDiv(scaler), {offset});

  OpFoldResult intraVectorOffset = affine::makeComposedFoldedAffineApply(
      builder, loc, addMulMap % scaler, offsetValues);

  return {{adjustBaseOffset, linearizedSize, intraVectorOffset},
          linearizedIndices};
}

static OpFoldResult
getLinearizedSrcIndices(OpBuilder &builder, Location loc, int64_t srcBits,
                        const SmallVector<OpFoldResult> &indices,
                        Value memref) {
  auto stridedMetadata =
      builder.create<memref::ExtractStridedMetadataOp>(loc, memref);
  OpFoldResult linearizedIndices;
  std::tie(std::ignore, linearizedIndices) =
      getLinearizedMemRefOffsetAndSize(
          builder, loc, srcBits, srcBits,
          stridedMetadata.getConstifiedMixedOffset(),
          stridedMetadata.getConstifiedMixedSizes(),
          stridedMetadata.getConstifiedMixedStrides(), indices);
  return linearizedIndices;
}

static SmallVector<int64_t> getLinearizedShape(MemRefType ty, int srcBits,
                                               int dstBits) {
  if (ty.getRank() == 0) return {};

  int64_t linearizedShape = 1;
  for (auto shape : ty.getShape()) {
    if (shape == ShapedType::kDynamic) return {ShapedType::kDynamic};
    linearizedShape *= shape;
  }
  int scale = dstBits / srcBits;
  // Scale the size to the ceilDiv(linearizedShape, scale)
  // to accomodate all the values.
  linearizedShape = (linearizedShape + scale - 1) / scale;
  return {linearizedShape};
}

static LogicalResult linearizeType(MemRefType memrefType,
                                   MemRefType &newMemrefType) {
  // Fetch linearized shape.
  // TODO(avarma): Take into account different src/dst bits.
  int srcBits = memrefType.getElementType().getIntOrFloatBitWidth();
  SmallVector<int64_t> linearizedShape =
      getLinearizedShape(memrefType, srcBits, srcBits);
  // Fetch offset and strides of the old memref.
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(memrefType, strides, offset)))
    return failure();
  if (!strides.empty() && strides.back() != 1) return failure();
  // Form layout for the linearized memref.
  StridedLayoutAttr layoutAttr;
  // If the offset is 0, we do not need a strided layout as the stride is
  // 1, so we only use the strided layout if the offset is not 0.
  if (offset != 0) {
    layoutAttr = StridedLayoutAttr::get(memrefType.getContext(), offset,
                                        ArrayRef<int64_t>{1});
  }
  Type elementType = memrefType.getElementType();
  newMemrefType = MemRefType::get(linearizedShape, elementType, layoutAttr,
                                  memrefType.getMemorySpace());
  return success();
}

static LogicalResult getLinearizedTypeFromSourceType(
    MemRefType currentTypeOfSourceMemref, MemRefType &linearizedType) {
  if (!currentTypeOfSourceMemref) return failure();
  if (currentTypeOfSourceMemref.getRank() < 2) return success();
  // Convert current type later.
  return linearizeType(currentTypeOfSourceMemref, linearizedType);
}

template <typename OpTy>
struct LinearizeMemrefAlloc : public OpRewritePattern<OpTy> {
  LinearizeMemrefAlloc(MLIRContext *context, PatternBenefit benefit = 10)
      : OpRewritePattern<OpTy>(context, benefit) {}

  LogicalResult matchAndRewrite(OpTy allocOp,
                                PatternRewriter &rewriter) const override {
    static_assert(std::is_same<OpTy, memref::AllocOp>() ||
                      std::is_same<OpTy, memref::AllocaOp>(),
                  "expected only memref::AllocOp or memref::AllocaOp");
    Location loc = allocOp->getLoc();
    MemRefType currentTypeOfSourceMemref =
        dyn_cast<MemRefType>(allocOp.getMemref().getType());
    MemRefType newTypeOfSourceMemref;
    if (failed(getLinearizedTypeFromSourceType(currentTypeOfSourceMemref,
                                               newTypeOfSourceMemref))) {
      return failure();
    }
    if (currentTypeOfSourceMemref.getRank() < 2) return success();

    auto elementType = currentTypeOfSourceMemref.getElementType();
    int srcBits = elementType.getIntOrFloatBitWidth();

    OpFoldResult zero = rewriter.getIndexAttr(0);

    // Get linearized type.
    int dstBits = srcBits;
    SmallVector<OpFoldResult> sizes = allocOp.getMixedSizes();

    memref::LinearizedMemRefInfo linearizedMemRefInfo =
        memref::getLinearizedMemRefOffsetAndSize(
            rewriter, loc, srcBits, dstBits, /*offset =*/zero, sizes);
    SmallVector<Value> dynamicLinearizedSize;
    if (!newTypeOfSourceMemref.hasStaticShape()) {
      dynamicLinearizedSize.push_back(getValueOrCreateConstantIndexOp(
          rewriter, loc, linearizedMemRefInfo.linearizedSize));
    }

    rewriter.replaceOpWithNewOp<OpTy>(
        allocOp, newTypeOfSourceMemref, dynamicLinearizedSize,
        allocOp.getSymbolOperands(), allocOp.getAlignmentAttr());
    return success();
  }
};

struct LinearizeMemrefLoad
    : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    Location loc = loadOp->getLoc();
    MemRefType currentTypeOfSourceMemref =
        dyn_cast<MemRefType>(loadOp.getMemref().getType());
    MemRefType newTypeOfSourceMemref;
    if (failed(getLinearizedTypeFromSourceType(currentTypeOfSourceMemref,
                                               newTypeOfSourceMemref))) {
      return failure();
    }
    if (currentTypeOfSourceMemref.getRank() < 2 &&
        loadOp.getIndices().size() < 2)
      return success();

    auto elementType = loadOp.getMemRefType().getElementType();
    int srcBits = elementType.getIntOrFloatBitWidth();
    Value linearizedIndices = rewriter.create<affine::AffineLinearizeIndexOp>(
        loc, loadOp.getIndices(), currentTypeOfSourceMemref.getShape(), true);
    auto reinterpretOp = rewriter.create<memref::ReinterpretCastOp>(
        loc, newTypeOfSourceMemref, loadOp.getMemref(), 0,
        newTypeOfSourceMemref.getShape(), ArrayRef<int64_t>({1}));
    Value linearizedLoad = rewriter.create<memref::LoadOp>(
        loc, reinterpretOp,
        getIndicesForLoadOrStore(rewriter, loc, linearizedIndices, srcBits,
                                 srcBits));

    rewriter.replaceOp(loadOp, {linearizedLoad});
    return success();
  }
};

class AMDAIELinearizeMemrefTypePass
    : public impl::AMDAIELinearizeMemrefTypeBase<
          AMDAIELinearizeMemrefTypePass> {
 public:
  AMDAIELinearizeMemrefTypePass() = default;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() override;
};

void AMDAIELinearizeMemrefTypePass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  RewritePatternSet patterns(context);
  patterns.add<LinearizeMemrefAlloc<memref::AllocOp>>(context);
  patterns.add<LinearizeMemrefAlloc<memref::AllocaOp>>(context);
  patterns.add<LinearizeMemrefLoad>(context);
  (void)applyPatternsAndFoldGreedily(moduleOp, std::move(patterns));

  return;
}

}  // namespace

std::unique_ptr<Pass> createAMDAIELinearizeMemrefTypePass() {
  return std::make_unique<AMDAIELinearizeMemrefTypePass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
