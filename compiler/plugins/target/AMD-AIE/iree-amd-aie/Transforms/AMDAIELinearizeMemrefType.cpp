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

// static memref::LinearizedMemRefInfo
// getLinearizedMemRefOffsetAndSize(OpBuilder &builder, Location loc, int srcBits,
//                                  int dstBits, OpFoldResult offset,
//                                  ArrayRef<OpFoldResult> sizes) {
//   SmallVector<OpFoldResult> strides(sizes.size());
//   if (!sizes.empty()) {
//     strides.back() = builder.getIndexAttr(1);
//     AffineExpr s0, s1;
//     bindSymbols(builder.getContext(), s0, s1);
//     for (int index = sizes.size() - 1; index > 0; --index) {
//       strides[index - 1] = affine::makeComposedFoldedAffineApply(
//           builder, loc, s0 * s1,
//           ArrayRef<OpFoldResult>{strides[index], sizes[index]});
//     }
//   }

//   memref::LinearizedMemRefInfo linearizedMemRefInfo;
//   std::tie(linearizedMemRefInfo, std::ignore) =
//       _getLinearizedMemRefOffsetAndSize(builder, loc, srcBits, dstBits, offset,
//                                        sizes, strides);
//   return linearizedMemRefInfo;
// }

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
    if (offset == ShapedType::kDynamic) {
      layoutAttr = StridedLayoutAttr::get(memrefType.getContext(), offset,
                                          ArrayRef<int64_t>{1});
    } else {
      // TODO(avarma): Take into account different src/dst bits.
      // // Check if the number of bytes are a multiple of the loadStoreWidth
      // // and if so, divide it by the loadStoreWidth to get the offset.
      // if ((offset * width) % loadStoreWidth != 0)
      //   return std::nullopt;
      // offset = (offset * width) / loadStoreWidth;

      layoutAttr = StridedLayoutAttr::get(memrefType.getContext(), offset,
                                          ArrayRef<int64_t>{1});
    }
  }
  Type elementType = memrefType.getElementType();
  newMemrefType = MemRefType::get(linearizedShape, elementType, layoutAttr,
                                  memrefType.getMemorySpace());
  return success();
}

static LogicalResult getLinearizedTypeFromSourceType(Type sourceType, MemRefType &linearizedType) {
  auto currentTypeOfSourceMemref = dyn_cast<MemRefType>(sourceType);
  if (!currentTypeOfSourceMemref) return failure();
  if (currentTypeOfSourceMemref.getRank() < 2) {
    linearizedType = currentTypeOfSourceMemref;
    return success();
  }
  // Convert current type later.
  if (failed(linearizeType(currentTypeOfSourceMemref, linearizedType))) {
    return failure();
  }
}

template <typename OpTy>
struct ConvertMemRefAllocation : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy allocOp,
                                PatternRewriter &rewriter) const override {
    static_assert(std::is_same<OpTy, memref::AllocOp>() ||
                      std::is_same<OpTy, memref::AllocaOp>(),
                  "expected only memref::AllocOp or memref::AllocaOp");
    Location loc = allocOp->getLoc();
    MemRefType newTypeOfSourceMemref;
    if (failed(getLinearizedTypeFromSourceType(allocOp.getMemref().getType()))) {
      return failure();
    }
    if (newTypeOfSourceMemref < 2) return success();
    
    auto elementType = loadOp.getMemRefType().getElementType();
    int srcBits = elementType.getIntOrFloatBitWidth();

    OpFoldResult zero = rewriter.getIndexAttr(0);
    SmallVector<OpFoldResult> indices(currentType.getRank(), zero);

    // Get linearized type.
    int srcBits = currentType.getElementType().getIntOrFloatBitWidth();
    int dstBits = newResultType.getElementType().getIntOrFloatBitWidth();
    SmallVector<OpFoldResult> sizes = op.getMixedSizes();

    memref::LinearizedMemRefInfo linearizedMemRefInfo =
        memref::getLinearizedMemRefOffsetAndSize(
            rewriter, loc, srcBits, dstBits, /*offset =*/zero, sizes);
    SmallVector<Value> dynamicLinearizedSize;
    if (!newResultType.hasStaticShape()) {
      dynamicLinearizedSize.push_back(getValueOrCreateConstantIndexOp(
          rewriter, loc, linearizedMemRefInfo.linearizedSize));
    }

    rewriter.replaceOpWithNewOp<OpTy>(op, newResultType, dynamicLinearizedSize,
                                      adaptor.getSymbolOperands(),
                                      adaptor.getAlignmentAttr());
    return success();
  }
};

struct LinearizeMemrefLoad
    : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    Location loc = loadOp->getLoc();
    MemRefType newTypeOfSourceMemref;
    if (failed(getLinearizedTypeFromSourceType(loadOp.getMemref().getType()))) {
      return failure();
    }
    if (newTypeOfSourceMemref < 2) return success();
    
    auto elementType = loadOp.getMemRefType().getElementType();
    int srcBits = elementType.getIntOrFloatBitWidth();
    // Linearize the indices of the original load instruction. Do not account
    // for the scaling yet. This will be accounted for later.
    OpFoldResult linearizedIndices = getLinearizedSrcIndices(
        rewriter, loc, srcBits, loadOp.getIndices(), loadOp.getMemRef());

    Value linearizedLoad = rewriter.create<memref::LoadOp>(
        loc, loadOp.getMemref(),
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
  patterns.add<LinearizeMemrefLoad>(context);
  (void)applyPatternsAndFoldGreedily(moduleOp, std::move(patterns));

  // moduleOp.walk([&](Operation *op) {
  //   if (isa<func::FuncOp, func::ReturnOp, arith::ConstantOp>(op))
  //     return WalkResult::skip();
  //   // TODO(avarma): Except funcOps. This will be improved later. And the
  //   // following will be pulled out to a PatterRewriter later.
  //   for (Value operand : op->getOperands()) {
  //     auto currentType = dyn_cast<MemRefType>(operand.getType());
  //     if (!currentType) {
  //       continue;
  //       // return rewriter.notifyMatchFailure(op->getLoc(),
  //       //                                    "unhandled non-memref types");
  //     }
  //     // Convert current type later.
  //     MemRefType newResultType;
  //     if (failed(linearizeType(currentType, newResultType)))
  //       return WalkResult::interrupt();

  //     // if (!newResultType) {
  //     //     return;
  //     //     // return rewriter.notifyMatchFailure(
  //     //     //     op->getLoc(),
  //     //     //     llvm::formatv("failed to legalize memref type: {0}",
  //     //     //     op.getType()));
  //     //   }
  //     // Location loc = op->getLoc();
  //     OpFoldResult zero = rewriter.getIndexAttr(0);
  //     SmallVector<OpFoldResult> indices(currentType.getRank(), zero);

  //     // Get linearized type.
  //     // int srcBits = currentType.getElementType().getIntOrFloatBitWidth();
  //     // int dstBits = newResultType.getElementType().getIntOrFloatBitWidth();
  //     llvm::outs() << "SRC TYPE := " << currentType << "\n";
  //     llvm::outs() << "NEW TYPE := " << newResultType << "\n";

  //     // Linearize the indices of the original load instruction. Do not account
  //     // for the scaling yet. This will be accounted for later.
  //     OpFoldResult linearizedIndices = getLinearizedSrcIndices(
  //         rewriter, loc, srcBits, adaptor.getIndices(), op.getMemRef());

  //     Value newLoad = rewriter.create<memref::LoadOp>(
  //         loc, adaptor.getMemref(),
  //         getIndicesForLoadOrStore(rewriter, loc, linearizedIndices, srcBits,
  //                                  dstBits));

  //     // OpFoldResult elementOffset;
  //     // Value byteOffset = adaptor.getByteOffset();
  //     // if (byteOffset && !matchPattern(byteOffset, m_Zero())) {
  //     //   elementOffset = convertByteOffsetToElementOffset(
  //     //       rewriter, loc, byteOffset, currentType.getElementType());
  //     // } else {
  //     //   elementOffset = rewriter.getIndexAttr(0);
  //     // }

  //     // llvm::outs()<<"AFFINE MAP :=
  //     // "<<currentType.getLayout().getAffineMap()<<"\n"; llvm::outs().flush();
  //     // SmallVector<OpFoldResult> sizes = getMixedValues(
  //     //     currentType.getShape(), adaptor.getDynamicDims(), rewriter);
  //     // memref::LinearizedMemRefInfo linearizedMemRefInfo =
  //     //     memref::getLinearizedMemRefOffsetAndSize(rewriter, loc, srcBits,
  //     //                                              dstBits, elementOffset,
  //     //                                              sizes);

  //     //   SmallVector<Value> dynamicLinearizedSize;
  //     //   if (newResultType.getRank() > 0 && !newResultType.hasStaticShape()) {
  //     //     dynamicLinearizedSize.push_back(getValueOrCreateConstantIndexOp(
  //     //         rewriter, loc, linearizedMemRefInfo.linearizedSize));
  //     //   }
  //   }
  //   return WalkResult::advance();
  // });
  return;
}

}  // namespace

std::unique_ptr<Pass> createAMDAIELinearizeMemrefTypePass() {
  return std::make_unique<AMDAIELinearizeMemrefTypePass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
