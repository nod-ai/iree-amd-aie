//===-VectorToVectorConversions.cpp - Conversions within Vector -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// This file contains conversions and rewrites to the Vector dialect to make
// it compatible with the available vector instructions in AIE architectures
//===----------------------------------------------------------------------===//

#include <limits>
#include <memory>

#include "Passes.h"
#include "aievec/AIEVecOps.h"
#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "aievec-canonicalization"

namespace mlir::iree_compiler::aievec {

namespace copied_from_mlir {

/// This code is copied from MLIR. It adds a single-line change in 2 patterns,
/// FlattenContiguousRowMajorTransfer[Read|Write]Pattern. The change allows the
/// memrefs of the transfer operations to be flattened to 1D memrefs, ie not
/// only the vectors are flattened. TODO(newling) consider upstreaming to reduce
/// code dup.

/// Creates a memref.collapse_shape collapsing all inner dimensions of the
/// input starting at `firstDimToCollapse`.
static Value collapseInnerDims(PatternRewriter &rewriter, mlir::Location loc,
                               Value input, int64_t firstDimToCollapse) {
  ShapedType inputType = cast<ShapedType>(input.getType());
  if (inputType.getRank() == 1) return input;
  SmallVector<ReassociationIndices> reassociation;
  for (int64_t i = 0; i < firstDimToCollapse; ++i)
    reassociation.push_back(ReassociationIndices{i});
  ReassociationIndices collapsedIndices;
  for (int64_t i = firstDimToCollapse; i < inputType.getRank(); ++i)
    collapsedIndices.push_back(i);
  reassociation.push_back(collapsedIndices);
  return rewriter.create<memref::CollapseShapeOp>(loc, input, reassociation);
}

/// Returns the new indices that collapses the inner dimensions starting from
/// the `firstDimToCollapse` dimension.
static SmallVector<Value> getCollapsedIndices(RewriterBase &rewriter,
                                              Location loc,
                                              ArrayRef<int64_t> shape,
                                              ValueRange indices,
                                              int64_t firstDimToCollapse) {
  assert(firstDimToCollapse < static_cast<int64_t>(indices.size()));

  // If all the collapsed indices are zero then no extra logic is needed.
  // Otherwise, a new offset/index has to be computed.
  SmallVector<Value> indicesAfterCollapsing(
      indices.begin(), indices.begin() + firstDimToCollapse);
  SmallVector<Value> indicesToCollapse(indices.begin() + firstDimToCollapse,
                                       indices.end());
  if (llvm::all_of(indicesToCollapse, isZeroIndex)) {
    indicesAfterCollapsing.push_back(indicesToCollapse[0]);
    return indicesAfterCollapsing;
  }

  // Compute the remaining trailing index/offset required for reading from
  // the collapsed memref:
  //
  //    offset = 0
  //    for (i = firstDimToCollapse; i < outputRank; ++i)
  //      offset += sourceType.getDimSize(i) * transferReadOp.indices[i]
  //
  // For this example:
  //   %2 = vector.transfer_read/write %arg4[%c0, %arg0, %c0] (...) :
  //      memref<1x43x2xi32>, vector<1x2xi32>
  // which would be collapsed to:
  //   %1 = vector.transfer_read/write %collapse_shape[%c0, %offset] (...) :
  //      memref<1x86xi32>, vector<2xi32>
  // one would get the following offset:
  //    %offset = %arg0 * 43
  OpFoldResult collapsedOffset =
      rewriter.create<arith::ConstantIndexOp>(loc, 0).getResult();

  auto collapsedStrides = computeSuffixProduct(
      ArrayRef<int64_t>(shape.begin() + firstDimToCollapse, shape.end()));

  // Compute the collapsed offset.
  auto &&[collapsedExpr, collapsedVals] =
      computeLinearIndex(collapsedOffset, collapsedStrides, indicesToCollapse);
  collapsedOffset = affine::makeComposedFoldedAffineApply(
      rewriter, loc, collapsedExpr, collapsedVals);

  if (collapsedOffset.is<Value>()) {
    indicesAfterCollapsing.push_back(collapsedOffset.get<Value>());
  } else {
    indicesAfterCollapsing.push_back(rewriter.create<arith::ConstantIndexOp>(
        loc, *getConstantIntValue(collapsedOffset)));
  }

  return indicesAfterCollapsing;
}

/// Rewrites contiguous row-major vector.transfer_read ops by inserting
/// memref.collapse_shape on the source so that the resulting
/// vector.transfer_read has a 1D source. Requires the source shape to be
/// already reduced i.e. without unit dims.
///
/// If `targetVectorBitwidth` is provided, the flattening will only happen if
/// the trailing dimension of the vector read is smaller than the provided
/// bitwidth.
class FlattenContiguousRowMajorTransferReadPattern
    : public OpRewritePattern<vector::TransferReadOp> {
 public:
  FlattenContiguousRowMajorTransferReadPattern(MLIRContext *context,
                                               unsigned vectorBitwidth,
                                               PatternBenefit benefit)
      : OpRewritePattern<vector::TransferReadOp>(context, benefit),
        targetVectorBitwidth(vectorBitwidth) {}

  LogicalResult matchAndRewrite(vector::TransferReadOp transferReadOp,
                                PatternRewriter &rewriter) const override {
    auto loc = transferReadOp.getLoc();
    Value vector = transferReadOp.getVector();
    VectorType vectorType = cast<VectorType>(vector.getType());
    auto source = transferReadOp.getSource();
    MemRefType sourceType = dyn_cast<MemRefType>(source.getType());

    // 0. Check pre-conditions
    // Contiguity check is valid on tensors only.
    if (!sourceType) return failure();
    // If this is already 0D/1D, there's nothing to do.
    if (vectorType.getRank() <= 1) return failure();
    if (!vectorType.getElementType().isSignlessIntOrFloat()) return failure();
    unsigned trailingVectorDimBitwidth =
        vectorType.getShape().back() * vectorType.getElementTypeBitWidth();
    if (trailingVectorDimBitwidth >= targetVectorBitwidth) return failure();
    if (!vector::isContiguousSlice(sourceType, vectorType)) return failure();
    // TODO: generalize this pattern, relax the requirements here.
    if (transferReadOp.hasOutOfBoundsDim()) return failure();
    if (!transferReadOp.getPermutationMap().isMinorIdentity()) return failure();
    if (transferReadOp.getMask()) return failure();

    // TODO(newling) This is the one line which changes from the original
    // MLIR function. Upstream this as an option to flatten the memref (and
    // not just the vector).
    // int64_t firstDimToCollapse = sourceType.getRank() - vectorType.getRank();
    int64_t firstDimToCollapse = 0;

    // 1. Collapse the source memref
    Value collapsedSource =
        collapseInnerDims(rewriter, loc, source, firstDimToCollapse);
    MemRefType collapsedSourceType =
        cast<MemRefType>(collapsedSource.getType());
    int64_t collapsedRank = collapsedSourceType.getRank();
    assert(collapsedRank == firstDimToCollapse + 1);

    // 2. Generate input args for a new vector.transfer_read that will read
    // from the collapsed memref.
    // 2.1. New dim exprs + affine map
    SmallVector<AffineExpr, 1> dimExprs{
        getAffineDimExpr(firstDimToCollapse, rewriter.getContext())};
    auto collapsedMap =
        AffineMap::get(collapsedRank, 0, dimExprs, rewriter.getContext());

    // 2.2 New indices
    SmallVector<Value> collapsedIndices =
        getCollapsedIndices(rewriter, loc, sourceType.getShape(),
                            transferReadOp.getIndices(), firstDimToCollapse);

    // 3. Create new vector.transfer_read that reads from the collapsed memref
    VectorType flatVectorType = VectorType::get({vectorType.getNumElements()},
                                                vectorType.getElementType());
    vector::TransferReadOp flatRead = rewriter.create<vector::TransferReadOp>(
        loc, flatVectorType, collapsedSource, collapsedIndices, collapsedMap);
    flatRead.setInBoundsAttr(rewriter.getBoolArrayAttr({true}));

    // 4. Replace the old transfer_read with the new one reading from the
    // collapsed shape
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        transferReadOp, cast<VectorType>(vector.getType()), flatRead);
    return success();
  }

 private:
  // Minimum bitwidth that the trailing vector dimension should have after
  // flattening.
  unsigned targetVectorBitwidth;
};

/// Rewrites contiguous row-major vector.transfer_write ops by inserting
/// memref.collapse_shape on the source so that the resulting
/// vector.transfer_write has a 1D source. Requires the source shape to be
/// already reduced i.e. without unit dims.
///
/// If `targetVectorBitwidth` is provided, the flattening will only happen if
/// the trailing dimension of the vector read is smaller than the provided
/// bitwidth.
class FlattenContiguousRowMajorTransferWritePattern
    : public OpRewritePattern<vector::TransferWriteOp> {
 public:
  FlattenContiguousRowMajorTransferWritePattern(MLIRContext *context,
                                                unsigned vectorBitwidth,
                                                PatternBenefit benefit)
      : OpRewritePattern<vector::TransferWriteOp>(context, benefit),
        targetVectorBitwidth(vectorBitwidth) {}

  LogicalResult matchAndRewrite(vector::TransferWriteOp transferWriteOp,
                                PatternRewriter &rewriter) const override {
    auto loc = transferWriteOp.getLoc();
    Value vector = transferWriteOp.getVector();
    VectorType vectorType = cast<VectorType>(vector.getType());
    Value source = transferWriteOp.getSource();
    MemRefType sourceType = dyn_cast<MemRefType>(source.getType());

    // 0. Check pre-conditions
    // Contiguity check is valid on tensors only.
    if (!sourceType) return failure();
    // If this is already 0D/1D, there's nothing to do.
    if (vectorType.getRank() <= 1)
      // Already 0D/1D, nothing to do.
      return failure();
    if (!vectorType.getElementType().isSignlessIntOrFloat()) return failure();
    unsigned trailingVectorDimBitwidth =
        vectorType.getShape().back() * vectorType.getElementTypeBitWidth();
    if (trailingVectorDimBitwidth >= targetVectorBitwidth) return failure();
    if (!vector::isContiguousSlice(sourceType, vectorType)) return failure();
    // TODO: generalize this pattern, relax the requirements here.
    if (transferWriteOp.hasOutOfBoundsDim()) return failure();
    if (!transferWriteOp.getPermutationMap().isMinorIdentity())
      return failure();
    if (transferWriteOp.getMask()) return failure();

    // TODO(newling) This is the one line which changes from the original
    // MLIR function. Upstream this as an option to flatten the memref (and
    // not just the vector).
    // int64_t firstDimToCollapse = sourceType.getRank() - vectorType.getRank();
    int64_t firstDimToCollapse = 0;

    // 1. Collapse the source memref
    Value collapsedSource =
        collapseInnerDims(rewriter, loc, source, firstDimToCollapse);
    MemRefType collapsedSourceType =
        cast<MemRefType>(collapsedSource.getType());
    int64_t collapsedRank = collapsedSourceType.getRank();
    assert(collapsedRank == firstDimToCollapse + 1);

    // 2. Generate input args for a new vector.transfer_read that will read
    // from the collapsed memref.
    // 2.1. New dim exprs + affine map
    SmallVector<AffineExpr, 1> dimExprs{
        getAffineDimExpr(firstDimToCollapse, rewriter.getContext())};
    auto collapsedMap =
        AffineMap::get(collapsedRank, 0, dimExprs, rewriter.getContext());

    // 2.2 New indices
    SmallVector<Value> collapsedIndices =
        getCollapsedIndices(rewriter, loc, sourceType.getShape(),
                            transferWriteOp.getIndices(), firstDimToCollapse);

    // 3. Create new vector.transfer_write that writes to the collapsed memref
    VectorType flatVectorType = VectorType::get({vectorType.getNumElements()},
                                                vectorType.getElementType());
    Value flatVector =
        rewriter.create<vector::ShapeCastOp>(loc, flatVectorType, vector);
    vector::TransferWriteOp flatWrite =
        rewriter.create<vector::TransferWriteOp>(
            loc, flatVector, collapsedSource, collapsedIndices, collapsedMap);
    flatWrite.setInBoundsAttr(rewriter.getBoolArrayAttr({true}));

    // 4. Replace the old transfer_write with the new one writing the
    // collapsed shape
    rewriter.eraseOp(transferWriteOp);
    return success();
  }

 private:
  // Minimum bitwidth that the trailing vector dimension should have after
  // flattening.
  unsigned targetVectorBitwidth;
};

}  // namespace copied_from_mlir

/// Utility to check if the indices provided are all 0.
static LogicalResult isAllZeroOffsetAccess(mlir::OperandRange indices) {
  if (!llvm::all_of(indices, [](Value val) {
        IntegerAttr attr;
        if (!matchPattern(val, m_Constant(&attr))) return false;
        return attr.getInt() == 0;
      })) {
    return failure();
  }
  return success();
}

/// Utility to convert OpFoldResult vector of offsets of a Subview op to
/// a vector of values.
static SmallVector<Value> opFoldResultsToValues(PatternRewriter &rewriter,
                                                Location loc,
                                                memref::SubViewOp subViewOp) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(subViewOp);
  SmallVector<Value> newIndices;
  for (OpFoldResult offset : subViewOp.getMixedOffsets()) {
    Value indexVal;
    if (auto attr = dyn_cast<Attribute>(offset)) {
      indexVal = rewriter.create<arith::ConstantIndexOp>(
          loc, cast<IntegerAttr>(attr).getInt());
    } else {
      indexVal = cast<Value>(offset);
    }
    newIndices.push_back(indexVal);
  }
  return newIndices;
}

/// A rewriter function to canonicalize the following :-
/// INPUT:
///       %b = memref.subview %a [offset0, offset1, ...]
///       %c = vector.transfer_read %b[0, 0, ...]
/// OUTPUT:
///       %c = vector.transfer_read %a[offset0, offset1, ...]
///
/// This is needed to enable other set of staged canonicalizations in this pass.
struct CanonicalizeTrivialReadAccessSubviewOpPattern
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    auto subViewOp = dyn_cast_if_present<memref::SubViewOp>(
        readOp.getSource().getDefiningOp());
    if (!subViewOp) return failure();
    if (failed(isAllZeroOffsetAccess(readOp.getIndices()))) return failure();
    SmallVector<Value> newIndices =
        opFoldResultsToValues(rewriter, readOp.getLoc(), subViewOp);
    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        readOp, readOp.getType(), subViewOp.getSource(), newIndices,
        readOp.getPadding(), readOp.getInBoundsValues());
    return success();
  }
};

/// A rewriter function to canonicalize the following :-
/// INPUT:
///       %b = memref.subview %a [offset0, offset1, ...]
///       vector.transfer_write %val, %b[0, 0, ...]
/// OUTPUT:
///       vector.transfer_write %val, %a[offset0, offset1, ...]
///
/// This is needed to enable other set of staged canonicalizations in this pass.
struct CanonicalizeTrivialWriteAccessSubviewOpPattern
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
    auto subViewOp = dyn_cast_if_present<memref::SubViewOp>(
        writeOp.getSource().getDefiningOp());
    if (!subViewOp) return failure();
    if (failed(isAllZeroOffsetAccess(writeOp.getIndices()))) return failure();
    SmallVector<Value> newIndices =
        opFoldResultsToValues(rewriter, writeOp.getLoc(), subViewOp);
    rewriter.create<vector::TransferWriteOp>(
        writeOp.getLoc(), writeOp.getVector(), subViewOp.getSource(),
        newIndices, writeOp.getInBoundsValues());
    rewriter.eraseOp(writeOp);
    return success();
  }
};

static bool isGemmBTransposedContractionOp(vector::ContractionOp op) {
  if (op.getKind() != vector::CombiningKind::ADD) return false;

  // Get and check shape of operands
  auto lhsShape = op.getLhsType().getShape();
  auto rhsShape = op.getRhsType().getShape();
  auto accShape = cast<ShapedType>(op.getAccType()).getShape();
  if (lhsShape.size() < 2 || rhsShape.size() < 2 || accShape.size() < 2)
    return false;

  // Check that the innermost iterators match gemm-like iterators
  SmallVector<vector::IteratorType> iterators = op.getIteratorTypesArray();
  if (iterators.size() < 3) return false;
  auto innerMostIterators =
      SmallVector<vector::IteratorType>(iterators.end() - 3, iterators.end());
  if (vector::IteratorType::parallel != innerMostIterators[0] ||
      vector::IteratorType::parallel != innerMostIterators[1] ||
      vector::IteratorType::reduction != innerMostIterators[2])
    return false;

  // Get indexing maps of iterators for operands
  SmallVector<AffineMap, 4> indexingMaps(op.getIndexingMapsArray());
  SmallVector<int64_t> outerMostResults;
  for (int64_t i = 0; i < indexingMaps[0].getNumResults() - 2; i++)
    outerMostResults.push_back(i);

  auto innerLhsMap = indexingMaps[0].dropResults(outerMostResults);
  auto innerRhsMap = indexingMaps[1].dropResults(outerMostResults);
  auto innerAccMap = indexingMaps[2].dropResults(outerMostResults);

  // Check whether they conform to a "transposed B" gemm
  auto ctx = op.getContext();
  auto mmAidxMap =
      AffineMap::getPermutationMap(ArrayRef<unsigned>{1, 0, 2}, ctx)
          .dropResults(0);
  auto mmBidxMap =
      AffineMap::getPermutationMap(ArrayRef<unsigned>{0, 1, 2}, ctx)
          .dropResults(0);
  auto mmCidxMap =
      AffineMap::getPermutationMap(ArrayRef<unsigned>{2, 0, 1}, ctx)
          .dropResults(0);
  int64_t numOuterMostDims = indexingMaps[0].getNumDims() - 3;
  return innerLhsMap == mmAidxMap.shiftDims(numOuterMostDims) &&
         innerRhsMap == mmBidxMap.shiftDims(numOuterMostDims) &&
         innerAccMap == mmCidxMap.shiftDims(numOuterMostDims);
}

// This pattern converts a `vector.transfer_read` with a splat permutation map
// into a contiguous `vector.transfer_read` followed by a `vector.extract` to
// obtain the splat value and a `vector.broadcast` to broadcast it into a
// vector of the right size.
struct ConvertSplatTransferReadToBroadcastPattern
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  ConvertSplatTransferReadToBroadcastPattern(MLIRContext *context)
      : OpRewritePattern<vector::TransferReadOp>(context) {}

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    AffineMap map = readOp.getPermutationMap();
    if (!map.isConstant()) return failure();

    Value srcMemRef = readOp.getSource();
    SmallVector<Value, 8> indices;
    Value newIdx;
    int64_t offset = 0;
    // If it's a zero-rank memory access
    if (cast<MemRefType>(srcMemRef.getType()).getRank() == 0) {
      srcMemRef = rewriter
                      .create<memref::ExpandShapeOp>(
                          readOp.getLoc(), SmallVector<int64_t, 1>({1}),
                          srcMemRef, SmallVector<ReassociationIndices, 1>({}))
                      .getResult();
      newIdx = rewriter.create<arith::ConstantOp>(readOp.getLoc(),
                                                  rewriter.getIndexAttr(0L));
      indices.push_back(newIdx);
    } else {
      indices.append(readOp.getIndices().begin(), readOp.getIndices().end());
      newIdx = indices[indices.size() - 1];
      // If the innermost index comes from an `affine.apply` op, take the base
      // as the new innermost index for the new `vector.transfer_read`, and the
      // offset as the index for the `aievec.broadcast` op.
      if (auto applyOp = newIdx.getDefiningOp<affine::AffineApplyOp>())
        if (applyOp.getAffineMap().getNumDims() == 1) {
          newIdx = applyOp.getMapOperands()[0];
          offset = applyOp.getAffineMap().compose(ArrayRef<int64_t>{0})[0];
        }
    }
    // XXX: We assume we are reading 1D vectors
    int64_t vlen = readOp.getVector().getType().getShape()[0];
    if (offset >= vlen) {
      // If the splat element is beyond the first vector, we calculate the
      // address of the vector containing the element.
      int64_t numElemsToSkip = vlen * (offset / vlen);
      offset = offset % vlen;
      auto newAddrMap = AffineMap::get(
          1, 0, getAffineDimExpr(0, readOp.getContext()) + numElemsToSkip);
      newIdx =
          rewriter
              .create<affine::AffineApplyOp>(readOp.getLoc(), newAddrMap,
                                             SmallVector<Value, 1>({newIdx}))
              .getResult();
    }
    indices[indices.size() - 1] = newIdx;
    auto newReadOp = rewriter.create<vector::TransferReadOp>(
        readOp.getLoc(), readOp.getVector().getType(), srcMemRef, indices,
        readOp.getPadding());
    auto extractOp = rewriter.create<vector::ExtractOp>(
        readOp.getLoc(), newReadOp.getResult(), ArrayRef<int64_t>{offset});
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        readOp, newReadOp.getVector().getType(), extractOp.getResult());
    return success();
  }
};

/// Given:
/// ```
/// %1 = elTypeChanger %0
/// %2 = shapeChanger %1
/// ```
///
/// Where 'elTypeChanger' changes the element type, and 'shapeChanger' changes
/// the shape, suppose we wanted to rewrite this as:
/// ```
/// %1 = shapeChanger %0
/// %2 = elTypeChanger %1
/// ```
///
/// What would the type of %1 be? That is what this function computes.
VectorType getIntermediateType(Operation *elTypeChanger,
                               Operation *shapeChanger) {
  assert(elTypeChanger->getNumOperands() == 1 && "require single operand");
  assert(shapeChanger->getNumResults() == 1 && "require single result");
  Value inValue = elTypeChanger->getOperand(0);
  Value outValue = shapeChanger->getResult(0);
  Type inType = inValue.getType();
  ShapedType inShapedType = dyn_cast<ShapedType>(inType);
  Type elType = inShapedType ? inShapedType.getElementType() : inType;
  ShapedType outType = dyn_cast<ShapedType>(outValue.getType());
  assert(outType && "require ShapedTypes");
  auto newType = VectorType::get(outType.getShape(), elType);
  return newType;
}

// This pattern swaps a UnaryOpA followed by UnaryOpB. This pattern can be used
// to improve pattern matching for mixed-type arithmetic ops, by getting sign
// extension ops closer to the single-type arithmetic operations.
template <class UnaryOpA, class UnaryOpB>
struct SwapUnaryOpsPattern : public OpRewritePattern<UnaryOpB> {
  using OpRewritePattern<UnaryOpB>::OpRewritePattern;
  // This function takes the chain of operations A->B, and returns the new type
  // between B and A after the swap.
  using InferTypeB2AFnTy = std::function<Type(UnaryOpA aOp, UnaryOpB bOp)>;
  InferTypeB2AFnTy inferTypeB2A = nullptr;

  SwapUnaryOpsPattern(MLIRContext *context, InferTypeB2AFnTy inferType)
      : OpRewritePattern<UnaryOpB>(context), inferTypeB2A(inferType) {}

  /// Construct with the default InferTypeB2AFnTy function.
  SwapUnaryOpsPattern(MLIRContext *context)
      : OpRewritePattern<UnaryOpB>(context),
        inferTypeB2A([](UnaryOpA aOp, UnaryOpB bOp) -> Type {
          return getIntermediateType(aOp, bOp);
        }) {}

  LogicalResult matchAndRewrite(UnaryOpB bOp,
                                PatternRewriter &rewriter) const override {
    static_assert(
        UnaryOpA::template hasTrait<OpTrait::OneOperand>(),
        "SwapUnaryOps can only be instantiated for single-operand ops");
    static_assert(
        UnaryOpB::template hasTrait<OpTrait::OneOperand>(),
        "SwapUnaryOps can only be instantiated for single-operand ops");
    UnaryOpA aOp = bOp.getOperand().template getDefiningOp<UnaryOpA>();
    if (!aOp)
      return rewriter.notifyMatchFailure(bOp, UnaryOpB::getOperationName() +
                                                  " not preceeded by " +
                                                  UnaryOpA::getOperationName());

    Type newA2BTy = inferTypeB2A(aOp, bOp);

    auto newA =
        rewriter.create<UnaryOpB>(bOp->getLoc(), SmallVector<Type>({newA2BTy}),
                                  aOp->getOperands(), bOp->getAttrs());
    auto newB = rewriter.create<UnaryOpA>(
        bOp->getLoc(), SmallVector<Type>({bOp.getResult().getType()}),
        newA->getResults(), aOp->getAttrs());
    rewriter.replaceOp(bOp, newB.getResult());
    return success();
  }
};

/// AffineMaps which are 'generalized' minor identities, where the results just
/// need to be in ascending order. Examples:
///
/// (d0, d1, d2) -> (d0, d2)    // return {0,2}
/// (d0, d1) -> (d0, d1)        // return {0,1}
/// (d0, d1) -> (d1)            // return {1}
/// (d0, d1) -> (d0)            // return {0}
/// (d0, d1) -> (d1, d0)        // failure
/// (d0, d1) -> (d0 + d1)       // failure
FailureOr<SmallVector<int64_t>> getDimsOfIdentitySubsampleMap(AffineMap perm) {
  ArrayRef<AffineExpr> results = perm.getResults();
  uint64_t nResults = results.size();
  SmallVector<int64_t> dims;
  for (uint64_t i = 0; i < nResults; ++i) {
    if (!isa<AffineDimExpr>(results[i])) return failure();
    auto nxtDim = cast<AffineDimExpr>(results[i]).getPosition();
    if (!dims.empty() && (nxtDim <= dims.back())) return failure();
    dims.push_back(nxtDim);
  }
  return dims;
}

template <typename TransferOp>
std::tuple<SmallVector<int64_t>, SmallVector<bool>>
getUnsqueezedShapeAndInBounds(uint64_t inputRank,
                              SmallVector<int64_t> oldDimensions,
                              PatternRewriter &rewriter, TransferOp op) {
  uint64_t newRank = inputRank - oldDimensions[0];
  SmallVector<int64_t> newShape(newRank, 1);
  SmallVector<bool> newInBounds(newRank, true);
  SmallVector<bool> oldInBounds = op.getInBoundsValues();
  ArrayRef<int64_t> oldShape = op.getVectorType().getShape();
  for (auto iter : enumerate(oldDimensions)) {
    uint64_t oldIndex = iter.index();
    uint64_t newIndex = iter.value() - oldDimensions[0];
    assert(oldIndex <= newIndex);
    newInBounds[newIndex] = oldInBounds[oldIndex];
    newShape[newIndex] = oldShape[oldIndex];
  }
  return {newShape, newInBounds};
}

/// This pattern rewrites a `vector.transfer_write` with a non minor identity
/// permutation map, with a minor-identity permutation map, if possible. For
/// example,
///
/// ```
/// #map = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
/// vector.transfer_write %v1, %alloc[%c0, %c0, %c0, %c0]
///    {permutation_map = #map} : vector<4x8xi8>, memref<2x4x1x8xi8>
/// ```
///
/// is rewritten as
/// ```
/// #map = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
/// vector.transfer_write %v2, %alloc[%c0, %c0, %c0, %c0]
///    {permutation_map = #map} : vector<4x1x8xi8>, memref<2x4x1x8xi8>
/// ```
///
/// with the new vector `v2` being a `vector.shape_cast` of `v1`.
///
/// This allows other upstream patterns to work on the transfer op, as they
/// expect minor-identity permutation maps.
struct ToMinorIdentityTransferWritePattern
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
    AffineMap perm = writeOp.getPermutationMap();

    // Already in target form:
    if (perm.isMinorIdentity()) return failure();

    FailureOr<SmallVector<int64_t>> maybeDims =
        getDimsOfIdentitySubsampleMap(perm);
    if (failed(maybeDims)) {
      return rewriter.notifyMatchFailure(
          writeOp, "cannot be expressed with a minor-identity permutation map");
    }

    TypedValue<ShapedType> source = writeOp.getSource();

    auto [newShape, newInBounds] = getUnsqueezedShapeAndInBounds(
        source.getType().getRank(), maybeDims.value(), rewriter, writeOp);

    VectorType newVectorType =
        VectorType::get(newShape, writeOp.getVectorType().getElementType());

    Value newVector = rewriter.create<vector::ShapeCastOp>(
        writeOp.getLoc(), newVectorType, writeOp.getVector());

    MemRefType sourceType = cast<MemRefType>(writeOp.getSource().getType());

    if (!mlir::vector::isContiguousSlice(sourceType, newVectorType))
      return failure();

    auto newWriteOp = rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        writeOp, newVector, source, writeOp.getIndices());

    newWriteOp.getProperties().setInBounds(
        rewriter.getBoolArrayAttr(newInBounds));

    assert(newWriteOp.getPermutationMap().isMinorIdentity() &&
           "Pattern failed to convert to minor identity");

    return success();
  }
};

/// This pattern rewrites a `vector.transfer_read` with a non minor identity
/// permutation map, with a minor-identity permutation map, if possible. For
/// example,
///
/// ```
/// #map = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
/// %0 = vector.transfer_read %alloc[%c0, %c0, %c0, %c0], %c0_i8
///    {in_bounds = [true, true], permutation_map = #map} : memref<2x4x1x8xi8>,
///    vector<4x8xi8>
/// ```
///
/// is rewritten as
///
/// ```
/// #map = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
/// %1 = vector.transfer_read %alloc[%c0, %c0, %c0, %c0], %c0_i8
///   {in_bounds = [true, true, true], permutation_map = #map} :
///   memref<2x4x1x8xi8>, vector<4x1x8xi8>
/// %2 = vector.shape_cast %1 : vector<4x1x8xi8> to vector<4x8xi8>
///   ```
///
struct ToMinorIdentityTransferReadPattern
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    AffineMap perm = readOp.getPermutationMap();

    // Already in target form:
    if (perm.isMinorIdentity()) return failure();

    // Cannot be converted into a minor identity map:
    FailureOr<SmallVector<int64_t>> maybeDims =
        getDimsOfIdentitySubsampleMap(perm);
    if (failed(maybeDims)) return failure();

    MemRefType sourceType = cast<MemRefType>(readOp.getSource().getType());

    auto [newShape, newInBounds] = getUnsqueezedShapeAndInBounds(
        sourceType.getRank(), maybeDims.value(), rewriter, readOp);

    VectorType newVectorTy =
        VectorType::get(newShape, readOp.getVectorType().getElementType());
    if (!vector::isContiguousSlice(sourceType, newVectorTy)) return failure();

    auto newReadOp = rewriter.create<vector::TransferReadOp>(
        readOp.getLoc(), newVectorTy, readOp.getSource(), readOp.getIndices());

    newReadOp.getProperties().setInBounds(
        rewriter.getBoolArrayAttr(newInBounds));

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        readOp, readOp.getVector().getType(), newReadOp);
    return success();
  }
};

// clang-format off
/// Pattern to linearize arith.truncf because later aievec.srs in AIEVecToLLVM is
/// expected to have 1-D source and target.
/// Refer: https://github.com/nod-ai/iree-amd-aie/blob/main/compiler/plugins/target/AMD-AIE/aievec/AIEVecToLLVM.cpp#L73-L74
///
/// Example of what this pattern achieves :-
/// INPUT
///     %0 = arith.truncf %inp : vector<2x3xf32> to vector<2x3xbf16>
/// OUTPUT
///     %0 = vector.shape_cast %inp : vector<2x3xf32> to vector<6xf32>
///     %1 = arith.truncf %0 : vector<6xf32> to vector<6xbf16>
///     %2 = vector.shape_cast %1 : vector<6xbf16> to vector<2x3xbf16>
// clang-format on
struct FlattenArithTruncFOpPattern : public OpRewritePattern<arith::TruncFOp> {
  using OpRewritePattern<arith::TruncFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::TruncFOp op,
                                PatternRewriter &rewriter) const override {
    // Get old shape type.
    auto oldShapedType = dyn_cast<VectorType>(op.getType());
    if (!oldShapedType) return failure();
    // Bail out if it's already linearized.
    if (oldShapedType.getRank() == 1) return failure();
    // Linearize the shape.
    int64_t linearizedSize = oldShapedType.getNumElements();
    // Fetch input.
    Value origInputOfTruncFOp = op.getIn();
    // Form linearized vector shape type for input and output.
    VectorType newVectorTypeForInput = VectorType::get(
        {linearizedSize},
        cast<ShapedType>(origInputOfTruncFOp.getType()).getElementType());
    VectorType newVectorTypeForOutput =
        VectorType::get({linearizedSize}, oldShapedType.getElementType());
    // Shape cast the original input to linearized shape type.
    Value newInputVector = rewriter.create<vector::ShapeCastOp>(
        op.getLoc(), newVectorTypeForInput, origInputOfTruncFOp);
    // Create new base operation with the linearized input/output.
    Value newTruncFOp = rewriter.create<arith::TruncFOp>(
        op.getLoc(), newVectorTypeForOutput, newInputVector);
    // Delinearize the output back to the original type.
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, op.getType(),
                                                     newTruncFOp);
    return success();
  }
};

// This pattern extracts an implicit transposition of the 2 innermost
// dimensions of `rhs` in a gemm-like contraction op, making it an explicit
// `vector.transpose` op.
// If `rhs` is coming from a widening op (`extf`/`extsi`/`extui`), the
// transposition will be hoisted above the widening op.
struct ExtractTransposeFromContractionOp
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  static VectorType getTransposedVectorType(VectorType vecTy) {
    SmallVector<int64_t> shape{vecTy.getShape()};
    auto nDim = shape.size();
    int64_t dimNm1 = shape[nDim - 1];
    shape[nDim - 1] = shape[nDim - 2];
    shape[nDim - 2] = dimNm1;
    auto elemTy = vecTy.getElementType();
    return VectorType::get(shape, elemTy);
  }

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {
    if (!isGemmBTransposedContractionOp(contractOp)) return failure();

    Location loc = contractOp.getLoc();
    auto ctx = rewriter.getContext();

    Value rhsVal = contractOp.getRhs();
    VectorType rhsVecTy = contractOp.getRhsType();
    Type rhsElemTy = rhsVecTy.getElementType();

    bool doExtF = false, doExtSI = false, doExtUI = false;
    if (auto extfRhsOp = rhsVal.getDefiningOp<arith::ExtFOp>()) {
      rhsVal = extfRhsOp.getIn();
      rhsVecTy = cast<VectorType>(rhsVal.getType());
      doExtF = true;
    } else if (auto extsiRhsOp = rhsVal.getDefiningOp<arith::ExtSIOp>()) {
      rhsVal = extsiRhsOp.getIn();
      rhsVecTy = cast<VectorType>(rhsVal.getType());
      doExtSI = true;
    } else if (auto extuiRhsOp = rhsVal.getDefiningOp<arith::ExtUIOp>()) {
      rhsVal = extuiRhsOp.getIn();
      rhsVecTy = cast<VectorType>(rhsVal.getType());
      doExtUI = true;
    }

    int64_t nDim = rhsVecTy.getShape().size();

    SmallVector<int64_t> rhsPermutation;
    for (int64_t i = 0; i < nDim - 2; i++) rhsPermutation.push_back(i);
    rhsPermutation.push_back(nDim - 1);
    rhsPermutation.push_back(nDim - 2);
    auto transpRhsVecTy = getTransposedVectorType(rhsVecTy);
    rhsVal = rewriter
                 .create<vector::TransposeOp>(loc, transpRhsVecTy, rhsVal,
                                              rhsPermutation)
                 .getResult();

    if (doExtF)
      rhsVal =
          rewriter
              .create<arith::ExtFOp>(
                  loc, VectorType::get(transpRhsVecTy.getShape(), rhsElemTy),
                  rhsVal)
              .getOut();
    if (doExtSI)
      rhsVal =
          rewriter
              .create<arith::ExtSIOp>(
                  loc, VectorType::get(transpRhsVecTy.getShape(), rhsElemTy),
                  rhsVal)
              .getOut();
    if (doExtUI)
      rhsVal =
          rewriter
              .create<arith::ExtUIOp>(
                  loc, VectorType::get(transpRhsVecTy.getShape(), rhsElemTy),
                  rhsVal)
              .getOut();

    SmallVector<AffineMap, 4> oldIdxMaps(contractOp.getIndexingMapsArray());

    nDim = oldIdxMaps[1].getNumDims();
    SmallVector<int64_t> innerDimPerm;
    for (int64_t i = 0; i < nDim - 2; i++) innerDimPerm.push_back(i);
    innerDimPerm.push_back(nDim - 1);
    innerDimPerm.push_back(nDim - 2);
    auto transpPermMap = AffineMap::getPermutationMap(innerDimPerm, ctx);

    auto newIdxMaps = rewriter.getAffineMapArrayAttr(
        {oldIdxMaps[0], oldIdxMaps[1].compose(transpPermMap), oldIdxMaps[2]});

    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        contractOp, contractOp.getResult().getType(), contractOp.getLhs(),
        rhsVal, contractOp.getAcc(), newIdxMaps, contractOp.getIteratorTypes());

    return success();
  }
};

static void configureCanonicalizeLegalizations(ConversionTarget &target) {
  target.addDynamicallyLegalOp<vector::TransferReadOp>(
      [](vector::TransferReadOp op) {
        return !op.getPermutationMap().isConstant() &&
               op.getVector().getType().getRank() < 2;
      });

  target.addDynamicallyLegalOp<vector::TransferWriteOp>(
      [](vector::TransferWriteOp op) {
        return cast<VectorType>(op.getVector().getType()).getRank() < 2;
      });
  target.addDynamicallyLegalOp<vector::ContractionOp>(
      [](vector::ContractionOp op) {
        return !isGemmBTransposedContractionOp(op);
      });
}

struct ConvertLeadingUnitDimInsertToReshapePattern
    : public OpRewritePattern<vector::InsertOp> {
  using OpRewritePattern<vector::InsertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::InsertOp insOp,
                                PatternRewriter &rewriter) const override {
    auto insSrcTy = dyn_cast<VectorType>(insOp.getSourceType());
    if (!insSrcTy) return failure();

    auto srcShape = insSrcTy.getShape();
    auto dstShape = insOp.getDestVectorType().getShape();

    unsigned long numLeadUnitDimDst = 0;
    while (numLeadUnitDimDst < dstShape.size() &&
           dstShape[numLeadUnitDimDst] == 1)
      numLeadUnitDimDst++;

    if (!numLeadUnitDimDst) return failure();

    unsigned long numLeadUnitDimSrc = 0;
    while (numLeadUnitDimSrc < srcShape.size() &&
           srcShape[numLeadUnitDimSrc] == 1)
      numLeadUnitDimSrc++;

    ArrayRef<int64_t> nonLeadUnitDimDstShape(
        dstShape.begin() + numLeadUnitDimDst, dstShape.end());
    ArrayRef<int64_t> nonLeadUnitDimSrcShape(
        srcShape.begin() + numLeadUnitDimSrc, srcShape.end());

    if (nonLeadUnitDimSrcShape != nonLeadUnitDimDstShape) return failure();

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        insOp, insOp.getDestVectorType(), insOp.getSource());
    return success();
  }
};

void populateBubbleSignExtensionsLate(RewritePatternSet &patterns) {
  patterns.add<SwapUnaryOpsPattern<arith::ExtSIOp, vector::BroadcastOp>,
               SwapUnaryOpsPattern<arith::ExtFOp, vector::BroadcastOp>,
               SwapUnaryOpsPattern<arith::ExtSIOp, vector::ShapeCastOp>,
               SwapUnaryOpsPattern<arith::ExtFOp, vector::ShapeCastOp>>(
      patterns.getContext());
}

// This pass converts standard vector ops into a subset of `Vector` ops more
// amenable to being converted to `AIEVec`.
struct CanonicalizeVectorForAIEVecPass
    : public PassWrapper<CanonicalizeVectorForAIEVecPass, OperationPass<>> {
  StringRef getArgument() const final {
    return "canonicalize-vector-for-aievec";
  }

  StringRef getDescription() const final {
    return "Canonicalize vector operations for AIEVec conversion";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    vector::VectorDialect, affine::AffineDialect>();
  }

  void runOnOperation() override {
    auto op = getOperation();
    MLIRContext *context = &getContext();

    {
      RewritePatternSet patterns(context);
      patterns.add<CanonicalizeTrivialReadAccessSubviewOpPattern,
                   CanonicalizeTrivialWriteAccessSubviewOpPattern>(context);
      (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
    }
    {
      // These must run before 'populateVectorBroadcastLoweringPatterns'
      // so that broadcasts can be matched before conversion to insert.
      RewritePatternSet patterns(context);
      populateBubbleSignExtensionsLate(patterns);
      (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
    }

    {
      RewritePatternSet patterns(context);
      patterns
          .add<ExtractTransposeFromContractionOp, FlattenArithTruncFOpPattern,
               ToMinorIdentityTransferReadPattern,
               ToMinorIdentityTransferWritePattern,
               ConvertLeadingUnitDimInsertToReshapePattern>(context);
      patterns.add<ConvertSplatTransferReadToBroadcastPattern>(context);
      patterns
          .add<copied_from_mlir::FlattenContiguousRowMajorTransferReadPattern,
               copied_from_mlir::FlattenContiguousRowMajorTransferWritePattern>(
              context, std::numeric_limits<unsigned>::max(), 1);
      mlir::vector::populateShapeCastFoldingPatterns(patterns);
      mlir::vector::populateDropUnitDimWithShapeCastPatterns(patterns);
      mlir::vector::populateVectorBroadcastLoweringPatterns(patterns);
      (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
    }

    {
      // These must run after 'populateFlattenVectorTransferPatterns' because
      // vector.shape_casts are introduced. Merging into a single pass creates
      // cycles.
      RewritePatternSet patterns(context);
      populateBubbleSignExtensionsLate(patterns);
      (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
    }
  }
};

/// Returns one of:
/// 1) failure, if there is definitely an error that should be propagated.
/// 2) a new transfer_read operation that is sufficiently aligned, if the old
///    transfer_read is determined to be insufficiently aligned and it is
///    possible to create a new transfer_read.
/// 3) the original transfer_read operation, otherwise.
FailureOr<Value> getAlignedTransferRead(
    vector::TransferReadOp readOp, IRRewriter &rewriter,
    const AMDAIE::AMDAIEDeviceModel &deviceModel) {
  uint32_t vectorLoadStoreAlignmentBits =
      deviceModel.getVectorLoadStoreAlignmentBits();
  uint32_t maxVectorSizeBits = deviceModel.getMaxVectorSizeBits();
  uint32_t shiftOperandBits = deviceModel.getShiftOperandBits();

  // Check that it's not a splat transfer read.
  if (readOp.getPermutationMap().isConstant()) return readOp.getVector();

  MLIRContext *ctx = readOp.getContext();
  VectorType shortType = readOp.getVectorType();
  Location loc = readOp.getLoc();
  Value padding = readOp.getPadding();
  ShapedType sourceType = readOp.getSource().getType();
  Type elementType = shortType.getElementType();

  if (sourceType.getRank() != 1 || shortType.getRank() != 1) {
    return readOp.emitOpError(
        "does not have rank-1 source and rank-1 vector type.");
  }

  uint32_t elementBits = elementType.getIntOrFloatBitWidth();
  int64_t shortLength = shortType.getShape().back();
  int64_t shortBits = shortLength * elementBits;
  uint32_t alignElements = vectorLoadStoreAlignmentBits / elementBits;

  rewriter.setInsertionPoint(readOp);

  AffineMap moduloMap =
      AffineMap::get(1, 0, getAffineDimExpr(0, ctx) % alignElements);

  Value oldIndex = readOp.getIndices().back();

  // Early exit case: If the current `tranfer_read` offset is already multiple
  // of alignment, can return without any modification.
  //
  // Below, we check if the offset is defined by `affine.apply`, if then if the
  // `affine.apply` is always a multiple of alignment.
  //
  // TODO(newling) generalize - what to case where the offset is not defined by
  //               `affine.apply`.
  // TODO(newling) make this reusable for canonicalization: a
  //               `transfer_read` followed by `aievec.ext` op can be simplified
  //               with this approach.
  if (auto offsetAffineApplyOp =
          oldIndex.getDefiningOp<affine::AffineApplyOp>()) {
    AffineMap affineMap = offsetAffineApplyOp.getAffineMap();
    assert(affineMap.getNumResults() == 1 &&
           "already established that destination of transfer_read is 1D");
    AffineExpr resultExpr = affineMap.getResult(0);
    int64_t largestKnownDivisor = resultExpr.getLargestKnownDivisor();
    if (largestKnownDivisor % alignElements == 0) return readOp.getVector();
  }

  Value offset = rewriter.createOrFold<affine::AffineApplyOp>(
      loc, moduloMap, SmallVector<Value, 1>{oldIndex});

  // If the offset is constant and zero, the read is already aligned.
  if (auto offsetConstantOp = offset.getDefiningOp<arith::ConstantIndexOp>())
    if (offsetConstantOp.getValue() == 0) return readOp.getVector();

  // Verify that we can load a vector 2x as long as the original vector.
  int64_t longBits = 2 * shortBits;
  int64_t longLength = 2 * shortLength;
  VectorType longType = VectorType::get(longLength, elementType);
  if (longBits > maxVectorSizeBits) {
    // Not returning failure, as it is possible that the read is already
    // aligned, and we just couldn't prove it.
    readOp.emitWarning()
        << "`transfer_read` can't be aligned with a read twice "
        << "as large because " << longBits
        << " bits is greater than the maximum vector size of "
        << maxVectorSizeBits << " bits.";

    return readOp.getVector();
  }

  SmallVector<bool> inBounds = readOp.getInBoundsValues();
  bool allInBounds =
      std::all_of(inBounds.begin(), inBounds.end(), [](bool b) { return b; });

  if (shortBits != shiftOperandBits / 2 && shortBits != shiftOperandBits) {
    // Not returning failure, as it is possible that the read is already
    // aligned, and we just couldn't prove it.
    readOp.emitWarning() << "`transfer_read` doesn't have a vector with "
                         << shiftOperandBits / 2 << " or " << shiftOperandBits
                         << " bits."
                         << "This case is not currently handled.";
    return readOp.getVector();
  }

  Value newIndex = rewriter.createOrFold<arith::SubIOp>(loc, oldIndex, offset);

  // Create the aligned transfer read for a vector 2x as long that covers the
  // elements of the unaligned vector.
  Value longVec = rewriter.create<vector::TransferReadOp>(
      loc, longType, readOp.getSource(), SmallVector<Value>{newIndex}, padding,
      SmallVector<bool>{allInBounds});

  Value elementBytes =
      rewriter.create<arith::ConstantIndexOp>(loc, elementBits / 8);

  Value offsetBytes =
      rewriter.createOrFold<arith::MulIOp>(loc, offset, elementBytes);

  Value offsetBytes_i32 = rewriter.createOrFold<arith::IndexCastOp>(
      loc, rewriter.getIntegerType(32), offsetBytes);

  Value replacement;
  if (shortBits == shiftOperandBits) {
    // - Extract lower 64 bytes
    // - Extract upper 64 bytes
    // - Apply shift to obtain new 64 bytes
    Value low = rewriter.create<ExtOp>(loc, shortType, longVec,
                                       rewriter.getI8IntegerAttr(0));
    Value upp = rewriter.create<ExtOp>(loc, shortType, longVec,
                                       rewriter.getI8IntegerAttr(1));
    replacement = rewriter.createOrFold<ShiftOp>(loc, shortType, low, upp,
                                                 offsetBytes_i32);
  } else if (shortBits == shiftOperandBits / 2) {
    // - Apply shift to obtain new 64 bytes, bottom 32 being the required ones
    // - Extract lower 32 bytes
    Value shift = rewriter.createOrFold<ShiftOp>(loc, longType, longVec,
                                                 longVec, offsetBytes_i32);
    replacement = rewriter.create<ExtOp>(loc, shortType, shift,
                                         rewriter.getI8IntegerAttr(0));
  } else {
    assert(false &&
           "unreachable: already checked that shortBytes is equal to or half "
           "of shiftOperandBytes");
  }

  rewriter.replaceOp(readOp, replacement);

  return replacement;
}

struct AlignTransferReadsPass
    : public PassWrapper<AlignTransferReadsPass, OperationPass<>> {
  StringRef getArgument() const final { return "align-transfer-reads"; }

  StringRef getDescription() const final {
    return "Align `vector.transfer_read` operations.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, memref::MemRefDialect,
                vector::VectorDialect, affine::AffineDialect, AIEVecDialect>();
  }

  void runOnOperation() override {
    Operation *op = getOperation();

    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
    std::optional<AMDAIE::AMDAIEDevice> maybeDevice =
        mlir::iree_compiler::AMDAIE::getConfigAMDAIEDevice(targetAttr);
    if (!maybeDevice) {
      op->emitOpError()
          << "has no AMDAIEDevice in the target attribute configuration. This "
             "device-specific information is required to determine what vector "
             "sizes and alignments are supported.";
      return signalPassFailure();
    }
    AMDAIE::AMDAIEDeviceModel deviceModel =
        AMDAIE::getDeviceModel(maybeDevice.value());

    IRRewriter rewriter(&getContext());
    op->walk([&](vector::TransferReadOp transferReadOp) {
      if (failed(
              getAlignedTransferRead(transferReadOp, rewriter, deviceModel))) {
        signalPassFailure();
      }
    });
  }
};

struct DetectNonCanonicalOpsPass
    : public PassWrapper<DetectNonCanonicalOpsPass, OperationPass<>> {
  StringRef getArgument() const final {
    return "detect-non-canonical-ops-for-aievec";
  }

  StringRef getDescription() const final {
    return "Detect non-canonical vector operations for AIEVec conversion. "
           "This pass will fail if vector dialect ops are not in "
           "a form that can be easily lowered to the AIEVec dialect.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    vector::VectorDialect, affine::AffineDialect>();
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    configureCanonicalizeLegalizations(target);
    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

void buildCanonicalizeVectorForAIEVec(OpPassManager &pm) {
  pm.addPass(createCanonicalizeVectorForAIEVecPass());
  pm.addPass(createAlignTransferReadsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(std::make_unique<DetectNonCanonicalOpsPass>());
}

std::unique_ptr<::mlir::Pass> createCanonicalizeVectorForAIEVecPass() {
  return std::make_unique<CanonicalizeVectorForAIEVecPass>();
}

void registerCanonicalizeVectorForAIEVecPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createCanonicalizeVectorForAIEVecPass();
  });
}

std::unique_ptr<::mlir::Pass> createAlignTransferReadsPass() {
  return std::make_unique<AlignTransferReadsPass>();
}

void registerAlignTransferReadsPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<AlignTransferReadsPass>();
  });
}

}  // namespace mlir::iree_compiler::aievec
