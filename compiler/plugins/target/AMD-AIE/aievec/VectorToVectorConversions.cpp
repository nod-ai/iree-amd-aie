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

#include <algorithm>
#include <memory>

#include "AIEVecUtils.h"
#include "Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "aievec-canonicalization"

using namespace mlir;
using namespace arith;
using namespace vector;
using namespace mlir::iree_compiler::aievec;

namespace mlir::iree_compiler::aievec {

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

static SmallVector<Value> collapseInnerMostDimIndices(PatternRewriter &b,
                                                      Location loc, int numDims,
                                                      ValueRange indices,
                                                      ArrayRef<int64_t> shape,
                                                      AffineMap layout) {
  // TODO: Don't assume trivial layout
  assert(layout.isMinorIdentity() &&
         "dimension collapse in non-identity layout is not implemented");
  auto newIdxExpr = b.getAffineDimExpr(numDims - 1);
  int64_t stride = 1;
  for (int64_t dim = numDims - 2; dim >= 0; dim--) {
    stride *= shape[shape.size() - (numDims - dim - 1)];
    newIdxExpr = newIdxExpr + b.getAffineDimExpr(dim) * stride;
  }
  auto newIndexMap = AffineMap::get(numDims, 0, newIdxExpr);
  Value newInnerMostIdxValue =
      b.create<affine::AffineApplyOp>(loc, newIndexMap,
                                      indices.take_back(numDims))
          .getResult();
  SmallVector<Value> newIdxRange;
  for (auto idx : indices.drop_back(numDims)) newIdxRange.push_back(idx);
  newIdxRange.push_back(newInnerMostIdxValue);
  return newIdxRange;
}

static Value collapseInnerMostShapeDims(PatternRewriter &b, Location loc,
                                        int numDims, Value val) {
  auto memRefTy = cast<MemRefType>(val.getType());
  auto shape = memRefTy.getShape();
  int64_t newInnerMostDim = std::accumulate(shape.end() - numDims, shape.end(),
                                            1, std::multiplies<>());
  SmallVector<int64_t, 4> newShape{shape.begin(), shape.end() - numDims + 1};
  newShape[shape.size() - numDims] = newInnerMostDim;
  auto newNumDims = newShape.size();
  auto ctx = b.getContext();
  auto newMemRefTy = MemRefType::get(
      newShape, memRefTy.getElementType(),
      AffineMap::getMinorIdentityMap(newNumDims, newNumDims, ctx),
      memRefTy.getMemorySpace());
  auto reassocIndices =
      getReassociationIndicesForCollapse(shape, newShape).value();
  return b
      .create<memref::CollapseShapeOp>(loc, newMemRefTy, val, reassocIndices)
      .getResult();
}

// This pattern flattens multidimensional `vector.transfer_read` operations
// replacing them with a `memref.collapse_shape`, a 1D `vector.transfer_read`,
// and a `vector.shape_cast`.
struct FlattenMultDimTransferReadPattern
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    // We can only deal with unmasked transfer ops with an identity permutation
    // map.
    if (!readOp.getPermutationMap().isMinorIdentity() || readOp.getMask())
      return failure();
    VectorType vectorTy = readOp.getVector().getType();
    if (vectorTy.getRank() < 2) return failure();
    // Work only on bufferized reads
    MemRefType memRefTy = dyn_cast<MemRefType>(readOp.getSource().getType());
    if (!memRefTy) return failure();
    auto memRefShape = memRefTy.getShape();
    auto vecShape = vectorTy.getShape();

    auto newVectorTy =
        VectorType::get({std::accumulate(vecShape.begin(), vecShape.end(), 1,
                                         std::multiplies<>())},
                        vectorTy.getElementType());
    AffineMap layout = memRefTy.getLayout().getAffineMap();
    auto newIndices =
        collapseInnerMostDimIndices(rewriter, readOp.getLoc(), vecShape.size(),
                                    readOp.getIndices(), memRefShape, layout);
    auto newSource = collapseInnerMostShapeDims(
        rewriter, readOp.getLoc(), vecShape.size(), readOp.getSource());
    auto newVector = rewriter.create<vector::TransferReadOp>(
        readOp.getLoc(), newVectorTy, newSource, newIndices);

    auto inBoundsArrayAttrOpt = readOp.getInBounds();
    if (inBoundsArrayAttrOpt) {
      SmallVector<bool> inBounds =
          llvm::to_vector(inBoundsArrayAttrOpt.getAsValueRange<BoolAttr>());
      SmallVector<bool> newInBounds({false});
      newInBounds[0] = std::all_of(inBounds.begin(), inBounds.end(),
                                   [](bool v) { return v; });
      newVector.getProperties().setInBounds(
          rewriter.getBoolArrayAttr(newInBounds));
    }

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(readOp, vectorTy,
                                                     newVector);

    return success();
  }
};

// This pattern flatten multidimensional `vector.transfer_write` operations
// replacing them with a `memref.collapse_shape`, a `vector.shape_cast`, and a
// 1D `vector.transfer_write`,
struct FlattenMultDimTransferWritePattern
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
    // We can only deal with unmasked transfer ops with an identity permutation
    // map.
    if (!writeOp.getPermutationMap().isMinorIdentity() || writeOp.getMask())
      return failure();
    VectorType vectorTy = cast<VectorType>(writeOp.getVector().getType());
    if (vectorTy.getRank() < 2) return failure();
    // Work only on bufferized reads
    MemRefType memRefTy = dyn_cast<MemRefType>(writeOp.getSource().getType());
    if (!memRefTy) return failure();
    auto memRefShape = memRefTy.getShape();
    auto vecShape = vectorTy.getShape();

    auto newVectorTy =
        VectorType::get({std::accumulate(vecShape.begin(), vecShape.end(), 1,
                                         std::multiplies<>())},
                        vectorTy.getElementType());
    AffineMap layout = memRefTy.getLayout().getAffineMap();
    auto newVector = rewriter
                         .create<vector::ShapeCastOp>(
                             writeOp.getLoc(), newVectorTy, writeOp.getVector())
                         .getResult();
    auto newIndices =
        collapseInnerMostDimIndices(rewriter, writeOp.getLoc(), vecShape.size(),
                                    writeOp.getIndices(), memRefShape, layout);
    auto newSource = collapseInnerMostShapeDims(
        rewriter, writeOp.getLoc(), vecShape.size(), writeOp.getSource());

    auto newOp = rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        writeOp, newVector, newSource, newIndices);

    auto inBoundsArrayAttrOpt = writeOp.getInBounds();
    if (inBoundsArrayAttrOpt) {
      SmallVector<bool> inBounds =
          llvm::to_vector(inBoundsArrayAttrOpt.getAsValueRange<BoolAttr>());
      SmallVector<bool> newInBounds({false});
      newInBounds[0] = std::all_of(inBounds.begin(), inBounds.end(),
                                   [](bool v) { return v; });
      newOp.getProperties().setInBounds(rewriter.getBoolArrayAttr(newInBounds));
    }

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
    RewritePatternSet patterns(context);

    patterns.add<ExtractTransposeFromContractionOp,
                 FlattenMultDimTransferReadPattern,
                 FlattenMultDimTransferWritePattern,
                 ConvertLeadingUnitDimInsertToReshapePattern>(context);

    patterns.add<ConvertSplatTransferReadToBroadcastPattern>(context);

    patterns.add<SwapUnaryOpsPattern<arith::ExtSIOp, vector::BroadcastOp>>(
        context, [](arith::ExtSIOp extOp, vector::BroadcastOp bcastOp) -> Type {
          Type extInElemTy = extOp.getIn().getType();
          auto extInVecTy = dyn_cast<VectorType>(extInElemTy);
          if (extInVecTy) extInElemTy = extInVecTy.getElementType();
          return VectorType::get(bcastOp.getResultVectorType().getShape(),
                                 extInElemTy);
        });

    populateVectorBroadcastLoweringPatterns(patterns);

    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
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
    auto op = getOperation();
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
  // Add `Vector` code canonicalization passes
  // TODO: Add passes to unroll vector with unsupported types
  // TODO: Add passes to split vectors that won't fit in registers
  pm.addPass(createCanonicalizeVectorForAIEVecPass());
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

}  // namespace mlir::iree_compiler::aievec
