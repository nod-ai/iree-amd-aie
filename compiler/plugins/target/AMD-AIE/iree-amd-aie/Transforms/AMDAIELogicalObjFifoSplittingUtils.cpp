// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AMDAIELogicalObjFifoSplittingUtils.h"

#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Iterators.h"

namespace mlir::iree_compiler::AMDAIE {

/// Utility to verify that the split dimensions for L2 are contiguous.
static LogicalResult verifySplitDimensionConstraint(
    SmallVector<unsigned> &splitDimsSetForL2) {
  unsigned dim = 0;
  for (unsigned splitDim : splitDimsSetForL2) {
    if (splitDim != dim) return failure();
    ++dim;
  }
  return success();
}

/*
  For L3 -> L2 DmaCpyNd :-
  From offset (0,0) we are extracting one 4x4 memref.
                  _______
                |. . . .|
                |. . . .|
                |. . . .|
                |. . . .|
                ---------

  After split we will extract four 2x2 memrefs.
  So, the corresponding offsets will be :-
  1. Offset (0,0) - extract 2x2 memref
        ___
      |. .|. .
      |. .|. .
      -----
        . . . .
        . . . .
  2. Offset (0,2) - extract 2x2 memref
            ___
        . .|. .|
        . .|. .|
          -----
        . . . .
        . . . .
  3. Offset (2,0) - extract 2x2 memref
        . . . .
        . . . .
        ___
      |. .|. .
      |. .|. .
      -----
  4. Offset (2,2) - extract 2x2 memref
        . . . .
        . . . .
            ___
        . .|. .|
        . .|. .|
          -----

  The following utility helps perform the computation of offsets for L3 source.
*/
static FailureOr<OpFoldResult> updateL3SourceOffset(IRRewriter &rewriter,
                                                    OpFoldResult oldL3Offset,
                                                    int64_t offsetToAdd,
                                                    MLIRContext *context) {
  OpFoldResult newL3AsSourceOffset;
  if (auto l3SourceOffsetAttr = dyn_cast<Attribute>(oldL3Offset)) {
    int64_t l3SourceOffsetIntVal =
        cast<IntegerAttr>(l3SourceOffsetAttr).getInt();
    int64_t newOffset = l3SourceOffsetIntVal + offsetToAdd;
    newL3AsSourceOffset = rewriter.getIndexAttr(newOffset);
  } else {
    auto l3SourceOffsetVal = cast<Value>(oldL3Offset);
    if (auto blockArg = dyn_cast<BlockArgument>(l3SourceOffsetVal)) {
      Operation *ownerOfBlockArg = blockArg.getOwner()->getParentOp();
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockArg.getOwner());
      AffineExpr affineExpr = rewriter.getAffineDimExpr(0);
      AffineExpr newAffineExpr = affineExpr + offsetToAdd;
      auto newAffineMap = AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                                         {newAffineExpr}, context);
      newL3AsSourceOffset =
          rewriter
              .create<affine::AffineApplyOp>(ownerOfBlockArg->getLoc(),
                                             newAffineMap, l3SourceOffsetVal)
              .getResult();
    } else {
      Operation *defOpOfL3SourceOffset = l3SourceOffsetVal.getDefiningOp();
      Location loc = defOpOfL3SourceOffset->getLoc();
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(defOpOfL3SourceOffset);
      if (auto applyOp =
              dyn_cast<affine::AffineApplyOp>(defOpOfL3SourceOffset)) {
        AffineExpr affineExpr = applyOp.getAffineMap().getResult(0);
        AffineExpr newAffineExpr = affineExpr + offsetToAdd;
        auto newAffineMap = AffineMap::get(
            /*dimCount=*/1, /*symbolCount=*/0, {newAffineExpr}, context);
        newL3AsSourceOffset =
            rewriter
                .create<affine::AffineApplyOp>(loc, newAffineMap,
                                               applyOp.getMapOperands())
                .getResult();
      } else if (auto constantOffset = getConstantIntValue(l3SourceOffsetVal)) {
        int64_t newOffset = *constantOffset + offsetToAdd;
        newL3AsSourceOffset = rewriter.getIndexAttr(newOffset);
      } else {
        return failure();
      }
    }
  }
  return newL3AsSourceOffset;
}

LogicalResult splitLogicalObjectFifos(
    IRRewriter &rewriter, SmallVector<AMDAIE::DmaCpyNdOp> &l2ToL1DmaOps,
    MLIRContext *context) {
  if (l2ToL1DmaOps.size() == 0) return success();

  SmallVector<OpFoldResult> baseSourceOffsets =
      l2ToL1DmaOps[0].getSourceMixedOffsets();
  LogicalObjectFifoFromMemrefOp sourceObjectFifo =
      l2ToL1DmaOps[0].getSourceObjectFifo();
  auto sourceAllocOp =
      sourceObjectFifo.getMemref().getDefiningOp<memref::AllocOp>();
  if (!sourceAllocOp) {
    sourceObjectFifo->emitOpError()
        << "expected alloc op as the defining op of source "
           "logicalobjectfifo.from_memref";
    return failure();
  }
  // We will now capture those dimensions where L2 memory was split. The way we
  // do this is by checking all L2->L1 DmaOps' source offset and marking those
  // dimensions which are not equal to at least one of the source offsets.
  DenseSet<unsigned> splitDimsSetForL2;
  SmallVector<unsigned> splitDimsForL2;
  for (unsigned i = 1, n = l2ToL1DmaOps.size(); i < n; i++) {
    if (l2ToL1DmaOps[i].getSourceObjectFifo() != sourceObjectFifo) {
      l2ToL1DmaOps[i]->emitRemark() << "has different source objectfifo";
      sourceObjectFifo->emitRemark() << "is the expected source objectfifo";
      return failure();
    }
    SmallVector<OpFoldResult> sourceOffsets =
        l2ToL1DmaOps[i].getSourceMixedOffsets();
    for (unsigned j = 0, m = baseSourceOffsets.size(); j < m; j++) {
      if (baseSourceOffsets[j] != sourceOffsets[j] &&
          !splitDimsSetForL2.contains(j)) {
        splitDimsForL2.push_back(j);
        splitDimsSetForL2.insert(j);
      }
    }
  }
  std::sort(splitDimsForL2.begin(), splitDimsForL2.end());

  if (failed(verifySplitDimensionConstraint(splitDimsForL2))) {
    l2ToL1DmaOps[0]->emitRemark()
        << "cannot split L2 logicalobjectfifo because of non-contiguous split "
           "dimensions inferred";
    return failure();
  }

  // Fetch the L3 -> L2 Dma Op corresponding to the L2 buffer as target.
  AMDAIE::DmaCpyNdOp l3ToL2DmaOp;
  DenseSet<Operation *> toBeErased;
  for (Operation *objFifoUserOp : sourceObjectFifo->getUsers()) {
    if (auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(objFifoUserOp);
        dmaOp.getTargetObjectFifo() == sourceObjectFifo) {
      l3ToL2DmaOp = dmaOp;
      toBeErased.insert(dmaOp);
      break;
    }
  }
  toBeErased.insert(sourceAllocOp);
  toBeErased.insert(sourceObjectFifo);

  SmallVector<OpFoldResult, 4> staticL2AsTargetOffsets =
      l3ToL2DmaOp.getTargetMixedOffsets();
  SmallVector<OpFoldResult, 4> staticL2AsTargetSizes =
      l3ToL2DmaOp.getTargetMixedSizes();
  SmallVector<OpFoldResult, 4> staticL2AsTargetStrides =
      l3ToL2DmaOp.getTargetMixedStrides();
  SmallVector<int64_t, 4> l2ShapeAsTarget = llvm::to_vector(
      cast<MemRefType>(l3ToL2DmaOp.getTargetObjectFifo().getMemref().getType())
          .getShape());
  OpFoldResult zeroVal = getAsIndexOpFoldResult(context, 0);
  OpFoldResult oneVal = getAsIndexOpFoldResult(context, 1);
  // Update split dimensions' offset/size for L2 as target . We can afford to do
  // this here because it's going to be the same for all L3->L2 splits. Here we
  // are setting offset = 0 and size = 1.
  for (unsigned dim : splitDimsForL2) {
    staticL2AsTargetOffsets[dim] = zeroVal;
    staticL2AsTargetSizes[dim] = oneVal;
    l2ShapeAsTarget[dim] = 1;
  }
  SmallVector<unsigned> nonSplitDimsForL2;
  for (unsigned dim = 0, n = staticL2AsTargetSizes.size(); dim < n; dim++) {
    if (splitDimsSetForL2.contains(dim)) continue;
    nonSplitDimsForL2.push_back(dim);
  }

  // Traverse each L2->L1 DmaCpyNd op and split them.
  for (AMDAIE::DmaCpyNdOp l2ToL1DmaOp : l2ToL1DmaOps) {
    SmallVector<OpFoldResult, 6> staticL2AsSourceOffsets =
        l2ToL1DmaOp.getSourceMixedOffsets();
    SmallVector<OpFoldResult, 6> staticL2AsSourceSizes =
        l2ToL1DmaOp.getSourceMixedSizes();
    SmallVector<OpFoldResult, 6> staticL2AsSourceStrides =
        l2ToL1DmaOp.getSourceMixedStrides();

    // Now we'll create a narrowed linearized L2 buffer.
    rewriter.setInsertionPoint(sourceAllocOp);
    LogicalObjectFifoFromMemrefOp targetObjectFifo =
        l2ToL1DmaOp.getTargetObjectFifo();
    Value targetAllocOp = targetObjectFifo.getMemref();
    auto oldSourceMemRefType = cast<MemRefType>(sourceAllocOp.getType());
    auto targetMemRefType = cast<MemRefType>(targetAllocOp.getType());
    MemRefType newAllocType = MemRefType::get(
        l2ShapeAsTarget, targetMemRefType.getElementType(),
        MemRefLayoutAttrInterface{}, oldSourceMemRefType.getMemorySpace());
    auto newAllocOp = rewriter.create<memref::AllocOp>(rewriter.getUnknownLoc(),
                                                       newAllocType);
    auto newDeallocOp = rewriter.create<memref::DeallocOp>(
        rewriter.getUnknownLoc(), newAllocOp);
    newDeallocOp->moveBefore(&newAllocOp->getBlock()->back());
    auto type = cast<MemRefType>(newAllocOp.getType());
    // Create new logicalobjectfifo.from_memref for the newly created L2 buffer.
    rewriter.setInsertionPoint(l2ToL1DmaOp.getSourceObjectFifo());
    auto source = rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
        rewriter.getUnknownLoc(), LogicalObjectFifoType::get(type),
        newAllocOp.getResult(), sourceObjectFifo.getTiles());

    // --------------------------------------------
    // ---------- L3 -> L2 splitting --------------
    // --------------------------------------------
    // Update L3 source offsets for non-split dimensions. Refer doc comment of
    // `updateL3SourceOffset` for the computation rationale involved.
    SmallVector<OpFoldResult, 4> staticL3AsSourceOffsets =
        l3ToL2DmaOp.getSourceMixedOffsets();
    for (auto &&[splitDim, nonSplitdim] :
         llvm::zip_equal(splitDimsForL2, nonSplitDimsForL2)) {
      std::optional<int64_t> constantOffset =
          getConstantIntValue(staticL2AsSourceOffsets[splitDim]);
      if (!constantOffset) {
        l2ToL1DmaOp->emitRemark()
            << "found a non-constant value for source offset at dim "
            << splitDim;
        return failure();
      }
      std::optional<int64_t> constantSize =
          getConstantIntValue(staticL2AsTargetSizes[nonSplitdim]);
      if (!constantSize) {
        l3ToL2DmaOp->emitRemark()
            << "found a non-constant value for target size at dim "
            << nonSplitdim;
        return failure();
      }
      int64_t offsetToAdd = constantOffset.value() * constantSize.value();
      FailureOr<OpFoldResult> newOffset = updateL3SourceOffset(
          rewriter, staticL3AsSourceOffsets[nonSplitdim], offsetToAdd, context);
      if (failed(newOffset)) {
        // TODO: Ideally we should be able to handle even +, -, *, /, etc.
        //       But handle this later (if at all!) as such cases aren't
        //       going to arise.
        l3ToL2DmaOp->emitRemark()
            << "Unhandled expression for source offset at dim " << nonSplitdim;
        return failure();
      }
      staticL3AsSourceOffsets[nonSplitdim] = *newOffset;
    }
    // Create new L3 -> L2 Dma Op.
    rewriter.setInsertionPoint(l3ToL2DmaOp);
    rewriter.create<AMDAIE::DmaCpyNdOp>(
        l3ToL2DmaOp.getLoc(), source, llvm::ArrayRef(staticL2AsTargetOffsets),
        llvm::ArrayRef(staticL2AsTargetSizes),
        llvm::ArrayRef(staticL2AsTargetStrides), l3ToL2DmaOp.getSource(),
        llvm::ArrayRef(staticL3AsSourceOffsets),
        llvm::ArrayRef(staticL2AsTargetSizes),
        l3ToL2DmaOp.getSourceMixedStrides());

    // --------------------------------------------
    // ---------- L2 -> L1 splitting --------------
    // --------------------------------------------
    // Update split dimensions' offset/size for L2 as target . Here we are
    // setting offset = 0 and size = 1.
    for (unsigned dim : splitDimsForL2) {
      staticL2AsSourceOffsets[dim] = zeroVal;
      staticL2AsSourceSizes[dim] = oneVal;
    }

    // Create new L2 -> L1 Input DmaOp.
    rewriter.setInsertionPoint(l2ToL1DmaOp);
    auto newL2ToL1DmaOp = rewriter.create<AMDAIE::DmaCpyNdOp>(
        l2ToL1DmaOp.getLoc(), l2ToL1DmaOp.getTarget(),
        l2ToL1DmaOp.getTargetMixedOffsets(), l2ToL1DmaOp.getTargetMixedSizes(),
        l2ToL1DmaOp.getTargetMixedStrides(), source,
        llvm::ArrayRef(staticL2AsSourceOffsets),
        llvm::ArrayRef(staticL2AsSourceSizes),
        llvm::ArrayRef(staticL2AsSourceStrides));
    rewriter.replaceOp(l2ToL1DmaOp, newL2ToL1DmaOp);

    // Remove old dealloc.
    memref::DeallocOp oldDeallocOp;
    for (Operation *userOp : sourceAllocOp->getUsers()) {
      if (auto deallocUser = dyn_cast<memref::DeallocOp>(userOp)) {
        oldDeallocOp = deallocUser;
      }
    }
    if (oldDeallocOp) {
      toBeErased.insert(oldDeallocOp);
    }
  }

  for (Operation *op : toBeErased) {
    op->dropAllUses();
    rewriter.eraseOp(op);
  }
  
  return success();
}

}  // namespace mlir::iree_compiler::AMDAIE
