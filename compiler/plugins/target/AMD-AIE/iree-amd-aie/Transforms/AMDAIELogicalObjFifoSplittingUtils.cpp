// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AMDAIELogicalObjFifoSplittingUtils.h"

#include <numeric>

#include "iree-amd-aie/Transforms/AMDAIEDmaUtils.h"
#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/Operation.h"

#define DEBUG_TYPE "iree-amdaie-logicalobjfifo-splitting-utils"

namespace mlir::iree_compiler::AMDAIE {

/// Utility to create a new logical objectfifo based on shape defined by
/// `newSizesOpFoldResultArr`.
static AMDAIE::LogicalObjectFifoFromMemrefOp createNewLogicalObjectFifo(
    IRRewriter &rewriter,
    AMDAIE::LogicalObjectFifoFromMemrefOp &oldLogicalObjectFifo,
    SmallVectorImpl<OpFoldResult> &newSizesOpFoldResultArr) {
  OpBuilder::InsertionGuard guard(rewriter);
  SmallVector<int64_t> newSizes = llvm::map_to_vector(
      newSizesOpFoldResultArr,
      [](OpFoldResult sizeVal) { return getConstantIndexOrAssert(sizeVal); });
  Value oldAllocOp = oldLogicalObjectFifo.getMemref();
  auto oldMemRefType = cast<MemRefType>(oldAllocOp.getType());
  MemRefType newAllocType = MemRefType::get(
      newSizes, oldMemRefType.getElementType(), MemRefLayoutAttrInterface{},
      oldMemRefType.getMemorySpace());
  assert(oldAllocOp.getDefiningOp() && "expected a defining op for the value");
  rewriter.setInsertionPoint(oldAllocOp.getDefiningOp());
  auto newAllocOp =
      rewriter.create<memref::AllocOp>(rewriter.getUnknownLoc(), newAllocType);
  auto newDeallocOp =
      rewriter.create<memref::DeallocOp>(rewriter.getUnknownLoc(), newAllocOp);
  newDeallocOp->moveBefore(&newAllocOp->getBlock()->back());
  auto type = cast<MemRefType>(newAllocOp.getType());
  // Create new logical objectfifo.
  rewriter.setInsertionPoint(oldLogicalObjectFifo);
  auto newLogicalObjectFifo =
      rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
          rewriter.getUnknownLoc(), LogicalObjectFifoType::get(type),
          newAllocOp.getResult(), oldLogicalObjectFifo.getTiles());
  return newLogicalObjectFifo;
}

/// Utility to help fetch those input DmaCpyNd Ops which needs to be split.
SmallVector<AMDAIE::DmaCpyNdOp> fetchDmaCpyNdOpsToSplitOrCombine(
    Operation *op) {
  SmallVector<AMDAIE::DmaCpyNdOp> l2ToL1DmaOps;
  // We are currently walking through CoreOps gathering 3rd Input DmaOp (if
  // applicable) from them.
  // TODO(avarma): We will generalize this later.
  op->walk([&](AMDAIE::CoreOp coreOp) {
    SmallVector<Value> inputDmas = coreOp.getInputDmas();
    if (inputDmas.size() != 3) return WalkResult::skip();
    auto dmaCpyNdOp = inputDmas[2].getDefiningOp<AMDAIE::DmaCpyNdOp>();
    assert(dmaCpyNdOp && "expected an amdaie.dma_cpy_nd op");
    l2ToL1DmaOps.push_back(dmaCpyNdOp);
    return WalkResult::advance();
  });
  return l2ToL1DmaOps;
}

/// Utility to verify that the split dimensions for L2 are contiguous.
static LogicalResult checkIsRangeFromZero(
    SmallVector<size_t> &splitDimsSetForL2) {
  for (auto &&[dim, splitDim] : llvm::enumerate(splitDimsSetForL2)) {
    if (splitDim != dim) return failure();
  }
  return success();
}

/// This utility helps to perform the computation of offsets for L3 source.
///
/// Example:
/// For L3 -> L2 DmaCpyNd :-
/// From offset (0,0) we are extracting one 4x4 memref.
///                _______
///               |. . . .|
///               |. . . .|
///               |. . . .|
///               |. . . .|
///               ---------
/// After split we will extract four 2x2 memrefs.
/// So, the corresponding offsets will be :-
/// 1. Offset (0,0) - extract 2x2 memref
///       ___
///      |. .|. .
///      |. .|. .
///      -----
///       . . . .
///       . . . .
/// 2. Offset (0,2) - extract 2x2 memref
///           ___
///       . .|. .|
///       . .|. .|
///          -----
///       . . . .
///       . . . .
/// 3. Offset (2,0) - extract 2x2 memref
///       . . . .
///       . . . .
///       ___
///      |. .|. .
///      |. .|. .
///      -----
/// 4. Offset (2,2) - extract 2x2 memref
///       . . . .
///       . . . .
///           ___
///       . .|. .|
///       . .|. .|
///          -----
static FailureOr<OpFoldResult> updateL3SourceOffset(IRRewriter &rewriter,
                                                    OpFoldResult oldL3Offset,
                                                    int64_t offsetToAdd,
                                                    MLIRContext *context) {
  auto createAffineMap = [&](AffineExpr affineExpr,
                             int64_t offsetToAdd) -> AffineMap {
    AffineExpr newAffineExpr = affineExpr + offsetToAdd;
    return AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, {newAffineExpr},
                          context);
  };
  OpFoldResult newL3AsSourceOffset;
  OpBuilder::InsertionGuard guard(rewriter);
  if (auto l3SourceOffsetAttr = dyn_cast<Attribute>(oldL3Offset)) {
    int64_t l3SourceOffsetIntVal =
        cast<IntegerAttr>(l3SourceOffsetAttr).getInt();
    int64_t newOffset = l3SourceOffsetIntVal + offsetToAdd;
    newL3AsSourceOffset = rewriter.getIndexAttr(newOffset);
  } else {
    auto l3SourceOffsetVal = cast<Value>(oldL3Offset);
    if (auto blockArg = dyn_cast<BlockArgument>(l3SourceOffsetVal)) {
      Operation *ownerOfBlockArg = blockArg.getOwner()->getParentOp();
      rewriter.setInsertionPointToStart(blockArg.getOwner());
      AffineExpr affineExpr = rewriter.getAffineDimExpr(0);
      AffineMap newAffineMap = createAffineMap(affineExpr, offsetToAdd);
      newL3AsSourceOffset =
          rewriter
              .create<affine::AffineApplyOp>(ownerOfBlockArg->getLoc(),
                                             newAffineMap, l3SourceOffsetVal)
              .getResult();
    } else {
      Operation *defOpOfL3SourceOffset = l3SourceOffsetVal.getDefiningOp();
      Location loc = defOpOfL3SourceOffset->getLoc();
      rewriter.setInsertionPoint(defOpOfL3SourceOffset);
      if (auto applyOp = dyn_cast_if_present<affine::AffineApplyOp>(
              defOpOfL3SourceOffset)) {
        AffineExpr affineExpr = applyOp.getAffineMap().getResult(0);
        AffineMap newAffineMap = createAffineMap(affineExpr, offsetToAdd);
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

/// Given a L2->L1 DmaCpyNd op, find the unique L3->L2 DmaCpyNd op.
static FailureOr<AMDAIE::DmaCpyNdOp> fetchL3ToL2DmaCpyNdOp(
    AMDAIE::DmaCpyNdOp l2ToL1DmaOp) {
  LogicalObjectFifoFromMemrefOp sourceObjectFifo =
      l2ToL1DmaOp.getSourceObjectFifo();
  SmallVector<AMDAIE::DmaCpyNdOp> l3ToL2DmaOps;
  AMDAIE::DmaCpyNdOp l3ToL2DmaOp;
  for (Operation *objFifoUserOp : sourceObjectFifo->getUsers()) {
    if (auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(objFifoUserOp);
        dmaOp.getTargetObjectFifo() == sourceObjectFifo) {
      l3ToL2DmaOps.push_back(dmaOp);
    }
  }
  if (l3ToL2DmaOps.size() == 0) {
    LLVM_DEBUG(llvm::dbgs() << "no corresponding L3->L2 dma op found for "
                            << sourceObjectFifo << "\n");
    return failure();
  }
  if (l3ToL2DmaOps.size() > 1) {
    LLVM_DEBUG(llvm::dbgs() << "found more than one L3->L2 dma ops for "
                            << sourceObjectFifo << "\n");
    return failure();
  }
  l3ToL2DmaOp = l3ToL2DmaOps[0];
  return l3ToL2DmaOp;
}

/// A struct utility to encapsulate all the data required to perform splitting
/// of logicalobjectfifos.
struct SplittingLogicalObjectFifoData {
  SmallVector<AMDAIE::DmaCpyNdOp> l2ToL1DmaOps;
  SmallVector<size_t> splitDimsForL2;
  SmallVector<size_t> nonSplitDimsForL2;
  AMDAIE::DmaCpyNdOp l3ToL2DmaOp;
};

/// Utility to check whether splitting of logicalobjectfifos can be performed.
/// If possible, it populates the struct `SplittingLogicalObjectFifoData` with
/// the data required to perform the actual splitting.
static LogicalResult checkWhetherSplitIsPossible(
    SplittingLogicalObjectFifoData &splittingLogicalObjectFifoData) {
  SmallVector<AMDAIE::DmaCpyNdOp> l2ToL1DmaOps =
      splittingLogicalObjectFifoData.l2ToL1DmaOps;

  if (l2ToL1DmaOps.size() == 0) return failure();

  SmallVector<OpFoldResult> baseSourceOffsets =
      l2ToL1DmaOps[0].getSourceMixedOffsets();
  LogicalObjectFifoFromMemrefOp sourceObjectFifo =
      l2ToL1DmaOps[0].getSourceObjectFifo();
  auto sourceAllocOp =
      sourceObjectFifo.getMemref().getDefiningOp<memref::AllocOp>();
  if (!sourceAllocOp) {
    LLVM_DEBUG(llvm::dbgs() << "expected alloc op as the defining op of "
                            << sourceObjectFifo << "\n");
    return failure();
  }

  // We will now capture those dimensions where L2 memory was split. The way we
  // do this is by checking all L2->L1 DmaOps' source offset and marking those
  // dimensions which are not equal to at least one of the source offsets.
  DenseSet<size_t> splitDimsSetForL2;
  SmallVector<size_t> splitDimsForL2;
  for (unsigned i = 1, n = l2ToL1DmaOps.size(); i < n; i++) {
    if (l2ToL1DmaOps[i].getSourceObjectFifo() != sourceObjectFifo) {
      LLVM_DEBUG(llvm::dbgs()
                 << l2ToL1DmaOps[i] << " does not have " << sourceObjectFifo
                 << " as the source objectfifo\n");
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

  if (failed(checkIsRangeFromZero(splitDimsForL2))) {
    LLVM_DEBUG(llvm::dbgs() << "cannot split L2 logicalobjectfifo because of "
                               "non-contiguous split dimensions inferred\n");
    return failure();
  }

  // Fetch the L3 -> L2 Dma Op corresponding to the L2 buffer as target.
  FailureOr<AMDAIE::DmaCpyNdOp> maybeL3ToL2DmaOp =
      fetchL3ToL2DmaCpyNdOp(l2ToL1DmaOps[0]);
  if (failed(maybeL3ToL2DmaOp)) return failure();
  AMDAIE::DmaCpyNdOp l3ToL2DmaOp = maybeL3ToL2DmaOp.value();
  if ((l3ToL2DmaOp.getTargetMixedOffsets().size() !=
       l3ToL2DmaOp.getSourceMixedOffsets().size()) ||
      (l3ToL2DmaOp.getTargetMixedSizes().size() !=
       l3ToL2DmaOp.getSourceMixedSizes().size()) ||
      (l3ToL2DmaOp.getTargetMixedStrides().size() !=
       l3ToL2DmaOp.getSourceMixedStrides().size())) {
    LLVM_DEBUG(llvm::dbgs() << "dimensionality of source and target's "
                               "offset/size/stride found different for "
                            << l3ToL2DmaOp << "\n");
    return failure();
  }

  SmallVector<OpFoldResult, 4> staticL2AsTargetSizes =
      l3ToL2DmaOp.getTargetMixedSizes();
  SmallVector<size_t> nonSplitDimsForL2(staticL2AsTargetSizes.size() -
                                        splitDimsForL2.size());
  std::iota(nonSplitDimsForL2.begin(), nonSplitDimsForL2.end(),
            splitDimsForL2.size());

  for (AMDAIE::DmaCpyNdOp l2ToL1DmaOp : l2ToL1DmaOps) {
    SmallVector<OpFoldResult, 6> staticL2AsSourceOffsets =
        l2ToL1DmaOp.getSourceMixedOffsets();
    for (auto &&[splitDim, nonSplitdim] :
         llvm::zip_equal(splitDimsForL2, nonSplitDimsForL2)) {
      std::optional<int64_t> constantVal =
          getConstantIntValue(staticL2AsSourceOffsets[splitDim]);
      if (!constantVal) {
        LLVM_DEBUG(llvm::dbgs()
                   << "found a non-constant value for source offset at dim "
                   << splitDim << " for " << l2ToL1DmaOp << "\n");
        return failure();
      }
      constantVal = getConstantIntValue(staticL2AsTargetSizes[nonSplitdim]);
      if (!constantVal) {
        LLVM_DEBUG(llvm::dbgs()
                   << "found a non-constant value for target size at dim "
                   << nonSplitdim << " for " << l3ToL2DmaOp << "\n");
        return failure();
      }
    }
  }
  splittingLogicalObjectFifoData.splitDimsForL2 = splitDimsForL2;
  splittingLogicalObjectFifoData.nonSplitDimsForL2 = nonSplitDimsForL2;
  splittingLogicalObjectFifoData.l3ToL2DmaOp = l3ToL2DmaOp;
  return success();
}

// Given a vector of L2->L1 Dma ops' perform the splitting :-
// 1. Check if the splitting can be performed or not. If not possible, bail out.
// 2. For the split dimension inferred set offset = 0 and size as 1 for L2 and
//    L3.
// 3. Now traverse each L2->L1 Dma op and perform the following :-
//    a) Create a new L2 AllocOp based on the updated size (step 3 above) and
//       create a logicalobjectfifo using the same.
//    b) Split L3->L2 Dma op.
//    c) SPlit L2->L1 Dma op.
// 4. Delete old L2->L1, L3->L2 and corresponding AllocOps.
LogicalResult splitLogicalObjectFifos(
    IRRewriter &rewriter, SmallVector<AMDAIE::DmaCpyNdOp> &l2ToL1DmaOps,
    MLIRContext *context) {
  SplittingLogicalObjectFifoData splittingLogicalObjectFifoData;
  splittingLogicalObjectFifoData.l2ToL1DmaOps = l2ToL1DmaOps;
  if (failed(checkWhetherSplitIsPossible(splittingLogicalObjectFifoData))) {
    LLVM_DEBUG(llvm::dbgs()
               << "Cannot perform splitting of logicalobjectfifos");
    return success();
  }
  OpBuilder::InsertionGuard guard(rewriter);
  SmallVector<size_t> splitDimsForL2 =
      splittingLogicalObjectFifoData.splitDimsForL2;
  SmallVector<size_t> nonSplitDimsForL2 =
      splittingLogicalObjectFifoData.nonSplitDimsForL2;
  AMDAIE::DmaCpyNdOp l3ToL2DmaOp = splittingLogicalObjectFifoData.l3ToL2DmaOp;

  LogicalObjectFifoFromMemrefOp sourceObjectFifo =
      l2ToL1DmaOps[0].getSourceObjectFifo();
  auto sourceAllocOp =
      sourceObjectFifo.getMemref().getDefiningOp<memref::AllocOp>();

  DenseSet<Operation *> toBeErased;
  toBeErased.insert(l3ToL2DmaOp);
  toBeErased.insert(sourceAllocOp);
  toBeErased.insert(sourceObjectFifo);

  SmallVector<OpFoldResult, 4> staticL2AsTargetOffsets =
      l3ToL2DmaOp.getTargetMixedOffsets();
  SmallVector<OpFoldResult, 4> staticL2AsTargetSizes =
      l3ToL2DmaOp.getTargetMixedSizes();
  SmallVector<OpFoldResult, 4> staticL3AsSourceOffsets =
      l3ToL2DmaOp.getSourceMixedOffsets();
  SmallVector<OpFoldResult, 4> staticL3AsSourceSizes =
      l3ToL2DmaOp.getSourceMixedSizes();
  OpFoldResult zeroVal = getAsIndexOpFoldResult(context, 0);
  OpFoldResult oneVal = getAsIndexOpFoldResult(context, 1);
  // Update split dimensions' offset/size for L2 as target and L3 as source. We
  // can afford to do this here because it's going to be the same for all L3->L2
  // splits. Here we are setting offset = 0 and size = 1.
  for (size_t dim : splitDimsForL2) {
    staticL2AsTargetOffsets[dim] = zeroVal;
    staticL2AsTargetSizes[dim] = oneVal;
    staticL3AsSourceOffsets[dim] = zeroVal;
    staticL3AsSourceSizes[dim] = oneVal;
  }

  // Traverse each L2->L1 DmaCpyNd op and split them.
  for (AMDAIE::DmaCpyNdOp l2ToL1DmaOp : l2ToL1DmaOps) {
    SmallVector<OpFoldResult, 6> staticL2AsSourceOffsets =
        l2ToL1DmaOp.getSourceMixedOffsets();
    SmallVector<OpFoldResult, 6> staticL2AsSourceSizes =
        l2ToL1DmaOp.getSourceMixedSizes();

    // Now we'll create a new L2 buffer based on the new shape inferred earlier
    // via `staticL2AsTargetSizes`.
    LogicalObjectFifoFromMemrefOp oldL2ObjectFifo =
        l2ToL1DmaOp.getSourceObjectFifo();
    AMDAIE::LogicalObjectFifoFromMemrefOp source = createNewLogicalObjectFifo(
        rewriter, oldL2ObjectFifo, staticL2AsTargetSizes);

    // --------------------------------------------
    // ---------- L3 -> L2 splitting --------------
    // --------------------------------------------
    // Update L3 source offsets for non-split dimensions. Refer doc comment of
    // `updateL3SourceOffset` for the computation rationale involved.
    SmallVector<OpFoldResult> staticL3AsSourceOffsets =
        l3ToL2DmaOp.getSourceMixedOffsets();
    for (auto &&[splitDim, nonSplitdim] :
         llvm::zip_equal(splitDimsForL2, nonSplitDimsForL2)) {
      std::optional<int64_t> constantOffset =
          getConstantIntValue(staticL2AsSourceOffsets[splitDim]);
      if (!constantOffset) {
        return l2ToL1DmaOp->emitOpError()
               << "found a non-constant value for source offset at dim "
               << splitDim;
      }
      std::optional<int64_t> constantSize =
          getConstantIntValue(staticL2AsTargetSizes[nonSplitdim]);
      if (!constantSize) {
        return l3ToL2DmaOp->emitOpError()
               << "found a non-constant value for target size at dim "
               << nonSplitdim;
      }
      int64_t offsetToAdd = constantOffset.value() * constantSize.value();
      FailureOr<OpFoldResult> newOffset = updateL3SourceOffset(
          rewriter, staticL3AsSourceOffsets[nonSplitdim], offsetToAdd, context);
      if (failed(newOffset)) {
        // TODO: Ideally we should be able to handle even +, -, *, /, etc.
        //       But handle this later (if at all!) as such cases might not
        //       arise.
        return l3ToL2DmaOp->emitOpError()
               << "Unhandled expression for source offset at dim "
               << nonSplitdim;
      }
      staticL3AsSourceOffsets[nonSplitdim] = *newOffset;
    }
    // Create new L3 -> L2 Dma Op.
    rewriter.setInsertionPoint(l3ToL2DmaOp);
    rewriter.create<AMDAIE::DmaCpyNdOp>(
        l3ToL2DmaOp.getLoc(), source, llvm::ArrayRef(staticL2AsTargetOffsets),
        llvm::ArrayRef(staticL2AsTargetSizes),
        l3ToL2DmaOp.getTargetMixedStrides(), l3ToL2DmaOp.getSource(),
        llvm::ArrayRef(staticL3AsSourceOffsets),
        llvm::ArrayRef(staticL3AsSourceSizes),
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
        l2ToL1DmaOp.getSourceMixedStrides());
    rewriter.replaceOp(l2ToL1DmaOp, newL2ToL1DmaOp);

    // Remove old dealloc.
    memref::DeallocOp oldDeallocOp;
    for (Operation *userOp : sourceAllocOp->getUsers()) {
      if (auto deallocUser = dyn_cast<memref::DeallocOp>(userOp))
        oldDeallocOp = deallocUser;
    }
    if (oldDeallocOp) toBeErased.insert(oldDeallocOp);
  }

  for (Operation *op : toBeErased) {
    op->dropAllUses();
    rewriter.eraseOp(op);
  }

  return success();
}

static int64_t fetchOffsetBias(OpFoldResult offsetOpFoldResult) {
  std::optional<int64_t> offset = getConstantIntValue(offsetOpFoldResult);
  if (offset) return offset.value();
  auto offsetVal = cast<Value>(offsetOpFoldResult);
  auto affineApplyOp =
      dyn_cast_if_present<affine::AffineApplyOp>(offsetVal.getDefiningOp());
  if (!affineApplyOp) return 0;
  AffineMap affineMap = affineApplyOp.getAffineMap();
  RetrieveScaleAndBias retriever;
  assert(!failed(retriever.visit(affineMap.getResult(0))) &&
         "failed to retrieve scale and bias");
  int64_t bias = 0;
  if (retriever.bias) {
    bias = retriever.bias.value();
  }
  return bias;
}

static LogicalResult combineL3ToL2AccessPatterns(
    RewriterBase &rewriter, const SmallVector<OpFoldResult> &offsetsA,
    const SmallVector<OpFoldResult> &sizesA,
    const SmallVector<OpFoldResult> &stridesA,
    const SmallVector<OpFoldResult> &offsetsB,
    const SmallVector<OpFoldResult> &sizesB,
    const SmallVector<OpFoldResult> &stridesB,
    SmallVector<OpFoldResult> &newOffsets, SmallVector<OpFoldResult> &newSizes,
    SmallVector<OpFoldResult> &newStrides, SmallVector<int64_t> &combiningDims,
    SmallVector<int64_t> &nonCombiningDims) {
  if (offsetsA.empty() && offsetsB.empty()) return success();

  int64_t newSize = 1;
  for (auto iter : llvm::enumerate(llvm::zip(offsetsA, offsetsB))) {
    if (iter.index() < combiningDims.size()) continue;
    const OpFoldResult &offsetA = std::get<0>(iter.value());
    const OpFoldResult &offsetB = std::get<1>(iter.value());
    if (offsetA != offsetB) {
      // Need to check the difference in bias here.
      int64_t biasA = fetchOffsetBias(offsetA);
      int64_t biasB = fetchOffsetBias(offsetB);
      std::optional<int64_t> sizeA = getConstantIntValue(sizesA[iter.index()]);
      assert(sizeA && "expected a constant integer value for size");
      if (sizeA != biasB - biasA) return failure();
      newSize++;
    }
  }
  newSizes[combiningDims.size() - 1] = rewriter.getI64IntegerAttr(newSize);
  return success();
}

static FailureOr<LogicalObjectFifoFromMemrefOp> combineL3ToL2Pair(
    IRRewriter &rewriter, DmaCpyNdOp dmaOpA, DmaCpyNdOp dmaOpB,
    SmallVector<int64_t> &combiningDims,
    SmallVector<int64_t> &nonCombiningDims) {
  OpBuilder::InsertionGuard guard(rewriter);
  SmallVector<OpFoldResult> sourceOffsetsA = dmaOpA.getSourceMixedOffsets();
  SmallVector<OpFoldResult> sourceSizesA = dmaOpA.getSourceMixedSizes();
  SmallVector<OpFoldResult> sourceStridesA = dmaOpA.getSourceMixedStrides();
  SmallVector<OpFoldResult> sourceOffsetsB = dmaOpB.getSourceMixedOffsets();
  SmallVector<OpFoldResult> sourceSizesB = dmaOpB.getSourceMixedSizes();
  SmallVector<OpFoldResult> sourceStridesB = dmaOpB.getSourceMixedStrides();

  SmallVector<OpFoldResult> targetOffsetsA = dmaOpA.getTargetMixedOffsets();
  SmallVector<OpFoldResult> targetSizesA = dmaOpA.getTargetMixedSizes();
  SmallVector<OpFoldResult> targetStridesA = dmaOpA.getTargetMixedStrides();
  SmallVector<OpFoldResult> targetOffsetsB = dmaOpB.getTargetMixedOffsets();
  SmallVector<OpFoldResult> targetSizesB = dmaOpB.getTargetMixedSizes();
  SmallVector<OpFoldResult> targetStridesB = dmaOpB.getTargetMixedStrides();

  SmallVector<OpFoldResult> newSourceOffsets = sourceOffsetsA;
  SmallVector<OpFoldResult> newSourceSizes = sourceSizesA;
  SmallVector<OpFoldResult> newSourceStrides = sourceStridesA;
  if (failed(combineL3ToL2AccessPatterns(
          rewriter, sourceOffsetsA, sourceSizesA, sourceStridesA,
          sourceOffsetsB, sourceSizesB, sourceStridesB, newSourceOffsets,
          newSourceSizes, newSourceStrides, combiningDims, nonCombiningDims))) {
    dmaOpA->emitOpError()
        << "L3->L2 pair cannot be combined because offset is not contiguous";
    return failure();
  }

  SmallVector<OpFoldResult> newTargetOffsets = targetOffsetsA;
  SmallVector<OpFoldResult> newTargetSizes = newSourceSizes;
  SmallVector<OpFoldResult> newTargetStrides = targetStridesA;
  // Now we need to create a new L2 buffer based on `newTargetSizes`.
  LogicalObjectFifoFromMemrefOp oldL2ObjectFifo = dmaOpA.getTargetObjectFifo();
  AMDAIE::LogicalObjectFifoFromMemrefOp newL2ObjectFifo =
      createNewLogicalObjectFifo(rewriter, oldL2ObjectFifo, newTargetSizes);

  // Create combined L3->L2 Dma.
  rewriter.setInsertionPoint(dmaOpA);
  auto combinedL3ToL2DmaOp = rewriter.create<AMDAIE::DmaCpyNdOp>(
      dmaOpA.getLoc(), newL2ObjectFifo, llvm::ArrayRef(newTargetOffsets),
      llvm::ArrayRef(newTargetSizes), llvm::ArrayRef(newTargetStrides),
      dmaOpA.getSource(), llvm::ArrayRef(newSourceOffsets),
      llvm::ArrayRef(newSourceSizes), llvm::ArrayRef(newSourceStrides));
  // Replace the uses of 2nd L3->L2 Dma with the new combined L3->L2 Dma
  // and erase the 1st L3->L2 Dma.
  rewriter.replaceOp(dmaOpB, combinedL3ToL2DmaOp);
  rewriter.eraseOp(dmaOpA);
  return newL2ObjectFifo;
}

/// Utility comparator function that compares two DmaCpyNd ops `a` and `b`.
/// Returns true if `a`'s offset is "less" than `b`'s. The following explains
/// the notion of one offset being "less" than the other :-
///     Offset A : N-dimension array.
///     Offset B : N-dimension array.
///     Then, A < B if :-
///             A[i] < B[i] for `i` in [0, N-1]
///     AND     A[0..i-1] == B[0..i-1]
static bool compareL3ToL2DmaPairOffsets(DmaCpyNdOp &a, DmaCpyNdOp &b) {
  SmallVector<OpFoldResult> sourceOffsetsA = a.getSourceMixedOffsets();
  SmallVector<OpFoldResult> sourceSizesA = a.getSourceMixedSizes();
  SmallVector<OpFoldResult> sourceOffsetsB = b.getSourceMixedOffsets();
  SmallVector<OpFoldResult> sourceSizesB = b.getSourceMixedSizes();
  // We'll add assertion checks on the size before invoking this function.
  for (int64_t i = 0, n = sourceOffsetsA.size(); i < n; i++) {
    std::optional<int64_t> offsetA = getConstantIntValue(sourceOffsetsA[i]);
    std::optional<int64_t> offsetB = getConstantIntValue(sourceOffsetsB[i]);
    if (offsetA && offsetB) {
      if (offsetA < offsetB) return true;
      if (offsetA > offsetB) return false;
      continue;
    }
    if (!offsetA && !offsetB) {
      auto offsetValA = cast<Value>(sourceOffsetsA[i]);
      auto offsetValB = cast<Value>(sourceOffsetsB[i]);
      auto affineApplyOpA = dyn_cast_if_present<affine::AffineApplyOp>(
          offsetValA.getDefiningOp());
      auto affineApplyOpB = dyn_cast_if_present<affine::AffineApplyOp>(
          offsetValB.getDefiningOp());
      // TODO(avarma): This should be handled better. The overall possibility
      // here already makes this complex enough.
      assert(affineApplyOpA && "expected affine.apply op");
      assert(affineApplyOpB && "expected affine.apply op");
      for (auto &&[valA, valB] :
           llvm::zip_equal(affineApplyOpA.getMapOperands(),
                           affineApplyOpB.getMapOperands())) {
        assert((valA == valB) &&
               "different base values being operated on between the L3->L2 Dma "
               "op pair");
      }
      AffineMap affineMapA = affineApplyOpA.getAffineMap();
      AffineMap affineMapB = affineApplyOpB.getAffineMap();
      RetrieveScaleAndBias retrieverA, retrieverB;
      assert(!failed(retrieverA.visit(affineMapA.getResult(0))) &&
             "failed to retrieve scale and bias");
      assert(!failed(retrieverB.visit(affineMapB.getResult(0))) &&
             "failed to retrieve scale and bias");
      int64_t biasA = 0, biasB = 0;
      if (retrieverA.bias) {
        biasA = retrieverA.bias.value();
      }
      if (retrieverB.bias) {
        biasB = retrieverB.bias.value();
      }
      // TODO(avarma): We should also check the scale value as well.
      if (biasA < biasB) return true;
      if (biasA > biasB) return false;
      continue;
    }
    assert(false &&
           "unexpected combination of offset val amongst L3->L2 Dma pair");
  }
  return false;
}

static bool areAccessPatternsCompatibleForCombining(
    AMDAIE::DmaCpyNdOp &l3ToL2DmaOpA, AMDAIE::DmaCpyNdOp &l3ToL2DmaOpB) {
  // Sources' access pattern check.
  SmallVector<OpFoldResult> sourceOffsetsA =
      l3ToL2DmaOpA.getSourceMixedOffsets();
  SmallVector<OpFoldResult> sourceSizesA = l3ToL2DmaOpA.getSourceMixedSizes();
  SmallVector<OpFoldResult> sourceStridesA =
      l3ToL2DmaOpA.getSourceMixedStrides();
  SmallVector<OpFoldResult> sourceOffsetsB =
      l3ToL2DmaOpB.getSourceMixedOffsets();
  SmallVector<OpFoldResult> sourceSizesB = l3ToL2DmaOpB.getSourceMixedSizes();
  SmallVector<OpFoldResult> sourceStridesB =
      l3ToL2DmaOpB.getSourceMixedStrides();
  if (sourceOffsetsA.size() != sourceOffsetsB.size() ||
      sourceSizesA.size() != sourceSizesB.size() ||
      sourceStridesA.size() != sourceStridesB.size() ||
      sourceOffsetsA.size() != sourceSizesA.size() ||
      sourceOffsetsA.size() != sourceStridesB.size()) {
    return false;
  }
  // Targets' access pattern check.
  SmallVector<OpFoldResult> targetOffsetsA =
      l3ToL2DmaOpA.getTargetMixedOffsets();
  SmallVector<OpFoldResult> targetSizesA = l3ToL2DmaOpA.getTargetMixedSizes();
  SmallVector<OpFoldResult> targetStridesA =
      l3ToL2DmaOpA.getTargetMixedStrides();
  SmallVector<OpFoldResult> targetOffsetsB =
      l3ToL2DmaOpB.getTargetMixedOffsets();
  SmallVector<OpFoldResult> targetSizesB = l3ToL2DmaOpB.getTargetMixedSizes();
  SmallVector<OpFoldResult> targetStridesB =
      l3ToL2DmaOpB.getTargetMixedStrides();
  if (targetOffsetsA.size() != targetOffsetsB.size() ||
      targetSizesA.size() != targetSizesB.size() ||
      targetStridesA.size() != targetStridesB.size() ||
      targetOffsetsA.size() != targetSizesA.size() ||
      targetOffsetsA.size() != targetStridesB.size()) {
    return false;
  }
  // Checking if targets' access pattern values are same.
  auto isSameValue = [](SmallVector<OpFoldResult> &accessPatternA,
                        SmallVector<OpFoldResult> &accessPatternB) -> bool {
    for (auto [a, b] : llvm::zip_equal(accessPatternA, accessPatternB)) {
      if (a != b) return false;
    }
    return true;
  };
  if (isSameValue(targetOffsetsA, targetOffsetsB) &&
      isSameValue(targetSizesA, targetSizesB) &&
      isSameValue(targetStridesA, targetStridesB)) {
    return true;
  }

  return false;
}

static LogicalResult fetchCombiningDimensions(
    SmallVector<AMDAIE::DmaCpyNdOp> &l3ToL2DmaOps,
    SmallVector<int64_t> &combiningDims,
    SmallVector<int64_t> &nonCombiningDims) {
  // Fetch combining/non-combining dimensions. Currently we look for a
  // continuous sequence of 0 offset dims with size as 1 to infer them as
  // combining dimensions.
  int64_t maxCombiningDimIndex = 0;
  for (unsigned i = 0, n = l3ToL2DmaOps.size(); i < n; i++) {
    SmallVector<OpFoldResult> sourceOffsets =
        l3ToL2DmaOps[i].getSourceMixedOffsets();
    SmallVector<OpFoldResult> sourceSizes =
        l3ToL2DmaOps[i].getSourceMixedSizes();
    unsigned j = 0, m = sourceOffsets.size();
    // Traverse through the i-th L3->L2 Dma op's source offset/size to find a
    // continuous sequence of 0 offset dims with size as 1.
    while (j < m) {
      std::optional<int64_t> constantOffset =
          getConstantIntValue(sourceOffsets[j]);
      if (!constantOffset || constantOffset.value() != 0) {
        break;
      }
      std::optional<int64_t> constantSize = getConstantIntValue(sourceSizes[j]);
      if (!constantSize || constantSize.value() != 1) {
        break;
      }
      j++;
    }
    if (i == 0) {
      maxCombiningDimIndex = j;
    } else if (maxCombiningDimIndex != j) {
      LLVM_DEBUG(llvm::dbgs()
                 << "incompatible combining dimensions across L3->L2\n");
      return failure();
    }
  }
  combiningDims.assign(maxCombiningDimIndex, 0);
  std::iota(combiningDims.begin(), combiningDims.end(), 0);
  nonCombiningDims.assign(maxCombiningDimIndex, 0);
  std::iota(nonCombiningDims.begin(), nonCombiningDims.end(),
            combiningDims.size());
  return success();
}

/// Given a vector of L2->L1 Dma Ops, combine the corresponding L3->L2 Dma Ops
/// and reuse the L2/L1 buffers.
/// TODO(avarma): Assign combined tiles while forming L2/L1 buffers which we'll
/// reuse.
LogicalResult combineLogicalObjectFifos(
    IRRewriter &rewriter, SmallVector<AMDAIE::DmaCpyNdOp> &l2ToL1DmaOps,
    MLIRContext *context) {
  if (l2ToL1DmaOps.size() == 0) return success();

  // Fetch the L3 -> L2 Dma Op corresponding to the first L2 buffer as target.
  SmallVector<AMDAIE::DmaCpyNdOp> l3ToL2DmaOps;
  FailureOr<AMDAIE::DmaCpyNdOp> maybeL3ToL2DmaOp =
      fetchL3ToL2DmaCpyNdOp(l2ToL1DmaOps[0]);
  if (failed(maybeL3ToL2DmaOp)) return failure();
  l3ToL2DmaOps.push_back(maybeL3ToL2DmaOp.value());

  // Check that all L3 buffer associated with the different L3->L2 Dma ops are
  // same.
  for (unsigned i = 1, n = l2ToL1DmaOps.size(); i < n; i++) {
    maybeL3ToL2DmaOp = fetchL3ToL2DmaCpyNdOp(l2ToL1DmaOps[i]);
    if (failed(maybeL3ToL2DmaOp)) return failure();
    l3ToL2DmaOps.push_back(maybeL3ToL2DmaOp.value());
    if (l3ToL2DmaOps[0].getSourceObjectFifo() !=
        l3ToL2DmaOps[i].getSourceObjectFifo()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Found different L3 objectFifo for " << l3ToL2DmaOps[0]
                 << " and " << l3ToL2DmaOps[i] << "\n");
      return failure();
    }
    if (!areAccessPatternsCompatibleForCombining(l3ToL2DmaOps[0],
                                                 l3ToL2DmaOps[i])) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "access patterns failed compatibility checks for combining\n");
      return failure();
    }
  }

  if (l2ToL1DmaOps.size() != l3ToL2DmaOps.size()) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "expected 1:1 correspondence between L3->L2 and L2->L1 Dma ops\n");
    return failure();
  }

  SmallVector<int64_t> combiningDims, nonCombiningDims;
  if (failed(fetchCombiningDimensions(l3ToL2DmaOps, combiningDims,
                                      nonCombiningDims))) {
    return failure();
  }

  // At this point it's nice to perhaps just sort the L3->L2 Dma ops based on
  // the "overlapping" offsets. And we'll sort the corresponding L2->L1 Dma ops
  // accordingly.
  for (int64_t i = 1, n = l3ToL2DmaOps.size(); i < n; i++) {
    DmaCpyNdOp currL3ToL2DmaOp = l3ToL2DmaOps[i];
    DmaCpyNdOp currL2ToL1DmaOp = l2ToL1DmaOps[i];
    int64_t j = i - 1;
    while (j >= 0 &&
           compareL3ToL2DmaPairOffsets(currL3ToL2DmaOp, l3ToL2DmaOps[j])) {
      l3ToL2DmaOps[j + 1] = l3ToL2DmaOps[j];
      l2ToL1DmaOps[j + 1] = l2ToL1DmaOps[j];
      j--;
    }
    l3ToL2DmaOps[j + 1] = currL3ToL2DmaOp;
    l2ToL1DmaOps[j + 1] = currL2ToL1DmaOp;
  }

  // Currently we have 4 cores so there are two pairs of DmaCpyNds to combine.
  // TODO(avarma): Revisit this later when we want to target more no. of cores.
  if (l3ToL2DmaOps.size() % 2 != 0) {
    LLVM_DEBUG(llvm::dbgs() << "found uneven L3->L2 ops for combining\n");
    return failure();
  }

  auto createL2ToL1ForReuse =
      [](IRRewriter &rewriter, DmaCpyNdOp &l2ToL1DmaOp,
         LogicalObjectFifoFromMemrefOp &reuseL1Buffer,
         LogicalObjectFifoFromMemrefOp &reuseL2Buffer,
         SmallVector<OpFoldResult> &newL2SourceOffsets) -> DmaCpyNdOp {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(l2ToL1DmaOp);
    auto newL2ToL1DmaOp = rewriter.create<AMDAIE::DmaCpyNdOp>(
        l2ToL1DmaOp.getLoc(), reuseL1Buffer,
        l2ToL1DmaOp.getTargetMixedOffsets(), l2ToL1DmaOp.getTargetMixedSizes(),
        l2ToL1DmaOp.getTargetMixedStrides(), reuseL2Buffer,
        llvm::ArrayRef(newL2SourceOffsets), l2ToL1DmaOp.getSourceMixedSizes(),
        l2ToL1DmaOp.getSourceMixedStrides());
    rewriter.replaceOp(l2ToL1DmaOp, newL2ToL1DmaOp);
    return newL2ToL1DmaOp;
  };
  // Till this point, we've created a L3->L2 DmaCpyNd chain that is sorted based
  // on increasing offsets. Refer to `compareL3ToL2DmaPairOffsets`'s doc comment
  // for the same. Now, we'll be picking up pairs of such DmaCpyNd ops from the
  // chain, eg: pair[0,1], then pair[2,3], etc. For each such pair[i, i+1] we
  // will make an attempt to combine the logical objectFifos as per the
  // following algorithm :-
  //  a. Combine i-th and i+1-th L3->L2 DmaCpyNd ops.
  //  b. Form reusable L1 buffer by assigning the cumulative tiles of the
  //     intended core ops.
  //  c. Since step a  would create a new L2 buffer (with combined shape), we
  //     will need to update the corresponding two L2->L1 Dma ops by indeed
  //     creating new ones. NOTE: Both of these new L2->L1 Dma ops will be
  //     reusing the same L1 buffers as well.
  //  d. Now pick the unique core ops corresponding to i-th and i+1-th L2->L1
  //     Dma ops and do the following :-
  //      1. For i-th CoreOp insert an AccessOp from the same L1 buffer towards
  //         the end.
  //      2. For i+1-th CoreOp insert an AccessOp from the same L1 buffer right
  //         before the corresponding AccessOp within the same CoreOp.
  for (unsigned i = 0, n = l3ToL2DmaOps.size(); i < n; i += 2) {
    // Step 1. Combine the picked L3->L2 DmaCpyNd pair.
    FailureOr<LogicalObjectFifoFromMemrefOp> maybeNewL2ObjectFifo =
        combineL3ToL2Pair(rewriter, l3ToL2DmaOps[i], l3ToL2DmaOps[i + 1],
                          combiningDims, nonCombiningDims);
    if (failed(maybeNewL2ObjectFifo)) return failure();
    LogicalObjectFifoFromMemrefOp newL2ObjectFifo =
        maybeNewL2ObjectFifo.value();

    LogicalObjectFifoFromMemrefOp oldFirstL1ObjFifoOp =
        l2ToL1DmaOps[i].getTargetObjectFifo();
    LogicalObjectFifoFromMemrefOp oldSecondL1ObjFifoOp =
        l2ToL1DmaOps[i + 1].getTargetObjectFifo();
    // Step 2. Form the reusable L1 buffer by assigning the cumulative tiles of
    // the intended core ops.
    LogicalObjectFifoFromMemrefOp reuseL1LogicalObjectFifoOp =
        l2ToL1DmaOps[i].getTargetObjectFifo();
    SmallVector<Value> tiles;
    auto addNewTileFrom = [&](CoreOp coreOp) -> LogicalResult {
      OpBuilder::InsertionGuard guard(rewriter);
      TileOp tileOp = coreOp.getTileOp();
      std::optional<int64_t> column = getConstantIntValue(tileOp.getCol());
      std::optional<int64_t> row = getConstantIntValue(tileOp.getRow());
      if (!column || !row) {
        return coreOp.emitOpError() << "has non-constant tile location";
      }
      rewriter.setInsertionPoint(reuseL1LogicalObjectFifoOp);
      auto colIndex = rewriter.create<arith::ConstantIndexOp>(
          rewriter.getUnknownLoc(), *column);
      auto rowIndex = rewriter.create<arith::ConstantIndexOp>(
          rewriter.getUnknownLoc(), *row);
      tileOp =
          rewriter.create<TileOp>(rewriter.getUnknownLoc(), colIndex, rowIndex);
      tiles.push_back(tileOp.getResult());
      return success();
    };
    std::optional<CoreOp> maybeFirstCoreOp = fetchUniqueCoreOp(l2ToL1DmaOps[i]);
    if (!maybeFirstCoreOp) return failure();
    CoreOp firstCoreOp = maybeFirstCoreOp.value();
    std::optional<CoreOp> maybeSecondCoreOp =
        fetchUniqueCoreOp(l2ToL1DmaOps[i + 1]);
    if (!maybeSecondCoreOp) return failure();
    CoreOp secondCoreOp = maybeSecondCoreOp.value();
    if (failed(addNewTileFrom(firstCoreOp)) ||
        failed(addNewTileFrom(secondCoreOp))) {
      return failure();
    }
    llvm::sort(tiles.begin(), tiles.end(),
               AMDAIE::TileOp::tileValueColumnAndRowComparator);
    rewriter.setInsertionPoint(reuseL1LogicalObjectFifoOp);
    reuseL1LogicalObjectFifoOp = rewriter.create<LogicalObjectFifoFromMemrefOp>(
        reuseL1LogicalObjectFifoOp.getLoc(),
        cast<LogicalObjectFifoType>(
            reuseL1LogicalObjectFifoOp.getOutput().getType()),
        reuseL1LogicalObjectFifoOp.getMemref(), tiles);

    // Step 3. We now have need to create two L2->L1 ops since the size has
    // changed. But for this we first need to find the new offset for L2 as
    // source.
    // TODO: For now I'm hardcoding the offsets but later it'd just depend
    // on combining/non-combining dimensions.
    // Offset = 0,0
    SmallVector<OpFoldResult> newL2AsSourceOffsets =
        l2ToL1DmaOps[i].getSourceMixedOffsets();
    createL2ToL1ForReuse(rewriter, l2ToL1DmaOps[i], reuseL1LogicalObjectFifoOp,
                         newL2ObjectFifo, newL2AsSourceOffsets);
    // Offset = 0, 1. NOTE here we'd use the same L1 logical objectFifo as
    // the first L2->L1 Dma.
    newL2AsSourceOffsets = l2ToL1DmaOps[i + 1].getSourceMixedOffsets();
    newL2AsSourceOffsets[1] = rewriter.getIndexAttr(1);
    createL2ToL1ForReuse(rewriter, l2ToL1DmaOps[i + 1],
                         reuseL1LogicalObjectFifoOp, newL2ObjectFifo,
                         newL2AsSourceOffsets);

    // Step 4. Pick the CoreOps associated with the 1:1 L2->L1.
    // For the first Core op we'll insert Read at the end. It doesn't matter
    // for now so we're gonna insert it right before amdaie.end op.
    firstCoreOp.walk([&](AMDAIE::LogicalObjectFifoAccessOp accessOp) {
      if (accessOp.getInput() == oldFirstL1ObjFifoOp) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfter(accessOp);
        rewriter.create<AMDAIE::LogicalObjectFifoAccessOp>(
            rewriter.getUnknownLoc(), reuseL1LogicalObjectFifoOp.getOutput(),
            accessOp.getAccessType());
      }
    });
    rewriter.replaceOp(oldFirstL1ObjFifoOp, reuseL1LogicalObjectFifoOp);
    // For the second Core op we'll insert `Read` right before the first read
    // from the corresponding L1 logicalobjectFifo.
    secondCoreOp.walk([&](AMDAIE::LogicalObjectFifoAccessOp accessOp) {
      if (accessOp.getInput() == oldSecondL1ObjFifoOp) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(accessOp);
        rewriter.create<AMDAIE::LogicalObjectFifoAccessOp>(
            rewriter.getUnknownLoc(), reuseL1LogicalObjectFifoOp.getOutput(),
            accessOp.getAccessType());
        // Need to insert the second one because THIS is what will actually
        // be used.
        auto secondAccessOp =
            rewriter.create<AMDAIE::LogicalObjectFifoAccessOp>(
                rewriter.getUnknownLoc(),
                reuseL1LogicalObjectFifoOp.getOutput(),
                accessOp.getAccessType());
        rewriter.replaceOp(accessOp, secondAccessOp);
      }
    });
  }

  return success();
}

}  // namespace mlir::iree_compiler::AMDAIE
