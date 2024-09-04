// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AMDAIELogicalObjFifoSplittingUtils.h"

#include <numeric>

#include "iree-amd-aie/Transforms/AMDAIEDmaUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/Operation.h"

///////////////////////////////////////////////////
#include "iree-amd-aie/Transforms/AMDAIEDmaUtils.h"
#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
///////////////////////////////////////////////////

#define DEBUG_TYPE "iree-amdaie-logicalobjfifo-splitting-utils"

namespace mlir::iree_compiler::AMDAIE {

/// Utility to create a new logical objectfifo based on shape defined by
/// `newSizesOpFoldResultArr`.
static AMDAIE::LogicalObjectFifoFromMemrefOp createNewLogicalObjectFifo(
    IRRewriter &rewriter,
    AMDAIE::LogicalObjectFifoFromMemrefOp &oldLogicalObjectFifo,
    SmallVector<OpFoldResult> &newSizesOpFoldResultArr) {
  OpBuilder::InsertionGuard guard(rewriter);
  SmallVector<int64_t> newSizes;
  for (OpFoldResult sizeVal : newSizesOpFoldResultArr) {
    newSizes.push_back(getConstantIndexOrAssert(sizeVal));
  }
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
    ModuleOp moduleOp) {
  SmallVector<AMDAIE::DmaCpyNdOp> l2ToL1DmaOps;
  // We are currently walking through CoreOps gathering 3rd Input DmaOp (if
  // applicable) from them.
  // TODO(avarma): We will generalize this later.
  moduleOp.walk([&](AMDAIE::CoreOp coreOp) {
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

  SmallVector<OpFoldResult> staticL2AsTargetOffsets =
      l3ToL2DmaOp.getTargetMixedOffsets();
  SmallVector<OpFoldResult> staticL2AsTargetSizes =
      l3ToL2DmaOp.getTargetMixedSizes();
  SmallVector<OpFoldResult> staticL3AsSourceOffsets =
      l3ToL2DmaOp.getSourceMixedOffsets();
  SmallVector<OpFoldResult> staticL3AsSourceSizes =
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
    SmallVector<OpFoldResult> staticL2AsSourceOffsets =
        l2ToL1DmaOp.getSourceMixedOffsets();
    SmallVector<OpFoldResult> staticL2AsSourceSizes =
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

static LogicalResult _TODOcombineAccessPatterns(
    RewriterBase &rewriter, const SmallVector<OpFoldResult> &offsetsA,
    const SmallVector<OpFoldResult> &sizesA,
    const SmallVector<OpFoldResult> &stridesA,
    const SmallVector<OpFoldResult> &offsetsB,
    const SmallVector<OpFoldResult> &sizesB,
    const SmallVector<OpFoldResult> &stridesB,
    SmallVector<OpFoldResult> &newOffsets, SmallVector<OpFoldResult> &newSizes,
    SmallVector<OpFoldResult> &newStrides) {
  // TODO: Move these checks later in a separate func.
  assert(offsetsA.size() == offsetsB.size() &&
         "expected same number of source offsets and target offsets");
  assert(offsetsA.size() == sizesA.size() &&
         "expected same number of source offsets and sizes");
  assert(offsetsA.size() == stridesA.size() &&
         "expected same number of source offsets and strides");
  assert(offsetsB.size() == sizesB.size() &&
         "expected same number of target offsets and sizes");
  assert(offsetsB.size() == stridesB.size() &&
         "expected same number of target offsets and strides");

  if (offsetsA.empty() && offsetsB.empty()) return success();

  for (auto iter : llvm::enumerate(llvm::zip(offsetsA, offsetsB))) {
    const OpFoldResult &offsetA = std::get<0>(iter.value());
    const OpFoldResult &offsetB = std::get<1>(iter.value());
    if (offsetA != offsetB) {
      // Need to check the difference in bias here.
    }
  }
  newSizes[1] = rewriter.getI64IntegerAttr(2);
  return success();
}

/// Utility to fetch a unique CoreOp associated with a L2->L1 Dma op.
static CoreOp fetchUniqueCoreOp(DmaCpyNdOp &l2ToL1DmaOp) {
  SmallVector<CoreOp> coreOps;
  for (Operation *userOp : l2ToL1DmaOp->getUsers()) {
    if (auto coreOp = dyn_cast<CoreOp>(userOp)) {
      coreOps.push_back(coreOp);
    }
  }
  assert(coreOps.size() == 1 &&
         "L2->L1 Dma op expected to have a unique Core op");
  return coreOps[0];
}

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
  }

  if (l2ToL1DmaOps.size() != l3ToL2DmaOps.size()) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "expected 1:1 correspondence between L3->L2 and L2->L1 Dma ops\n");
    return failure();
  }

  // For now pick the first two L3->L2 Dma op and try to combine them. Later
  // we'll implement the selector.
  ////////////////////////////////////////////////
  ////////////// PICK logic TODO /////////////////
  ////////////////////////////////////////////////
  auto op = l3ToL2DmaOps[0];
  auto nextStridedOp = l3ToL2DmaOps[1];
  ////////////////////////////////////////////////
  /////// COMBINE the picked L3->L2 pair /////////
  ////////////////////////////////////////////////
  {
    /// The maximum number of addressing dimensions on the source side of the
    /// DMA.
    // int64_t sourceMaxNbDims{0};
    // /// The maximum number of addressing dimensions on the target side of the
    // DMA.
    // int64_t targetMaxNbDims{0};
    OpBuilder::InsertionGuard guard(rewriter);
    // rewriter.setInsertionPoint(op);
    SmallVector<OpFoldResult> sourceOffsetsA = op.getSourceMixedOffsets();
    SmallVector<OpFoldResult> sourceSizesA = op.getSourceMixedSizes();
    SmallVector<OpFoldResult> sourceStridesA = op.getSourceMixedStrides();
    SmallVector<OpFoldResult> sourceOffsetsB =
        nextStridedOp.getSourceMixedOffsets();
    SmallVector<OpFoldResult> sourceSizesB =
        nextStridedOp.getSourceMixedSizes();
    SmallVector<OpFoldResult> sourceStridesB =
        nextStridedOp.getSourceMixedStrides();
    bool areSourcesCombinable = true;

    SmallVector<OpFoldResult> targetOffsetsA = op.getTargetMixedOffsets();
    SmallVector<OpFoldResult> targetSizesA = op.getTargetMixedSizes();
    SmallVector<OpFoldResult> targetStridesA = op.getTargetMixedStrides();
    SmallVector<OpFoldResult> targetOffsetsB =
        nextStridedOp.getTargetMixedOffsets();
    SmallVector<OpFoldResult> targetSizesB =
        nextStridedOp.getTargetMixedSizes();
    SmallVector<OpFoldResult> targetStridesB =
        nextStridedOp.getTargetMixedStrides();
    bool areTargetsCombinable = true;

    if (areSourcesCombinable && areTargetsCombinable) {
      SmallVector<OpFoldResult> newSourceOffsets = sourceOffsetsA;
      SmallVector<OpFoldResult> newSourceSizes = sourceSizesA;
      SmallVector<OpFoldResult> newSourceStrides = sourceStridesA;
      if (failed(_TODOcombineAccessPatterns(
              rewriter, sourceOffsetsA, sourceSizesA, sourceStridesA,
              sourceOffsetsB, sourceSizesB, sourceStridesB, newSourceOffsets,
              newSourceSizes, newSourceStrides))) {
        return failure();
      }
      llvm::outs() << "Combined sources\n";
      llvm::outs().flush();

      SmallVector<OpFoldResult> newTargetOffsets = targetOffsetsA;
      SmallVector<OpFoldResult> newTargetSizes = targetSizesA;
      SmallVector<OpFoldResult> newTargetStrides = targetStridesA;
      if (failed(_TODOcombineAccessPatterns(
              rewriter, targetOffsetsA, targetSizesA, targetStridesA,
              targetOffsetsB, targetSizesB, targetStridesB, newTargetOffsets,
              newTargetSizes, newTargetStrides))) {
        return failure();
      }
      llvm::outs() << "Combined target\n";
      llvm::outs().flush();
      // Now we need to create a new L2 buffer based on `newTargetSizes`.
      LogicalObjectFifoFromMemrefOp oldL2ObjectFifo = op.getTargetObjectFifo();
      AMDAIE::LogicalObjectFifoFromMemrefOp newL2ObjectFifo =
          createNewLogicalObjectFifo(rewriter, oldL2ObjectFifo, newTargetSizes);

      // Create combined L3->L2 Dma.
      rewriter.setInsertionPoint(op);
      auto combinedL3ToL2DmaOp = rewriter.create<AMDAIE::DmaCpyNdOp>(
          op.getLoc(), newL2ObjectFifo, llvm::ArrayRef(newTargetOffsets),
          llvm::ArrayRef(newTargetSizes), llvm::ArrayRef(newTargetStrides),
          op.getSource(), llvm::ArrayRef(newSourceOffsets),
          llvm::ArrayRef(newSourceSizes), llvm::ArrayRef(newSourceStrides));
      // Replace the uses of 2nd L3->L2 Dma with the new combined L3->L2 Dma and
      // erase the 1st L3->L2 Dma.
      rewriter.replaceOp(nextStridedOp, combinedL3ToL2DmaOp);
      rewriter.eraseOp(op);

      // We now have need to create two L2->L1 ops since the size has changed.
      // But for this we first need to find the new offset for L2 as source.
      // TODO: For now I'm hardcoding the offsets but later it'd just depend on
      //       split/non-split dimensions.
      // Offset = 0,0
      auto firstL2ToL1DmaOp = l2ToL1DmaOps[0];
      rewriter.setInsertionPoint(firstL2ToL1DmaOp);
      LogicalObjectFifoFromMemrefOp reuseL1LogicalObjectFifoOp =
          firstL2ToL1DmaOp.getTargetObjectFifo();
      SmallVector<OpFoldResult> newL2AsSourceOffsets =
          firstL2ToL1DmaOp.getSourceMixedOffsets();
      auto newFirstL2ToL1DmaOp = rewriter.create<AMDAIE::DmaCpyNdOp>(
          firstL2ToL1DmaOp.getLoc(), reuseL1LogicalObjectFifoOp,
          firstL2ToL1DmaOp.getTargetMixedOffsets(),
          firstL2ToL1DmaOp.getTargetMixedSizes(),
          firstL2ToL1DmaOp.getTargetMixedStrides(), newL2ObjectFifo,
          llvm::ArrayRef(newL2AsSourceOffsets),
          firstL2ToL1DmaOp.getSourceMixedSizes(),
          firstL2ToL1DmaOp.getSourceMixedStrides());
      rewriter.replaceOp(firstL2ToL1DmaOp, newFirstL2ToL1DmaOp);
      // Offset = 0, 1. NOTE here we'd use the same L1 logical objectFifo as the
      // first L2->L1 Dma.
      auto secondL2ToL1DmaOp = l2ToL1DmaOps[1];
      rewriter.setInsertionPoint(secondL2ToL1DmaOp);
      newL2AsSourceOffsets = secondL2ToL1DmaOp.getSourceMixedOffsets();
      newL2AsSourceOffsets[1] = rewriter.getIndexAttr(1);
      auto newSecondL2ToL1DmaOp = rewriter.create<AMDAIE::DmaCpyNdOp>(
          secondL2ToL1DmaOp.getLoc(), reuseL1LogicalObjectFifoOp,
          secondL2ToL1DmaOp.getTargetMixedOffsets(),
          secondL2ToL1DmaOp.getTargetMixedSizes(),
          secondL2ToL1DmaOp.getTargetMixedStrides(), newL2ObjectFifo,
          llvm::ArrayRef(newL2AsSourceOffsets),
          secondL2ToL1DmaOp.getSourceMixedSizes(),
          secondL2ToL1DmaOp.getSourceMixedStrides());
      rewriter.replaceOp(secondL2ToL1DmaOp, newSecondL2ToL1DmaOp);

      /////////////////////////////////////////////////////////
      //// PICK the CoreOps associated with the 1:1 L2->L1 ////
      /////////////////////////////////////////////////////////
      // For the first Core op we'll insert Read at the end. It doesn't matter
      // for now so we're gonna insert it right before amdaie.end op.
      CoreOp firstCoreOp = fetchUniqueCoreOp(newFirstL2ToL1DmaOp);
      firstCoreOp.walk([&](AMDAIE::EndOp endOp) {
        OpBuilder::InsertionGuard guard(rewriter);
        // Hardcoding to `AMDAIE::MemoryAccess::Read`.
        rewriter.setInsertionPoint(endOp);
        rewriter.create<AMDAIE::LogicalObjectFifoAccessOp>(
            rewriter.getUnknownLoc(), reuseL1LogicalObjectFifoOp.getOutput(),
            AMDAIE::MemoryAccess::Read);
      });
      // For the seconf Core op we'll insert Read right before the first read
      // from the corresponding L1 logicalobjectFifo.
      CoreOp secondCoreOp = fetchUniqueCoreOp(newSecondL2ToL1DmaOp);
      secondCoreOp.walk([&](AMDAIE::LogicalObjectFifoAccessOp accessOp) {
        if (accessOp.getInput() == l2ToL1DmaOps[1].getTargetObjectFifo()) {
          OpBuilder::InsertionGuard guard(rewriter);
          // Hardcoding to `AMDAIE::MemoryAccess::Read`.
          rewriter.setInsertionPoint(accessOp);
          rewriter.create<AMDAIE::LogicalObjectFifoAccessOp>(
              rewriter.getUnknownLoc(), reuseL1LogicalObjectFifoOp.getOutput(),
              AMDAIE::MemoryAccess::Read);
          // Need to insert the second one because THIS is what will actually be
          // used.
          auto secondAccessOp =
              rewriter.create<AMDAIE::LogicalObjectFifoAccessOp>(
                  rewriter.getUnknownLoc(),
                  reuseL1LogicalObjectFifoOp.getOutput(),
                  AMDAIE::MemoryAccess::Read);
          rewriter.replaceOp(accessOp, secondAccessOp);
        }
      });
    }
    // llvm::outs() << "NOT Compatible\n";
    // llvm::outs().flush();
  }

  return success();
}

}  // namespace mlir::iree_compiler::AMDAIE
