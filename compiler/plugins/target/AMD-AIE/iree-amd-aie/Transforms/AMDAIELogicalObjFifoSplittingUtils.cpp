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
    const SmallVectorImpl<int64_t> &newSizes) {
  OpBuilder::InsertionGuard guard(rewriter);
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

static AMDAIE::LogicalObjectFifoFromMemrefOp createNewLogicalObjectFifo(
    IRRewriter &rewriter,
    AMDAIE::LogicalObjectFifoFromMemrefOp &oldLogicalObjectFifo,
    const SmallVectorImpl<OpFoldResult> &newSizesOpFoldResultArr) {
  OpBuilder::InsertionGuard guard(rewriter);
  SmallVector<int64_t> newSizes = llvm::map_to_vector(
      newSizesOpFoldResultArr,
      [](OpFoldResult sizeVal) { return getConstantIndexOrAssert(sizeVal); });
  return createNewLogicalObjectFifo(rewriter, oldLogicalObjectFifo, newSizes);
}

/// Utility to help fetch those input DmaCpyNd Ops which needs to be split.
SmallVector<AMDAIE::DmaCpyNdOp> fetchDmaCpyNdOpsToSplitOrCombine(
    Operation *op) {
  SmallVector<AMDAIE::DmaCpyNdOp> l2ToL1DmaOps;
  // We are currently walking through CoreOps gathering 3rd Input DmaOp (if
  // applicable) from them.
  // TODO(avarma): We will generalize this later.
  op->walk([&](AMDAIE::CoreOp coreOp) {
    // SmallVector<Value> inputDmas = coreOp.getInputDmas();
    // if (inputDmas.size() != 3) return WalkResult::skip();
    // auto dmaCpyNdOp = inputDmas[2].getDefiningOp<AMDAIE::DmaCpyNdOp>();
    // assert(dmaCpyNdOp && "expected an amdaie.dma_cpy_nd op");
    // l2ToL1DmaOps.push_back(dmaCpyNdOp);
    SmallVector<Value> outputDmas = coreOp.getOutputDmas();
    if (outputDmas.size() != 1) return WalkResult::skip();
    auto dmaCpyNdOp = outputDmas[2].getDefiningOp<AMDAIE::DmaCpyNdOp>();
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
static FailureOr<OpFoldResult> addToOffset(IRRewriter &rewriter,
                                           OpFoldResult oldL3Offset,
                                           int64_t offsetToAdd) {
  auto createAffineMap = [&](AffineExpr affineExpr,
                             int64_t offsetToAdd) -> AffineMap {
    AffineExpr newAffineExpr = affineExpr + offsetToAdd;
    return AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, {newAffineExpr},
                          rewriter.getContext());
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

  LogicalObjectFifoFromMemrefOp l2TargetObjectFifo =
      l3ToL2DmaOp.getTargetObjectFifo();
  ArrayRef<int64_t> l2TargetShape =
      l2TargetObjectFifo.getMemrefType().getShape();
  if (l2TargetShape.size() != staticL2AsTargetSizes.size()) {
    LLVM_DEBUG(llvm::dbgs() << "L2 target size should be the same");
    return failure();
  }

  // Check if the L3->L2 dma is transposed on the target side.
  bool dmaTransposeOnSource = true;
  for (auto [s1, s2] : llvm::zip_equal(l2TargetShape, staticL2AsTargetSizes)) {
    if (s1 != getConstantIntValue(s2)) {
      dmaTransposeOnSource = false;
      break;
    }
  }
  if (staticL3AsSourceSizes.size() != staticL2AsTargetSizes.size()) {
    dmaTransposeOnSource = false;
  }

  OpFoldResult zeroVal = getAsIndexOpFoldResult(context, 0);
  OpFoldResult oneVal = getAsIndexOpFoldResult(context, 1);

  if (dmaTransposeOnSource) {
    // Update split dimensions' offset/size for L2 as target and L3 as source.
    // We can afford to do this here because it's going to be the same for all
    // L3->L2 splits. Here we are setting offset = 0 and size = 1.
    for (size_t dim : splitDimsForL2) {
      staticL2AsTargetOffsets[dim] = zeroVal;
      staticL2AsTargetSizes[dim] = oneVal;
      staticL3AsSourceOffsets[dim] = zeroVal;
      staticL3AsSourceSizes[dim] = oneVal;
    }
  } else {
    // The L2 target side has transposed dimensions, while the L3 source side
    // data are continuous and don't have `nonSplitDim`. Then the L3 source
    // sizes need to be modified to match the new L2 target sizes.
    // Hardcoded the transposed dimensions for now.
    const SmallVector<size_t> transposeDim = {0, 2, 1, 3};
    for (auto &&[splitDim, nonSplitdim] :
         llvm::zip_equal(splitDimsForL2, nonSplitDimsForL2)) {
      staticL2AsTargetOffsets[transposeDim[splitDim]] = zeroVal;
      staticL2AsTargetSizes[transposeDim[splitDim]] = oneVal;
      staticL3AsSourceSizes[splitDim] =
          staticL2AsTargetSizes[transposeDim[nonSplitdim]];
    }
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
    // If the dma transpose is on the source(target) side, then the L2
    // target(source) side has the sizes in order.
    SmallVector<OpFoldResult> newL2Sizes =
        dmaTransposeOnSource ? staticL2AsTargetSizes : staticL2AsSourceSizes;
    AMDAIE::LogicalObjectFifoFromMemrefOp source =
        createNewLogicalObjectFifo(rewriter, oldL2ObjectFifo, newL2Sizes);

    // --------------------------------------------
    // ---------- L3 -> L2 splitting --------------
    // --------------------------------------------
    // Update L3 source offsets for non-split dimensions. Refer doc comment of
    // `addToOffset` for the computation rationale involved.
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
          getConstantIntValue(newL2Sizes[nonSplitdim]);
      if (!constantSize) {
        return l3ToL2DmaOp->emitOpError()
               << "found a non-constant value for target size at dim "
               << nonSplitdim;
      }
      int64_t offsetToAdd = constantOffset.value() * constantSize.value();

      // If the dma transpose is on the target side, L3 source side data are
      // continuous and don't have `nonSplitDim`.
      size_t dim = dmaTransposeOnSource ? nonSplitdim : splitDim;
      FailureOr<OpFoldResult> newOffset = addToOffset(
          rewriter, staticL3AsSourceOffsets[dim], offsetToAdd);
      if (failed(newOffset)) {
        // TODO: Ideally we should be able to handle even +, -, *, /, etc.
        //       But handle this later (if at all!) as such cases might not
        //       arise.
        return l3ToL2DmaOp->emitOpError()
               << "Unhandled expression for source offset at dim "
               << nonSplitdim;
      }
      staticL3AsSourceOffsets[dim] = *newOffset;
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

template <typename T>
LogicalResult getDoublyStridedCopyOpProducersAndConsumers(
    AMDAIE::LogicalObjectFifoFromMemrefOp op, SmallVector<T> &producers,
    SmallVector<T> &consumers) {
  for (Operation *userOp : op->getUsers()) {
    if (auto stridedCopyOp = dyn_cast<T>(userOp)) {
      if (dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromMemrefOp>(
              stridedCopyOp.getTarget().getDefiningOp()) == op) {
        producers.push_back(stridedCopyOp);
      } else if (dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromMemrefOp>(
                     stridedCopyOp.getSource().getDefiningOp()) == op) {
        consumers.push_back(stridedCopyOp);
      } else {
        return op.emitOpError()
               << "has non-consumer, non-producer doubly strided copy op user";
      }
    } else {
      return op.emitOpError() << "has non-doubly strided copy op user";
    }
  }
  return success();
}

LogicalResult splitObjFifo(IRRewriter &rewriter,
                           AMDAIE::LogicalObjectFifoFromMemrefOp op,
                           size_t splitDim, int64_t splitFactor) {
  llvm::outs() << "splitObjFifo\n";
  assert(splitDim == 0 &&
         "splitting of DMAs only supported on the outermost dimension");
  MemRefType type = op.getMemrefType();
  SmallVector<int64_t> shape = llvm::to_vector(type.getShape());
  assert(splitDim < shape.size() &&
         "the dimension to be split on should be smaller than the number of "
         "dimensions in the shape");
  int64_t splitSize = shape[splitDim];
  if (ShapedType::isDynamic(splitSize)) {
    return op.emitOpError()
           << "a dynamic size on the split dimension is not supported";
  }
  // TODO(jornt): use DoublyStridedCopyOpInterface
  SmallVector<AMDAIE::DmaCpyNdOp> producers;
  SmallVector<AMDAIE::DmaCpyNdOp> consumers;
  if (failed(getDoublyStridedCopyOpProducersAndConsumers(op, producers,
                                                         consumers))) {
    return failure();
  }
  if (splitFactor == -1) {
    splitFactor = splitSize;  // std::max(producers.size(), consumers.size());
  } else if (splitSize % splitFactor != 0) {
    return op.emitOpError()
           << "the size of the dimension to be split: " << splitSize
           << " is not divisible by the provided splitting factor: "
           << splitFactor;
  }
  shape[splitDim] = splitSize / splitFactor;
  int64_t splitOffset = std::accumulate(shape.begin() + splitDim + 1,
                                        shape.end(), 1, std::multiplies<>());

  llvm::outs() << "splitOffset: " << splitOffset << "\n";
  SmallVector<AMDAIE::LogicalObjectFifoFromMemrefOp> newObjFifos;
  // SmallVector<SmallPtrSet<Operation *, 16>> newObjFifoStridedOps;
  newObjFifos.reserve(splitFactor);
  // newObjFifos.reserve(splitFactor);
  for (int i = 0; i < splitFactor; i++) {
    newObjFifos.push_back(createNewLogicalObjectFifo(rewriter, op, shape));
  }
  for (AMDAIE::DmaCpyNdOp producer : producers) {
    SmallVector<OpFoldResult> targetOffsets = producer.getTargetMixedOffsets();
    SmallVector<OpFoldResult> targetSizes = producer.getTargetMixedSizes();
    SmallVector<OpFoldResult> targetStrides = producer.getTargetMixedStrides();
    SmallVector<int64_t> offsetIndices;
    for (auto [idx, stride] : llvm::enumerate(targetStrides)) {
      if (std::optional<int64_t> strideVal = getConstantIntValue(stride);
          strideVal && strideVal.value() == splitOffset) {
        offsetIndices.push_back(idx);
      }
    }
    if (offsetIndices.size() != 1) {
      return producer.emitOpError()
             << "expected exactly one target stride equal to the slit offset";
    }
    int64_t offsetIdx = offsetIndices[0];
    std::optional<int64_t> targetSize =
        getConstantIntValue(targetSizes[offsetIdx]);
    std::optional<int64_t> targetOffset =
        getConstantIntValue(targetOffsets[offsetIdx]);
    if (!targetSize || !targetOffset) {
      return producer.emitOpError()
             << "expected a static target offset and size on index: "
             << offsetIdx;
    }
    if (targetSize.value() != 1) {
      return producer.emitOpError() << "only a static size of 1 is currently "
                                       "supported on the split index";
    }
    // TODO(jornt): support splitFactor != splitSize.
    assert(targetOffset < newObjFifos.size());
    AMDAIE::LogicalObjectFifoFromMemrefOp newObjFifo =
        newObjFifos[targetOffset.value()];
    targetOffsets[offsetIdx] = rewriter.getIndexAttr(0);
    rewriter.setInsertionPoint(producer);
    auto newDmaOp = rewriter.create<AMDAIE::DmaCpyNdOp>(
        producer.getLoc(), newObjFifo, targetOffsets, targetSizes,
        targetStrides, producer.getSource(), producer.getSourceMixedOffsets(),
        producer.getSourceMixedSizes(), producer.getSourceMixedStrides());
    rewriter.replaceOp(producer, newDmaOp);
  }
  for (AMDAIE::DmaCpyNdOp consumer : consumers) {
    SmallVector<OpFoldResult> sourceOffsets = consumer.getSourceMixedOffsets();
    SmallVector<OpFoldResult> sourceSizes = consumer.getSourceMixedSizes();
    SmallVector<OpFoldResult> sourceStrides = consumer.getSourceMixedStrides();
    SmallVector<int64_t> offsetIndices;
    for (auto [idx, stride] : llvm::enumerate(sourceStrides)) {
      if (std::optional<int64_t> strideVal = getConstantIntValue(stride);
          strideVal && strideVal.value() == splitOffset) {
        offsetIndices.push_back(idx);
      }
    }
    if (offsetIndices.size() != 1) {
      return consumer.emitOpError()
             << "expected exactly one target stride equal to the slit offset";
    }
    int64_t offsetIdx = offsetIndices[0];
    std::optional<int64_t> sourceSize =
        getConstantIntValue(sourceSizes[offsetIdx]);
    std::optional<int64_t> sourceOffset =
        getConstantIntValue(sourceOffsets[offsetIdx]);
    if (!sourceSize || !sourceOffset) {
      return consumer.emitOpError()
             << "expected a static source offset and size on index: "
             << offsetIdx;
    }
    if (sourceSize.value() != 1) {
      return consumer.emitOpError() << "only a static size of 1 is currently "
                                       "supported on the split index";
    }
    // TODO(jornt): support splitFactor != splitSize.
    assert(sourceOffset < newObjFifos.size());
    AMDAIE::LogicalObjectFifoFromMemrefOp newObjFifo =
        newObjFifos[sourceOffset.value()];
    sourceOffsets[offsetIdx] = rewriter.getIndexAttr(0);
    rewriter.setInsertionPoint(consumer);
    auto newDmaOp = rewriter.create<AMDAIE::DmaCpyNdOp>(
        consumer.getLoc(), consumer.getTarget(),
        consumer.getTargetMixedOffsets(), consumer.getTargetMixedSizes(),
        consumer.getTargetMixedStrides(), newObjFifo, sourceOffsets,
        sourceSizes, sourceStrides);
    rewriter.replaceOp(consumer, newDmaOp);
  }

  return success();
}

LogicalResult splitDoublyStridedOp(IRRewriter &rewriter,
                                   AMDAIE::DoublyStridedOpInterface op,
                                   size_t splitDim, int64_t splitFactor) {
  assert(splitDim == 0 &&
         "splitting of DMAs only supported on the outermost dimension");
  if (!op->use_empty())
    return op.emitOpError() << "can't be split because it has uses";
  SmallVector<OpFoldResult> sourceOffsets = op.getSourceMixedOffsets();
  SmallVector<OpFoldResult> sourceSizes = op.getSourceMixedSizes();
  SmallVector<OpFoldResult> sourceStrides = op.getSourceMixedStrides();
  SmallVector<OpFoldResult> targetOffsets = op.getTargetMixedOffsets();
  SmallVector<OpFoldResult> targetSizes = op.getTargetMixedSizes();
  SmallVector<OpFoldResult> targetStrides = op.getTargetMixedStrides();
  assert(splitDim < sourceOffsets.size() &&
         "the dimension to be split on should be smaller than the number of "
         "source dimensions");
  assert(splitDim < targetOffsets.size() &&
         "the dimension to be split on should be smaller than the number of "
         "target dimensions");
  std::optional<int64_t> sourceSize =
      getConstantIntValue(sourceSizes[splitDim]);
  std::optional<int64_t> targetSize =
      getConstantIntValue(targetSizes[splitDim]);
  if (!sourceSize) {
    return op.emitOpError()
           << "does not have a static source size on dim: " << splitDim;
  }
  if (!targetSize) {
    return op.emitOpError()
           << "does not have a static target size on dim: " << splitDim;
  }
  if (splitFactor <= 0) {
    splitFactor = std::gcd(sourceSize.value(), targetSize.value());
  } else if (sourceSize.value() % splitFactor != 0 ||
             targetSize.value() % splitFactor != 0) {
    return op.emitOpError() << "the target or source size is not divisible by "
                               "the provided splitting factor: "
                            << splitFactor;
  }
  int64_t newSourceSize = sourceSize.value() / splitFactor;
  int64_t newTargetSize = targetSize.value() / splitFactor;
  sourceSizes[splitDim] = rewriter.getIndexAttr(newSourceSize);
  targetSizes[splitDim] = rewriter.getIndexAttr(newTargetSize);
  rewriter.setInsertionPoint(op);
  for (int i = 0; i < splitFactor; ++i) {
    FailureOr<OpFoldResult> newSourceOffset =
        addToOffset(rewriter, sourceOffsets[splitDim], i * newSourceSize);
    FailureOr<OpFoldResult> newTargetOffset =
        addToOffset(rewriter, targetOffsets[splitDim], i * newTargetSize);
    if (failed(newSourceOffset))
      return op.emitOpError() << "could not create a new source offset";
    if (failed(newTargetOffset))
      return op.emitOpError() << "could not create a new target offset";
    sourceOffsets[splitDim] = newSourceOffset.value();
    targetOffsets[splitDim] = newTargetOffset.value();
    op.createDoublyStridedOp(rewriter, targetOffsets, targetSizes,
                             targetStrides, sourceOffsets, sourceSizes,
                             sourceStrides);
  }
  rewriter.eraseOp(op);
  return success();
}

}  // namespace mlir::iree_compiler::AMDAIE
