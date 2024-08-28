// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIEDmaUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-split-logical-objectfifos"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Utility to help fetch those input DmaCpyNd Ops which needs to be split.
/// TODO(avarma): Move this as a reusable component for other transformations ?
static SmallVector<AMDAIE::DmaCpyNdOp> fetchDmaCpyNdOpsToSplit(
    ModuleOp moduleOp) {
  SmallVector<AMDAIE::DmaCpyNdOp> l2ToL1DmaOps;
  // We are currently walking through CoreOps gathering 3rd Input DmaOp (if
  // applicable) from them.
  // TODO: We will generalize this later.
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

class AMDAIESplitLogicalObjectFifosPass
    : public impl::AMDAIESplitLogicalObjectFifosBase<
          AMDAIESplitLogicalObjectFifosPass> {
 public:
  using AMDAIESplitLogicalObjectFifosBase::AMDAIESplitLogicalObjectFifosBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }
  void runOnOperation() override;
};

void AMDAIESplitLogicalObjectFifosPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  SmallVector<AMDAIE::DmaCpyNdOp> l2ToL1DmaOps =
      fetchDmaCpyNdOpsToSplit(moduleOp);

  if (l2ToL1DmaOps.size() == 0) return;

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
    return signalPassFailure();
  }
  // We will now capture those dimensions where L2 memory was split. The way we
  // do this is by checking all L2->L1 DmaOps' source offset and marking those
  // dimensions which are not equal to at least one of the source offsets.
  DenseSet<unsigned> splitDimensionsSet;
  for (unsigned i = 1, n = l2ToL1DmaOps.size(); i < n; i++) {
    if (l2ToL1DmaOps[i].getSourceObjectFifo() != sourceObjectFifo) {
      l2ToL1DmaOps[i]->emitRemark() << "has different source objectfifo";
      sourceObjectFifo->emitRemark() << "is the expected source objectfifo";
      return signalPassFailure();
    }
    SmallVector<OpFoldResult> sourceOffsets =
        l2ToL1DmaOps[i].getSourceMixedOffsets();
    for (unsigned j = 0, m = baseSourceOffsets.size(); j < m; j++) {
      if (baseSourceOffsets[j] != sourceOffsets[j]) {
        splitDimensionsSet.insert(j);
      }
    }
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

  OpFoldResult zeroVal = getAsIndexOpFoldResult(context, 0);
  OpFoldResult oneVal = getAsIndexOpFoldResult(context, 1);
  // Traverse each L2->L1 DmaCpyNd op and split them.
  for (AMDAIE::DmaCpyNdOp l2ToL1DmaOp : l2ToL1DmaOps) {
    LogicalObjectFifoFromMemrefOp targetObjectFifo =
        l2ToL1DmaOp.getTargetObjectFifo();
    Value targetAllocOp = targetObjectFifo.getMemref();

    SmallVector<OpFoldResult, 6> staticL2AsSourceOffsets =
        l2ToL1DmaOp.getSourceMixedOffsets();
    SmallVector<OpFoldResult, 6> staticL2AsSourceSizes =
        l2ToL1DmaOp.getSourceMixedSizes();
    SmallVector<OpFoldResult, 6> staticL2AsSourceStrides =
        l2ToL1DmaOp.getSourceMixedStrides();
    SmallVector<OpFoldResult, 4> staticL2AsTargetOffsets =
        l3ToL2DmaOp.getTargetMixedOffsets();
    SmallVector<OpFoldResult, 4> staticL2AsTargetSizes =
        l3ToL2DmaOp.getTargetMixedSizes();
    SmallVector<OpFoldResult, 4> staticL2AsTargetStrides =
        l3ToL2DmaOp.getTargetMixedStrides();
    SmallVector<int64_t, 4> l2ShapeAsTarget = llvm::to_vector(
        cast<MemRefType>(
            l3ToL2DmaOp.getTargetObjectFifo().getMemref().getType())
            .getShape());
    // We traverse through the split dimensions we captured earlier and for each
    // such dimension we perform the following updates :-
    // 1. Maintain a map: DIM -> CONST_OFFSET_TO_ADD. This is done with the
    //    assumption that L3<->L2 is 4D and L2<->L1 is 6D. `DIM` here is split
    //    dimension + 2. `CONST_OFFSET_TO_ADD` is the constant we get by
    //    multiplying L2 as source's offset at split dimension with L2 as
    //    target's size at split dimension + 2. We are maintaining this to later
    //    update the extraction offset of L3 -> L2.
    //    split dimension + 2 is basically the dimension of L3's source offset
    //    that we need to modify.
    // 2. Update L2 as source/target offset => 0.
    // 3. Update L2 as source/target size => 1.
    // 4. Compute the shape of L2 buffer after split.
    DenseMap<int64_t, int64_t> dimToOffsetMapForL3AsSource;
    for (unsigned dim : splitDimensionsSet) {
      std::optional<int64_t> constantOffset =
          getConstantIntValue(staticL2AsSourceOffsets[dim]);
      if (!constantOffset) {
        l2ToL1DmaOp->emitRemark()
            << "found a non-constant value for source offset at dim " << dim;
        return signalPassFailure();
      }
      std::optional<int64_t> constantSize =
          getConstantIntValue(staticL2AsTargetSizes[dim + 2]);
      if (!constantSize) {
        l3ToL2DmaOp->emitRemark()
            << "found a non-constant value for target size at dim "
            << (dim + 2);
        return signalPassFailure();
      }
      dimToOffsetMapForL3AsSource.insert(
          {dim + 2, constantOffset.value() * constantSize.value()});
      staticL2AsSourceOffsets[dim] = zeroVal;
      staticL2AsSourceSizes[dim] = oneVal;
      staticL2AsTargetOffsets[dim] = zeroVal;
      staticL2AsTargetSizes[dim] = oneVal;
      l2ShapeAsTarget[dim] = 1;
    }

    // Now we'll create a narrowed linearized L2 buffer.
    rewriter.setInsertionPoint(sourceAllocOp);
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

    SmallVector<OpFoldResult, 4> staticL3AsSourceOffsets =
        l3ToL2DmaOp.getSourceMixedOffsets();
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

      The following logic performs this computation of offsets for L3 source.
    */
    for (auto [dim, offsetToAdd] : dimToOffsetMapForL3AsSource) {
      auto l3SourceOffsetVal = cast<Value>(staticL3AsSourceOffsets[dim]);
      Operation *defOpOfL3SourceOffset = l3SourceOffsetVal.getDefiningOp();
      if (!defOpOfL3SourceOffset) {
        // TODO: Handle this case better later.
        l3ToL2DmaOp->emitRemark()
            << "source offset at dim " << dim << " is a block argument";
        return signalPassFailure();
      }
      Location loc = defOpOfL3SourceOffset->getLoc();
      rewriter.setInsertionPoint(defOpOfL3SourceOffset);
      OpBuilder::InsertionGuard guard(rewriter);
      Value newL3AsSourceOffsetVal;
      if (auto applyOp =
              dyn_cast<affine::AffineApplyOp>(defOpOfL3SourceOffset)) {
        AffineExpr affineExpr = applyOp.getAffineMap().getResult(0);
        AffineExpr newAffineExpr = affineExpr + offsetToAdd;
        auto newAffineMap = AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                                           {newAffineExpr}, context);
        newL3AsSourceOffsetVal = rewriter.create<affine::AffineApplyOp>(
            loc, newAffineMap, applyOp.getMapOperands());
      } else if (auto constantOffset = getConstantIntValue(l3SourceOffsetVal)) {
        int64_t newOffset = *constantOffset + offsetToAdd;
        newL3AsSourceOffsetVal = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIndexAttr(newOffset));
      } else {
        // TODO: Ideally we should be able to handle even +, -, *, /, etc.
        //       But handle this later (if at all!) as such cases aren't going
        //       to arise.
        l3ToL2DmaOp->emitRemark()
            << "Unhandled expression for source offset at dim " << dim;
        return signalPassFailure();
      }
      staticL3AsSourceOffsets[dim] = newL3AsSourceOffsetVal;
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
}

}  // namespace

std::unique_ptr<Pass> createAMDAIESplitLogicalObjectFifosPass() {
  return std::make_unique<AMDAIESplitLogicalObjectFifosPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
