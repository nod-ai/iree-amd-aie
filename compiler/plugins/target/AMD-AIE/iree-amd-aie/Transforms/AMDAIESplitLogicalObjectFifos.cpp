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

  SmallVector<AMDAIE::DmaCpyNdOp> l2ToL1DmaOps;
  // We are currently walking through CoreOps gathering 3rd Input DmaOp (if
  // applicable) from them.
  // TODO: We will generalize this later.
  moduleOp.walk([&](AMDAIE::CoreOp coreOp) {
    SmallVector<Value> inputDmas = coreOp.getInputDmas();
    if (inputDmas.size() < 3) return WalkResult::skip();
    l2ToL1DmaOps.push_back(inputDmas[2].getDefiningOp<AMDAIE::DmaCpyNdOp>());
    return WalkResult::advance();
  });

  if (l2ToL1DmaOps.size() == 0) return;

  SmallVector<Value> baseSourceOffsets = l2ToL1DmaOps[0].getSourceOffsets();
  DenseSet<unsigned> splitDimensionsSet;
  for (unsigned i = 1, n = l2ToL1DmaOps.size(); i < n; i++) {
    SmallVector<Value> sourceOffsets = l2ToL1DmaOps[i].getSourceOffsets();
    for (unsigned j = 0, m = baseSourceOffsets.size(); j < m; j++) {
      if (baseSourceOffsets[j] != sourceOffsets[j]) {
        splitDimensionsSet.insert(j);
      }
    }
  }

  OpFoldResult zeroVal = getAsIndexOpFoldResult(context, 0);
  OpFoldResult oneVal = getAsIndexOpFoldResult(context, 1);
  DenseSet<Operation *> toBeErased;
  for (AMDAIE::DmaCpyNdOp l2ToL1DmaOp : l2ToL1DmaOps) {
    LogicalObjectFifoFromMemrefOp sourceObjectFifo =
        l2ToL1DmaOp.getSourceObjectFifo();
    auto sourceAllocOp =
        sourceObjectFifo.getMemref().getDefiningOp<memref::AllocOp>();
    uint64_t sourceMemrefSpace = sourceObjectFifo.getMemorySpaceAsUInt();
    if (!sourceAllocOp || sourceMemrefSpace != 1) continue;
    LogicalObjectFifoFromMemrefOp targetObjectFifo =
        l2ToL1DmaOp.getTargetObjectFifo();
    Value targetAllocOp = targetObjectFifo.getMemref();

    // Fetch the L3 -> L2 Dma Op corresponding to the L2 buffer as target.
    AMDAIE::DmaCpyNdOp l3ToL2DmaOp;
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
    llvm::outs() << "FOR " << l2ToL1DmaOp << "\n";
    llvm::outs().flush();
    // SmallVector<int64_t> splitDimensionConstants;
    DenseMap<int64_t, int64_t> dimToOffsetMapForL3AsSource;
    for (unsigned dim : splitDimensionsSet) {
      // llvm::outs()<<"\tdim :
      // "<<dim<<"\n\t\t"<<staticL2AsSourceOffsets[dim]<<"\n";
      // llvm::outs().flush();
      std::optional<int64_t> constantOffset =
          getConstantIntValue(staticL2AsSourceOffsets[dim]);
      if (!constantOffset) {
        l2ToL1DmaOp->emitOpError("found a non-constant value for offset");
        return signalPassFailure();
      }
      dimToOffsetMapForL3AsSource.insert(
          {dim + 2,
           constantOffset.value() *
               (getConstantIntValue(staticL2AsTargetSizes[dim + 2]).value())});
      // splitDimensions.push_back(dim);
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
    for (auto [dim, offsetToAdd] : dimToOffsetMapForL3AsSource) {
      llvm::outs() << "Dim = " << dim << ", offsetToAdd = " << offsetToAdd
                   << "\n";
      llvm::outs().flush();
      auto applyOp = cast<affine::AffineApplyOp>(
          cast<Value>(staticL3AsSourceOffsets[dim]).getDefiningOp());
      AffineMap affineMap = applyOp.getAffineMap();
      llvm::outs() << "AffineMap = " << affineMap << "\n";
      llvm::outs().flush();
    }
    // Create new L3 -> L2 Dma Op.
    rewriter.setInsertionPoint(l3ToL2DmaOp);
    rewriter.create<AMDAIE::DmaCpyNdOp>(
        l3ToL2DmaOp.getLoc(), source, llvm::ArrayRef(staticL2AsTargetOffsets),
        llvm::ArrayRef(staticL2AsTargetSizes),
        llvm::ArrayRef(staticL2AsTargetStrides), l3ToL2DmaOp.getSource(),
        l3ToL2DmaOp.getSourceMixedOffsets(),
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
    // We have to discard non-zero offsets as subview has been replaced by a
    // dedicated allocated memref.
    SmallVector<int64_t> allocShape(type.getShape());
    (void)discardAllNonZeroOffsets<CopyOpOperateOn::Source>(
        rewriter,
        cast<AMDAIE::DoublyStridedOpInterface>(newL2ToL1DmaOp.getOperation()),
        allocShape);

    // Remove old dealloc.
    memref::DeallocOp oldDeallocOp;
    for (Operation *userOp : sourceAllocOp->getUsers()) {
      if (auto deallocUser = dyn_cast<memref::DeallocOp>(userOp)) {
        oldDeallocOp = deallocUser;
      }
    }
    if (oldDeallocOp) {
      rewriter.eraseOp(oldDeallocOp);
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
