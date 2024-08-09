// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIEDmaUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Common.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#define DEBUG_TYPE "iree-amdaie-split-buffers"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIESplitBuffersPass
    : public impl::AMDAIESplitBuffersBase<AMDAIESplitBuffersPass> {
 public:
  using AMDAIESplitBuffersBase::AMDAIESplitBuffersBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }
  void runOnOperation() override;
};

void AMDAIESplitBuffersPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  IRRewriter rewriter(moduleOp.getContext());

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

    // Now we'll create a narrowed L2 buffer.
    rewriter.setInsertionPoint(sourceAllocOp);
    auto oldSourceMemRefType = cast<MemRefType>(sourceAllocOp.getType());
    auto targetMemRefType = cast<MemRefType>(targetAllocOp.getType());
    MemRefType newAllocType = MemRefType::get(
        targetMemRefType.getNumElements(), targetMemRefType.getElementType(),
        MemRefLayoutAttrInterface{}, oldSourceMemRefType.getMemorySpace());
    auto newAllocOp = rewriter.create<memref::AllocOp>(rewriter.getUnknownLoc(),
                                                       newAllocType);
    auto newDeallocOp = rewriter.create<memref::DeallocOp>(
        rewriter.getUnknownLoc(), newAllocOp);
    newDeallocOp->moveBefore(&newAllocOp->getBlock()->back());

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

    auto type = cast<MemRefType>(newAllocOp.getType());
    // Create new logicalobjectfifo.from_memref for the newly created L2 buffer.
    rewriter.setInsertionPoint(l2ToL1DmaOp.getSourceObjectFifo());
    auto source = rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
        rewriter.getUnknownLoc(), LogicalObjectFifoType::get(type),
        newAllocOp.getResult(), sourceObjectFifo.getTiles());

    // Create new L3 -> L2 Dma Op.
    rewriter.setInsertionPoint(l3ToL2DmaOp);
    rewriter.create<AMDAIE::DmaCpyNdOp>(
        l3ToL2DmaOp.getLoc(), source, l3ToL2DmaOp.getTargetMixedOffsets(),
        l3ToL2DmaOp.getTargetMixedSizes(), l3ToL2DmaOp.getTargetMixedStrides(),
        l3ToL2DmaOp.getSource(), l3ToL2DmaOp.getSourceMixedOffsets(),
        l3ToL2DmaOp.getSourceMixedSizes(), l3ToL2DmaOp.getSourceMixedStrides());

    // Create new L2 -> L1 Input DmaOp.
    rewriter.setInsertionPoint(l2ToL1DmaOp);
    auto newL2ToL1DmaOp = rewriter.create<AMDAIE::DmaCpyNdOp>(
        l2ToL1DmaOp.getLoc(), l2ToL1DmaOp.getTarget(),
        l2ToL1DmaOp.getTargetMixedOffsets(), l2ToL1DmaOp.getTargetMixedSizes(),
        l2ToL1DmaOp.getTargetMixedStrides(), source,
        l2ToL1DmaOp.getSourceMixedOffsets(), l2ToL1DmaOp.getSourceMixedSizes(),
        l2ToL1DmaOp.getSourceMixedStrides());
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

std::unique_ptr<Pass> createAMDAIESplitBuffersPass() {
  return std::make_unique<AMDAIESplitBuffersPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
