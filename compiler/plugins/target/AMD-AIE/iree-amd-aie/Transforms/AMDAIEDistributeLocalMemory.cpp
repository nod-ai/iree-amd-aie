// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-distribute-local-memory"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Distribute local memory accesses through subviews by allocating a single
/// smaller memory. This is needed because cores can't operate on one larger L1
/// memory.
LogicalResult distributeLocalMemory(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
  SmallVector<Operation *> toBeErased;
  // Map from alloc operations to a new alloc operations to be used.
  DenseMap<memref::AllocOp, memref::AllocOp> memrefToNew;

  moduleOp->walk([&](memref::AllocOp allocOp) {
    // Only consider local memory (L1).
    Attribute memSpace =
        cast<MemRefType>(allocOp.getResult().getType()).getMemorySpace();
    if (!memSpace || dyn_cast<IntegerAttr>(memSpace).getInt() != 2)
      return WalkResult::advance();

    llvm::outs().flush();
    LLVM_DEBUG(llvm::dbgs()
               << "DistributeLocalMemory for: " << allocOp << "\n");

    SmallVector<AMDAIE::DmaCpyNdOp> dmaUsers;
    for (Operation *userOp : allocOp->getUsers()) {
      if (auto logicalObjectFifo =
              dyn_cast<AMDAIE::LogicalObjectFifoFromMemrefOp>(userOp)) {
        for (Operation *objFifoUserOp : logicalObjectFifo->getUsers()) {
          if (auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(objFifoUserOp);
              dmaOp.getSourceObjectFifo() == logicalObjectFifo) {
            dmaUsers.push_back(dmaOp);
          }
        }
      }
    }
    if (dmaUsers.empty()) return WalkResult::advance();
    LLVM_DEBUG(llvm::dbgs() << "DMA users: " << dmaUsers.size() << "\n");

    for (Operation *userOp : allocOp->getUsers()) {
      auto subviewOp = dyn_cast<memref::SubViewOp>(userOp);
      if (!subviewOp) continue;

      if (!memrefToNew.contains(allocOp)) {
        LLVM_DEBUG(llvm::dbgs() << "Create new allocate\n");
        rewriter.setInsertionPoint(allocOp);
        auto memRefType = cast<MemRefType>(subviewOp.getResult().getType());
        MemRefType allocType = MemRefType::get(
            memRefType.getShape(), memRefType.getElementType(),
            MemRefLayoutAttrInterface{}, memRefType.getMemorySpace());
        auto newAllocOp = rewriter.create<memref::AllocOp>(
            rewriter.getUnknownLoc(), allocType);
        auto newDeallocOp = rewriter.create<memref::DeallocOp>(
            rewriter.getUnknownLoc(), newAllocOp);
        newDeallocOp->moveBefore(&newAllocOp->getBlock()->back());
        memrefToNew[allocOp] = newAllocOp;
      }
      auto newAlloc = memrefToNew[allocOp];
      rewriter.replaceAllUsesWith(subviewOp, newAlloc);
      toBeErased.push_back(subviewOp);
    }

    // Update the alloc's DMA users.
    if (memrefToNew.contains(allocOp)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Update allocate DMA users: " << dmaUsers.size() << "\n");
      auto newAlloc = memrefToNew[allocOp];
      auto type = cast<MemRefType>(newAlloc.getType());
      for (AMDAIE::DmaCpyNdOp dmaOp : dmaUsers) {
        SmallVector<Value> empty;
        rewriter.setInsertionPoint(dmaOp.getSourceObjectFifo());
        auto source = rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
            rewriter.getUnknownLoc(), LogicalObjectFifoType::get(type),
            newAlloc.getResult());
        rewriter.replaceOp(dmaOp.getSourceObjectFifo(), source);
        rewriter.setInsertionPoint(dmaOp);
        auto newDmaOp = rewriter.create<AMDAIE::DmaCpyNdOp>(
            dmaOp.getLoc(), dmaOp.getTarget(), dmaOp.getTargetOffsets(),
            dmaOp.getTargetSizes(), dmaOp.getTargetStrides(), source,
            dmaOp.getSourceOffsets(), dmaOp.getSourceSizes(),
            dmaOp.getSourceStrides());
        rewriter.replaceOp(dmaOp, newDmaOp);
      }

      // Insert dealloc
      memref::DeallocOp deallocOp;
      for (Operation *userOp : allocOp->getUsers()) {
        if (auto deallocUser = dyn_cast<memref::DeallocOp>(userOp)) {
          deallocOp = deallocUser;
        }
      }
      if (deallocOp) {
        toBeErased.push_back(deallocOp);
      }
      toBeErased.push_back(allocOp);
    }
    return WalkResult::advance();
  });

  for (auto *op : toBeErased) {
    op->dropAllUses();
    rewriter.eraseOp(op);
  }
  return success();
}

class AMDAIEDistributeLocalMemoryPass
    : public impl::AMDAIEDistributeLocalMemoryBase<
          AMDAIEDistributeLocalMemoryPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEDistributeLocalMemoryPass() = default;
  AMDAIEDistributeLocalMemoryPass(
      const AMDAIEDistributeLocalMemoryPass &pass){};
  void runOnOperation() override;
};

void AMDAIEDistributeLocalMemoryPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  if (failed(distributeLocalMemory(moduleOp))) {
    moduleOp.emitOpError() << "local memory distribution failed";
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEDistributeLocalMemoryPass() {
  return std::make_unique<AMDAIEDistributeLocalMemoryPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
