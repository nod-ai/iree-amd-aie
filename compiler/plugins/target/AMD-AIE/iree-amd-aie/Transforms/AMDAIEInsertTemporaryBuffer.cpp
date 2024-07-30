// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Iterators.h"

#define DEBUG_TYPE "iree-amdaie-insert-temporary-buffer"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Walk `None` type access operations within each core operation, and replace
/// the users of such access ops with a temporary buffer.
LogicalResult noneAccessOpToTemporaryBuffer(Operation *parentOp) {
  IRRewriter rewriter(parentOp->getContext());

  SmallVector<AMDAIE::CoreOp> coreOps;
  parentOp->walk([&](AMDAIE::CoreOp coreOp) { coreOps.push_back(coreOp); });

  for (AMDAIE::CoreOp coreOp : coreOps) {
    // Map from a logical objectFifo to a temporary buffer.
    DenseMap<Value, memref::AllocOp> logicalObjectFifoToAlloc;

    WalkResult res =
        coreOp->walk([&](AMDAIE::LogicalObjectFifoAccessOp accessOp) {
          if (accessOp.getAccessType() != AMDAIE::MemoryAccess::None)
            return WalkResult::advance();

          memref::AllocOp newAllocOp;
          if (!logicalObjectFifoToAlloc.contains(accessOp.getInput())) {
            // Insert a new alloc op at the point of the first `None` type
            // access op
            rewriter.setInsertionPoint(accessOp);
            auto memRefType = cast<MemRefType>(accessOp.getOutput().getType());
            MemRefType allocType = MemRefType::get(
                memRefType.getShape(), memRefType.getElementType(),
                MemRefLayoutAttrInterface{}, memRefType.getMemorySpace());
            newAllocOp = rewriter.create<memref::AllocOp>(
                rewriter.getUnknownLoc(), allocType);
            logicalObjectFifoToAlloc[accessOp.getInput()] = newAllocOp;

            // Insert a dealloc op at the end of the block
            auto newDeallocOp = rewriter.create<memref::DeallocOp>(
                rewriter.getUnknownLoc(), newAllocOp);
            newDeallocOp->moveBefore(&newAllocOp->getBlock()->back());
          } else {
            newAllocOp = logicalObjectFifoToAlloc[accessOp.getInput()];
            if (!newAllocOp) {
              accessOp.emitOpError()
                  << "No alloc op is mapped from the input of access op";
              return WalkResult::interrupt();
            }
          }

          rewriter.replaceAllUsesWith(accessOp.getResult(),
                                      newAllocOp.getResult());
          return WalkResult::advance();
        });
    if (res.wasInterrupted()) return failure();
  }
  return success();
}

class AMDAIEInsertTemporaryBufferPass
    : public impl::AMDAIEInsertTemporaryBufferBase<
          AMDAIEInsertTemporaryBufferPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect, memref::MemRefDialect>();
  }

  AMDAIEInsertTemporaryBufferPass() = default;
  AMDAIEInsertTemporaryBufferPass(
      const AMDAIEInsertTemporaryBufferPass &pass){};
  void runOnOperation() override;
};

void AMDAIEInsertTemporaryBufferPass::runOnOperation() {
  Operation *parentOp = getOperation();
  if (failed(noneAccessOpToTemporaryBuffer(parentOp))) {
    parentOp->emitOpError() << "failed to convert `None` type access "
                               "operations to temporary buffers";
    return signalPassFailure();
  }
  // Erase old access operations.
  IRRewriter rewriter(parentOp->getContext());
  parentOp->walk([&](AMDAIE::LogicalObjectFifoAccessOp accessOp) {
    if (accessOp->getUses().empty()) {
      rewriter.eraseOp(accessOp);
    }
  });
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEInsertTemporaryBufferPass() {
  return std::make_unique<AMDAIEInsertTemporaryBufferPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
