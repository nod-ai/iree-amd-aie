// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"

#define DEBUG_TYPE "iree-amdaie-none-access-to-temporary-buffer"

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
          auto iter = logicalObjectFifoToAlloc.find(accessOp.getInput());
          if (iter == logicalObjectFifoToAlloc.end()) {
            Location loc = accessOp.getLoc();

            // Insert an alloc op at the start of the parent core op's block.
            rewriter.setInsertionPointToStart(coreOp.getBody(0));
            auto memRefType = cast<MemRefType>(accessOp.getOutput().getType());
            MemRefType allocType = MemRefType::get(
                memRefType.getShape(), memRefType.getElementType(),
                MemRefLayoutAttrInterface{}, memRefType.getMemorySpace());
            newAllocOp = rewriter.create<memref::AllocOp>(loc, allocType);
            logicalObjectFifoToAlloc.insert({accessOp.getInput(), newAllocOp});

            // Insert a dealloc just before amdaie.end
            rewriter.setInsertionPoint(coreOp.getBody(0)->getTerminator());
            rewriter.create<memref::DeallocOp>(loc, newAllocOp);
          } else {
            newAllocOp = iter->second;
            assert(newAllocOp && "how was a null value inserted into map?");
          }
          rewriter.replaceAllUsesWith(accessOp.getResult(),
                                      newAllocOp.getResult());
          return WalkResult::advance();
        });
    if (res.wasInterrupted()) return failure();
  }
  return success();
}

class AMDAIENoneAccessToTemporaryBufferPass
    : public impl::AMDAIENoneAccessToTemporaryBufferBase<
          AMDAIENoneAccessToTemporaryBufferPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect, memref::MemRefDialect>();
  }

  AMDAIENoneAccessToTemporaryBufferPass() = default;
  AMDAIENoneAccessToTemporaryBufferPass(
      const AMDAIENoneAccessToTemporaryBufferPass &pass){};
  void runOnOperation() override;
};

void AMDAIENoneAccessToTemporaryBufferPass::runOnOperation() {
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

std::unique_ptr<Pass> createAMDAIENoneAccessToTemporaryBufferPass() {
  return std::make_unique<AMDAIENoneAccessToTemporaryBufferPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
