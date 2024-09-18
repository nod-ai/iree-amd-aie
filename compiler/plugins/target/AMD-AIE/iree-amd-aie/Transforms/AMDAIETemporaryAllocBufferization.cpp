// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"

#define DEBUG_TYPE "iree-amdaie-temporary-alloc-bufferization"

namespace mlir::iree_compiler::AMDAIE {

namespace {

static std::optional<BufferOp> createBufferForTemporaryAllocOp(
    IRRewriter &rewriter, WorkgroupOp workgroupOp, memref::AllocOp allocOp,
    CoreOp coreOp, unsigned index) {
  OpBuilder::InsertionGuard g(rewriter);
  TileOp tileOp = coreOp.getTileOp();
  // Reset rewriter's location to after last tile's declaration.
  auto tiles = workgroupOp.getBody()->getOps<TileOp>();
  assert(!tiles.empty() && "no tiles in workgroupOp");
  rewriter.setInsertionPointAfter(*std::prev(tiles.end(), 1));
  auto bufferType = cast<MemRefType>(allocOp.getType());
  auto bufferOp = rewriter.create<AMDAIE::BufferOp>(
      rewriter.getUnknownLoc(), bufferType, tileOp, nullptr);
  return bufferOp;
}

static LogicalResult bufferizeTemporaryAllocInCoreOp(
    IRRewriter &rewriter, WorkgroupOp workgroupOp, CoreOp coreOp,
    SmallVector<Operation *> &toBeErased) {
  // Step 1. Get all buffers within a CoreOp.
  SmallVector<memref::AllocOp> allocOps;
  coreOp.walk([&](Operation *op) {
    if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      allocOps.push_back(allocOp);
      toBeErased.push_back(allocOp);
    } else if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
      toBeErased.push_back(deallocOp);
    }
  });
  // Bail out early in case of no temporary buffers.
  if (allocOps.size() == 0) return success();
  // Step 2. Traverse unique allocOps and create an aie.buffer for them.
  SmallVector<BufferOp> temporaryBuffers;
  unsigned tempBufferIndex = 0;
  for (memref::AllocOp allocOp : allocOps) {
    std::optional<BufferOp> temporaryBuffer = createBufferForTemporaryAllocOp(
        rewriter, workgroupOp, allocOp, coreOp, tempBufferIndex++);
    if (!temporaryBuffer) {
      return failure();
    }
    allocOp.replaceAllUsesWith(temporaryBuffer.value().getResult());
  }
  return success();
}

class AMDAIETemporaryAllocBufferizationPass
    : public impl::AMDAIETemporaryAllocBufferizationBase<
          AMDAIETemporaryAllocBufferizationPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  void runOnOperation() override;
};

void AMDAIETemporaryAllocBufferizationPass::runOnOperation() {
  Operation *parentOp = getOperation();
  IRRewriter rewriter(&getContext());

  SmallVector<Operation *> toBeErased;
  WalkResult res = parentOp->walk([&](WorkgroupOp workgroupOp) {
    for (CoreOp coreOp : workgroupOp.getOps<CoreOp>()) {
      if (failed(bufferizeTemporaryAllocInCoreOp(rewriter, workgroupOp, coreOp,
                                                 toBeErased)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return signalPassFailure();

  for (Operation *op : toBeErased) {
    op->dropAllUses();
    rewriter.eraseOp(op);
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIETemporaryAllocBufferizationPass() {
  return std::make_unique<AMDAIETemporaryAllocBufferizationPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
