// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Iterators.h"

#define DEBUG_TYPE "iree-amdaie-produce-consume-to-acquire-release"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Utility to find the parent operation of the provided operation within the
/// same block as the provided one if it exists.
Operation *getParentOpInBlock(Block *block, Operation *op) {
  if (!op || op->getBlock() == block) return op;
  auto parentOp = op->getParentOp();
  return getParentOpInBlock(block, parentOp);
}

/// Walk all consume/produce operations within the core operations and insert
/// semaphore operations.
template <typename OpTy>
LogicalResult consumeProduceToAcquireRelease(Operation *parentOp) {
  using IteratorType = std::conditional_t<
      std::is_same<OpTy, AMDAIE::LogicalObjectFifoConsume>::value,
      ForwardIterator, ReverseIterator>;
  using SemaphoreTypeAtOp = std::conditional_t<
      std::is_same<OpTy, AMDAIE::LogicalObjectFifoConsume>::value,
      AMDAIE::LogicalObjectFifoAcquire, AMDAIE::LogicalObjectFifoRelease>;
  using SemaphoreTypeAtOtherEndOfBlock = std::conditional_t<
      std::is_same<OpTy, AMDAIE::LogicalObjectFifoConsume>::value,
      AMDAIE::LogicalObjectFifoRelease, AMDAIE::LogicalObjectFifoAcquire>;

  IRRewriter rewriter(parentOp->getContext());
  auto walkResult = parentOp->walk([&](AMDAIE::CoreOp coreOp) {
    IRMapping mapper;
    coreOp->walk<WalkOrder::PostOrder, IteratorType>([&](OpTy op) {
      rewriter.setInsertionPoint(op);
      rewriter.create<SemaphoreTypeAtOp>(rewriter.getUnknownLoc(), op.getDma(),
                                         op.getPort());

      // Retrieve the DMA operation for this consume/produce and check whether
      // it was encountered before. Add it to the map and advance if not.
      Operation *dmaOp = op.getDma().getDefiningOp();
      if (!mapper.contains(dmaOp)) {
        mapper.map(dmaOp, op.getOperation());
        return WalkResult::advance();
      }

      // Find the new consume/produce operation's parent operation within the
      // same block as the previous operation of the same type and operating on
      // the same DMA. Use this parent operation in the same block to set the
      // insertion point either before or after depending on whether the
      // iteration is happening in forward or backward fashion.
      auto parentOpInBlock =
          getParentOpInBlock(mapper.lookup(dmaOp)->getBlock(), op);
      if (parentOpInBlock) {
        if (std::is_same<IteratorType, ForwardIterator>::value) {
          rewriter.setInsertionPoint(parentOpInBlock);
        } else {
          rewriter.setInsertionPointAfter(parentOpInBlock);
        }
      } else {
        if (std::is_same<IteratorType, ForwardIterator>::value) {
          rewriter.setInsertionPoint(
              mapper.lookup(dmaOp)->getBlock()->getTerminator());
        } else {
          rewriter.setInsertionPointToStart(mapper.lookup(dmaOp)->getBlock());
        }
      }

      // Insert the other semaphore operation and erase the produce/consume
      // operation.
      rewriter.create<SemaphoreTypeAtOtherEndOfBlock>(
          rewriter.getUnknownLoc(), op.getDma(), op.getPort());
      rewriter.eraseOp(mapper.lookup(dmaOp));
      mapper.map(dmaOp, op.getOperation());
      return WalkResult::advance();
    });

    // Add `SemaphoreTypeAtOtherEndOfBlock` operations for remaining
    // consume/produce operations at the other end of the blocks.
    for (auto &&[keyOp, valueOp] : mapper.getOperationMap()) {
      auto produceConsumeOp = dyn_cast<OpTy>(valueOp);
      if (std::is_same<OpTy, AMDAIE::LogicalObjectFifoConsume>::value) {
        rewriter.setInsertionPoint(
            produceConsumeOp->getBlock()->getTerminator());
      } else {
        rewriter.setInsertionPointToStart(produceConsumeOp->getBlock());
      }
      rewriter.create<SemaphoreTypeAtOtherEndOfBlock>(
          rewriter.getUnknownLoc(), produceConsumeOp.getDma(),
          produceConsumeOp.getPort());
      rewriter.eraseOp(produceConsumeOp);
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) return failure();
  return success();
}

class AMDAIEConsumeProduceToAcquireReleasePass
    : public impl::AMDAIEConsumeProduceToAcquireReleaseBase<
          AMDAIEConsumeProduceToAcquireReleasePass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEConsumeProduceToAcquireReleasePass() = default;
  AMDAIEConsumeProduceToAcquireReleasePass(
      const AMDAIEConsumeProduceToAcquireReleasePass &pass){};
  void runOnOperation() override;
};

void AMDAIEConsumeProduceToAcquireReleasePass::runOnOperation() {
  if (failed(consumeProduceToAcquireRelease<AMDAIE::LogicalObjectFifoConsume>(
          getOperation()))) {
    return signalPassFailure();
  }
  if (failed(consumeProduceToAcquireRelease<AMDAIE::LogicalObjectFifoProduce>(
          getOperation()))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEConsumeProduceToAcquireReleasePass() {
  return std::make_unique<AMDAIEConsumeProduceToAcquireReleasePass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
