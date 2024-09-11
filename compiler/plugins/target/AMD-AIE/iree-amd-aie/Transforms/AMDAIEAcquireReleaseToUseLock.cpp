// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"

#define DEBUG_TYPE "iree-amdaie-acquire-release-to-use-lock"

namespace mlir::iree_compiler::AMDAIE {

template <typename T>
FailureOr<AMDAIE::LogicalObjectFifoFromBuffersOp> getLogicalObjFifoOperatedOn(
    T op) {
  auto copyOp =
      dyn_cast_if_present<CopyOpInterface>(op.getDma().getDefiningOp());
  if (!copyOp)
    return op.emitOpError() << "should operate on a copy-like operation";
  auto logicalObjFifo =
      op.getPort() == LogicalObjectFifoPort::Consume
          ? dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromBuffersOp>(
                copyOp.getTarget().getDefiningOp())
          : dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromBuffersOp>(
                copyOp.getSource().getDefiningOp());
  if (!logicalObjFifo) {
    return copyOp.emitOpError()
           << "should operate on an `amdaie.logicalobjectfifo.from_buffers` op";
  }
  return logicalObjFifo;
}

/// Unroll the scf.for loops inside the core operations based on the depths of
/// the acquired objFifos.
LogicalResult coreLoopUnroll(RewriterBase &rewriter, AMDAIE::CoreOp coreOp) {
  WalkResult res = coreOp.walk([&](scf::ForOp forOp) {
    llvm::SmallDenseSet<uint8_t> depths;
    for (auto acqOp :
         forOp.getBody()->getOps<AMDAIE::LogicalObjectFifoAcquire>()) {
      FailureOr<AMDAIE::LogicalObjectFifoFromBuffersOp> maybeLogicalObjFifo =
          getLogicalObjFifoOperatedOn<AMDAIE::LogicalObjectFifoAcquire>(acqOp);
      if (failed(maybeLogicalObjFifo)) return WalkResult::interrupt();
      AMDAIE::LogicalObjectFifoFromBuffersOp logicalObjFifo =
          maybeLogicalObjFifo.value();
      depths.insert(logicalObjFifo.getDepth());
    }
    int unrollFactor =
        std::accumulate(depths.begin(), depths.end(), 1, std::lcm<int, int>);
    if (unrollFactor > 1 &&
        failed(mlir::loopUnrollByFactor(forOp, unrollFactor))) {
      forOp.emitOpError() << "could not be unrolled with unrollFactor: "
                          << unrollFactor << "\n";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return failure();
  return success();
}

FailureOr<AMDAIE::LockOp> getLockToBeUsed(
    AMDAIE::LogicalObjectFifoFromBuffersOp logicalObjFifo,
    AMDAIE::TileOp tileOp, LogicalObjectFifoPort port, LockAction lockAction) {
  // Retrieve the lock to be used based on the port and lock action.
  SmallVector<AMDAIE::LockOp> consumerLocks =
      logicalObjFifo.getConsumerLocksOnTile(tileOp);
  if (consumerLocks.size() != 1) {
    return logicalObjFifo.emitOpError()
           << "expected a single consumer lock for tile: "
           << tileOp.getResult();
  }
  SmallVector<AMDAIE::LockOp> producerLocks =
      logicalObjFifo.getProducerLocksOnTile(tileOp);
  if (producerLocks.size() != 1) {
    return logicalObjFifo.emitOpError()
           << "expected a single producer lock for tile: "
           << tileOp.getResult();
  }
  AMDAIE::LockOp lockOp;
  if (lockAction == LockAction::Acquire ||
      lockAction == LockAction::AcquireGreaterOrEqual) {
    lockOp = port == LogicalObjectFifoPort::Consume ? consumerLocks[0]
                                                    : producerLocks[0];
  } else if (lockAction == LockAction::Release) {
    lockOp = port == LogicalObjectFifoPort::Consume ? producerLocks[0]
                                                    : consumerLocks[0];
  } else {
    return logicalObjFifo.emitOpError()
           << "used in unsupported lock action: " << stringifyEnum(lockAction);
  }
  return lockOp;
}

LogicalResult acquireToUseLock(RewriterBase &rewriter, AMDAIE::CoreOp coreOp) {
  OpBuilder::InsertionGuard g(rewriter);
  AMDAIE::TileOp tileOp = coreOp.getTileOp();
  DenseMap<AMDAIE::LogicalObjectFifoFromBuffersOp, size_t>
      logicalObjFifoToIndex;
  SmallVector<Operation *> toBeErased;
  WalkResult res = coreOp.walk([&](AMDAIE::LogicalObjectFifoAcquire acqOp) {
    LLVM_DEBUG(llvm::dbgs()
               << "Convert acquire op: " << acqOp.getOutput() << "\n");
    std::optional<int> maybeAcqSize = acqOp.getSize();
    assert(maybeAcqSize && maybeAcqSize.value() == 1 &&
           "logic currently only handles size set and equal to 1");
    int acqSize = maybeAcqSize.value();

    FailureOr<AMDAIE::LogicalObjectFifoFromBuffersOp> maybeLogicalObjFifo =
        getLogicalObjFifoOperatedOn<AMDAIE::LogicalObjectFifoAcquire>(acqOp);
    if (failed(maybeLogicalObjFifo)) return WalkResult::interrupt();
    AMDAIE::LogicalObjectFifoFromBuffersOp logicalObjFifo =
        maybeLogicalObjFifo.value();

    FailureOr<AMDAIE::LockOp> maybeLockOp =
        getLockToBeUsed(logicalObjFifo, tileOp, acqOp.getPort(),
                        LockAction::AcquireGreaterOrEqual);
    if (failed(maybeLockOp)) return WalkResult::interrupt();

    rewriter.setInsertionPoint(acqOp);
    rewriter.create<AMDAIE::UseLockOp>(acqOp.getLoc(), maybeLockOp.value(),
                                       LockAction::AcquireGreaterOrEqual,
                                       acqSize);

    // Rotate through buffers based on access index.
    SmallVector<AMDAIE::BufferOp> buffers =
        logicalObjFifo.getBuffersOnTile(tileOp);
    if (!logicalObjFifoToIndex.contains(logicalObjFifo))
      logicalObjFifoToIndex[logicalObjFifo] = 0;
    size_t bufferIndex = logicalObjFifoToIndex[logicalObjFifo] % buffers.size();
    for (Operation *userOp : acqOp->getUsers()) {
      auto accessOp = dyn_cast<AMDAIE::LogicalObjectFifoAccessOp>(userOp);
      if (!accessOp) {
        acqOp.emitOpError() << "currently only supports "
                               "`amdaie.logicalobjectfifo.access` users";
        return WalkResult::interrupt();
      }
      AMDAIE::BufferOp bufferOp = buffers[bufferIndex];
      accessOp.getResult().replaceAllUsesWith(bufferOp.getResult());
      toBeErased.push_back(accessOp);
    }
    logicalObjFifoToIndex[logicalObjFifo] += acqSize;
    toBeErased.push_back(acqOp);
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return failure();
  for (Operation *op : toBeErased) {
    op->dropAllUses();
    rewriter.eraseOp(op);
  }
  return success();
}

LogicalResult releaseToUseLock(RewriterBase &rewriter, AMDAIE::CoreOp coreOp) {
  OpBuilder::InsertionGuard g(rewriter);
  AMDAIE::TileOp tileOp = coreOp.getTileOp();
  SmallVector<Operation *> toBeErased;
  WalkResult res = coreOp.walk([&](AMDAIE::LogicalObjectFifoRelease relOp) {
    LLVM_DEBUG(llvm::dbgs() << "Convert release op: " << relOp << "\n");
    std::optional<int> maybeRelSize = relOp.getSize();
    assert(maybeRelSize && maybeRelSize.value() == 1 &&
           "logic currently only handles size set and equal to 1");
    int relSize = maybeRelSize.value();

    FailureOr<AMDAIE::LogicalObjectFifoFromBuffersOp> maybeLogicalObjFifo =
        getLogicalObjFifoOperatedOn<AMDAIE::LogicalObjectFifoRelease>(relOp);
    if (failed(maybeLogicalObjFifo)) return WalkResult::interrupt();

    FailureOr<AMDAIE::LockOp> maybeLockOp =
        getLockToBeUsed(maybeLogicalObjFifo.value(), tileOp, relOp.getPort(),
                        LockAction::Release);
    if (failed(maybeLockOp)) return WalkResult::interrupt();

    rewriter.setInsertionPoint(relOp);
    rewriter.create<AMDAIE::UseLockOp>(relOp.getLoc(), maybeLockOp.value(),
                                       LockAction::Release, relSize);
    toBeErased.push_back(relOp);
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return failure();
  for (Operation *op : toBeErased) {
    op->dropAllUses();
    rewriter.eraseOp(op);
  }
  return success();
}

namespace {

struct AMDAIEAcquireReleaseToUseLockPass
    : public impl::AMDAIEAcquireReleaseToUseLockBase<
          AMDAIEAcquireReleaseToUseLockPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  void runOnOperation() override {
    Operation *parentOp = getOperation();
    IRRewriter rewriter(parentOp->getContext());

    WalkResult res = parentOp->walk([&](AMDAIE::CoreOp coreOp) {
      // Loops need to be unrolled based on on the depths of the logical
      // objectFifos so `amdaie.use_lock` ops can be inserted correctly for
      // double buffering purposes, without need for a dependency on the loop
      // induction variable.
      if (failed(coreLoopUnroll(rewriter, coreOp))) {
        return WalkResult::interrupt();
      }
      if (failed(acquireToUseLock(rewriter, coreOp))) {
        return WalkResult::interrupt();
      }
      if (failed(releaseToUseLock(rewriter, coreOp))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (res.wasInterrupted()) return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIEAcquireReleaseToUseLockPass() {
  return std::make_unique<AMDAIEAcquireReleaseToUseLockPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
