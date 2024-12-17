// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEDmaUtils.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/IR/Iterators.h"
#define DEBUG_TYPE "iree-amdaie-fold-dma-waits"

namespace mlir::iree_compiler::AMDAIE {

namespace {

using DmaBdIdKey = std::pair<AMDAIE::TileOp, AMDAIE::ConnectionOp>;

/// Utility function to erase the DMA wait operations in the queue, except for
/// the last one.
LogicalResult eraseQueueOperations(IRRewriter &rewriter,
                                   SmallVector<AMDAIE::NpuDmaWaitOp> &waitOps) {
  // Skip if there are less than two DMA wait operations in the queue.
  if (waitOps.size() < 2) return success();

  Operation *parentOp = waitOps.back()->getParentOp();
  // Do not modify the last wait op, it will be kept.
  waitOps.pop_back();

  for (AMDAIE::NpuDmaWaitOp waitOp : waitOps) {
    if (waitOp->getParentOp() != parentOp) {
      return waitOp.emitError(
          "DMA operations to be queued must belong to the same scope");
    }
    // Erase the wait op.
    SmallVector<Value> asyncTokens(waitOp.getAsyncTokens());
    rewriter.eraseOp(waitOp);
    for (Value token : asyncTokens) {
      auto dmaOp =
          dyn_cast_if_present<AMDAIE::NpuHalfDmaCpyNdOp>(token.getDefiningOp());
      if (!dmaOp)
        waitOp.emitError("expected to operate on an `amdaie.half_dma_cpy_nd`");
      if (dmaOp.use_empty()) {
        rewriter.setInsertionPoint(dmaOp);
        TypeRange resultTypeRange = TypeRange{};
        // Nullify the result to avoid issuing a token.
        rewriter.create<AMDAIE::NpuHalfDmaCpyNdOp>(
            dmaOp.getLoc(), resultTypeRange, dmaOp.getConnection(),
            dmaOp.getInput(), dmaOp.getMixedOffsets(), dmaOp.getMixedSizes(),
            dmaOp.getMixedStrides(), dmaOp.getBdId(), dmaOp.getChannel(),
            dmaOp.getNextBd(), dmaOp.getStartBd());
        rewriter.eraseOp(dmaOp);
      }
    }
  }
  return success();
}

/// Utility function to determine whether a DMA wait op can be folded into a
/// queue based on its half DMA copy operation.
FailureOr<bool> canFoldByQueue(
    const AMDAIE::AMDAIEDeviceModel &deviceModel, Operation *queueParentOp,
    AMDAIE::NpuHalfDmaCpyNdOp &npuHalfDmaCpyNdOp,
    DenseMap<DmaBdIdKey, DenseSet<uint32_t>> &dmaBdIdsMap) {
  // Check if the current operation is in the same scope as the rest of the
  // queue.
  bool isSameScope = npuHalfDmaCpyNdOp->getParentOp() == queueParentOp;

  // Retrieve the connection op.
  std::optional<AMDAIE::ConnectionOp> maybeConnectionOp =
      npuHalfDmaCpyNdOp.getConnectionOp();
  if (!maybeConnectionOp) {
    return npuHalfDmaCpyNdOp.emitOpError()
           << "expected to operate on an `amdaie.connection`";
  }
  AMDAIE::ConnectionOp connectionOp = maybeConnectionOp.value();

  // Retrieve the flow op.
  std::optional<AMDAIE::FlowOp> maybeFlowOp = connectionOp.getFlowOp();
  if (!maybeFlowOp) {
    return connectionOp.emitOpError()
           << "expected to operate on an `amdaie.flow`";
  }
  AMDAIE::FlowOp flowOp = maybeFlowOp.value();
  bool isPacketFlow = flowOp.getIsPacketFlow();

  // Retrieve the BD ID op.
  std::optional<AMDAIE::BdIdOp> maybeBdIdOp = npuHalfDmaCpyNdOp.getBdIdOp();
  if (!maybeBdIdOp) {
    return npuHalfDmaCpyNdOp.emitOpError()
           << "must have a BD ID op to lower to "
              "`amdaie.npu.write_bd`";
  }
  AMDAIE::BdIdOp bdIdOp = maybeBdIdOp.value();

  // Retrieve the tile op.
  AMDAIE::TileOp tileOp =
      dyn_cast_if_present<AMDAIE::TileOp>(bdIdOp.getTile().getDefiningOp());
  if (!tileOp) {
    return bdIdOp.emitOpError() << "must operate on an `amdaie.tile`";
  }

  // Get the maximum queue size.
  uint32_t col = getConstantIndexOrAssert(tileOp.getCol());
  uint32_t row = getConstantIndexOrAssert(tileOp.getRow());
  uint32_t maxQueueSize = deviceModel.getDmaMaxQueueSize(col, row);

  uint32_t bdId = getConstantIndexOrAssert(bdIdOp.getValue());
  bool isDuplicateBdId = llvm::any_of(dmaBdIdsMap, [&](const auto &entry) {
    return entry.first.first == tileOp && entry.second.contains(bdId);
  });
  DenseSet<uint32_t> &bdIds = dmaBdIdsMap[{tileOp, connectionOp}];
  bool canFold = true;
  // Can't fold wait op if:
  // (1) the current BD ID on the same tile already occurs in the queue, or
  // (2) the current operation is a packet flow, or
  // (3) reaches the maximum queue size, or
  // (4) the queue is empty, or
  // (5) the current operation is not in the same scope as the queue.
  if (isDuplicateBdId || isPacketFlow || bdIds.size() >= maxQueueSize ||
      bdIds.empty() || !isSameScope) {
    bdIds.clear();
    canFold = false;
  }
  bdIds.insert(bdId);
  return canFold;
}

/// Traverses the control code in reverse, ensuring that for each connection,
/// only one DMA wait op is retained for every maximum queue size.
///
/// Example Output: assuming a maximum queue size of 4.
///   dma_cpy_nd(connection=0, bd_id=0)
///   %0 = dma_cpy_nd(connection=0, bd_id=1)
///   dma_wait(%0)
///   dma_cpy_nd(connection=0, bd_id=2)
///   dma_cpy_nd(connection=0, bd_id=3)
///   dma_cpy_nd(connection=0, bd_id=4)
///   %1 = dma_cpy_nd(connection=0, bd_id=5)
///   dma_wait(%1)
/// From the bottom up, for every four DMA copy operations, only one DMA wait
/// operation is retained.
///
/// Reverse traversal simplifies handling duplicate BD IDs, preventing
/// the need to revisit and modify earlier operations after processing later
/// ones.
LogicalResult foldDmaWaitsByQueue(const AMDAIE::AMDAIEDeviceModel &deviceModel,
                                  AMDAIE::ControlCodeOp controlCodeOp) {
  IRRewriter rewriter(controlCodeOp->getContext());
  SmallVector<SmallVector<AMDAIE::NpuDmaWaitOp>> waitOpQueues;
  DenseMap<DmaBdIdKey, DenseSet<uint32_t>> dmaBdIdsMap;
  // Traverse the control code in reverse.
  WalkResult res = controlCodeOp->walk<WalkOrder::PostOrder, ReverseIterator>(
      [&](AMDAIE::NpuDmaWaitOp waitOp) {
        bool toFold = true;
        Operation *queueParentOp =
            waitOpQueues.empty() ? waitOp->getParentOp()
                                 : waitOpQueues.back().front()->getParentOp();
        for (Value token : waitOp.getAsyncTokens()) {
          if (auto npuHalfDmaCpyNdOp =
                  dyn_cast_if_present<AMDAIE::NpuHalfDmaCpyNdOp>(
                      token.getDefiningOp())) {
            FailureOr<bool> result = canFoldByQueue(
                deviceModel, queueParentOp, npuHalfDmaCpyNdOp, dmaBdIdsMap);
            if (failed(result)) return WalkResult::interrupt();
            toFold &= *result;
          }
        }
        // Store all the queues, and modify later to avoid invalidating the
        // iterator.
        if (toFold) {
          // Append the wait op to the last queue if it can be folded.
          waitOpQueues.back().push_back(waitOp);
        } else {
          // Create a new queue if the wait op cannot be folded.
          waitOpQueues.push_back(SmallVector<AMDAIE::NpuDmaWaitOp>{waitOp});
        }
        return WalkResult::advance();
      });
  if (res.wasInterrupted()) return failure();
  for (SmallVector<AMDAIE::NpuDmaWaitOp> &waitOps : waitOpQueues) {
    // Since the controlcode is traversed in reverse order, we need to
    // restore the original order of the DMA operations.
    std::reverse(waitOps.begin(), waitOps.end());
    if (failed(eraseQueueOperations(rewriter, waitOps))) return failure();
  }
  return success();
}

/// For each batch, combine the async tokens into a single NpuDmaWaitOp.
LogicalResult updateBatchTokens(IRRewriter &rewriter,
                                SmallVector<AMDAIE::NpuDmaWaitOp> &waitOps) {
  // Skip if there are less than two DMA wait operations.
  if (waitOps.size() < 2) return success();

  SmallVector<Value> asyncTokens;
  Operation *parentOp = waitOps[0]->getParentOp();
  for (AMDAIE::NpuDmaWaitOp waitOp : waitOps) {
    if (waitOp->getParentOp() != parentOp) {
      return waitOp.emitError(
          "DMA operations to be batched must belong to the same scope");
    }
    asyncTokens.append(waitOp.getAsyncTokens().begin(),
                       waitOp.getAsyncTokens().end());
  }

  rewriter.setInsertionPointAfter(waitOps.back());
  rewriter.create<AMDAIE::NpuDmaWaitOp>(waitOps.back().getLoc(), asyncTokens);
  for (AMDAIE::NpuDmaWaitOp waitOp : waitOps) rewriter.eraseOp(waitOp);
  return success();
}

/// Utility function to determine if a DMA wait operation can be folded into a
/// a batch based on its half DMA copy operation.
FailureOr<bool> canFoldByBatch(
    Operation *batchParentOp, AMDAIE::NpuHalfDmaCpyNdOp npuHalfDmaCpyNdOp,
    DenseSet<AMDAIE::ConnectionOp> &connectionOps,
    DenseMap<DmaBdIdKey, DenseSet<uint32_t>> &dmaBdIdsMap) {
  // Check if the current operation is in the same scope as the rest of the
  // batch.
  bool isSameScope = npuHalfDmaCpyNdOp->getParentOp() == batchParentOp;

  // Retrieve the connection op.
  std::optional<AMDAIE::ConnectionOp> maybeConnectionOp =
      npuHalfDmaCpyNdOp.getConnectionOp();
  if (!maybeConnectionOp) {
    return npuHalfDmaCpyNdOp.emitOpError()
           << "expected to operate on an `amdaie.connection`";
  }
  AMDAIE::ConnectionOp connectionOp = maybeConnectionOp.value();

  // Retrieve the flow op.
  std::optional<AMDAIE::FlowOp> maybeFlowOp = connectionOp.getFlowOp();
  if (!maybeFlowOp) {
    return connectionOp.emitOpError()
           << "expected to operate on an `amdaie.flow`";
  }
  AMDAIE::FlowOp flowOp = maybeFlowOp.value();
  bool isPacketFlow = flowOp.getIsPacketFlow();

  // Retrieve the BD ID op.
  std::optional<AMDAIE::BdIdOp> maybeBdIdOp = npuHalfDmaCpyNdOp.getBdIdOp();
  if (!maybeBdIdOp) {
    return npuHalfDmaCpyNdOp.emitOpError()
           << "must have a BD ID op to lower to "
              "`amdaie.npu.write_bd`";
  }
  AMDAIE::BdIdOp bdIdOp = maybeBdIdOp.value();

  // Retrieve the tile op.
  AMDAIE::TileOp tileOp =
      dyn_cast_if_present<AMDAIE::TileOp>(bdIdOp.getTile().getDefiningOp());
  if (!tileOp) {
    return bdIdOp.emitOpError() << "must operate on an `amdaie.tile`";
  }

  bool isDuplicateConnection = connectionOps.contains(connectionOp);
  uint32_t bdId = getConstantIndexOrAssert(bdIdOp.getValue());
  bool isDuplicateBdId = llvm::any_of(dmaBdIdsMap, [&](const auto &entry) {
    return entry.first.first == tileOp && entry.second.contains(bdId);
  });

  bool canFold = true;
  // Can't fold wait op if:
  // (1) the current connection op already occurs in the batch, or
  // (2) the current BD ID on the same tile already occurs in the batch, or
  // (3) the current operation is a packet flow, or
  // (4) the batch is empty, or
  // (5) the current operation is not in the same scope as the batch.
  if (isDuplicateConnection || isDuplicateBdId || isPacketFlow ||
      connectionOps.empty() || !isSameScope) {
    // Clear the BD IDs for all the connections in the batch.
    for (auto &entry : dmaBdIdsMap) {
      ConnectionOp connectionOp = entry.first.second;
      DenseSet<uint32_t> &bdIds = entry.second;
      if (connectionOps.contains(connectionOp)) bdIds.clear();
    }
    connectionOps.clear();
    canFold = false;
  }
  connectionOps.insert(connectionOp);
  dmaBdIdsMap[{tileOp, connectionOp}].insert(bdId);
  return canFold;
}

/// Traverses the control code in reverse, ensuring that only one DMA wait op is
/// retained for every batch of DMA copy operations.
///
/// Example Input:
///   %0 = dma_cpy_nd(connection0)
///   dma_wait(%0)
///   %1 = dma_cpy_nd(connection1)
///   %2 = dma_cpy_nd(connection2)
///   %3 = dma_cpy_nd(connection3)
///   dma_wait(%1)
///   dma_wait(%2)
///   dma_wait(%3)
/// Example Output:
///   %0 = dma_cpy_nd(connection0)
///   %1 = dma_cpy_nd(connection1)
///   %2 = dma_cpy_nd(connection2)
///   %3 = dma_cpy_nd(connection3)
///   dma_wait(%0, %1, %2, %3)
/// Reverse traversal simplifies handling duplicate connections, preventing
/// the need to revisit and modify earlier operations after processing later
/// ones.
LogicalResult foldDmaWaitsByBatch(AMDAIE::ControlCodeOp controlCodeOp) {
  IRRewriter rewriter(controlCodeOp->getContext());
  SmallVector<AMDAIE::NpuDmaWaitOp> waitOps;
  DenseSet<AMDAIE::ConnectionOp> connectionOps;
  DenseMap<DmaBdIdKey, DenseSet<uint32_t>> dmaBdIdsMap;
  WalkResult res = controlCodeOp->walk<WalkOrder::PostOrder, ReverseIterator>(
      [&](AMDAIE::NpuDmaWaitOp waitOp) {
        bool toBatch = true;
        Operation *batchParentOp =
            waitOps.empty() ? waitOp->getParentOp() : waitOps[0]->getParentOp();
        for (Value token : waitOp.getAsyncTokens()) {
          if (auto npuHalfDmaCpyNdOp =
                  dyn_cast_if_present<AMDAIE::NpuHalfDmaCpyNdOp>(
                      token.getDefiningOp())) {
            FailureOr<bool> result = canFoldByBatch(
                batchParentOp, npuHalfDmaCpyNdOp, connectionOps, dmaBdIdsMap);
            if (failed(result)) return WalkResult::interrupt();
            toBatch &= *result;
          }
        }
        // Process the previous batch of wait ops, and start a new batch.
        if (!toBatch) {
          // Since the controlcode is traversed in reverse order, we need to
          // restore the original order of the DMA operations.
          std::reverse(waitOps.begin(), waitOps.end());
          if (failed(updateBatchTokens(rewriter, waitOps)))
            return WalkResult::interrupt();
          waitOps.clear();
        }
        waitOps.push_back(waitOp);
        return WalkResult::advance();
      });

  if (res.wasInterrupted()) return failure();
  // Process the remaining wait ops.
  std::reverse(waitOps.begin(), waitOps.end());
  if (failed(updateBatchTokens(rewriter, waitOps))) return failure();
  return success();
}

class AMDAIEFoldDmaWaitsPass
    : public impl::AMDAIEFoldDmaWaitsBase<AMDAIEFoldDmaWaitsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEFoldDmaWaitsPass() = default;
  AMDAIEFoldDmaWaitsPass(const AMDAIEFoldDmaWaitsPass &pass){};
  void runOnOperation() override;
};

void AMDAIEFoldDmaWaitsPass::runOnOperation() {
  Operation *parentOp = getOperation();

  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(parentOp);
  std::optional<AMDAIEDevice> maybeDevice = getConfigAMDAIEDevice(targetAttr);
  if (!maybeDevice) {
    parentOp->emitOpError()
        << "has no AMDAIEDevice in the target attribute configuration. This "
           "device-specific information is required to fold DMA wait "
           "ops.";
    return signalPassFailure();
  }
  AMDAIE::AMDAIEDeviceModel deviceModel =
      AMDAIE::getDeviceModel(maybeDevice.value());

  WalkResult res = parentOp->walk([&](AMDAIE::WorkgroupOp workgroupOp) {
    AMDAIE::ControlCodeOp controlCodeOp = workgroupOp.getControlCode();
    if (failed(foldDmaWaitsByQueue(deviceModel, controlCodeOp))) {
      return WalkResult::interrupt();
    }
    if (failed(foldDmaWaitsByBatch(controlCodeOp))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEFoldDmaWaitsPass() {
  return std::make_unique<AMDAIEFoldDmaWaitsPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
