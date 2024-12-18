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
using DmaBdIdPair = std::pair<DmaBdIdKey, uint32_t>;

/// Utility function to retrieve TileOp, ConnectionOp, and BD ID from a given
/// half DMA copy operation.
FailureOr<DmaBdIdPair> retrieveDmaBdIdPair(
    AMDAIE::NpuHalfDmaCpyNdOp &npuHalfDmaCpyNdOp) {
  // Retrieve the connection op.
  std::optional<AMDAIE::ConnectionOp> maybeConnectionOp =
      npuHalfDmaCpyNdOp.getConnectionOp();
  if (!maybeConnectionOp) {
    return npuHalfDmaCpyNdOp.emitOpError()
           << "expected to operate on an `amdaie.connection`";
  }
  AMDAIE::ConnectionOp connectionOp = maybeConnectionOp.value();

  // Retrieve the BD ID op.
  std::optional<AMDAIE::BdIdOp> maybeBdIdOp = npuHalfDmaCpyNdOp.getBdIdOp();
  if (!maybeBdIdOp) {
    return npuHalfDmaCpyNdOp.emitOpError()
           << "must have a BD ID op to lower to "
              "`amdaie.npu.write_bd`";
  }
  AMDAIE::BdIdOp bdIdOp = maybeBdIdOp.value();
  uint32_t currBdIdVal = getConstantIndexOrAssert(bdIdOp.getValue());

  // Retrieve the tile op.
  AMDAIE::TileOp tileOp =
      dyn_cast_if_present<AMDAIE::TileOp>(bdIdOp.getTile().getDefiningOp());
  if (!tileOp) {
    return bdIdOp.emitOpError() << "must operate on an `amdaie.tile`";
  }

  DmaBdIdKey currBdIdKey = {tileOp, connectionOp};
  return DmaBdIdPair{currBdIdKey, currBdIdVal};
}

/// Utility function to determine whether a DMA wait op can be folded based on
/// its half DMA copy operation.
FailureOr<bool> canFoldByQueue(
    const AMDAIE::AMDAIEDeviceModel &deviceModel,
    AMDAIE::NpuHalfDmaCpyNdOp &npuHalfDmaCpyNdOp,
    DenseMap<DmaBdIdKey, SmallVector<uint32_t>> &tileConnectToBdIdQueue) {
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
    return connectionOp->emitOpError()
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

  // Keep wait op if, either reaches the maximum queue size, or a
  // duplicate BD ID in the same tile, or packet flow, or the queue is
  // empty
  uint32_t bdId = getConstantIndexOrAssert(bdIdOp.getValue());
  bool isDuplicateBdId =
      llvm::any_of(tileConnectToBdIdQueue, [&](const auto &entry) {
        return entry.first.first == tileOp &&
               llvm::is_contained(entry.second, bdId);
      });
  SmallVector<uint32_t> &bdIdQueue =
      tileConnectToBdIdQueue[{tileOp, connectionOp}];
  bool canFold = true;
  if (isDuplicateBdId || isPacketFlow || bdIdQueue.size() >= maxQueueSize ||
      bdIdQueue.empty()) {
    bdIdQueue.clear();
    canFold = false;
  }
  bdIdQueue.push_back(bdId);
  return canFold;
}

/// Traverses the control code in reverse, ensuring that for each connection,
/// only one DMA wait op is retained for every maximum queue size.
///
/// Example Output: assuming a maximum queue size of 4.
///   dma_cpy_nd
///   %0 = dma_cpy_nd
///   dma_wait(%0)
///   dma_cpy_nd
///   dma_cpy_nd
///   dma_cpy_nd
///   %1 = dma_cpy_nd
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
  std::vector<AMDAIE::NpuDmaWaitOp> waitOpsToErase;
  DenseMap<DmaBdIdKey, SmallVector<uint32_t>> tileConnectToBdIdQueue;
  // Traverse the control code in reverse.
  WalkResult res = controlCodeOp->walk<WalkOrder::PostOrder, ReverseIterator>(
      [&](AMDAIE::NpuDmaWaitOp waitOp) {
        bool toErase = true;
        for (Value token : waitOp.getAsyncTokens()) {
          if (auto npuHalfDmaCpyNdOp =
                  dyn_cast_if_present<AMDAIE::NpuHalfDmaCpyNdOp>(
                      token.getDefiningOp())) {
            FailureOr<bool> result = canFoldByQueue(
                deviceModel, npuHalfDmaCpyNdOp, tileConnectToBdIdQueue);
            if (failed(result)) return WalkResult::interrupt();
            toErase &= *result;
          }
        }
        // Erase later to avoid invalidating the iterator.
        if (toErase) waitOpsToErase.push_back(waitOp);
        return WalkResult::advance();
      });
  if (res.wasInterrupted()) return failure();

  for (AMDAIE::NpuDmaWaitOp waitOp : waitOpsToErase) {
    SmallVector<Value> asyncTokens(waitOp.getAsyncTokens());
    // Erase the wait op.
    rewriter.eraseOp(waitOp);
    for (Value token : asyncTokens) {
      if (auto op = dyn_cast_if_present<AMDAIE::NpuHalfDmaCpyNdOp>(
              token.getDefiningOp())) {
        if (op.use_empty()) {
          rewriter.setInsertionPoint(op);
          TypeRange resultTypeRange = TypeRange{};
          // Nullify the result to avoid issuing a token.
          rewriter.create<AMDAIE::NpuHalfDmaCpyNdOp>(
              op.getLoc(), resultTypeRange, op.getConnection(), op.getInput(),
              op.getMixedOffsets(), op.getMixedSizes(), op.getMixedStrides(),
              op.getBdId(), op.getChannel(), op.getNextBd(), op.getStartBd());
          rewriter.eraseOp(op);
        }
      }
    }
  }

  return success();
}

/// For each batch, combine the async tokens into a single NpuDmaWaitOp.
LogicalResult eraseBatchOperations(IRRewriter &rewriter,
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
/// Can't fold wait op if:
/// (1) the current operation is not in the same scope as the batch, or
/// (2) the current connection op already occurs in the batch, or
/// (3) the batch is empty, or
/// (4) the current operation is a packet flow, or
/// (5) the current BD ID on the same tile already occurs in the batch.
FailureOr<bool> canFoldByBatch(
    const Operation *batchParentOp,
    const DenseSet<AMDAIE::ConnectionOp> &connectionOps,
    const DenseMap<DmaBdIdKey, DenseSet<uint32_t>> &dmaBdIdsMap,
    AMDAIE::NpuHalfDmaCpyNdOp currHalfDmaCpyNdOp, DmaBdIdPair currBdIdPair) {
  // Not in the same scope? Can't fold.
  if (currHalfDmaCpyNdOp->getParentOp() != batchParentOp) return false;

  // Connection op already in the batch, or an empty batch? Can't fold.
  AMDAIE::ConnectionOp connectionOp = currBdIdPair.first.second;
  if (connectionOps.contains(connectionOp) || connectionOps.empty())
    return false;

  // Packet flow? Can't fold.
  std::optional<AMDAIE::FlowOp> maybeFlowOp = connectionOp.getFlowOp();
  if (!maybeFlowOp) {
    return connectionOp.emitOpError()
           << "expected to operate on an `amdaie.flow`";
  }
  AMDAIE::FlowOp flowOp = maybeFlowOp.value();
  if (flowOp.getIsPacketFlow()) return false;

  // Duplicate BD ID on the same tile? Can't fold.
  AMDAIE::TileOp tileOp = currBdIdPair.first.first;
  uint32_t currBdIdVal = currBdIdPair.second;
  bool isDuplicateBdId = llvm::any_of(dmaBdIdsMap, [&](const auto &entry) {
    return entry.first.first == tileOp && entry.second.contains(currBdIdVal);
  });
  if (isDuplicateBdId) return false;

  // Can fold.
  return true;
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

  auto updateWithCurrBdId =
      [&](bool canFold, DmaBdIdPair currBdIdPair,
          DenseSet<AMDAIE::ConnectionOp> &connectionOps,
          DenseMap<DmaBdIdKey, DenseSet<uint32_t>> &dmaBdIdsMap) {
        DmaBdIdKey currBdIdKey = currBdIdPair.first;
        uint32_t currBdIdVal = currBdIdPair.second;
        if (!canFold) {
          // Clear the BD IDs for all the connections in the batch.
          for (auto &entry : dmaBdIdsMap) {
            ConnectionOp connectionOp = entry.first.second;
            DenseSet<uint32_t> &bdIds = entry.second;
            if (connectionOps.contains(connectionOp)) bdIds.clear();
          }
          connectionOps.clear();
        }
        connectionOps.insert(currBdIdKey.second);
        dmaBdIdsMap[currBdIdKey].insert(currBdIdVal);
      };

  // Traverse the control code in reverse.
  WalkResult res = controlCodeOp->walk<WalkOrder::PostOrder, ReverseIterator>(
      [&](AMDAIE::NpuDmaWaitOp waitOp) {
        bool toBatch = true;
        Operation *batchParentOp =
            waitOps.empty() ? waitOp->getParentOp() : waitOps[0]->getParentOp();
        for (Value token : waitOp.getAsyncTokens()) {
          if (auto npuHalfDmaCpyNdOp =
                  dyn_cast_if_present<AMDAIE::NpuHalfDmaCpyNdOp>(
                      token.getDefiningOp())) {
            // Retrieve the TileOp, ConnectionOp, and BD ID.
            FailureOr<DmaBdIdPair> currBdIdPair =
                retrieveDmaBdIdPair(npuHalfDmaCpyNdOp);
            if (failed(currBdIdPair)) return WalkResult::interrupt();
            // Check if the current DMA wait op can be folded into the batch.
            FailureOr<bool> canFold =
                canFoldByBatch(batchParentOp, connectionOps, dmaBdIdsMap,
                               npuHalfDmaCpyNdOp, *currBdIdPair);
            if (failed(canFold)) return WalkResult::interrupt();
            // Update the `connectionOps` and `dmaBdIdsMap`.
            updateWithCurrBdId(*canFold, *currBdIdPair, connectionOps,
                               dmaBdIdsMap);
            toBatch &= *canFold;
          }
        }
        // Process the previous batch of wait ops, and start a new batch.
        if (!toBatch) {
          // Since the controlcode is traversed in reverse order, we need to
          // restore the original order of the DMA operations.
          std::reverse(waitOps.begin(), waitOps.end());
          if (failed(eraseBatchOperations(rewriter, waitOps)))
            return WalkResult::interrupt();
          waitOps.clear();
        }
        waitOps.push_back(waitOp);
        return WalkResult::advance();
      });

  if (res.wasInterrupted()) return failure();
  // Process the remaining wait ops.
  std::reverse(waitOps.begin(), waitOps.end());
  if (failed(eraseBatchOperations(rewriter, waitOps))) return failure();
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
