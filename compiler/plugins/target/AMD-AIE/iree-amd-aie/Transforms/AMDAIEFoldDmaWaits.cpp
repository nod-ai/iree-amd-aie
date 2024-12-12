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

/// Utility function to determine whether a DMA wait op can be folded based on
/// its half DMA copy operation.
FailureOr<bool> canFoldByConnection(
    const AMDAIE::AMDAIEDeviceModel &deviceModel,
    AMDAIE::NpuHalfDmaCpyNdOp &npuHalfDmaCpyNdOp,
    DenseMap<std::pair<AMDAIE::TileOp, AMDAIE::ConnectionOp>,
             SmallVector<uint32_t>> &tileConnectToBdIdQueue) {
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
LogicalResult foldDmaWaitsByConnection(
    const AMDAIE::AMDAIEDeviceModel &deviceModel,
    AMDAIE::ControlCodeOp controlCodeOp) {
  IRRewriter rewriter(controlCodeOp->getContext());
  std::vector<AMDAIE::NpuDmaWaitOp> waitOpsToErase;
  DenseMap<std::pair<AMDAIE::TileOp, AMDAIE::ConnectionOp>,
           SmallVector<uint32_t>>
      tileConnectToBdIdQueue;
  // Traverse the control code in reverse.
  WalkResult res = controlCodeOp->walk<WalkOrder::PostOrder, ReverseIterator>(
      [&](AMDAIE::NpuDmaWaitOp waitOp) {
        bool toErase = true;
        for (Value token : waitOp.getAsyncTokens()) {
          if (auto npuHalfDmaCpyNdOp =
                  dyn_cast_if_present<AMDAIE::NpuHalfDmaCpyNdOp>(
                      token.getDefiningOp())) {
            FailureOr<bool> result = canFoldByConnection(
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

struct DmaColumnBatch {
  uint32_t row;
  uint32_t channel;
  AMDAIE::DMAChannelDir direction;

  // Sorted by column.
  std::map<uint32_t, AMDAIE::NpuDmaWaitOp> colWaitOpMap;
};

/// Updates a batch of asynchronous DMA wait operations by combining their
/// async tokens into a single NpuDmaWaitOp.
void updateColumnBatchTokens(
    IRRewriter &rewriter,
    std::map<uint32_t, AMDAIE::NpuDmaWaitOp> &colWaitOpMap) {
  if (colWaitOpMap.size() < 2) return;

  // Check if there is any discontinuity in the columns, and if so, split into
  // separate batches.
  SmallVector<SmallVector<AMDAIE::NpuDmaWaitOp>> waitOpsList;
  uint32_t prevCol = 0;
  for (auto &entry : colWaitOpMap) {
    uint32_t col = entry.first;
    AMDAIE::NpuDmaWaitOp waitOp = entry.second;
    if (waitOpsList.empty() || col != prevCol + 1) {
      waitOpsList.push_back({});
    }
    waitOpsList.back().push_back(waitOp);
    prevCol = col;
  }

  for (SmallVector<AMDAIE::NpuDmaWaitOp> &waitOps : waitOpsList) {
    // For each batch, combine the async tokens into a single NpuDmaWaitOp.
    SmallVector<Value> asyncTokens;
    for (AMDAIE::NpuDmaWaitOp waitOp : waitOps) {
      asyncTokens.append(waitOp.getAsyncTokens().begin(),
                         waitOp.getAsyncTokens().end());
    }
    rewriter.setInsertionPointAfter(waitOps.back());
    rewriter.create<AMDAIE::NpuDmaWaitOp>(waitOps.back().getLoc(), asyncTokens);
    for (AMDAIE::NpuDmaWaitOp waitOp : waitOps) {
      rewriter.eraseOp(waitOp);
    }
  }
}

/// Utility function to determine if a DMA wait operation can be folded.
/// This is achieved by verifying whether it shares the same row, channel,
/// and direction with preceding wait operations.
LogicalResult foldByColumn(IRRewriter &rewriter, DmaColumnBatch &dmaBatch,
                           AMDAIE::NpuHalfDmaCpyNdOp dmaOp,
                           AMDAIE::NpuDmaWaitOp waitOp) {
  // Get the row and column.
  std::optional<AMDAIE::BdIdOp> maybeBdIdOp = dmaOp.getBdIdOp();
  if (!maybeBdIdOp) return dmaOp.emitOpError() << "must have a BD ID op";
  AMDAIE::BdIdOp bdIdOp = maybeBdIdOp.value();
  AMDAIE::TileOp tileOp =
      dyn_cast_if_present<AMDAIE::TileOp>(bdIdOp.getTile().getDefiningOp());
  if (!tileOp)
    return bdIdOp.emitOpError() << "must operate on an `amdaie.tile`";
  uint32_t col = getConstantIndexOrAssert(tileOp.getCol());
  uint32_t row = getConstantIndexOrAssert(tileOp.getRow());

  // Get the channel.
  std::optional<AMDAIE::ChannelOp> maybeChannelOp = dmaOp.getChannelOp();
  if (!maybeChannelOp)
    return dmaOp.emitOpError() << "found non-`amdaie.channel` channel";
  AMDAIE::ChannelOp channelOp = maybeChannelOp.value();
  std::optional<AMDAIE::DMAChannelDir> maybeDirection =
      channelOp.getDirection();
  std::optional<uint32_t> maybeChannel = channelOp.getValue();
  if (!maybeDirection || !maybeChannel)
    return channelOp.emitOpError() << "direction and channel needed";
  AMDAIE::DMAChannelDir direction = maybeDirection.value();
  uint32_t channel = maybeChannel.value();

  if (dmaBatch.colWaitOpMap.empty() || row != dmaBatch.row ||
      channel != dmaBatch.channel || direction != dmaBatch.direction) {
    updateColumnBatchTokens(rewriter, dmaBatch.colWaitOpMap);
    dmaBatch = {row, channel, direction, {}};
  }
  dmaBatch.colWaitOpMap[col] = waitOp;
  return success();
}

/// Traverses the control code forward, ensuring that only one DMA wait op is
/// retained for all the columns.
///
/// Example Input:
///   %0 = dma_cpy_nd(col=0)
///   %1 = dma_cpy_nd(col=1)
///   %2 = dma_cpy_nd(col=2)
///   %3 = dma_cpy_nd(col=3)
///   dma_wait(%0)
///   dma_wait(%1)
///   dma_wait(%2)
///   dma_wait(%3)
/// Example Output:
///   %0 = dma_cpy_nd(col=0)
///   %1 = dma_cpy_nd(col=1)
///   %2 = dma_cpy_nd(col=2)
///   %3 = dma_cpy_nd(col=3)
///   dma_wait(%0, %1, %2, %3)
LogicalResult foldDmaWaitsByColumn(const AMDAIE::AMDAIEDeviceModel &deviceModel,
                                   AMDAIE::ControlCodeOp controlCodeOp) {
  IRRewriter rewriter(controlCodeOp->getContext());
  DmaColumnBatch dmaBatch = {};

  WalkResult res = controlCodeOp->walk([&](Operation *op) {
    auto waitOp = dyn_cast<AMDAIE::NpuDmaWaitOp>(op);
    // Skip if not a DMA wait op or if it already has multiple async tokens.
    if (!waitOp || waitOp.getAsyncTokens().size() != 1) {
      updateColumnBatchTokens(rewriter, dmaBatch.colWaitOpMap);
      dmaBatch.colWaitOpMap.clear();
      return WalkResult::advance();
    }

    // Get the half DMA copy operation.
    Value token = waitOp.getAsyncTokens().front();
    auto npuHalfDmaCpyNdOp =
        dyn_cast_if_present<AMDAIE::NpuHalfDmaCpyNdOp>(token.getDefiningOp());
    if (!npuHalfDmaCpyNdOp) {
      waitOp.emitOpError() << "expected to operate on an "
                              "`amdaie.npu.half_dma_cpy_nd`";
      return WalkResult::interrupt();
    }

    // Check if the DMA wait op can be folded into the column batch.
    if (succeeded(
            foldByColumn(rewriter, dmaBatch, npuHalfDmaCpyNdOp, waitOp))) {
      return WalkResult::advance();
    } else {
      return WalkResult::interrupt();
    }
  });

  // Process the remaining wait ops.
  updateColumnBatchTokens(rewriter, dmaBatch.colWaitOpMap);
  if (res.wasInterrupted()) return failure();
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
    if (failed(foldDmaWaitsByConnection(deviceModel, controlCodeOp))) {
      return WalkResult::interrupt();
    }
    if (failed(foldDmaWaitsByColumn(deviceModel, controlCodeOp))) {
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
