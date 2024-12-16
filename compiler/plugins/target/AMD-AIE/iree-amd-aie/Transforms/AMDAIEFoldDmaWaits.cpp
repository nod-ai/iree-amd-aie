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

using DmaQueue = std::pair<AMDAIE::TileOp, AMDAIE::ConnectionOp>;

/// Utility function to determine whether a DMA wait op can be folded into a
/// queue based on its half DMA copy operation.
FailureOr<bool> canFoldByQueue(
    const AMDAIE::AMDAIEDeviceModel &deviceModel,
    AMDAIE::NpuHalfDmaCpyNdOp &npuHalfDmaCpyNdOp,
    DenseMap<DmaQueue, SmallVector<uint32_t>> &dmaQueueToBdIds) {
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

  // Keep wait op if, either reaches the maximum queue size, or a
  // duplicate BD ID in the same tile, or packet flow, or the queue is
  // empty
  uint32_t bdId = getConstantIndexOrAssert(bdIdOp.getValue());
  bool isDuplicateBdId = llvm::any_of(dmaQueueToBdIds, [&](const auto &entry) {
    return entry.first.first == tileOp &&
           llvm::is_contained(entry.second, bdId);
  });
  SmallVector<uint32_t> &bdIds = dmaQueueToBdIds[{tileOp, connectionOp}];
  bool canFold = true;
  if (isDuplicateBdId || isPacketFlow || bdIds.size() >= maxQueueSize ||
      bdIds.empty()) {
    bdIds.clear();
    canFold = false;
  }
  bdIds.push_back(bdId);
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
  std::vector<AMDAIE::NpuDmaWaitOp> waitOpsToErase;
  DenseMap<DmaQueue, SmallVector<uint32_t>> dmaQueueToBdIds;
  // Traverse the control code in reverse.
  WalkResult res = controlCodeOp->walk<WalkOrder::PostOrder, ReverseIterator>(
      [&](AMDAIE::NpuDmaWaitOp waitOp) {
        bool toErase = true;
        for (Value token : waitOp.getAsyncTokens()) {
          if (auto npuHalfDmaCpyNdOp =
                  dyn_cast_if_present<AMDAIE::NpuHalfDmaCpyNdOp>(
                      token.getDefiningOp())) {
            FailureOr<bool> result =
                canFoldByQueue(deviceModel, npuHalfDmaCpyNdOp, dmaQueueToBdIds);
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
  for (AMDAIE::NpuDmaWaitOp waitOp : waitOps) {
    rewriter.eraseOp(waitOp);
  }
  return success();
}

/// Utility function to determine if a DMA wait operation can be folded into a
/// a batch based on its half DMA copy operation.
FailureOr<bool> canFoldByBatch(
    AMDAIE::NpuHalfDmaCpyNdOp npuHalfDmaCpyNdOp,
    SmallVector<AMDAIE::ConnectionOp> &connectionOps) {
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

  bool canFold = true;
  // Can't fold if the current connection op already occurs in the batch, or
  // if the current operation is a packet flow, or if the batch is empty.
  if (llvm::is_contained(connectionOps, connectionOp) || isPacketFlow ||
      connectionOps.empty()) {
    connectionOps.clear();
    canFold = false;
  }
  connectionOps.push_back(connectionOp);
  return canFold;
}

/// Traverses the control code forward, ensuring that only one DMA wait op is
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
LogicalResult foldDmaWaitsByBatch(AMDAIE::ControlCodeOp controlCodeOp) {
  IRRewriter rewriter(controlCodeOp->getContext());
  SmallVector<AMDAIE::NpuDmaWaitOp> waitOps;
  SmallVector<AMDAIE::ConnectionOp> connectionOps;
  WalkResult res = controlCodeOp->walk([&](AMDAIE::NpuDmaWaitOp waitOp) {
    bool toBatch = true;
    for (Value token : waitOp.getAsyncTokens()) {
      if (auto npuHalfDmaCpyNdOp =
              dyn_cast_if_present<AMDAIE::NpuHalfDmaCpyNdOp>(
                  token.getDefiningOp())) {
        FailureOr<bool> result =
            canFoldByBatch(npuHalfDmaCpyNdOp, connectionOps);
        if (failed(result)) return WalkResult::interrupt();
        toBatch &= *result;
      }
    }
    // Process the previous batch of wait ops, and start a new batch.
    if (!toBatch) {
      if (failed(updateBatchTokens(rewriter, waitOps)))
        return WalkResult::interrupt();
      waitOps.clear();
    }
    waitOps.push_back(waitOp);
    return WalkResult::advance();
  });

  if (res.wasInterrupted()) return failure();
  // Process the remaining wait ops.
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
