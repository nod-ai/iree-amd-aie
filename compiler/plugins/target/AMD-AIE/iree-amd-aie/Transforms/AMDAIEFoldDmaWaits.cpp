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
FailureOr<bool> canFoldBasedOnHalfDmaCpy(
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
LogicalResult foldDmaWaits(const AMDAIE::AMDAIEDeviceModel &deviceModel,
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
            FailureOr<bool> result = canFoldBasedOnHalfDmaCpy(
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
              op.getBdId(), op.getChannel());
          rewriter.eraseOp(op);
        }
      }
    }
  }

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
    if (failed(foldDmaWaits(deviceModel, controlCodeOp))) {
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
