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

/// Traverses the control code in reverse, ensuring that for each connection,
/// only one DMA wait op is retained for every maximum queue size.
LogicalResult foldDmaWaits(const AMDAIE::AMDAIEDeviceModel &deviceModel,
                           AMDAIE::ControlCodeOp controlCodeOp) {
  IRRewriter rewriter(controlCodeOp->getContext());
  std::vector<AMDAIE::NpuDmaWaitOp> waitOpsToErase;
  DenseMap<std::pair<AMDAIE::TileOp, AMDAIE::ConnectionOp>,
           SmallVector<uint32_t>>
      tileConnectionToBdIdQueueMap;
  // Traverse the control code in reverse.
  WalkResult res = controlCodeOp->walk<WalkOrder::PostOrder, ReverseIterator>(
      [&](AMDAIE::NpuDmaWaitOp waitOp) {
        bool toErase = true;
        for (Value token : waitOp.getAsyncTokens()) {
          if (auto npuHalfDmaCpyNdOp =
                  dyn_cast_if_present<AMDAIE::NpuHalfDmaCpyNdOp>(
                      token.getDefiningOp())) {
            // Retrieve the connection op.
            std::optional<AMDAIE::ConnectionOp> maybeConnectionOp =
                npuHalfDmaCpyNdOp.getConnectionOp();
            if (!maybeConnectionOp) {
              npuHalfDmaCpyNdOp.emitOpError()
                  << "expected to operate on an `amdaie.connection`";
              return WalkResult::interrupt();
            }
            AMDAIE::ConnectionOp connectionOp = maybeConnectionOp.value();
            // Retrieve the flow op.
            std::optional<AMDAIE::FlowOp> maybeFlowOp =
                maybeConnectionOp->getFlowOp();
            if (!maybeFlowOp) {
              maybeConnectionOp->emitOpError()
                  << "expected to operate on an `amdaie.flow`";
              return WalkResult::interrupt();
            }
            bool isPacketFlow = maybeFlowOp->getIsPacketFlow();
            // Retrieve the BD ID op.
            std::optional<AMDAIE::BdIdOp> maybeBdIdOp =
                npuHalfDmaCpyNdOp.getBdIdOp();
            if (!maybeBdIdOp) {
              npuHalfDmaCpyNdOp.emitOpError()
                  << "must have a BD ID op to lower to "
                     "`amdaie.npu.write_bd`";
              return WalkResult::interrupt();
            }
            AMDAIE::BdIdOp bdIdOp = maybeBdIdOp.value();
            // Retrieve the tile op.
            AMDAIE::TileOp tileOp = dyn_cast_if_present<AMDAIE::TileOp>(
                bdIdOp.getTile().getDefiningOp());
            if (!tileOp) {
              bdIdOp.emitOpError() << "must operate on an `amdaie.tile`";
              return WalkResult::interrupt();
            }
            // Get the maximum queue size.
            uint32_t col = getConstantIndexOrAssert(tileOp.getCol());
            uint32_t row = getConstantIndexOrAssert(tileOp.getRow());
            uint32_t maxQueueSize = deviceModel.getDmaMaxQueueSize(col, row);
            // Keep wait op if, either reaches the maximum queue size, or there
            // is a duplicate BD ID in the same tile.
            uint32_t bdId = getConstantIndexOrAssert(bdIdOp.getValue());
            bool isDuplicateBdId = llvm::any_of(
                tileConnectionToBdIdQueueMap, [&](const auto &entry) {
                  return entry.first.first == tileOp &&
                         llvm::is_contained(entry.second, bdId);
                });
            SmallVector<uint32_t> &bdIdQueue =
                tileConnectionToBdIdQueueMap[std::make_pair(tileOp,
                                                            connectionOp)];
            if (isDuplicateBdId || isPacketFlow ||
                bdIdQueue.size() >= maxQueueSize) {
              bdIdQueue.clear();
            }
            if (bdIdQueue.empty()) toErase = false;
            bdIdQueue.push_back(bdId);
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
