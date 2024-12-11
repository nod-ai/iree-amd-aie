// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEDmaUtils.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/Utils/ChannelBdIdGenerator.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/IR/Iterators.h"
#define DEBUG_TYPE "iree-amdaie-insert-dma-bd-chain"

namespace mlir::iree_compiler::AMDAIE {

namespace {

using DmaChain = std::pair<AMDAIE::TileOp, AMDAIE::ConnectionOp>;

/// Utility function to update `next_bd` and `start_bd` operands.
LogicalResult updateChainOperands(
    IRRewriter &rewriter, SmallVector<AMDAIE::NpuHalfDmaCpyNdOp> &dmaOps) {
  // Nothing to do if the DMA chain length is one or less.
  if (dmaOps.size() < 2) return success();

  Value startBdId = dmaOps[0].getBdId();
  Operation *parentOp = dmaOps[0]->getParentOp();
  // Chain the DMA ops.
  for (unsigned i = 0; i < dmaOps.size() - 1; ++i) {
    AMDAIE::NpuHalfDmaCpyNdOp currDmaOp = dmaOps[i];
    if (currDmaOp->getParentOp() != parentOp) {
      return currDmaOp.emitError(
          "DMA operations to be chained must belong to the same scope");
    }
    Value nextBdId = dmaOps[i + 1].getBdId();
    // No token is produced at the beginning or middle of a chain.
    TypeRange token = TypeRange{};
    rewriter.setInsertionPointAfter(currDmaOp);
    rewriter.create<AMDAIE::NpuHalfDmaCpyNdOp>(
        currDmaOp.getLoc(), token, currDmaOp.getConnection(),
        currDmaOp.getInput(), currDmaOp.getMixedOffsets(),
        currDmaOp.getMixedSizes(), currDmaOp.getMixedStrides(),
        currDmaOp.getBdId(), currDmaOp.getChannel(), nextBdId, startBdId);
    for (auto &use : currDmaOp->getUses()) rewriter.eraseOp(use.getOwner());
    rewriter.eraseOp(currDmaOp);
  }
  // Last DMA op in the chain.
  AMDAIE::NpuHalfDmaCpyNdOp lastDmaOp = dmaOps.back();
  if (lastDmaOp->getParentOp() != parentOp) {
    return lastDmaOp.emitError(
        "DMA operations to be chained must belong to the same scope");
  }
  Value nextBdId = nullptr;
  rewriter.setInsertionPointAfter(lastDmaOp);
  auto lastDmaOpChained = rewriter.create<AMDAIE::NpuHalfDmaCpyNdOp>(
      lastDmaOp.getLoc(), lastDmaOp.getResultTypes(), lastDmaOp.getConnection(),
      lastDmaOp.getInput(), lastDmaOp.getMixedOffsets(),
      lastDmaOp.getMixedSizes(), lastDmaOp.getMixedStrides(),
      lastDmaOp.getBdId(), lastDmaOp.getChannel(), nextBdId, startBdId);
  rewriter.replaceOp(lastDmaOp, lastDmaOpChained.getResults());
  return success();
}

/// Utility function to determine if chains can grow further
/// or require breaking.
///
/// Example:
/// - Chain X currently holds BD IDs: [4, 5, 6, 7]
/// - Chain Y currently holds BD IDs: [0, 1, 2, 3]
/// - A new BD ID (0) needs to be added to the front (due to reverse
/// traversing) of chain X.
///
/// Conflict resolution:
/// - Chain Y must be broken because BD ID 0 is already assigned to it
/// and must be released.
/// - Chain X is also broken to prevent the new added BD ID (0) from
/// invalidating chain Y.
///
/// Result:
/// - Break both chains X and Y.
///   - Chain X: [0] (the newly added BD ID).
///   - Chain Y: [] (emptied after breaking).
void checkForChainsToBeBroken(
    const uint32_t &currbdId, const DmaChain &currDmaChain,
    const DenseMap<DmaChain, DenseSet<uint32_t>> &dmaChainToBdIds,
    SmallVector<DmaChain> &chainsToBreak) {
  for (auto &[entry, bdIds] : dmaChainToBdIds) {
    if (entry.first == currDmaChain.first && bdIds.contains(currbdId)) {
      // Break the chain that contains the duplicate BD ID.
      chainsToBreak.push_back(entry);
      if (entry != currDmaChain) {
        // Break the current chain as well.
        chainsToBreak.push_back(currDmaChain);
      }
      break;
    }
  }
}

/// Traverse the control code in reverse order to create DMA BD chains. Reverse
/// traversal simplifies handling duplicate BD IDs, preventing the need to
/// revisit and modify earlier operations after processing later ones.
LogicalResult insertDmaBdChain(const AMDAIE::AMDAIEDeviceModel &deviceModel,
                               AMDAIE::ControlCodeOp controlCodeOp) {
  IRRewriter rewriter(controlCodeOp->getContext());

  // Move all BdIdOps to the beginning of the control code.
  // This is to avoid dominance issues when chaining BD IDs.
  SmallVector<Operation *> bdIdOps;
  WalkResult res = controlCodeOp->walk([&](Operation *op) {
    if (auto bdIdOp = dyn_cast<AMDAIE::BdIdOp>(op)) {
      bdIdOps.push_back(op);
    }
    return WalkResult::advance();
  });
  for (Operation *op : llvm::reverse(bdIdOps)) {
    op->moveBefore(&controlCodeOp.front());
  }

  // BD IDs that have been assigned in each tile.
  DenseMap<DmaChain, DenseSet<uint32_t>> dmaChainToBdIds;
  // Buffers the DMA ops that will be chained.
  DenseMap<DmaChain, SmallVector<AMDAIE::NpuHalfDmaCpyNdOp>> dmaChainToDmaOps;

  res = controlCodeOp->walk<WalkOrder::PostOrder,
                            ReverseIterator>([&](Operation *op) {
    if (auto npuHalfDmaCpyNdOp = dyn_cast<AMDAIE::NpuHalfDmaCpyNdOp>(op)) {
      // Not shim, will be erased at ControlcodeLowering, ignore.
      if (npuHalfDmaCpyNdOp.getMemorySpaceAsUInt() != 0) {
        return WalkResult::advance();
      }

      // Get the connection op.
      std::optional<AMDAIE::ConnectionOp> maybeConnectionOp =
          npuHalfDmaCpyNdOp.getConnectionOp();
      if (!maybeConnectionOp) {
        npuHalfDmaCpyNdOp.emitOpError()
            << "expected to operate on an `amdaie.connection`";
        return WalkResult::interrupt();
      }
      AMDAIE::ConnectionOp connectionOp = maybeConnectionOp.value();

      // Packet flow, do not chain BDs.
      std::optional<AMDAIE::FlowOp> maybeFlowOp = connectionOp.getFlowOp();
      if (!maybeFlowOp) {
        connectionOp->emitOpError()
            << "expected to operate on an `amdaie.flow`";
        return WalkResult::interrupt();
      }
      AMDAIE::FlowOp flowOp = maybeFlowOp.value();
      bool isPacketFlow = flowOp.getIsPacketFlow();
      if (isPacketFlow) return WalkResult::advance();

      // Repeat count > 1, do not chain BDs.
      int32_t repeatCount = 1;
      uint8_t numIntraAddrDim = deviceModel.getDmaProp<uint8_t>(
          AMDAIE::AMDAIETileType::SHIMNOC, AMDAIE::AMDAIEDmaProp::NumAddrDim);
      uint8_t numAddrDim =
          numIntraAddrDim + deviceModel.deviceConfig.dmaNbInterDims;
      auto sizes = npuHalfDmaCpyNdOp.getMixedSizes();
      auto strides = npuHalfDmaCpyNdOp.getMixedStrides();
      if (!sizes.empty() && !strides.empty()) {
        int64_t size = getConstantIndexOrAssert(sizes[0]);
        int64_t stride = getConstantIndexOrAssert(strides[0]);
        if (sizes.size() == numAddrDim || stride == 0) {
          repeatCount = size;
        }
      }
      if (repeatCount > 1) return WalkResult::advance();

      // Get the BD ID and tile op.
      std::optional<AMDAIE::BdIdOp> maybeBdIdOp = npuHalfDmaCpyNdOp.getBdIdOp();
      if (!maybeBdIdOp) {
        npuHalfDmaCpyNdOp.emitOpError() << "must have a BD ID op";
        return WalkResult::interrupt();
      }
      AMDAIE::BdIdOp bdIdOp = maybeBdIdOp.value();
      uint32_t bdId = getConstantIndexOrAssert(bdIdOp.getValue());
      AMDAIE::TileOp tileOp =
          dyn_cast_if_present<AMDAIE::TileOp>(bdIdOp.getTile().getDefiningOp());
      if (!tileOp) {
        bdIdOp.emitOpError() << "must operate on an `amdaie.tile`";
        return WalkResult::interrupt();
      }

      // Any duplicate BD ID from the same tile indicates that the chain
      // cannot grow further and requires breaking to release the
      // conflicting BD ID.
      SmallVector<DmaChain> chainsToBreak;
      DmaChain currDmaChain = {tileOp, connectionOp};
      checkForChainsToBeBroken(bdId, currDmaChain, dmaChainToBdIds,
                               chainsToBreak);

      // If the chains are not to be continued, update DMA operands using
      // the `updateChainOperands` function.
      if (!chainsToBreak.empty()) {
        for (auto &entry : chainsToBreak) {
          if (failed(updateChainOperands(rewriter, dmaChainToDmaOps[entry])))
            WalkResult::interrupt();
          dmaChainToBdIds[entry].clear();
          dmaChainToDmaOps[entry].clear();
        }
      }

      dmaChainToBdIds[currDmaChain].insert(bdId);
      // Insert at the front, as we are walking in reverse order.
      dmaChainToDmaOps[currDmaChain].insert(
          dmaChainToDmaOps[currDmaChain].begin(), npuHalfDmaCpyNdOp);
    }
    return WalkResult::advance();
  });

  // Build the remaining chains.
  for (auto &[entry, _] : dmaChainToBdIds) {
    if (failed(updateChainOperands(rewriter, dmaChainToDmaOps[entry])))
      return failure();
  }

  if (res.wasInterrupted()) return failure();
  return success();
}

class AMDAIEInsertDmaBdChainPass
    : public impl::AMDAIEInsertDmaBdChainBase<AMDAIEInsertDmaBdChainPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEInsertDmaBdChainPass() = default;
  AMDAIEInsertDmaBdChainPass(const AMDAIEInsertDmaBdChainPass &pass){};
  void runOnOperation() override;
};

void AMDAIEInsertDmaBdChainPass::runOnOperation() {
  Operation *parentOp = getOperation();

  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(parentOp);
  std::optional<AMDAIEDevice> maybeDevice = getConfigAMDAIEDevice(targetAttr);
  if (!maybeDevice) {
    parentOp->emitOpError()
        << "has no AMDAIEDevice in the target attribute configuration. This "
           "device-specific information is required to lower control code "
           "ops.";
    return signalPassFailure();
  }
  AMDAIE::AMDAIEDeviceModel deviceModel =
      AMDAIE::getDeviceModel(maybeDevice.value());

  WalkResult res = parentOp->walk([&](AMDAIE::WorkgroupOp workgroupOp) {
    AMDAIE::ControlCodeOp controlCodeOp = workgroupOp.getControlCode();
    if (failed(insertDmaBdChain(deviceModel, controlCodeOp))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEInsertDmaBdChainPass() {
  return std::make_unique<AMDAIEInsertDmaBdChainPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
