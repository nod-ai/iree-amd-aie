// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"

#define DEBUG_TYPE "iree-amdaie-insert-dma-out-of-order-block"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Find the MM2S DMA start op corresponding to the given connection.
FailureOr<AMDAIE::DMAStartOp> getMm2sDmaStart(
    AMDAIE::ConnectionOp connectionOp,
    DenseMap<AMDAIE::ChannelOp, SmallVector<AMDAIE::DMAStartOp>>
        &mm2sChannelToDmaStarts) {
  SmallVector<Value> srcChannels = connectionOp.getSourceChannels();
  if (srcChannels.size() != 1) {
    return connectionOp.emitOpError()
           << "expected exactly one source channel for connection op";
  }
  auto channelOp = dyn_cast<AMDAIE::ChannelOp>(srcChannels[0].getDefiningOp());
  if (!channelOp)
    return connectionOp.emitOpError() << "expected an `amdaie.channel` op";
  if (mm2sChannelToDmaStarts.count(channelOp) == 0) {
    return connectionOp.emitOpError()
           << "source channel does not have corresponding MM2S DMA start op";
  }
  if (mm2sChannelToDmaStarts[channelOp].size() != 1) {
    return connectionOp.emitOpError()
           << "expected exactly one MM2S DMA start op";
  }
  return mm2sChannelToDmaStarts[channelOp][0];
}

/// For the given S2MM channel, merge all the corresponding MM2S DMA start ops
/// into a single out-of-order block.
LogicalResult insertDmaOutOfOrder(
    const AMDAIE::AMDAIEDeviceModel &deviceModel, IRRewriter &rewriter,
    AMDAIE::ChannelOp s2mmChannelOp,
    SmallVectorImpl<AMDAIE::DMAStartOp> &s2mmDmaStartOps,
    DenseMap<AMDAIE::ChannelOp, SmallVector<AMDAIE::DMAStartOp>>
        &mm2sChannelToDmaStarts) {
  IRRewriter::InsertionGuard insertGuard(rewriter);
  // If there is only one DMA start op, do nothing.
  if (s2mmDmaStartOps.size() <= 1) return success();
  // Merge the given DMA start ops into a single out-of-order block.
  uint32_t newRepeatCount = 0;
  SmallVector<Value> newConnections;
  SmallVector<SmallVector<Operation *>> lockAndBdOps;
  for (AMDAIE::DMAStartOp s2mmDmaStartOp : s2mmDmaStartOps) {
    // Reject the merge if the DMA start op already has more than one connection
    // or if it is not packet-switched.
    SmallVector<Value> connections = s2mmDmaStartOp.getConnections();
    newConnections.append(connections.begin(), connections.end());
    if (connections.size() != 1) {
      return s2mmDmaStartOp.emitOpError()
             << "expected exactly one connection for DMA start op, but found "
             << connections.size() << "\n";
    }
    AMDAIE::ConnectionOp connectionOp =
        dyn_cast<AMDAIE::ConnectionOp>(connections[0].getDefiningOp());
    if (connectionOp.getConnectionType() != AMDAIE::ConnectionType::Packet) {
      return s2mmDmaStartOp.emitOpError()
             << "out-of-order mode requires DMA to be packet-switched\n";
    }
    // Find the corresponding MM2S DMA start op.
    // TODO(zhewen): Add support for npu.write_bd when the MM2S channel is
    // located in the shim tile.
    FailureOr<AMDAIE::DMAStartOp> maybeMm2sDmaStartOp =
        getMm2sDmaStart(connectionOp, mm2sChannelToDmaStarts);
    if (failed(maybeMm2sDmaStartOp)) {
      return s2mmDmaStartOp.emitOpError()
             << "could not find corresponding MM2S DMA start op";
    }
    // Get the blocks in the S2MM and MM2S DMA start ops.
    SmallVector<Block *> s2mmBlocks = llvm::to_vector(llvm::map_range(
        s2mmDmaStartOp.getBody().getBlocks(), [](Block &b) { return &b; }));
    SmallVector<Block *> mm2sBlocks = llvm::to_vector(
        llvm::map_range(maybeMm2sDmaStartOp.value().getBody().getBlocks(),
                        [](Block &b) { return &b; }));
    if (s2mmBlocks.size() != mm2sBlocks.size()) {
      return s2mmDmaStartOp.emitOpError()
             << "expected the same number of blocks in S2MM and MM2S DMA start "
                "ops";
    }
    // Pass the S2MM BD ID to the corresponding MM2S, and configure it in MM2S
    // packet header.
    uint32_t numBDs = 0;
    for (auto &&[s2mmBlock, mm2sBlock] : llvm::zip(s2mmBlocks, mm2sBlocks)) {
      lockAndBdOps.push_back({});
      std::optional<uint32_t> maybeS2mmBdId;
      for (Operation &op : *s2mmBlock) {
        // Find the DMA BDs and locks in the S2MM block.
        if (isa<AMDAIE::DMABDOp, AMDAIE::UseLockOp>(op))
          lockAndBdOps.back().push_back(&op);
        if (auto bdOp = dyn_cast<AMDAIE::DMABDOp>(op)) {
          numBDs++;
          maybeS2mmBdId = bdOp.getBdId();
        }
      }
      // No DMABDOps are found, continue to the next block.
      if (lockAndBdOps.back().size() == 0) {
        lockAndBdOps.pop_back();
        continue;
      }
      if (!maybeS2mmBdId.has_value()) {
        return s2mmDmaStartOp.emitOpError()
               << "BD ID is not assigned in S2MM block";
      }
      // Embed the S2MM BD ID in the corresponding MM2S packer header.
      auto maybeMm2sDmaBdPacketOps = mm2sBlock->getOps<AMDAIE::DmaBdPacketOp>();
      if (llvm::range_size(maybeMm2sDmaBdPacketOps) != 1) {
        return maybeMm2sDmaStartOp.value().emitOpError()
               << "expected exactly one `amdaie.dma_bd_packet` op in each MM2S "
                  "block";
      }
      AMDAIE::DmaBdPacketOp mm2sDmaBdPacketOp =
          *maybeMm2sDmaBdPacketOps.begin();
      mm2sDmaBdPacketOp.setOutOfOrderBdId(maybeS2mmBdId.value());
    }
    // For out-of-order mode, repeat count should represent the total number
    // of transmissions across all BDs.
    newRepeatCount += s2mmDmaStartOp.getRepeatCount() * numBDs;
  }
  // Check the new repeat count is within the max repeat count.
  AMDAIE::TileOp tileOp = s2mmChannelOp.getTileOp();
  uint32_t col = getConstantIndexOrAssert(tileOp.getCol());
  uint32_t row = getConstantIndexOrAssert(tileOp.getRow());
  AMDAIE::AMDAIETileType tileType = deviceModel.getTileType(col, row);
  uint32_t maxRepeatCount = deviceModel.getMaxRepeatCount(tileType);
  if (newRepeatCount > maxRepeatCount) {
    return s2mmDmaStartOps[0]->emitOpError()
           << "The maximum repeat count is " << maxRepeatCount << ", but got "
           << newRepeatCount;
  }
  // Create a new DMA start op with the merged connections and BDs.
  rewriter.setInsertionPoint(s2mmDmaStartOps[0]);
  auto newS2mmDmaStartOp = rewriter.create<AMDAIE::DMAStartOp>(
      rewriter.getUnknownLoc(), s2mmChannelOp, newConnections, newRepeatCount,
      /*enableOutOfOrder=*/rewriter.getBoolAttr(true));
  IRMapping mapper;
  for (SmallVector<Operation *> ops : lockAndBdOps) {
    rewriter.setInsertionPointToStart(
        &newS2mmDmaStartOp.getRegion().emplaceBlock());
    for (Operation *op : ops) {
      rewriter.clone(*op, mapper);
      // If it is a DMABD op, set the next_bd_id to nullopt to clear any
      // existing BD chain.
      if (auto bdOp = dyn_cast<AMDAIE::DMABDOp>(op))
        bdOp.setNextBdId(std::nullopt);
    }
    rewriter.create<AMDAIE::EndOp>(rewriter.getUnknownLoc());
  }
  // Set the keep_pkt_header attribute on the flow ops, as those headers are
  // required for the out-of-order mode.
  for (Value connection : newConnections) {
    auto connectionOp =
        dyn_cast<AMDAIE::ConnectionOp>(connection.getDefiningOp());
    std::optional<AMDAIE::FlowOp> maybeFlowOp = connectionOp.getFlowOp();
    if (!maybeFlowOp) return connectionOp.emitOpError() << "has no flow op";
    maybeFlowOp->setKeepPktHeader(true);
  }
  // Erase the old DMA start ops.
  for (AMDAIE::DMAStartOp s2mmDmaStartOp : s2mmDmaStartOps) {
    s2mmDmaStartOp->dropAllUses();
    rewriter.eraseOp(s2mmDmaStartOp);
  }
  return success();
}

class AMDAIEInsertDmaOutOfOrderBlockPass
    : public impl::AMDAIEInsertDmaOutOfOrderBlockBase<
          class AMDAIEInsertDmaOutOfOrderBlockPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEInsertDmaOutOfOrderBlockPass() = default;
  AMDAIEInsertDmaOutOfOrderBlockPass(
      const AMDAIEInsertDmaOutOfOrderBlockPass &pass){};
  void runOnOperation() override;
};

void AMDAIEInsertDmaOutOfOrderBlockPass::runOnOperation() {
  Operation *parentOp = getOperation();

  // Get the device model.
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(parentOp);
  std::optional<AMDAIEDevice> maybeDevice = getConfigAMDAIEDevice(targetAttr);
  if (!maybeDevice) {
    parentOp->emitOpError()
        << "has no AMDAIEDevice in the target attribute configuration.";
    return signalPassFailure();
  }
  AMDAIE::AMDAIEDeviceModel deviceModel =
      AMDAIE::getDeviceModel(maybeDevice.value());

  // Group all DMA start ops based on their corresponding channel.
  DenseMap<AMDAIE::ChannelOp, SmallVector<AMDAIE::DMAStartOp>>
      s2mmChannelToDmaStarts, mm2sChannelToDmaStarts;
  WalkResult res = parentOp->walk([&](AMDAIE::DMAStartOp dmaStartOp) {
    auto channelOp = dmaStartOp.getChannel().getDefiningOp<AMDAIE::ChannelOp>();
    if (channelOp.getDirection() == AMDAIE::DMAChannelDir::S2MM) {
      s2mmChannelToDmaStarts[channelOp].push_back(dmaStartOp);
    } else if (channelOp.getDirection() == AMDAIE::DMAChannelDir::MM2S) {
      mm2sChannelToDmaStarts[channelOp].push_back(dmaStartOp);
    } else {
      llvm::errs() << "Unsupported channel direction\n";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return signalPassFailure();
  // For each S2MM channel, merge all corresponding DMA start ops togther into a
  // single out-of-order block.
  IRRewriter rewriter(parentOp->getContext());
  for (auto &[s2mmChannelOp, s2mmDmaStartOps] : s2mmChannelToDmaStarts) {
    if (failed(insertDmaOutOfOrder(deviceModel, rewriter, s2mmChannelOp,
                                   s2mmDmaStartOps, mm2sChannelToDmaStarts))) {
      return signalPassFailure();
    }
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEInsertDmaOutOfOrderBlockPass() {
  return std::make_unique<AMDAIEInsertDmaOutOfOrderBlockPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
