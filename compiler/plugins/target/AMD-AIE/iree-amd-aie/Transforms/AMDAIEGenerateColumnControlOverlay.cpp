// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/Utils/ChannelGenerator.h"

#define DEBUG_TYPE "iree-amdaie-generate-column-control-overlay"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Initializes the channel generators for the shim tiles, excluding any
/// channels that are already in use by existing circuit flows.
LogicalResult initializeChannelsGenerators(
    AMDAIE::WorkgroupOp workgroupOp, const AMDAIEDeviceModel &deviceModel,
    const DenseSet<TileOp> &shimTileOps,
    DenseMap<Value, ChannelGenerator> &shimTileToGeneratorMap) {
  // Check the number of DMA channels available for the shim tile.
  uint8_t numShimDmaChannels = deviceModel.getDmaProp<uint8_t>(
      AMDAIETileType::SHIMNOC, AMDAIEDmaProp::NumChannels);
  std::for_each(shimTileOps.begin(), shimTileOps.end(), [&](TileOp shimTileOp) {
    shimTileToGeneratorMap[shimTileOp.getResult()] =
        ChannelGenerator(numShimDmaChannels, numShimDmaChannels);
  });
  // Exclude those channels that are already used by a circuit flow.
  workgroupOp->walk([&](AMDAIE::FlowOp flowOp) {
    if (flowOp.getIsPacketFlow()) return WalkResult::advance();
    SmallVector<AMDAIE::ChannelOp> sourceChannels;
    for (Value source : flowOp.getSources()) {
      if (auto channelOp =
              dyn_cast<AMDAIE::ChannelOp>(source.getDefiningOp())) {
        sourceChannels.push_back(channelOp);
      }
    }
    for (AMDAIE::ChannelOp channelOp : sourceChannels) {
      AMDAIE::TileOp tileOp = channelOp.getTileOp();
      uint8_t channel = channelOp.getValue();
      StrmSwPortType portType = channelOp.getPortType();
      AMDAIE::DMAChannelDir direction = channelOp.getDirection();
      if (shimTileOps.contains(tileOp) && portType == StrmSwPortType::DMA) {
        // Assign to exclude.
        if (direction == AMDAIE::DMAChannelDir::MM2S) {
          shimTileToGeneratorMap[tileOp.getResult()].assignProducerDMAChannel(
              channel);
        } else {
          shimTileToGeneratorMap[tileOp.getResult()].assignConsumerDMAChannel(
              channel);
        }
      }
    }
    return WalkResult::advance();
  });
  return success();
}

LogicalResult generateColumnControlOverlay(AMDAIE::WorkgroupOp workgroupOp,
                                           bool routeShimToTileCtrl,
                                           bool routeShimCtrlToTct) {
  // Get the device model.
  std::optional<AMDAIEDevice> device = getConfigAMDAIEDevice(workgroupOp);
  if (!device) {
    return workgroupOp->emitOpError()
           << "could not find an AMDAIEDevice attribute";
  }
  AMDAIEDeviceModel deviceModel = AMDAIE::getDeviceModel(device.value());

  IRRewriter rewriter(workgroupOp->getContext());
  DenseSet<uint32_t> occupiedCols;
  DenseMap<uint32_t, AMDAIE::TileOp> columnToShimTile;
  workgroupOp->walk([&](AMDAIE::TileOp tileOp) {
    uint32_t col = getConstantIndexOrAssert(tileOp.getCol());
    uint32_t row = getConstantIndexOrAssert(tileOp.getRow());
    occupiedCols.insert(col);
    if (deviceModel.getTileType(col, row) == AMDAIETileType::SHIMNOC)
      columnToShimTile[col] = tileOp;
  });

  // If the column is occupied, but the shim tile op is not present, then create
  // one.
  rewriter.setInsertionPoint(workgroupOp.getControlCode());
  for (uint32_t col : occupiedCols) {
    if (!columnToShimTile.count(col)) {
      auto colIndex = rewriter.create<arith::ConstantIndexOp>(
          rewriter.getUnknownLoc(), col);
      auto rowIndex =
          rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), 0);
      columnToShimTile[col] = rewriter.create<AMDAIE::TileOp>(
          rewriter.getUnknownLoc(), colIndex, rowIndex);
    }
  }

  // Create a packet flow from the shim DMA to the tile CTRL, for sending
  // control packets.
  if (routeShimToTileCtrl) {
    DenseMap<Value, ChannelGenerator> shimTileToGeneratorMap;
    DenseSet<TileOp> shimTileOps;
    for (const auto &pair : columnToShimTile) shimTileOps.insert(pair.second);
    if (failed(initializeChannelsGenerators(
            workgroupOp, deviceModel, shimTileOps, shimTileToGeneratorMap))) {
      return failure();
    }
    WalkResult res = workgroupOp->walk([&](AMDAIE::TileOp tileOp) {
      uint32_t col = getConstantIndexOrAssert(tileOp.getCol());
      TileOp shimTileOp = columnToShimTile[col];
      // Get the available channel, but do not assign it. Allow it to be
      // shared across multiple packet flows as needed.
      std::optional<uint8_t> maybeChannel =
          shimTileToGeneratorMap[shimTileOp.getResult()]
              .getProducerDMAChannel();
      if (!maybeChannel) {
        shimTileOp.emitOpError() << "no producer DMA channel available";
        return WalkResult::interrupt();
      }
      auto shimDmaChannelOp = rewriter.create<AMDAIE::ChannelOp>(
          rewriter.getUnknownLoc(), shimTileOp, maybeChannel.value(),
          StrmSwPortType::DMA, AMDAIE::DMAChannelDir::MM2S);
      auto tileCtrlChannelOp = rewriter.create<AMDAIE::ChannelOp>(
          rewriter.getUnknownLoc(), tileOp, 0, StrmSwPortType::CTRL,
          AMDAIE::DMAChannelDir::S2MM);
      rewriter.create<AMDAIE::FlowOp>(
          rewriter.getUnknownLoc(), ValueRange{shimDmaChannelOp},
          ValueRange{tileCtrlChannelOp},
          /*isPacketFlow*/ true, /*packetId*/ nullptr);
      return WalkResult::advance();
    });
    if (res.wasInterrupted()) return failure();
  }

  // Create a circuit flow from the shim CTRL to the shim SOUTH 0, for sending
  // Task Completion Tokens (TCTs).
  if (routeShimCtrlToTct) {
    for (auto [_, shimTileOp] : columnToShimTile) {
      auto shimCtrlChannelOp = rewriter.create<AMDAIE::ChannelOp>(
          rewriter.getUnknownLoc(), shimTileOp, 0, StrmSwPortType::CTRL,
          AMDAIE::DMAChannelDir::MM2S);
      auto shimSouthChannelOp = rewriter.create<AMDAIE::ChannelOp>(
          rewriter.getUnknownLoc(), shimTileOp, 0, StrmSwPortType::SOUTH,
          AMDAIE::DMAChannelDir::S2MM);
      rewriter.create<AMDAIE::FlowOp>(
          rewriter.getUnknownLoc(), ValueRange{shimCtrlChannelOp},
          ValueRange{shimSouthChannelOp},
          /*isPacketFlow*/ false, /*packetId*/ nullptr);
    }
  }

  return success();
}

class AMDAIEGenerateColumnControlOverlayPass
    : public impl::AMDAIEGenerateColumnControlOverlayBase<
          AMDAIEGenerateColumnControlOverlayPass> {
 public:
  AMDAIEGenerateColumnControlOverlayPass(
      const AMDAIEGenerateColumnControlOverlayOptions &options)
      : AMDAIEGenerateColumnControlOverlayBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  void runOnOperation() override;
};

void AMDAIEGenerateColumnControlOverlayPass::runOnOperation() {
  Operation *parentOp = getOperation();
  WalkResult res = parentOp->walk([&](AMDAIE::WorkgroupOp workgroupOp) {
    if (failed(generateColumnControlOverlay(workgroupOp, routeShimToTileCtrl,
                                            routeShimCtrlToTct))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (res.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEGenerateColumnControlOverlayPass(
    AMDAIEGenerateColumnControlOverlayOptions options) {
  return std::make_unique<AMDAIEGenerateColumnControlOverlayPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
