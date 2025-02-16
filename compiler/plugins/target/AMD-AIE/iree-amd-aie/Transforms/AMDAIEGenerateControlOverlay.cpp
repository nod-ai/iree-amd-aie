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

#define DEBUG_TYPE "iree-amdaie-generate-control-overlay"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Initializes channel generators for shim tiles, ensuring that no shim DMA
/// MM2S channels have been assigned before. This guarantees priority for the
/// control overlay.
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
  // Ensure that shim DMA MM2S channels are not already assigned.
  WalkResult res = workgroupOp->walk([&](AMDAIE::ChannelOp channelOp) {
    if (shimTileOps.contains(channelOp.getTileOp()) &&
        channelOp.getPortType() == StrmSwPortType::DMA &&
        channelOp.getDirection() == AMDAIE::DMAChannelDir::MM2S) {
      channelOp.emitOpError()
          << "shim DMA MM2S channel must remain unassigned before "
             "control overlay generation.";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return failure();
  return success();
}

LogicalResult generateControlOverlay(AMDAIE::WorkgroupOp workgroupOp,
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
    if (deviceModel.isShimNOCTile(col, row)) columnToShimTile[col] = tileOp;
  });

  AMDAIE::ControlCodeOp controlCodeOp = workgroupOp.getControlCode();
  rewriter.setInsertionPoint(controlCodeOp);
  // If the column is occupied, but the shim tile op is not present, then create
  // one.
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

  // Create a packet-mode connection from the shim DMA to the tile CTRL, for
  // sending control packets.
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
      // Get the available DMA channel for the shim tile, and assign it for the
      // packet flow.
      std::optional<uint8_t> maybeChannel =
          shimTileToGeneratorMap[shimTileOp.getResult()]
              .getAndAssignProducerDMAChannel(
                  ChannelAssignmentMode::RoundRobinPacketFlow);
      if (!maybeChannel) {
        shimTileOp.emitOpError() << "no producer DMA channel available";
        return WalkResult::interrupt();
      }

      rewriter.setInsertionPoint(controlCodeOp);
      auto sourceChannelOp = rewriter.create<AMDAIE::ChannelOp>(
          rewriter.getUnknownLoc(), shimTileOp, maybeChannel.value(),
          StrmSwPortType::DMA, AMDAIE::DMAChannelDir::MM2S);
      auto targetChannelOp = rewriter.create<AMDAIE::ChannelOp>(
          rewriter.getUnknownLoc(), tileOp, 0, StrmSwPortType::CTRL,
          AMDAIE::DMAChannelDir::S2MM);

      // Get the objectfifo placeholder for both the source and target.
      MemRefType elementType =
          MemRefType::get(ShapedType::kDynamic, rewriter.getI32Type());
      auto sourcePlaceholder =
          rewriter.create<AMDAIE::LogicalObjectFifoPlaceholderOp>(
              rewriter.getUnknownLoc(), LogicalObjectFifoType::get(elementType),
              ValueRange(shimTileOp));
      auto targetPlaceholder =
          rewriter.create<AMDAIE::LogicalObjectFifoPlaceholderOp>(
              rewriter.getUnknownLoc(), LogicalObjectFifoType::get(elementType),
              ValueRange(tileOp));

      auto connectionOp = rewriter.create<AMDAIE::ConnectionOp>(
          rewriter.getUnknownLoc(), targetPlaceholder,
          ValueRange{targetChannelOp}, sourcePlaceholder,
          ValueRange{sourceChannelOp},
          ConnectionTypeAttr::get(rewriter.getContext(),
                                  ConnectionType::Packet),
          /*flow=*/nullptr);

      rewriter.setInsertionPoint(controlCodeOp.getBody()->getTerminator());
      rewriter.create<AMDAIE::NpuDmaPlaceHolderOp>(rewriter.getUnknownLoc(),
                                                   connectionOp.getResult());
      return WalkResult::advance();
    });
    if (res.wasInterrupted()) return failure();
  }

  // Create a circuit-mode connection from the shim CTRL to the shim SOUTH 0,
  // for sending Task Completion Tokens (TCTs).
  if (routeShimCtrlToTct) {
    for (auto [_, shimTileOp] : columnToShimTile) {
      rewriter.setInsertionPoint(controlCodeOp);
      auto sourceChannelOp = rewriter.create<AMDAIE::ChannelOp>(
          rewriter.getUnknownLoc(), shimTileOp, 0, StrmSwPortType::CTRL,
          AMDAIE::DMAChannelDir::MM2S);
      auto targetChannelOp = rewriter.create<AMDAIE::ChannelOp>(
          rewriter.getUnknownLoc(), shimTileOp, 0, StrmSwPortType::SOUTH,
          AMDAIE::DMAChannelDir::S2MM);

      // Get the objectfifo placeholder for both the source and target.
      // Set the shape to dynamic because the size of the control packet
      // sequence is unknown and may vary based on the reconfiguration content.
      MemRefType elementType =
          MemRefType::get(ShapedType::kDynamic, rewriter.getI32Type());
      auto sourcePlaceholder =
          rewriter.create<AMDAIE::LogicalObjectFifoPlaceholderOp>(
              rewriter.getUnknownLoc(), LogicalObjectFifoType::get(elementType),
              ValueRange(shimTileOp));
      auto targetPlaceholder =
          rewriter.create<AMDAIE::LogicalObjectFifoPlaceholderOp>(
              rewriter.getUnknownLoc(), LogicalObjectFifoType::get(elementType),
              ValueRange(shimTileOp));

      auto connectionOp = rewriter.create<AMDAIE::ConnectionOp>(
          rewriter.getUnknownLoc(), targetPlaceholder,
          ValueRange{targetChannelOp}, sourcePlaceholder,
          ValueRange{sourceChannelOp},
          ConnectionTypeAttr::get(rewriter.getContext(),
                                  ConnectionType::Circuit),
          /*flow=*/nullptr);

      rewriter.setInsertionPoint(controlCodeOp.getBody()->getTerminator());
      rewriter.create<AMDAIE::NpuDmaPlaceHolderOp>(rewriter.getUnknownLoc(),
                                                   connectionOp.getResult());
    }
  }

  return success();
}

class AMDAIEGenerateControlOverlayPass
    : public impl::AMDAIEGenerateControlOverlayBase<
          AMDAIEGenerateControlOverlayPass> {
 public:
  AMDAIEGenerateControlOverlayPass(
      const AMDAIEGenerateControlOverlayOptions &options)
      : AMDAIEGenerateControlOverlayBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  void runOnOperation() override;
};

void AMDAIEGenerateControlOverlayPass::runOnOperation() {
  Operation *parentOp = getOperation();
  WalkResult res = parentOp->walk([&](AMDAIE::WorkgroupOp workgroupOp) {
    if (failed(generateControlOverlay(workgroupOp, routeShimToTileCtrl,
                                      routeShimCtrlToTct))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (res.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEGenerateControlOverlayPass(
    AMDAIEGenerateControlOverlayOptions options) {
  return std::make_unique<AMDAIEGenerateControlOverlayPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
