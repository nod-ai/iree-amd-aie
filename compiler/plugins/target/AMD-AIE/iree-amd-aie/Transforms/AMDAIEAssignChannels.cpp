// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/Utils/ChannelGenerator.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"

#define DEBUG_TYPE "iree-amdaie-assign-channels"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Assign channels to `amdaie.connection` ops.
LogicalResult assignChannels(AMDAIE::WorkgroupOp workgroupOp) {
  IRRewriter rewriter(workgroupOp->getContext());

  // Get the device model.
  std::optional<AMDAIEDevice> device = getConfigAMDAIEDevice(workgroupOp);
  if (!device) {
    return workgroupOp->emitOpError()
           << "could not find an AMDAIEDevice attribute";
  }
  AMDAIEDeviceModel deviceModel = AMDAIE::getDeviceModel(device.value());

  // Get the number of producer and consumer channels for each tile.
  DenseMap<Value, ChannelGenerator> tileToGeneratorMap;
  workgroupOp.walk([&](AMDAIE::TileOp tileOp) {
    uint32_t col = getConstantIndexOrAssert(tileOp.getCol());
    uint32_t row = getConstantIndexOrAssert(tileOp.getRow());
    AMDAIETileType tileType = deviceModel.getTileType(col, row);
    uint8_t numDmaChannels =
        deviceModel.getDmaProp<uint8_t>(tileType, AMDAIEDmaProp::NumChannels);
    tileToGeneratorMap[tileOp.getResult()] =
        ChannelGenerator(numDmaChannels, numDmaChannels);
  });

  SmallVector<AMDAIE::ConnectionOp> connectionOps;
  workgroupOp->walk([&](AMDAIE::ConnectionOp connectionOp) {
    connectionOps.push_back(connectionOp);
  });
  for (AMDAIE::ConnectionOp connectionOp : connectionOps) {
    auto sourceLogicalObjFifo =
        dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
            connectionOp.getSource().getDefiningOp());
    if (!sourceLogicalObjFifo) {
      return connectionOp.emitOpError()
             << "expected a `LogicalObjFifoOpInterface` source";
    }
    auto targetLogicalObjFifo =
        dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
            connectionOp.getTarget().getDefiningOp());
    if (!targetLogicalObjFifo) {
      return connectionOp.emitOpError()
             << "expected a `LogicalObjFifoOpInterface` target";
    }
    std::optional<AMDAIE::ConnectionType> connectionType =
        connectionOp.getConnectionType();
    bool isPacketFlow = connectionType && connectionType.value() ==
                                              AMDAIE::ConnectionType::Packet;

    rewriter.setInsertionPoint(connectionOp);
    SmallVector<Value> sourceChannels;
    for (Value tile : sourceLogicalObjFifo.getTiles()) {
      assert(tileToGeneratorMap.contains(tile) &&
             "no channel generator found for tile");
      std::optional<uint8_t> maybeChannel =
          tileToGeneratorMap[tile].getProducerDMAChannel();
      if (!maybeChannel) {
        return connectionOp.emitOpError()
               << "no producer DMA channel available";
      }
      // Only assign the channel if it is for circuit flow.
      if (!isPacketFlow)
        tileToGeneratorMap[tile].assignProducerDMAChannel(maybeChannel.value());
      auto channelOp = rewriter.create<AMDAIE::ChannelOp>(
          rewriter.getUnknownLoc(), tile, maybeChannel.value(),
          StrmSwPortType::DMA, AMDAIE::DMAChannelDir::MM2S);
      sourceChannels.push_back(channelOp.getResult());
    }
    SmallVector<Value> targetChannels;
    for (Value tile : targetLogicalObjFifo.getTiles()) {
      assert(tileToGeneratorMap.contains(tile) &&
             "no channel generator found for tile");
      std::optional<uint8_t> maybeChannel =
          tileToGeneratorMap[tile].getConsumerDMAChannel();
      if (!maybeChannel) {
        return connectionOp.emitOpError()
               << "no consumer DMA channel available";
      }
      // Only assign the channel if it is for circuit flow.
      if (!isPacketFlow)
        tileToGeneratorMap[tile].assignConsumerDMAChannel(maybeChannel.value());
      auto channelOp = rewriter.create<AMDAIE::ChannelOp>(
          rewriter.getUnknownLoc(), tile, maybeChannel.value(),
          StrmSwPortType::DMA, AMDAIE::DMAChannelDir::S2MM);
      targetChannels.push_back(channelOp.getResult());
    }
    rewriter.replaceOpWithNewOp<AMDAIE::ConnectionOp>(
        connectionOp, connectionOp.getTarget(), targetChannels,
        connectionOp.getSource(), sourceChannels,
        connectionOp.getConnectionTypeAttr(), /*flow*/ nullptr);
  }
  return success();
}

class AMDAIEAssignChannelsPass
    : public impl::AMDAIEAssignChannelsBase<AMDAIEAssignChannelsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  void runOnOperation() override;
};

void AMDAIEAssignChannelsPass::runOnOperation() {
  Operation *parentOp = getOperation();
  SmallVector<AMDAIE::WorkgroupOp> workgroupOps;
  parentOp->walk([&](AMDAIE::WorkgroupOp workgroupOp) {
    workgroupOps.push_back(workgroupOp);
  });
  for (AMDAIE::WorkgroupOp workgroupOp : workgroupOps) {
    if (failed(assignChannels(workgroupOp))) return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEAssignChannelsPass() {
  return std::make_unique<AMDAIEAssignChannelsPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
