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

/// Initializes channel generators for tiles by detecting DMA channels
/// previously assigned by other passes (e.g., for control packets) and
/// registering them to prevent conflicts.
LogicalResult initializeChannelsGenerators(
    AMDAIE::WorkgroupOp workgroupOp, const AMDAIEDeviceModel &deviceModel,
    DenseMap<Value, ChannelGenerator> &tileToGeneratorMap) {
  // Get the number of producer and consumer channels for each tile.
  workgroupOp.walk([&](AMDAIE::TileOp tileOp) {
    uint32_t col = getConstantIndexOrAssert(tileOp.getCol());
    uint32_t row = getConstantIndexOrAssert(tileOp.getRow());
    AMDAIETileType tileType = deviceModel.getTileType(col, row);
    uint8_t numDmaChannels =
        deviceModel.getDmaProp<uint8_t>(tileType, AMDAIEDmaProp::NumChannels);
    tileToGeneratorMap[tileOp.getResult()] =
        ChannelGenerator(numDmaChannels, numDmaChannels);
  });

  WalkResult res = workgroupOp.walk([&](AMDAIE::ConnectionOp connectionOp) {
    ChannelAssignmentMode mode =
        (connectionOp.getConnectionType() == AMDAIE::ConnectionType::Packet)
            ? ChannelAssignmentMode::RoundRobinPacketFlow
            : ChannelAssignmentMode::FirstAvailableCircuitFlow;
    // Check source DMA channels previously assigned by other passes,
    // and register them in `ChannelGenerator` using `assignProducerDMAChannel`.
    for (Value source : connectionOp.getSourceChannels()) {
      auto channelOp = dyn_cast<AMDAIE::ChannelOp>(source.getDefiningOp());
      if (!channelOp) {
        connectionOp.emitOpError() << "expected a `amdaie.channel` op source";
        return WalkResult::interrupt();
      }
      if (channelOp.getPortType() == StrmSwPortType::DMA) {
        Value tile = channelOp.getTileOp().getResult();
        tileToGeneratorMap[tile].assignProducerDMAChannel(channelOp.getValue(),
                                                          mode);
      }
    }
    // Check target DMA channels previously assigned by other passes,
    // and register them in `ChannelGenerator` using `assignConsumerDMAChannel`.
    for (Value target : connectionOp.getTargetChannels()) {
      auto channelOp = dyn_cast<AMDAIE::ChannelOp>(target.getDefiningOp());
      if (!channelOp) {
        connectionOp.emitOpError() << "expected a `amdaie.channel` op target";
        return WalkResult::interrupt();
      }
      if (channelOp.getPortType() == StrmSwPortType::DMA) {
        Value tile = channelOp.getTileOp().getResult();
        tileToGeneratorMap[tile].assignConsumerDMAChannel(channelOp.getValue(),
                                                          mode);
      }
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return failure();
  return success();
}

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
  // Initialize channel generators for tiles.
  DenseMap<Value, ChannelGenerator> tileToGeneratorMap;
  if (failed(initializeChannelsGenerators(workgroupOp, deviceModel,
                                          tileToGeneratorMap))) {
    return failure();
  }
  // Get all `amdaie.connection` ops.
  SmallVector<AMDAIE::ConnectionOp> circuitConnections, packetConnections;
  workgroupOp->walk([&](AMDAIE::ConnectionOp op) {
    if (op.getConnectionType() == AMDAIE::ConnectionType::Packet) {
      packetConnections.push_back(op);
    } else {
      circuitConnections.push_back(op);
    }
  });
  SmallVector<AMDAIE::ConnectionOp> connectionOps;
  connectionOps.reserve(circuitConnections.size() + packetConnections.size());
  // Append circuit connections first, so that they are also assigned first.
  connectionOps.append(circuitConnections.begin(), circuitConnections.end());
  connectionOps.append(packetConnections.begin(), packetConnections.end());

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
    ChannelAssignmentMode mode =
        (connectionOp.getConnectionType() == AMDAIE::ConnectionType::Packet)
            ? ChannelAssignmentMode::RoundRobinPacketFlow
            : ChannelAssignmentMode::FirstAvailableCircuitFlow;
    rewriter.setInsertionPoint(connectionOp);
    SmallVector<Value> sourceChannels = connectionOp.getSourceChannels();
    // Assign source (producer) DMA channels if not already assigned.
    if (sourceChannels.empty()) {
      for (Value tile : sourceLogicalObjFifo.getTiles()) {
        assert(tileToGeneratorMap.contains(tile) &&
               "no channel generator found for tile");
        std::optional<uint8_t> maybeChannel =
            tileToGeneratorMap[tile].getAndAssignProducerDMAChannel(mode);
        if (!maybeChannel) {
          return connectionOp.emitOpError()
                 << "no producer DMA channel available";
        }
        auto channelOp = rewriter.create<AMDAIE::ChannelOp>(
            rewriter.getUnknownLoc(), tile, maybeChannel.value(),
            StrmSwPortType::DMA, AMDAIE::DMAChannelDir::MM2S);
        sourceChannels.push_back(channelOp.getResult());
      }
    }
    // Assign target (consumer) DMA channels if not already assigned.
    SmallVector<Value> targetChannels = connectionOp.getTargetChannels();
    if (targetChannels.empty()) {
      for (Value tile : targetLogicalObjFifo.getTiles()) {
        assert(tileToGeneratorMap.contains(tile) &&
               "no channel generator found for tile");
        std::optional<uint8_t> maybeChannel =
            tileToGeneratorMap[tile].getAndAssignConsumerDMAChannel(mode);
        if (!maybeChannel) {
          return connectionOp.emitOpError()
                 << "no consumer DMA channel available";
        }
        auto channelOp = rewriter.create<AMDAIE::ChannelOp>(
            rewriter.getUnknownLoc(), tile, maybeChannel.value(),
            StrmSwPortType::DMA, AMDAIE::DMAChannelDir::S2MM);
        targetChannels.push_back(channelOp.getResult());
      }
    }
    // Replace the `amdaie.connection` op with newly assigned `sourceChannels`
    // and `targetChannels`.
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
