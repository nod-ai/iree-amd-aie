// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/iree_aie_router.h"

#define DEBUG_TYPE "iree-amdaie-assign-connection-types"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIEAssignConnectionTypesPass
    : public impl::AMDAIEAssignConnectionTypesBase<
          AMDAIEAssignConnectionTypesPass> {
 public:
  AMDAIEAssignConnectionTypesPass(
      const AMDAIEAssignConnectionTypesOptions &options)
      : AMDAIEAssignConnectionTypesBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  void runOnOperation() override;
};

/// Utility function to retrieve the endpoint information (tile location, port
/// type and direction) from a connection operation. It handles both the
/// source(s) and target(s) of the connection.
FailureOr<SmallVector<PhysPortType>> getAllPhysPortTypes(
    AMDAIE::ConnectionOp connectionOp) {
  SmallVector<PhysPortType> physPortTypes;

  auto process = [&](AMDAIE::DMAChannelDir direction) -> LogicalResult {
    SmallVector<Value> channels = (direction == AMDAIE::DMAChannelDir::MM2S)
                                      ? connectionOp.getSourceChannels()
                                      : connectionOp.getTargetChannels();
    Value logicalObjFifo = (direction == AMDAIE::DMAChannelDir::MM2S)
                               ? connectionOp.getSource()
                               : connectionOp.getTarget();
    if (channels.empty()) {
      // If channels are not assigned, use the logicalObjFifo to get the tiles,
      // and assume the port type is `StrmSwPortType::DMA`.
      auto logicalObjFifoOp =
          dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
              logicalObjFifo.getDefiningOp());
      if (!logicalObjFifoOp) {
        return connectionOp.emitOpError()
               << "expected a `LogicalObjFifoOpInterface`";
      }
      for (Value value : logicalObjFifoOp.getTiles()) {
        auto tileOp = dyn_cast<AMDAIE::TileOp>(value.getDefiningOp());
        if (!tileOp)
          return connectionOp.emitOpError() << "expected an `amdaie.tile` op";
        int32_t col = getConstantIndexOrAssert(tileOp.getCol());
        int32_t row = getConstantIndexOrAssert(tileOp.getRow());
        physPortTypes.push_back(
            {{col, row}, AMDAIE::StrmSwPortType::DMA, direction});
      }
    } else {
      for (Value channel : channels) {
        auto channelOp = dyn_cast<AMDAIE::ChannelOp>(channel.getDefiningOp());
        if (!channelOp) {
          return connectionOp.emitOpError()
                 << "expected an `amdaie.channel` op";
        }
        AMDAIE::TileOp tileOp = channelOp.getTileOp();
        int32_t col = getConstantIndexOrAssert(tileOp.getCol());
        int32_t row = getConstantIndexOrAssert(tileOp.getRow());
        physPortTypes.push_back(
            {{col, row}, channelOp.getPortType(), direction});
      }
    }
    return success();
  };

  // Process both the source and target of the connection.
  if (failed(process(AMDAIE::DMAChannelDir::MM2S)) ||
      failed(process(AMDAIE::DMAChannelDir::S2MM))) {
    return failure();
  }
  return physPortTypes;
}

void updateConnectionType(IRRewriter &rewriter,
                          AMDAIE::ConnectionOp connectionOp,
                          AMDAIE::ConnectionType connectionType) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(connectionOp);
  ConnectionTypeAttr connectionTypeAttr =
      ConnectionTypeAttr::get(rewriter.getContext(), connectionType);
  rewriter.replaceOpWithNewOp<AMDAIE::ConnectionOp>(
      connectionOp, connectionOp.getTarget(), connectionOp.getTargetChannels(),
      connectionOp.getSource(), connectionOp.getSourceChannels(),
      connectionTypeAttr, connectionOp.getFlow());
}

/// Assigns connection types to the circuit mode unless a congestion is
/// discovered.
LogicalResult congestionAwareAutoAssignment(Operation *parentOp,
                                            IRRewriter &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  // Get the device model.
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(parentOp);
  std::optional<AMDAIEDevice> maybeDevice = getConfigAMDAIEDevice(targetAttr);
  if (!maybeDevice) {
    return parentOp->emitOpError()
           << "has no AMDAIEDevice in the target attribute configuration.";
  }
  AMDAIEDeviceModel deviceModel = AMDAIE::getDeviceModel(maybeDevice.value());
  // Get all the connection ops and sort them into two groups: those that have
  // already been assigned a type and those that have not.
  SmallVector<AMDAIE::ConnectionOp> preAssignedConnections;
  SmallVector<AMDAIE::ConnectionOp> unassignedConnections;
  parentOp->walk([&](AMDAIE::ConnectionOp connectionOp) {
    if (connectionOp.getConnectionType().has_value()) {
      preAssignedConnections.push_back(connectionOp);
    } else {
      unassignedConnections.push_back(connectionOp);
    }
  });
  SmallVector<AMDAIE::ConnectionOp> connectionOps;
  connectionOps.reserve(preAssignedConnections.size() +
                        unassignedConnections.size());
  connectionOps.append(preAssignedConnections.begin(),
                       preAssignedConnections.end());
  connectionOps.append(unassignedConnections.begin(),
                       unassignedConnections.end());
  // `circuitUsage` and `packetUsage` are used to track the number of
  // connections assigned to each physical port type. `maxCircuitUsage` is used
  // to track the maximum number of circuit connections allowed for each
  // physical port type, i.e. the number of channels available.
  DenseMap<PhysPortType, uint32_t> circuitUsage, packetUsage, maxCircuitUsage;
  auto getOrInitMaxUsage = [&](const PhysPortType &b) -> uint32_t {
    if (maxCircuitUsage.count(b)) return maxCircuitUsage[b];
    // Cannot just use `getNumSource/DestSwitchBoxConnections` due to shimmux.
    if (b.portType == AMDAIE::StrmSwPortType::DMA) {
      AMDAIETileType tileType =
          deviceModel.getTileType(b.tileLoc.col, b.tileLoc.row);
      maxCircuitUsage[b] =
          deviceModel.getDmaProp<uint8_t>(tileType, AMDAIEDmaProp::NumChannels);
    } else {
      maxCircuitUsage[b] = (b.direction == AMDAIE::DMAChannelDir::MM2S)
                               ? deviceModel.getNumSourceSwitchBoxConnections(
                                     b.tileLoc.col, b.tileLoc.row, b.portType)
                               : deviceModel.getNumDestSwitchBoxConnections(
                                     b.tileLoc.col, b.tileLoc.row, b.portType);
    }
    return maxCircuitUsage[b];
  };

  // Record the final connection index associated with each physical port type.
  DenseMap<PhysPortType, uint32_t> physPortTypeToFinalConnectionIndex;
  for (auto [i, connectionOp] : llvm::enumerate(connectionOps)) {
    FailureOr<SmallVector<PhysPortType>> physPortTypes =
        getAllPhysPortTypes(connectionOp);
    if (failed(physPortTypes)) return failure();
    for (PhysPortType &physPortType : *physPortTypes)
      physPortTypeToFinalConnectionIndex[physPortType] = i;
  }
  // Iterate through the connections and assign types based on the
  // congestion status of the physical port types.
  for (auto [i, connectionOp] : llvm::enumerate(connectionOps)) {
    AMDAIE::ConnectionType connectionType = AMDAIE::ConnectionType::Circuit;
    FailureOr<SmallVector<PhysPortType>> physPortTypes =
        getAllPhysPortTypes(connectionOp);
    if (failed(physPortTypes)) return failure();
    // If the connection type is pre-assigned, use it.
    if (connectionOp.getConnectionType().has_value()) {
      connectionType = *connectionOp.getConnectionType();
    } else {
      // Evaluate each physical port type associated with the connection.
      for (PhysPortType &physPortType : *physPortTypes) {
        // Check if the maximum allowed circuit usage reached, and if so, set
        // the connection type to packet.
        uint32_t maxUsage = getOrInitMaxUsage(physPortType);
        if (circuitUsage[physPortType] + packetUsage[physPortType] >=
            maxUsage) {
          connectionType = AMDAIE::ConnectionType::Packet;
          break;
        }
        // If the current physical port type is one usage away from its maximum,
        // and there are upcoming unassigned connections that will use this
        // port type, then set the current connection type to packet.
        if (circuitUsage[physPortType] == maxUsage - 1 &&
            physPortTypeToFinalConnectionIndex[physPortType] > i) {
          connectionType = AMDAIE::ConnectionType::Packet;
          break;
        }
      }
      // Update with the assigned connection type.
      updateConnectionType(rewriter, connectionOp, connectionType);
    }
    // Update the circuit and packet usage.
    DenseMap<PhysPortType, uint32_t> &usage =
        (connectionType == AMDAIE::ConnectionType::Circuit) ? circuitUsage
                                                            : packetUsage;
    for (PhysPortType &physPortType : *physPortTypes) usage[physPortType]++;
  }
  return success();
}

LogicalResult simpleManualAssignment(Operation *parentOp, IRRewriter &rewriter,
                                     PacketFlowStrategy packetFlowStrategy) {
  OpBuilder::InsertionGuard g(rewriter);
  bool enableInputPacketFlow, enableOutputPacketFlow;
  switch (packetFlowStrategy) {
    case PacketFlowStrategy::None:
      enableInputPacketFlow = false;
      enableOutputPacketFlow = false;
      break;
    case PacketFlowStrategy::Inputs:
      enableInputPacketFlow = true;
      enableOutputPacketFlow = false;
      break;
    case PacketFlowStrategy::Outputs:
      enableInputPacketFlow = false;
      enableOutputPacketFlow = true;
      break;
    case PacketFlowStrategy::All:
      enableInputPacketFlow = true;
      enableOutputPacketFlow = true;
      break;
    default:
      return parentOp->emitError() << "invalid packet flow strategy.";
  }

  WalkResult res = parentOp->walk([&](AMDAIE::ConnectionOp connectionOp) {
    // If the connection type is pre-assigned, use it.
    if (connectionOp.getConnectionType().has_value())
      return WalkResult::advance();

    rewriter.setInsertionPoint(connectionOp);

    // Determine the source and target memory spaces of the connection.
    auto sourceLogicalObjFifo =
        dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
            connectionOp.getSource().getDefiningOp());
    auto targetLogicalObjFifo =
        dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
            connectionOp.getTarget().getDefiningOp());
    if (!sourceLogicalObjFifo || !targetLogicalObjFifo) {
      connectionOp.emitError(
          "source and target of connection must be logical object fifos");
      return WalkResult::interrupt();
    }
    uint8_t sourceMemSpace = sourceLogicalObjFifo.getMemorySpaceAsUInt();
    uint8_t targetMemSpace = targetLogicalObjFifo.getMemorySpaceAsUInt();

    // Default connection type is circuit.
    AMDAIE::ConnectionType connectionType = AMDAIE::ConnectionType::Circuit;
    // Use the memory space to determine if the connetion belongs to the kernel
    // input or output, and set the connection type accordingly.
    if (((sourceMemSpace < targetMemSpace) && enableInputPacketFlow) ||
        ((sourceMemSpace > targetMemSpace) && enableOutputPacketFlow)) {
      connectionType = AMDAIE::ConnectionType::Packet;
    }

    ConnectionTypeAttr connectionTypeAttr =
        ConnectionTypeAttr::get(rewriter.getContext(), connectionType);
    rewriter.replaceOpWithNewOp<AMDAIE::ConnectionOp>(
        connectionOp, connectionOp.getTarget(),
        connectionOp.getTargetChannels(), connectionOp.getSource(),
        connectionOp.getSourceChannels(), connectionTypeAttr,
        connectionOp.getFlow());
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return failure();
  return success();
}

void AMDAIEAssignConnectionTypesPass::runOnOperation() {
  Operation *parentOp = getOperation();
  IRRewriter rewriter(parentOp->getContext());

  LogicalResult result = success();
  if (packetFlowStrategy == PacketFlowStrategy::Auto) {
    result = congestionAwareAutoAssignment(parentOp, rewriter);
  } else {
    result = simpleManualAssignment(parentOp, rewriter, packetFlowStrategy);
  }

  if (failed(result)) return signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEAssignConnectionTypesPass(
    AMDAIEAssignConnectionTypesOptions options) {
  return std::make_unique<AMDAIEAssignConnectionTypesPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
