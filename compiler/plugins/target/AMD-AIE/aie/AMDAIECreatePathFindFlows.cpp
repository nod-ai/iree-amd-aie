// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cassert>
#include <set>

#include "AIEDialect.h"
#include "Passes.h"
#include "iree-amd-aie/aie_runtime/iree_aie_router.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::iree_compiler::AMDAIE;

using xilinx::AIE::AMSelOp;
using xilinx::AIE::ConnectOp;
using xilinx::AIE::DeviceOp;
using xilinx::AIE::DMAChannelDir;
using xilinx::AIE::EndOp;
using xilinx::AIE::FlowOp;
using xilinx::AIE::MasterSetOp;
using xilinx::AIE::PacketDestOp;
using xilinx::AIE::PacketFlowOp;
using xilinx::AIE::PacketRuleOp;
using xilinx::AIE::PacketRulesOp;
using xilinx::AIE::PacketSourceOp;
using xilinx::AIE::ShimMuxOp;
using xilinx::AIE::SwitchboxOp;
using xilinx::AIE::TileOp;

#define DEBUG_TYPE "amdaie-create-pathfinder-flows"

namespace mlir::iree_compiler::AMDAIE {

TileOp getOrCreateTile(OpBuilder &builder, DeviceOp &device, int col, int row) {
  for (auto tile : device.getOps<TileOp>()) {
    if (tile.getCol() == col && tile.getRow() == row) return tile;
  }
  OpBuilder::InsertionGuard g(builder);
  return builder.create<TileOp>(builder.getUnknownLoc(), col, row);
}

SwitchboxOp getOrCreateSwitchbox(OpBuilder &builder, DeviceOp &device, int col,
                                 int row) {
  auto tile = getOrCreateTile(builder, device, col, row);
  for (auto i : tile.getResult().getUsers()) {
    if (llvm::isa<SwitchboxOp>(*i)) return llvm::cast<SwitchboxOp>(*i);
  }
  OpBuilder::InsertionGuard g(builder);
  auto sbOp = builder.create<SwitchboxOp>(builder.getUnknownLoc(), tile);
  SwitchboxOp::ensureTerminator(sbOp.getConnections(), builder,
                                builder.getUnknownLoc());
  return sbOp;
}

ShimMuxOp getOrCreateShimMux(OpBuilder &builder, DeviceOp &device, int col,
                             int row) {
  auto tile = getOrCreateTile(builder, device, col, row);
  for (auto i : tile.getResult().getUsers()) {
    if (auto shim = llvm::dyn_cast<ShimMuxOp>(*i)) return shim;
  }
  OpBuilder::InsertionGuard g(builder);
  auto shimMuxOp = builder.create<ShimMuxOp>(builder.getUnknownLoc(), tile);
  ShimMuxOp::ensureTerminator(shimMuxOp.getConnections(), builder,
                              builder.getUnknownLoc());
  return shimMuxOp;
}

ConnectOp getOrCreateConnect(OpBuilder &builder, Operation *parentOp,
                             StrmSwPortType srcBundle, int srcChannel,
                             StrmSwPortType destBundle, int destChannel) {
  Block &b = parentOp->getRegion(0).front();
  for (auto connect : b.getOps<ConnectOp>()) {
    if (connect.getSourceBundle() == srcBundle &&
        connect.getSourceChannel() == srcChannel &&
        connect.getDestBundle() == destBundle &&
        connect.getDestChannel() == destChannel)
      return connect;
  }
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(b.getTerminator());
  return builder.create<ConnectOp>(builder.getUnknownLoc(), srcBundle,
                                   srcChannel, destBundle, destChannel);
}

AMSelOp getOrCreateAMSel(OpBuilder &builder, SwitchboxOp &swboxOp,
                         int arbiterID, int msel) {
  Block &b = swboxOp.getConnections().front();
  builder.setInsertionPoint(b.getTerminator());
  for (auto amsel : swboxOp.getOps<AMSelOp>()) {
    builder.setInsertionPointAfter(amsel);
    if (amsel.getArbiterID() == arbiterID && amsel.getMsel() == msel)
      return amsel;
  }
  return builder.create<AMSelOp>(builder.getUnknownLoc(), arbiterID, msel);
}

MasterSetOp getOrCreateMasterSet(OpBuilder &builder, SwitchboxOp &swboxOp,
                                 StrmSwPortType bundle, int channel,
                                 ArrayRef<Value> amselVals) {
  Block &b = swboxOp.getConnections().front();
  builder.setInsertionPoint(b.getTerminator());
  for (auto masterSet : swboxOp.getOps<MasterSetOp>()) {
    builder.setInsertionPointAfter(masterSet);
    if (masterSet.getDestBundle() == bundle &&
        masterSet.getDestChannel() == channel &&
        masterSet.getAmsels() == amselVals)
      return masterSet;
  }
  return builder.create<MasterSetOp>(builder.getUnknownLoc(),
                                     builder.getIndexType(), bundle, channel,
                                     amselVals);
}

PacketRulesOp getOrCreatePacketRules(OpBuilder &builder, SwitchboxOp &swboxOp,
                                     StrmSwPortType bundle, int channel) {
  Block &b = swboxOp.getConnections().front();
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(b.getTerminator());
  for (auto packetRules : swboxOp.getOps<PacketRulesOp>()) {
    builder.setInsertionPointAfter(packetRules);
    if (packetRules.getSourceBundle() == bundle &&
        packetRules.getSourceChannel() == channel) {
      return packetRules;
    }
  }
  auto packetRules =
      builder.create<PacketRulesOp>(builder.getUnknownLoc(), bundle, channel);
  PacketRulesOp::ensureTerminator(packetRules.getRules(), builder,
                                  builder.getUnknownLoc());
  return packetRules;
}

LogicalResult runOnCircuitFlow(
    DeviceOp device, std::vector<FlowOp> &flowOps,
    const std::map<PathEndPoint, SwitchSettings> &flowSolutions) {
  OpBuilder builder(device.getContext());
  AMDAIEDeviceModel deviceModel =
      getDeviceModel(static_cast<AMDAIEDevice>(device.getDevice()));
  std::set<PathEndPoint> processedFlows;

  for (FlowOp flowOp : flowOps) {
    auto srcTile = llvm::cast<TileOp>(flowOp.getSource().getDefiningOp());
    TileLoc srcCoords = {static_cast<int>(srcTile.getCol()),
                         static_cast<int>(srcTile.getRow())};
    StrmSwPortType srcBundle = (flowOp.getSourceBundle());
    int srcChannel = flowOp.getSourceChannel();
    Port srcPort = {srcBundle, srcChannel};
    PathEndPoint srcPe{srcCoords.col, srcCoords.row, srcPort};
    if (processedFlows.count(srcPe)) {
      flowOp.erase();
      continue;
    }

    builder.setInsertionPointAfter(flowOp);
    for (auto &[curr, conns] :
         emitConnections(flowSolutions, srcPe, deviceModel)) {
      SwitchboxOp switchboxOp =
          getOrCreateSwitchbox(builder, device, curr.col, curr.row);
      for (const auto &conn : conns) {
        // Create switchboxes eagerly just to agree with mlir-aie tests.
        Operation *op;
        switch (conn.interconnect) {
          case Connect::Interconnect::SHIMMUX:
            op = getOrCreateShimMux(builder, device, conn.col, conn.row)
                     .getOperation();
            break;
          case Connect::Interconnect::SWB:
            op = switchboxOp.getOperation();
            break;
          case Connect::Interconnect::NOCARE:
            return flowOp->emitOpError("unsupported/unknown interconnect");
        }
        getOrCreateConnect(builder, op, conn.src.bundle, conn.src.channel,
                           conn.dst.bundle, conn.dst.channel);
      }
    }

    processedFlows.insert(srcPe);
    flowOp.erase();
  }

  return success();
}

struct AMDAIERouteFlowsWithPathfinderPass
    : public impl::AMDAIERouteFlowsWithPathfinderBase<
          AMDAIERouteFlowsWithPathfinderPass> {
  AMDAIERouteFlowsWithPathfinderPass(
      const AMDAIERouteFlowsWithPathfinderOptions &options)
      : AMDAIERouteFlowsWithPathfinderBase(options) {}

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<xilinx::AIE::AIEDialect>();
  }

  void runOnOperation() override;
};

FailureOr<std::tuple<PacketFlowMapT, SmallVector<PhysPortAndID>>>
getRoutedPacketFlows(DeviceOp device, AMDAIEDeviceModel &deviceModel) {
  PacketFlowMapT routedPacketFlows;
  SmallVector<PhysPortAndID> routedSlavePorts;
  // Iterate over all switchboxes and collect the packet flows.
  for (SwitchboxOp switchboxOp : device.getOps<SwitchboxOp>()) {
    TileOp t = xilinx::AIE::getTileOp(*switchboxOp.getOperation());
    TileLoc loc = {t.getCol(), t.getRow()};
    DenseMap<std::pair<uint8_t, uint8_t>, SmallVector<PhysPort>>
        amselToDestPhysPorts;
    DenseMap<std::pair<uint8_t, uint8_t>, SmallVector<PhysPortAndID>>
        amselToSrcPhysPortAndIDs;
    // Inside each switchbox, look for `MasterSetOp` and `PacketRulesOp`.
    for (Operation &op : switchboxOp.getConnections().getOps()) {
      if (auto masterSetOp = dyn_cast<MasterSetOp>(op)) {
        // Recover the original DMA port + channel from the special shim mux
        // port.
        if (std::optional<std::pair<StrmSwPortType, uint8_t>> mappedDmaPort =
                deviceModel.getDmaFromShimMuxPortMapping(
                    getConnectingBundle(masterSetOp.getDestBundle()),
                    masterSetOp.getDestChannel(), DMAChannelDir::S2MM);
            mappedDmaPort && deviceModel.isShimTile(t.getCol(), t.getRow())) {
          masterSetOp.setDestBundle(mappedDmaPort->first);
          masterSetOp.setDestChannel(mappedDmaPort->second);
        }
        // Get the destination for a packet flow.
        PhysPort destPhyPort = {
            loc,
            Port{masterSetOp.getDestBundle(), masterSetOp.getDestChannel()},
            PhysPort::Direction::DST};
        for (Value val : masterSetOp.getAmsels()) {
          AMSelOp amselOp = cast<AMSelOp>(val.getDefiningOp());
          uint8_t arbiterID = amselOp.getArbiterID();
          uint8_t msel = amselOp.getMsel();
          amselToDestPhysPorts[{arbiterID, msel}].push_back(destPhyPort);
        }
      } else if (auto packetRulesOp = dyn_cast<PacketRulesOp>(op)) {
        // Recover the original DMA port + channel from the special shim mux
        // port.
        if (std::optional<std::pair<StrmSwPortType, uint8_t>> mappedDmaPort =
                deviceModel.getDmaFromShimMuxPortMapping(
                    getConnectingBundle(packetRulesOp.getSourceBundle()),
                    packetRulesOp.getSourceChannel(), DMAChannelDir::MM2S);
            mappedDmaPort && deviceModel.isShimTile(t.getCol(), t.getRow())) {
          packetRulesOp.setSourceBundle(mappedDmaPort->first);
          packetRulesOp.setSourceChannel(mappedDmaPort->second);
        }
        // Get the source for a packet flow.
        PhysPort sourcePhyPort = {loc,
                                  Port{packetRulesOp.getSourceBundle(),
                                       packetRulesOp.getSourceChannel()},
                                  PhysPort::Direction::SRC};
        Block &block = packetRulesOp.getRules().front();
        for (auto packetRuleOp : block.getOps<PacketRuleOp>()) {
          AMSelOp amselOp =
              cast<AMSelOp>(packetRuleOp.getAmsel().getDefiningOp());
          uint8_t arbiterID = amselOp.getArbiterID();
          uint8_t msel = amselOp.getMsel();
          std::optional<ArrayRef<int>> maybePacketIds =
              packetRuleOp.getPacketIds();
          if (!maybePacketIds.has_value()) return failure();
          for (int pktId : *maybePacketIds) {
            amselToSrcPhysPortAndIDs[{arbiterID, msel}].push_back(
                {sourcePhyPort, pktId});
          }
        }
      }
    }
    // Infer the connections between source and destination ports, by matching
    // the amsel.
    for (const auto &[amsel, srcPhysPortAndIDs] : amselToSrcPhysPortAndIDs) {
      auto &destPhysPorts = amselToDestPhysPorts[amsel];
      for (const auto &srcPhysPortAndID : srcPhysPortAndIDs) {
        for (const auto &destPhysPort : destPhysPorts) {
          routedPacketFlows[srcPhysPortAndID].insert(
              {destPhysPort, srcPhysPortAndID.id});
        }
        routedSlavePorts.push_back(srcPhysPortAndID);
      }
    }
  }
  return std::make_tuple(routedPacketFlows, routedSlavePorts);
}

LogicalResult runOnPacketFlow(
    DeviceOp device, std::vector<PacketFlowOp> &pktFlowOps,
    const std::map<PathEndPoint, SwitchSettings> &flowSolutions,
    const PacketFlowMapT &existingPacketFlows,
    ArrayRef<PhysPortAndID> existingSlavePorts) {
  OpBuilder builder(device.getContext());
  AMDAIEDeviceModel deviceModel =
      getDeviceModel(static_cast<AMDAIEDevice>(device.getDevice()));
  mlir::DenseMap<TileLoc, TileOp> tiles;
  for (auto tileOp : device.getOps<TileOp>()) {
    int col = tileOp.getCol();
    int row = tileOp.getRow();
    tiles[{col, row}] = tileOp;
  }

  DenseMap<PhysPort, Attribute> keepPktHeaderAttr;
  TileLocToConnectionFlowIDT switchboxes;
  for (PacketFlowOp pktFlowOp : pktFlowOps) {
    int flowID = pktFlowOp.getID();
    Port srcPort{StrmSwPortType::SS_PORT_TYPE_MAX, -1};
    TileOp srcTile;
    TileLoc srcCoords{-1, -1};

    Block &b = pktFlowOp.getPorts().front();
    for (Operation &Op : b.getOperations()) {
      if (auto pktSource = llvm::dyn_cast<PacketSourceOp>(Op)) {
        srcTile = llvm::cast<TileOp>(pktSource.getTile().getDefiningOp());
        srcPort = {(pktSource.getBundle()),
                   static_cast<int>(pktSource.getChannel())};
        srcCoords = {srcTile.getCol(), srcTile.getRow()};
      } else if (auto pktDest = llvm::dyn_cast<PacketDestOp>(Op)) {
        TileOp destTile = llvm::cast<TileOp>(pktDest.getTile().getDefiningOp());
        Port destPort = {(pktDest.getBundle()), pktDest.getChannel()};
        TileLoc destCoord = {destTile.getCol(), destTile.getRow()};
        if (pktFlowOp->hasAttr("keep_pkt_header"))
          keepPktHeaderAttr[PhysPort{destCoord, destPort,
                                     PhysPort::Direction::DST}] =
              StringAttr::get(Op.getContext(), "true");
        assert(srcPort.bundle != StrmSwPortType::SS_PORT_TYPE_MAX &&
               srcPort.channel != -1 && "expected srcPort to have been set");
        assert(srcCoords.col != -1 && srcCoords.row != -1 &&
               "expected srcCoords to have been set");
        PathEndPoint srcPoint = {TileLoc{srcCoords.col, srcCoords.row},
                                 srcPort};
        // TODO(max): when does this happen???
        if (!flowSolutions.count(srcPoint)) continue;
        SwitchSettings settings = flowSolutions.at(srcPoint);
        // add connections for all the Switchboxes in SwitchSettings
        for (const auto &[curr, setting] : settings) {
          TileLoc currTile = {curr.col, curr.row};
          assert(setting.srcs.size() == setting.dsts.size());
          for (size_t i = 0; i < setting.srcs.size(); i++) {
            Port src = setting.srcs[i];
            Port dst = setting.dsts[i];
            // reject false broadcast
            if (!existsPathToDest(settings, currTile, dst.bundle, dst.channel,
                                  destCoord, destPort.bundle,
                                  destPort.channel)) {
              continue;
            }
            Connect connect = {Port{src.bundle, src.channel},
                               Port{dst.bundle, dst.channel},
                               Connect::Interconnect::NOCARE,
                               static_cast<uint8_t>(currTile.col),
                               static_cast<uint8_t>(currTile.row)};
            ConnectionAndFlowIDT connFlow = {connect, flowID};
            switchboxes[currTile].insert(connFlow);
            if (tiles.count(currTile) == 0) {
              builder.setInsertionPoint(device.getBody()->getTerminator());
              tiles[currTile] =
                  getOrCreateTile(builder, device, currTile.col, currTile.row);
            }
          }
        }
      }
    }
    pktFlowOp.erase();
  }

  SmallVector<TileLoc> tileLocs = llvm::map_to_vector(
      tiles, [](const llvm::detail::DenseMapPair<TileLoc, Operation *> &p) {
        return p.getFirst();
      });

  // Convert switchbox connections into packet flow maps from source/slave
  // 'port and id's to master/destination 'port and id's and keep track of all
  // source/slave ports.
  PacketFlowMapT packetFlows;
  SmallVector<PhysPortAndID> slavePorts;
  for (const auto &[tileId, connects] : switchboxes) {
    for (const auto &[conn, flowID] : connects) {
      PhysPortAndID sourceFlow = {
          PhysPort{tileId, conn.src, PhysPort::Direction::SRC}, flowID};
      packetFlows[sourceFlow].insert(
          {PhysPort{tileId, conn.dst, PhysPort::Direction::DST}, flowID});
      slavePorts.push_back(sourceFlow);
    }
  }

  auto maybeRoutingConfiguration = emitPacketRoutingConfiguration(
      deviceModel, packetFlows, existingPacketFlows);
  if (failed(maybeRoutingConfiguration)) {
    return device.emitOpError()
           << "could not create a valid routing configuration";
  }
  auto [masterSets, slaveAMSels] = maybeRoutingConfiguration.value();
  auto [slaveGroups, slaveMasks] = emitSlaveGroupsAndMasksRoutingConfig(
      slavePorts, packetFlows, existingSlavePorts, existingPacketFlows,
      deviceModel.getPacketIdMaskWidth());

  // Erase any duplicate packet rules in exising, pre-routed packet flows.
  std::vector<PacketRulesOp> existingPacketRules;
  for (SwitchboxOp switchboxOp : device.getOps<SwitchboxOp>()) {
    TileOp t = xilinx::AIE::getTileOp(*switchboxOp.getOperation());
    TileLoc loc = {t.getCol(), t.getRow()};
    for (Operation &op : switchboxOp.getConnections().getOps()) {
      if (auto packetRulesOp = dyn_cast<PacketRulesOp>(op)) {
        PhysPort sourcePhyPort = {loc,
                                  Port{packetRulesOp.getSourceBundle(),
                                       packetRulesOp.getSourceChannel()},
                                  PhysPort::Direction::SRC};
        if (slaveGroups.count(sourcePhyPort))
          existingPacketRules.push_back(packetRulesOp);
      }
    }
  }
  for (PacketRulesOp packetRulesOp : existingPacketRules) packetRulesOp.erase();

  // Realize the routes in MLIR
  for (auto &[tileLoc, tileOp] : tiles) {
    uint8_t numArbiters =
        1 + deviceModel.getStreamSwitchArbiterMax(tileLoc.col, tileLoc.row);
    uint8_t numMSels =
        1 + deviceModel.getStreamSwitchMSelMax(tileLoc.col, tileLoc.row);
    // Create a switchbox for the routes and insert inside it.
    builder.setInsertionPointAfter(tileOp);
    SwitchboxOp swbox =
        getOrCreateSwitchbox(builder, device, tileOp.getCol(), tileOp.getRow());
    SwitchboxOp::ensureTerminator(swbox.getConnections(), builder,
                                  builder.getUnknownLoc());
    std::vector<std::vector<bool>> amselOpNeededVector(
        numArbiters, std::vector<bool>(numMSels, false));
    for (const auto &[physPort, masterSet] : masterSets) {
      if (tileLoc != physPort.tileLoc) continue;
      for (auto [arbiter, msel] : masterSet)
        amselOpNeededVector.at(arbiter).at(msel) = true;
    }
    // Create all the amsel Ops
    DenseMap<std::pair<uint8_t, uint8_t>, AMSelOp> amselOps;
    for (int i = 0; i < numMSels; i++) {
      for (int a = 0; a < numArbiters; a++) {
        if (amselOpNeededVector.at(a).at(i))
          amselOps[{a, i}] = getOrCreateAMSel(builder, swbox, a, i);
      }
    }

    // Create all the master set Ops
    // First collect the master sets for this tile.
    SmallVector<Port> tileMasters;
    for (const auto &[physPort, _] : masterSets) {
      if (tileLoc != physPort.tileLoc) continue;
      tileMasters.push_back(physPort.port);
    }
    // Sort them so we get a reasonable order
    std::sort(tileMasters.begin(), tileMasters.end());
    for (Port tileMaster : tileMasters) {
      std::vector<std::pair<uint8_t, uint8_t>> amsels =
          masterSets[{tileLoc, tileMaster, PhysPort::Direction::DST}];
      std::vector<Value> amselVals;
      for (std::pair<uint8_t, uint8_t> amsel : amsels) {
        assert(amselOps.count(amsel) == 1 && "expected amsel in amselOps");
        amselVals.push_back(amselOps[amsel]);
      }
      MasterSetOp msOp = getOrCreateMasterSet(builder, swbox, tileMaster.bundle,
                                              tileMaster.channel, amselVals);
      if (auto pktFlowAttrs = keepPktHeaderAttr[{tileLoc, tileMaster,
                                                 PhysPort::Direction::DST}])
        msOp->setAttr("keep_pkt_header", pktFlowAttrs);
    }

    // Generate the packet rules.
    uint32_t numPacketRuleSlots =
        deviceModel.getNumPacketRuleSlots(tileLoc.col, tileLoc.row);
    DenseMap<Port, PacketRulesOp> slaveRules;
    for (auto &[physPort, groups] : slaveGroups) {
      if (tileLoc != physPort.tileLoc) continue;
      Port slave = physPort.port;
      for (std::set<uint32_t> &group : groups) {
        PhysPortAndID physPortAndId(physPort, *group.begin());
        uint32_t mask = slaveMasks[physPortAndId];
        uint32_t maskedId = physPortAndId.id & mask;

#ifndef NDEBUG
        // Verify that we actually map all the ID's correctly.
        for (uint32_t _pktId : group) assert((_pktId & mask) == maskedId);
#endif

        Value amsel = amselOps[slaveAMSels[physPortAndId]];
        PacketRulesOp packetrules =
            getOrCreatePacketRules(builder, swbox, slave.bundle, slave.channel);
        // Ensure the number of packet rules does not exceed the allowed slots.
        if (groups.size() > numPacketRuleSlots) {
          return packetrules.emitOpError()
                 << "Exceeded packet rule limit. Allowed: "
                 << numPacketRuleSlots << " Required: " << groups.size();
        }
        Block &rules = packetrules.getRules().front();
        builder.setInsertionPoint(rules.getTerminator());
        builder.create<PacketRuleOp>(
            builder.getUnknownLoc(), mask, maskedId, amsel,
            builder.getDenseI32ArrayAttr(
                std::vector<int>(group.begin(), group.end())));
      }
    }
  }

  // Add special shim mux connections for DMA/NOC streams.
  for (auto switchbox : make_early_inc_range(device.getOps<SwitchboxOp>())) {
    auto retVal = switchbox->getOperand(0);
    auto tileOp = retVal.getDefiningOp<TileOp>();
    // Only requires special connection for Shim/NOC tile.
    if (!deviceModel.isShimNOCTile(tileOp.getCol(), tileOp.getRow())) continue;
    // Skip any empty switchbox.
    if (&switchbox.getBody()->front() == switchbox.getBody()->getTerminator())
      continue;
    // Get the shim mux operation.
    builder.setInsertionPointAfter(tileOp);
    ShimMuxOp shimMuxOp =
        getOrCreateShimMux(builder, device, tileOp.getCol(), tileOp.getRow());
    for (Operation &op : switchbox.getConnections().getOps()) {
      if (auto packetRulesOp = dyn_cast<PacketRulesOp>(op)) {
        // Found the source (MM2S) of a packet flow.
        StrmSwPortType srcBundle = packetRulesOp.getSourceBundle();
        uint8_t srcChannel = packetRulesOp.getSourceChannel();
        std::optional<std::pair<StrmSwPortType, uint8_t>> mappedShimMuxPort =
            deviceModel.getShimMuxPortMappingForDmaOrNoc(srcBundle, srcChannel,
                                                         DMAChannelDir::MM2S);
        if (!mappedShimMuxPort) continue;
        StrmSwPortType newSrcBundle = mappedShimMuxPort->first;
        uint8_t newSrcChannel = mappedShimMuxPort->second;
        // Add a special connection from `srcBundle/srcChannel` to
        // `newSrcBundle/newSrcChannel`.
        getOrCreateConnect(builder, shimMuxOp, srcBundle, srcChannel,
                           newSrcBundle, newSrcChannel);
        // Replace the source bundle and channel. `getConnectingBundle` is
        // used to update bundle direction from shim mux to shim switchbox.
        packetRulesOp.setSourceBundle(getConnectingBundle(newSrcBundle));
        packetRulesOp.setSourceChannel(newSrcChannel);

      } else if (auto masterSetOp = dyn_cast<MasterSetOp>(op)) {
        // Found the destination (S2MM) of a packet flow.
        StrmSwPortType destBundle = masterSetOp.getDestBundle();
        uint8_t destChannel = masterSetOp.getDestChannel();
        std::optional<std::pair<StrmSwPortType, uint8_t>> mappedShimMuxPort =
            deviceModel.getShimMuxPortMappingForDmaOrNoc(
                destBundle, destChannel, DMAChannelDir::S2MM);
        if (!mappedShimMuxPort) continue;
        StrmSwPortType newDestBundle = mappedShimMuxPort->first;
        uint8_t newDestChannel = mappedShimMuxPort->second;
        // Add a special connection from `newDestBundle/newDestChannel` to
        // `destBundle/destChannel`.
        getOrCreateConnect(builder, shimMuxOp, newDestBundle, newDestChannel,
                           destBundle, destChannel);
        // Replace the destination bundle and channel. `getConnectingBundle` is
        // used to update bundle direction from shim mux to shim switchbox.
        masterSetOp.setDestBundle(getConnectingBundle(newDestBundle));
        masterSetOp.setDestChannel(newDestChannel);
      }
    }
  }
  return success();
}

/// Rough outline:
/// 1. find aie.flow ops and translate them into the structs/data-structures
/// the
///    router expects
/// 2. find aie.packet_flow ops and translate them into the
///    structs/data-structures the router expects
/// 3. add existing ("fixed") internal switchbox connections
/// 4. run the router (findPaths)
/// 5. run the ConvertFlowsToInterconnect rewrite pattern that uses the router
///    results to translate flow ops into aie.switchbox ops that can contain
///    aie.connects
/// 6. feed the router results to runOnPacketFlow to insert packet routing ops
/// (aie.packet_rules, aie.amsel, etc) into aie.switchboxes
void AMDAIERouteFlowsWithPathfinderPass::runOnOperation() {
  DeviceOp device = getOperation();
  std::map<PathEndPoint, SwitchSettings> flowSolutions;
  // don't be clever and remove these initializations because
  // then you're doing a max against garbage data...
  uint8_t maxCol = 0, maxRow = 0;
  for (TileOp tileOp : device.getOps<TileOp>()) {
    maxCol = std::max(maxCol, tileOp.getCol());
    maxRow = std::max(maxRow, tileOp.getRow());
  }

  AMDAIEDeviceModel deviceModel =
      getDeviceModel(static_cast<AMDAIEDevice>(device.getDevice()));
  Router pathfinder(maxCol, maxRow);
  pathfinder.initialize(deviceModel);

  // for each flow in the device, add it to pathfinder
  // each source can map to multiple different destinations (fanout)
  std::vector<FlowOp> circuitFlowOps;
  for (FlowOp flowOp : device.getOps<FlowOp>()) {
    // Skip flows based on control/non-control routing flags.
    bool isCtrlFlow = (flowOp.getSourceBundle() == StrmSwPortType::CTRL) ||
                      (flowOp.getDestBundle() == StrmSwPortType::CTRL);
    if ((isCtrlFlow && !routeCtrl) || (!isCtrlFlow && !routeNonCtrl)) continue;

    // Not skipped, add the flow to the pathfinder.
    TileOp srcTile = llvm::cast<TileOp>(flowOp.getSource().getDefiningOp());
    TileOp dstTile = llvm::cast<TileOp>(flowOp.getDest().getDefiningOp());
    TileLoc srcCoords = {srcTile.getCol(), srcTile.getRow()};
    TileLoc dstCoords = {dstTile.getCol(), dstTile.getRow()};
    Port srcPort = {(flowOp.getSourceBundle()), flowOp.getSourceChannel()};
    Port dstPort = {(flowOp.getDestBundle()), flowOp.getDestChannel()};
    pathfinder.addFlow(srcCoords, srcPort, dstCoords, dstPort,
                       /*isPacketFlow=*/false);
    circuitFlowOps.push_back(flowOp);
  }

  std::vector<PacketFlowOp> packetFlowOps;
  for (PacketFlowOp pktFlowOp : device.getOps<PacketFlowOp>()) {
    Region &r = pktFlowOp.getPorts();
    Block &b = r.front();
    SmallVector<PacketSourceOp> pktSrcs;
    SmallVector<PacketDestOp> pktDests;
    for (Operation &op : b.getOperations()) {
      if (auto pktSourceOp = llvm::dyn_cast<PacketSourceOp>(op)) {
        pktSrcs.push_back(pktSourceOp);
      } else if (auto pktDestOp = llvm::dyn_cast<PacketDestOp>(op)) {
        pktDests.push_back(pktDestOp);
      }
    }
    if (pktSrcs.size() != 1 || pktDests.size() < 1) {
      pktFlowOp.emitOpError()
          << "expected exactly one source and at least one dest";
      return signalPassFailure();
    }
    // Skip flows based on control/non-control routing flags.For desitnation,
    // we only need to check the first one, as the `CTRL` flow would never be
    // a broadcast.
    bool isCtrlFlow = (pktSrcs[0].getBundle() == StrmSwPortType::CTRL) ||
                      (pktDests[0].getBundle() == StrmSwPortType::CTRL);
    if ((isCtrlFlow && !routeCtrl) || (!isCtrlFlow && !routeNonCtrl)) continue;

    // Not skipped, add the flow to the pathfinder.
    TileOp srcTile = llvm::cast<TileOp>(pktSrcs[0].getTile().getDefiningOp());
    TileLoc srcCoords = {srcTile.getCol(), srcTile.getRow()};
    Port srcPort = {(pktSrcs[0].getBundle()), pktSrcs[0].getChannel()};
    for (PacketDestOp pktDest : pktDests) {
      TileOp dstTile = llvm::cast<TileOp>(pktDest.getTile().getDefiningOp());
      TileLoc dstCoords = {dstTile.getCol(), dstTile.getRow()};
      Port dstPort = {(pktDest.getBundle()), pktDest.getChannel()};
      pathfinder.addFlow(srcCoords, srcPort, dstCoords, dstPort,
                         /*isPacketFlow=*/true);
    }
    packetFlowOps.push_back(pktFlowOp);
  }

  // Mark existing circuit-mode connections as unavailable in Pathfinder.
  // These pre-routed connections are fixed and will not be modified.
  for (SwitchboxOp switchboxOp : device.getOps<SwitchboxOp>()) {
    std::vector<std::tuple<Port, Port>> connects;
    for (ConnectOp connectOp : switchboxOp.getOps<ConnectOp>()) {
      Port src(connectOp.getSourceBundle(), connectOp.getSourceChannel());
      Port dst(connectOp.getDestBundle(), connectOp.getDestChannel());
      connects.emplace_back(src, dst);
    }
    TileOp t = xilinx::AIE::getTileOp(*switchboxOp.getOperation());
    if (!pathfinder.addFixedCircuitConnection(t.getCol(), t.getRow(),
                                              connects)) {
      switchboxOp.emitOpError() << "Unable to add fixed circuit connections";
      return signalPassFailure();
    }
  }

  // Recover packet flows that already routed by looking for existing `amsel`
  // + `packet_rules` and `masterset` operations.
  PacketFlowMapT existingPacketFlows;
  SmallVector<PhysPortAndID> existingSlavePorts;
  auto result = getRoutedPacketFlows(device, deviceModel);
  if (failed(result)) {
    device.emitError("Unable to recover existing packet flows");
    return signalPassFailure();
  } else {
    std::tie(existingPacketFlows, existingSlavePorts) = result.value();
  }
  // Add fixed packet connections to the pathfinder.
  for (const auto &[srcPhysPortAndID, destPhysPortAndIDs] :
       existingPacketFlows) {
    for (const PhysPortAndID &destPhysPortAndID : destPhysPortAndIDs) {
      if (!pathfinder.addFixedPacketConnection(srcPhysPortAndID.physPort,
                                               destPhysPortAndID.physPort)) {
        device.emitError("Unable to add fixed packet connections");
        return signalPassFailure();
      }
    }
  }

  // all flows are now populated, call the congestion-aware pathfinder
  // algorithm
  // check whether the pathfinder algorithm creates a legal routing
  if (auto maybeFlowSolutions = pathfinder.findPaths(/*maxIterations=*/1000)) {
    flowSolutions.swap(maybeFlowSolutions.value());
  } else {
    device.emitError("Unable to find a legal routing");
    return signalPassFailure();
  }

  if (routeCircuit &&
      failed(runOnCircuitFlow(device, circuitFlowOps, flowSolutions))) {
    device.emitError("failed to convert circuit flows to interconnects");
    return signalPassFailure();
  }

  if (routePacket &&
      failed(runOnPacketFlow(device, packetFlowOps, flowSolutions,
                             existingPacketFlows, existingSlavePorts))) {
    device.emitError("failed to convert packet flows to amsels and rules");
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<DeviceOp>>
createAMDAIERouteFlowsWithPathfinderPass(
    AMDAIERouteFlowsWithPathfinderOptions options) {
  return std::make_unique<AMDAIERouteFlowsWithPathfinderPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
