// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cassert>
#include <list>
#include <set>

#include "Passes.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "iree-amd-aie/aie_runtime/iree_aie_router.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_os_ostream.h"
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
using xilinx::AIE::Interconnect;
using xilinx::AIE::MasterSetOp;
using xilinx::AIE::PacketDestOp;
using xilinx::AIE::PacketFlowOp;
using xilinx::AIE::PacketRuleOp;
using xilinx::AIE::PacketRulesOp;
using xilinx::AIE::PacketSourceOp;
using xilinx::AIE::ShimMuxOp;
using xilinx::AIE::SwitchboxOp;
using xilinx::AIE::TileOp;
using xilinx::AIE::WireOp;

#define DEBUG_TYPE "amdaie-create-pathfinder-flows"
#define OVER_CAPACITY_COEFF 0.02
#define USED_CAPACITY_COEFF 0.02
#define DEMAND_COEFF 1.1

namespace mlir::iree_compiler::AMDAIE {

StrmSwPortType toStrmT(xilinx::AIE::WireBundle w) {
  switch (w) {
    case xilinx::AIE::WireBundle::Core:
      return StrmSwPortType::CORE;
    case xilinx::AIE::WireBundle::DMA:
      return StrmSwPortType::DMA;
    case xilinx::AIE::WireBundle::FIFO:
      return StrmSwPortType::FIFO;
    case xilinx::AIE::WireBundle::South:
      return StrmSwPortType::SOUTH;
    case xilinx::AIE::WireBundle::West:
      return StrmSwPortType::WEST;
    case xilinx::AIE::WireBundle::North:
      return StrmSwPortType::NORTH;
    case xilinx::AIE::WireBundle::East:
      return StrmSwPortType::EAST;
    case xilinx::AIE::WireBundle::NOC:
      return StrmSwPortType::NOC;
    case xilinx::AIE::WireBundle::Trace:
      return StrmSwPortType::TRACE;
    case xilinx::AIE::WireBundle::Ctrl:
      return StrmSwPortType::CTRL;
    default:
      llvm::report_fatal_error("unhandled xilinx::AIE::WireBundle");
  }
}

xilinx::AIE::WireBundle toWireB(StrmSwPortType w) {
  switch (w) {
    case StrmSwPortType::CORE:
      return xilinx::AIE::WireBundle::Core;
    case StrmSwPortType::DMA:
      return xilinx::AIE::WireBundle::DMA;
    case StrmSwPortType::FIFO:
      return xilinx::AIE::WireBundle::FIFO;
    case StrmSwPortType::SOUTH:
      return xilinx::AIE::WireBundle::South;
    case StrmSwPortType::WEST:
      return xilinx::AIE::WireBundle::West;
    case StrmSwPortType::NORTH:
      return xilinx::AIE::WireBundle::North;
    case StrmSwPortType::EAST:
      return xilinx::AIE::WireBundle::East;
    case StrmSwPortType::TRACE:
      return xilinx::AIE::WireBundle::Trace;
    case StrmSwPortType::NOC:
      return xilinx::AIE::WireBundle::NOC;
    case StrmSwPortType::CTRL:
      return xilinx::AIE::WireBundle::Ctrl;
    default:
      llvm::report_fatal_error("unhandled xilinx::AIE::WireBundle");
  }
}

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

ShimMuxOp getOrCreateShimMux(OpBuilder &builder, DeviceOp &device, int col) {
  auto tile = getOrCreateTile(builder, device, col, /*row*/ 0);
  for (auto i : tile.getResult().getUsers()) {
    if (auto shim = llvm::dyn_cast<ShimMuxOp>(*i)) return shim;
  }
  OpBuilder::InsertionGuard g(builder);
  auto shmuxOp = builder.create<ShimMuxOp>(builder.getUnknownLoc(), tile);
  ShimMuxOp::ensureTerminator(shmuxOp.getConnections(), builder,
                              builder.getUnknownLoc());
  return shmuxOp;
}

struct ConvertFlowsToInterconnect : OpConversionPattern<FlowOp> {
  using OpConversionPattern::OpConversionPattern;
  const std::map<PathEndPoint, SwitchSettings> flowSolutions;
  std::set<PathEndPoint> processedFlows;

  ConvertFlowsToInterconnect(
      MLIRContext *context,
      const std::map<PathEndPoint, SwitchSettings> &flowSolutions,
      std::set<PathEndPoint> &processedFlows, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit),
        flowSolutions(flowSolutions),
        processedFlows(processedFlows) {}

  LogicalResult matchAndRewrite(
      FlowOp flowOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto srcTile = llvm::cast<TileOp>(flowOp.getSource().getDefiningOp());
    TileLoc srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
    StrmSwPortType srcBundle = toStrmT(flowOp.getSourceBundle());
    int srcChannel = flowOp.getSourceChannel();
    Port srcPort = {srcBundle, srcChannel};
    PathEndPoint srcPe{srcCoords.col, srcCoords.row, srcPort};
    if (processedFlows.count(srcPe)) {
      rewriter.eraseOp(flowOp);
      return success();
    }

    DeviceOp device = flowOp->getParentOfType<DeviceOp>();
    AMDAIEDeviceModel deviceModel =
        getDeviceModel(static_cast<AMDAIEDevice>(device.getDevice()));
    for (auto &[curr, conns] :
         emitConnections(flowSolutions, srcPe, deviceModel)) {
      SwitchboxOp switchboxOp =
          getOrCreateSwitchbox(rewriter, device, curr.col, curr.row);
      for (const auto &conn : conns) {
        // create switchboxes eagerly just to agree with mlir-aie tests
        Operation *op;
        switch (conn.interconnect) {
          case Connect::Interconnect::SHIMMUX:
            op = getOrCreateShimMux(rewriter, device, conn.col).getOperation();
            break;
          case Connect::Interconnect::SWB:
            op = switchboxOp.getOperation();
            break;
          case Connect::Interconnect::NOCARE:
            return flowOp->emitOpError("unsupported/unknown interconnect");
        }

        Block &b = op->getRegion(0).front();
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(b.getTerminator());
        rewriter.create<ConnectOp>(rewriter.getUnknownLoc(),
                                   toWireB(conn.src.bundle), conn.src.channel,
                                   toWireB(conn.dst.bundle), conn.dst.channel);
      }
    }

    const_cast<ConvertFlowsToInterconnect *>(this)->processedFlows.insert(
        srcPe);
    rewriter.eraseOp(flowOp);
    return success();
  }
};

struct AIEPathfinderPass
    : PassWrapper<AIEPathfinderPass, OperationPass<DeviceOp>> {
  AIEPathfinderPass() = default;
  AIEPathfinderPass(const AIEPathfinderPass &pass) : PassWrapper(pass) {}
  AIEPathfinderPass(const AIERoutePathfinderFlowsOptions &options)
      : AIEPathfinderPass() {
    clRouteCircuit = options.clRouteCircuit;
    clRoutePacket = options.clRoutePacket;
  }

  llvm::StringRef getArgument() const override {
    return "amdaie-create-pathfinder-flows";
  }

  void runOnOperation() override;

  mlir::Pass::Option<bool> clRouteCircuit{
      *this, "route-circuit",
      llvm::cl::desc("Flag to enable aie.flow lowering."),
      llvm::cl::init(true)};
  mlir::Pass::Option<bool> clRoutePacket{
      *this, "route-packet",
      llvm::cl::desc("Flag to enable aie.packetflow lowering."),
      llvm::cl::init(true)};
};

LogicalResult runOnPacketFlow(
    DeviceOp device, OpBuilder &builder,
    const std::map<PathEndPoint, SwitchSettings> &flowSolutions) {
  mlir::DenseMap<TileLoc, TileOp> tiles;
  for (auto tileOp : device.getOps<TileOp>()) {
    int col = tileOp.colIndex();
    int row = tileOp.rowIndex();
    tiles[{col, row}] = tileOp;
  }

  DenseMap<PhysPort, Attribute> keepPktHeaderAttr;
  SwitchBoxToConnectionFlowIDT switchboxes;
  for (PacketFlowOp pktFlowOp : device.getOps<PacketFlowOp>()) {
    int flowID = pktFlowOp.IDInt();
    Port srcPort{StrmSwPortType::SS_PORT_TYPE_MAX, -1};
    TileOp srcTile;
    TileLoc srcCoords{-1, -1};

    Block &b = pktFlowOp.getPorts().front();
    for (Operation &Op : b.getOperations()) {
      if (auto pktSource = llvm::dyn_cast<PacketSourceOp>(Op)) {
        srcTile = llvm::cast<TileOp>(pktSource.getTile().getDefiningOp());
        srcPort = {toStrmT(pktSource.port().bundle), pktSource.port().channel};
        srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
      } else if (auto pktDest = llvm::dyn_cast<PacketDestOp>(Op)) {
        TileOp destTile = llvm::cast<TileOp>(pktDest.getTile().getDefiningOp());
        Port destPort = {toStrmT(pktDest.port().bundle),
                         pktDest.port().channel};
        TileLoc destCoord = {destTile.colIndex(), destTile.rowIndex()};
        if (pktFlowOp->hasAttr("keep_pkt_header"))
          keepPktHeaderAttr[PhysPort{destCoord, destPort}] =
              StringAttr::get(Op.getContext(), "true");
        assert(srcPort.bundle != StrmSwPortType::SS_PORT_TYPE_MAX &&
               srcPort.channel != -1 && "expected srcPort to have been set");
        assert(srcCoords.col != -1 && srcCoords.row != -1 &&
               "expected srcCoords to have been set");
        PathEndPoint srcPoint = {SwitchBox{srcCoords.col, srcCoords.row},
                                 srcPort};
        // TODO(max): when does this happen???
        if (!flowSolutions.count(srcPoint)) continue;
        SwitchSettings settings = flowSolutions.at(srcPoint);
        // add connections for all the Switchboxes in SwitchSettings
        for (const auto &[curr, setting] : settings) {
          for (const auto &[bundle, channel] : setting.dsts) {
            TileLoc currTile = {curr.col, curr.row};
            // reject false broadcast
            if (!existsPathToDest(settings, currTile, bundle, channel,
                                  destCoord, destPort.bundle,
                                  destPort.channel)) {
              continue;
            }
            Connect connect = {Port{setting.src.bundle, setting.src.channel},
                               Port{bundle, channel},
                               Connect::Interconnect::NOCARE};
            ConnectionAndFlowIDT connFlow = {connect, flowID};
            switchboxes[currTile].insert(connFlow);
          }
        }
      }
    }
  }

  SmallVector<TileLoc> tileLocs = llvm::map_to_vector(
      tiles, [](const llvm::detail::DenseMapPair<TileLoc, Operation *> &p) {
        return p.getFirst();
      });

  int numMsels = 4;
  int numArbiters = 6;
  auto [masterSets, slaveGroups, slaveMasks, slaveAMSels] =
      emitPacketRoutingConfiguration(numMsels, numArbiters, switchboxes,
                                     tileLocs);

  // Realize the routes in MLIR
  for (auto &[tileLoc, tileOp] : tiles) {
    // Create a switchbox for the routes and insert inside it.
    builder.setInsertionPointAfter(tileOp);
    SwitchboxOp swbox =
        getOrCreateSwitchbox(builder, device, tileOp.getCol(), tileOp.getRow());
    SwitchboxOp::ensureTerminator(swbox.getConnections(), builder,
                                  builder.getUnknownLoc());
    Block &b = swbox.getConnections().front();
    builder.setInsertionPoint(b.getTerminator());

    std::vector<bool> amselOpNeededVector(32);
    for (const auto &[physPort, masterSet] : masterSets) {
      if (tileLoc != physPort.tileLoc) continue;
      for (int value : masterSet) amselOpNeededVector[value] = true;
    }
    // Create all the amsel Ops
    DenseMap<int, AMSelOp> amselOps;
    for (int i = 0; i < 32; i++) {
      if (!amselOpNeededVector[i]) continue;
      int arbiterID = i % numArbiters;
      int msel = i / numArbiters;
      auto amsel =
          builder.create<AMSelOp>(builder.getUnknownLoc(), arbiterID, msel);
      amselOps[i] = amsel;
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
      SmallVector<int> msels = masterSets[{tileLoc, tileMaster}];
      SmallVector<Value> amsels;
      for (int msel : msels) {
        assert(amselOps.count(msel) == 1 && "expected msel in amselOps");
        amsels.push_back(amselOps[msel]);
      }
      auto msOp = builder.create<MasterSetOp>(
          builder.getUnknownLoc(), builder.getIndexType(),
          toWireB(tileMaster.bundle), tileMaster.channel, amsels);
      if (auto pktFlowAttrs = keepPktHeaderAttr[{tileLoc, tileMaster}])
        msOp->setAttr("keep_pkt_header", pktFlowAttrs);
    }

    // Generate the packet rules
    DenseMap<Port, PacketRulesOp> slaveRules;
    for (SmallVector<PhysPortAndID> group : slaveGroups) {
      builder.setInsertionPoint(b.getTerminator());
      PhysPort physPort = group.front().physPort;
      if (tileLoc != physPort.tileLoc) continue;
      Port slave = physPort.port;
      StrmSwPortType bundle = slave.bundle;
      int channel = slave.channel;
      int mask = slaveMasks[group.front()];
      int ID = group.front().id & mask;

#ifndef NDEBUG
      // Verify that we actually map all the ID's correctly.
      for (PhysPortAndID _slave : group) assert((_slave.id & mask) == ID);
#endif

      Value amsel = amselOps[slaveAMSels[group.front()]];
      PacketRulesOp packetrules;
      if (slaveRules.count(slave) == 0) {
        packetrules = builder.create<PacketRulesOp>(builder.getUnknownLoc(),
                                                    toWireB(bundle), channel);
        PacketRulesOp::ensureTerminator(packetrules.getRules(), builder,
                                        builder.getUnknownLoc());
        slaveRules[slave] = packetrules;
      } else {
        packetrules = slaveRules[slave];
      }

      Block &rules = packetrules.getRules().front();
      builder.setInsertionPoint(rules.getTerminator());
      builder.create<PacketRuleOp>(builder.getUnknownLoc(), mask, ID, amsel);
    }
  }

  // Add support for shimDMA
  // From shimDMA to BLI: 1) shimDMA 0 --> North 3
  //                      2) shimDMA 1 --> North 7
  // From BLI to shimDMA: 1) North   2 --> shimDMA 0
  //                      2) North   3 --> shimDMA 1
  AMDAIEDeviceModel deviceModel =
      getDeviceModel(static_cast<AMDAIEDevice>(device.getDevice()));
  for (auto switchbox : make_early_inc_range(device.getOps<SwitchboxOp>())) {
    auto retVal = switchbox->getOperand(0);
    auto tileOp = retVal.getDefiningOp<TileOp>();

    if (!deviceModel.isShimNOCTile(tileOp.getCol(), tileOp.getRow())) continue;
    // Check if the switchbox is empty
    if (&switchbox.getBody()->front() == switchbox.getBody()->getTerminator())
      continue;

    ShimMuxOp shimMuxOp = nullptr;
    for (auto shimmux : device.getOps<ShimMuxOp>()) {
      if (shimmux.getTile() != tileOp) continue;
      shimMuxOp = shimmux;
      break;
    }
    if (!shimMuxOp) {
      builder.setInsertionPointAfter(tileOp);
      shimMuxOp = getOrCreateShimMux(builder, device, tileOp.getCol());
    }

    for (Operation &op : switchbox.getConnections().getOps()) {
      // check if there is MM2S DMA in the switchbox of the 0th row
      if (auto pktrules = llvm::dyn_cast<PacketRulesOp>(op);
          pktrules &&
          toStrmT(pktrules.getSourceBundle()) == StrmSwPortType::DMA) {
        // If there is, then it should be put into the corresponding shimmux
        OpBuilder::InsertionGuard g(builder);
        Block &b0 = shimMuxOp.getConnections().front();
        builder.setInsertionPointToStart(&b0);
        pktrules.setSourceBundle(toWireB(StrmSwPortType::SOUTH));
        if (pktrules.getSourceChannel() == 0) {
          pktrules.setSourceChannel(3);
          builder.create<ConnectOp>(builder.getUnknownLoc(),
                                    toWireB(StrmSwPortType::DMA), 0,
                                    toWireB(StrmSwPortType::NORTH), 3);
        } else if (pktrules.getSourceChannel() == 1) {
          pktrules.setSourceChannel(7);
          builder.create<ConnectOp>(builder.getUnknownLoc(),
                                    toWireB(StrmSwPortType::DMA), 1,
                                    toWireB(StrmSwPortType::NORTH), 7);
        }
      }

      // check if there is S2MM DMA in the switchbox of the 0th row
      if (auto mtset = llvm::dyn_cast<MasterSetOp>(op);
          mtset && toStrmT(mtset.getDestBundle()) == StrmSwPortType::DMA) {
        // If there is, then it should be put into the corresponding shimmux
        OpBuilder::InsertionGuard g(builder);
        Block &b0 = shimMuxOp.getConnections().front();
        builder.setInsertionPointToStart(&b0);
        mtset.setDestBundle(toWireB(StrmSwPortType::SOUTH));
        if (mtset.getDestChannel() == 0) {
          mtset.setDestChannel(2);
          builder.create<ConnectOp>(builder.getUnknownLoc(),
                                    toWireB(StrmSwPortType::NORTH), 2,
                                    toWireB(StrmSwPortType::DMA), 0);
        } else if (mtset.getDestChannel() == 1) {
          mtset.setDestChannel(3);
          builder.create<ConnectOp>(builder.getUnknownLoc(),
                                    toWireB(StrmSwPortType::NORTH), 3,
                                    toWireB(StrmSwPortType::DMA), 1);
        }
      }
    }
  }
  return success();
}

/// Rough outline:
/// 1. find aie.flow ops and translate them into the structs/data-structures the
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
void AIEPathfinderPass::runOnOperation() {
  DeviceOp device = getOperation();
  Router pathfinder;
  std::map<PathEndPoint, SwitchSettings> flowSolutions;
  std::set<PathEndPoint> processedFlows;
  // don't be clever and remove these initializations because
  // then you're doing a max against garbage data...
  int maxCol = 0, maxRow = 0;
  for (TileOp tileOp : device.getOps<TileOp>()) {
    maxCol = std::max(maxCol, tileOp.colIndex());
    maxRow = std::max(maxRow, tileOp.rowIndex());
  }

  AMDAIEDeviceModel deviceModel =
      getDeviceModel(static_cast<AMDAIEDevice>(device.getDevice()));
  pathfinder.initialize(maxCol, maxRow, deviceModel);

  // for each flow in the device, add it to pathfinder
  // each source can map to multiple different destinations (fanout)
  for (FlowOp flowOp : device.getOps<FlowOp>()) {
    TileOp srcTile = llvm::cast<TileOp>(flowOp.getSource().getDefiningOp());
    TileOp dstTile = llvm::cast<TileOp>(flowOp.getDest().getDefiningOp());
    TileLoc srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
    TileLoc dstCoords = {dstTile.colIndex(), dstTile.rowIndex()};
    Port srcPort = {toStrmT(flowOp.getSourceBundle()),
                    flowOp.getSourceChannel()};
    Port dstPort = {toStrmT(flowOp.getDestBundle()), flowOp.getDestChannel()};
    pathfinder.addFlow(srcCoords, srcPort, dstCoords, dstPort, false);
  }

  for (PacketFlowOp pktFlowOp : device.getOps<PacketFlowOp>()) {
    Region &r = pktFlowOp.getPorts();
    Block &b = r.front();
    Port srcPort{StrmSwPortType::SS_PORT_TYPE_MAX, -1};
    TileOp srcTile;
    TileLoc srcCoords{-1, -1};
    for (Operation &op : b.getOperations()) {
      if (auto pktSource = llvm::dyn_cast<PacketSourceOp>(op)) {
        srcTile = llvm::cast<TileOp>(pktSource.getTile().getDefiningOp());
        srcPort = {toStrmT(pktSource.port().bundle), pktSource.port().channel};
        srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
      } else if (auto pktDest = llvm::dyn_cast<PacketDestOp>(op)) {
        TileOp dstTile = llvm::cast<TileOp>(pktDest.getTile().getDefiningOp());
        Port dstPort = {toStrmT(pktDest.port().bundle), pktDest.port().channel};
        TileLoc dstCoords = {dstTile.colIndex(), dstTile.rowIndex()};
        assert(srcPort.bundle != StrmSwPortType::SS_PORT_TYPE_MAX &&
               srcPort.channel != -1 && "expected srcPort to have been set");
        assert(srcCoords.col != -1 && srcCoords.row != -1);
        pathfinder.addFlow(srcCoords, srcPort, dstCoords, dstPort, true);
      }
    }
  }

  // add existing connections so Pathfinder knows which resources are
  // available search all existing SwitchBoxOps for exising connections
  for (SwitchboxOp switchboxOp : device.getOps<SwitchboxOp>()) {
    std::vector<std::tuple<StrmSwPortType, int, StrmSwPortType, int>> connects;
    for (ConnectOp connectOp : switchboxOp.getOps<ConnectOp>()) {
      connects.emplace_back(toStrmT(connectOp.sourcePort().bundle),
                            connectOp.sourcePort().channel,
                            toStrmT(connectOp.destPort().bundle),
                            connectOp.destPort().channel);
    }
    if (!pathfinder.addFixedConnection(switchboxOp.colIndex(),
                                       switchboxOp.rowIndex(), connects)) {
      switchboxOp.emitOpError() << "Unable to add fixed connections";
      return signalPassFailure();
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

  OpBuilder builder = OpBuilder::atBlockEnd(device.getBody());

  ConversionTarget target(getContext());
  target.addLegalOp<TileOp>();
  target.addLegalOp<ConnectOp>();
  target.addLegalOp<SwitchboxOp>();
  target.addLegalOp<ShimMuxOp>();
  target.addLegalOp<EndOp>();

  RewritePatternSet patterns(&getContext());
  patterns.insert<ConvertFlowsToInterconnect>(&getContext(), flowSolutions,
                                              processedFlows);
  if (clRouteCircuit &&
      failed(applyPartialConversion(device, target, std::move(patterns)))) {
    device.emitError("failed to convert routed flows to interconnects");
    return signalPassFailure();
  }

  if (clRoutePacket && failed(runOnPacketFlow(device, builder, flowSolutions)))
    return signalPassFailure();
}

std::unique_ptr<OperationPass<DeviceOp>> createAMDAIEPathfinderPass() {
  return std::make_unique<AIEPathfinderPass>();
}

void registerAMDAIERoutePathfinderFlows() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createAMDAIEPathfinderPass();
  });
}
}  // namespace mlir::iree_compiler::AMDAIE
