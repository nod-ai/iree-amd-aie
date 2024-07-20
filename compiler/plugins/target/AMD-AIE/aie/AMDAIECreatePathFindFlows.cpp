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
using xilinx::AIE::PLIOOp;
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
    case xilinx::AIE::WireBundle::PLIO:
      return StrmSwPortType::PLIO;
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
    case StrmSwPortType::PLIO:
      return xilinx::AIE::WireBundle::PLIO;
    case StrmSwPortType::NOC:
      return xilinx::AIE::WireBundle::NOC;
    case StrmSwPortType::CTRL:
      return xilinx::AIE::WireBundle::Ctrl;
    default:
      llvm::report_fatal_error("unhandled xilinx::AIE::WireBundle");
  }
}

}  // namespace mlir::iree_compiler::AMDAIE

namespace mlir::iree_compiler::AMDAIE {

// DynamicTileAnalysis integrates the Pathfinder class into the MLIR
// environment. It passes flows to the Pathfinder as ordered pairs of ints.
// Detailed routing is received as SwitchboxSettings
// It then converts these settings to MLIR operations
class DynamicTileAnalysis {
 public:
  int maxCol, maxRow;
  Pathfinder pathfinder;
  std::map<PathEndPoint, SwitchSettings> flowSolutions;
  std::set<PathEndPoint> processedFlows;

  llvm::DenseMap<TileLoc, TileOp> coordToTile;
  llvm::DenseMap<TileLoc, SwitchboxOp> coordToSwitchbox;
  llvm::DenseMap<TileLoc, ShimMuxOp> coordToShimMux;
  llvm::DenseMap<int, PLIOOp> coordToPLIO;

  const int maxIterations = 1000;  // how long until declared unroutable

  DynamicTileAnalysis() = default;
  DynamicTileAnalysis(const Pathfinder &p) : pathfinder(std::move(p)) {}

  mlir::LogicalResult runAnalysis(DeviceOp &device);

  int getMaxCol() const { return maxCol; }
  int getMaxRow() const { return maxRow; }

  TileOp getTile(mlir::OpBuilder &builder, int col, int row);

  SwitchboxOp getSwitchbox(mlir::OpBuilder &builder, int col, int row);

  ShimMuxOp getShimMux(mlir::OpBuilder &builder, int col);
};

}  // namespace mlir::iree_compiler::AMDAIE
namespace mlir::iree_compiler::AMDAIE {
LogicalResult DynamicTileAnalysis::runAnalysis(DeviceOp &device) {
  LLVM_DEBUG(llvm::dbgs() << "\t---Begin DynamicTileAnalysis Constructor---\n");
  // find the maxCol and maxRow
  maxCol = 0;
  maxRow = 0;
  for (TileOp tileOp : device.getOps<TileOp>()) {
    maxCol = std::max(maxCol, tileOp.colIndex());
    maxRow = std::max(maxRow, tileOp.rowIndex());
  }

  AMDAIEDeviceModel targetModel =
      getDeviceModel(static_cast<AMDAIEDevice>(device.getDevice()));
  pathfinder.initialize(maxCol, maxRow, targetModel);

  // for each flow in the device, add it to pathfinder
  // each source can map to multiple different destinations (fanout)
  for (FlowOp flowOp : device.getOps<FlowOp>()) {
    TileOp srcTile = cast<TileOp>(flowOp.getSource().getDefiningOp());
    TileOp dstTile = cast<TileOp>(flowOp.getDest().getDefiningOp());
    TileLoc srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
    TileLoc dstCoords = {dstTile.colIndex(), dstTile.rowIndex()};
    Port srcPort = {toStrmT(flowOp.getSourceBundle()),
                    flowOp.getSourceChannel()};
    Port dstPort = {toStrmT(flowOp.getDestBundle()), flowOp.getDestChannel()};
    LLVM_DEBUG(llvm::dbgs()
               << "\tAdding Flow: (" << srcCoords.col << ", " << srcCoords.row
               << ")" << srcPort.bundle << srcPort.channel << " -> ("
               << dstCoords.col << ", " << dstCoords.row << ")"
               << dstPort.bundle << dstPort.channel << "\n");
    pathfinder.addFlow(srcCoords, srcPort, dstCoords, dstPort, false);
  }

  for (PacketFlowOp pktFlowOp : device.getOps<PacketFlowOp>()) {
    Region &r = pktFlowOp.getPorts();
    Block &b = r.front();
    Port srcPort, dstPort;
    TileOp srcTile, dstTile;
    TileLoc srcCoords, dstCoords;
    for (Operation &Op : b.getOperations()) {
      if (auto pktSource = dyn_cast<PacketSourceOp>(Op)) {
        srcTile = dyn_cast<TileOp>(pktSource.getTile().getDefiningOp());
        srcPort = {toStrmT(pktSource.port().bundle), pktSource.port().channel};
        srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
      } else if (auto pktDest = dyn_cast<PacketDestOp>(Op)) {
        dstTile = dyn_cast<TileOp>(pktDest.getTile().getDefiningOp());
        dstPort = {toStrmT(pktDest.port().bundle), pktDest.port().channel};
        dstCoords = {dstTile.colIndex(), dstTile.rowIndex()};
        LLVM_DEBUG(llvm::dbgs()
                   << "\tAdding Packet Flow: (" << srcCoords.col << ", "
                   << srcCoords.row << ")" << srcPort.bundle << srcPort.channel
                   << " -> (" << dstCoords.col << ", " << dstCoords.row << ")"
                   << dstPort.bundle << dstPort.channel << "\n");
        // todo: support many-to-one & many-to-many?
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
                                       switchboxOp.rowIndex(), connects))
      return switchboxOp.emitOpError() << "Unable to add fixed connections";
  }

  // all flows are now populated, call the congestion-aware pathfinder
  // algorithm
  // check whether the pathfinder algorithm creates a legal routing
  if (auto maybeFlowSolutions = pathfinder.findPaths(maxIterations))
    flowSolutions.swap(maybeFlowSolutions.value());
  else
    return device.emitError("Unable to find a legal routing");

  // fill in coords to TileOps, SwitchboxOps, and ShimMuxOps
  for (auto tileOp : device.getOps<TileOp>()) {
    int col, row;
    col = tileOp.colIndex();
    row = tileOp.rowIndex();
    maxCol = std::max(maxCol, col);
    maxRow = std::max(maxRow, row);
    assert(coordToTile.count({col, row}) == 0);
    coordToTile[{col, row}] = tileOp;
  }
  for (auto switchboxOp : device.getOps<SwitchboxOp>()) {
    int col = switchboxOp.colIndex();
    int row = switchboxOp.rowIndex();
    assert(coordToSwitchbox.count({col, row}) == 0);
    coordToSwitchbox[{col, row}] = switchboxOp;
  }
  for (auto shimmuxOp : device.getOps<ShimMuxOp>()) {
    int col = shimmuxOp.colIndex();
    int row = shimmuxOp.rowIndex();
    assert(coordToShimMux.count({col, row}) == 0);
    coordToShimMux[{col, row}] = shimmuxOp;
  }

  LLVM_DEBUG(llvm::dbgs() << "\t---End DynamicTileAnalysis Constructor---\n");
  return success();
}

TileOp DynamicTileAnalysis::getTile(OpBuilder &builder, int col, int row) {
  if (coordToTile.count({col, row})) {
    return coordToTile[{col, row}];
  }
  auto tileOp = builder.create<TileOp>(builder.getUnknownLoc(), col, row);
  coordToTile[{col, row}] = tileOp;
  maxCol = std::max(maxCol, col);
  maxRow = std::max(maxRow, row);
  return tileOp;
}

SwitchboxOp DynamicTileAnalysis::getSwitchbox(OpBuilder &builder, int col,
                                              int row) {
  assert(col >= 0);
  assert(row >= 0);
  if (coordToSwitchbox.count({col, row})) {
    return coordToSwitchbox[{col, row}];
  }
  auto switchboxOp = builder.create<SwitchboxOp>(builder.getUnknownLoc(),
                                                 getTile(builder, col, row));
  SwitchboxOp::ensureTerminator(switchboxOp.getConnections(), builder,
                                builder.getUnknownLoc());
  coordToSwitchbox[{col, row}] = switchboxOp;
  maxCol = std::max(maxCol, col);
  maxRow = std::max(maxRow, row);
  return switchboxOp;
}

ShimMuxOp DynamicTileAnalysis::getShimMux(OpBuilder &builder, int col) {
  assert(col >= 0);
  int row = 0;
  if (coordToShimMux.count({col, row})) {
    return coordToShimMux[{col, row}];
  }
  //  assert(getTile(builder, col, row).isShimNOCTile());
  auto switchboxOp = builder.create<ShimMuxOp>(builder.getUnknownLoc(),
                                               getTile(builder, col, row));
  SwitchboxOp::ensureTerminator(switchboxOp.getConnections(), builder,
                                builder.getUnknownLoc());
  coordToShimMux[{col, row}] = switchboxOp;
  maxCol = std::max(maxCol, col);
  maxRow = std::max(maxRow, row);
  return switchboxOp;
}
}  // namespace mlir::iree_compiler::AMDAIE

namespace mlir::iree_compiler::AMDAIE {

struct ConvertFlowsToInterconnect : OpConversionPattern<FlowOp> {
  using OpConversionPattern::OpConversionPattern;
  DeviceOp &device;
  DynamicTileAnalysis &analyzer;
  ConvertFlowsToInterconnect(MLIRContext *context, DeviceOp &d,
                             DynamicTileAnalysis &a, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), device(d), analyzer(a) {}

  LogicalResult matchAndRewrite(
      FlowOp flowOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto srcTile = cast<TileOp>(flowOp.getSource().getDefiningOp());
    TileLoc srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
    auto srcBundle = toStrmT(flowOp.getSourceBundle());
    auto srcChannel = flowOp.getSourceChannel();
    Port srcPort = {srcBundle, srcChannel};
    SwitchboxNode srcSB =
        analyzer.pathfinder.getSwitchboxNode({srcCoords.col, srcCoords.row});
    PathEndPoint srcPe{srcSB, srcPort};
    if (analyzer.processedFlows.count(srcPe)) {
      rewriter.eraseOp(flowOp);
      return success();
    }

    AMDAIEDeviceModel targetModel =
        getDeviceModel(static_cast<AMDAIEDevice>(device.getDevice()));
    for (auto &[curr, conn] :
         emitConnections(analyzer.flowSolutions, srcPe, targetModel)) {
      // create switchboxes eagerly just to agree with mlir-aie tests
      SwitchboxOp switchboxOp =
          analyzer.getSwitchbox(rewriter, curr.col, curr.row);
      Operation *op;
      switch (conn.interconnect) {
        case Connect::Interconnect::shimMuxOp:
          op = analyzer.getShimMux(rewriter, conn.col).getOperation();
          break;
        case Connect::Interconnect::swOp:
          op = switchboxOp.getOperation();
          break;
        case Connect::Interconnect::unk:
          return flowOp->emitOpError("unsupported/unknown interconnect");
      }

      Region &r = op->getRegion(0);
      Block &b = r.front();
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(b.getTerminator());
      rewriter.create<ConnectOp>(rewriter.getUnknownLoc(),
                                 toWireB(conn.src.bundle), conn.src.channel,
                                 toWireB(conn.dst.bundle), conn.dst.channel);
    }

    analyzer.processedFlows.insert(srcPe);
    rewriter.eraseOp(flowOp);
    return success();
  }
};

struct AIEPathfinderPass
    : PassWrapper<AIEPathfinderPass, OperationPass<DeviceOp>> {
  DynamicTileAnalysis analyzer;
  mlir::DenseMap<TileLoc, mlir::Operation *> tiles;

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
  void runOnFlow(DeviceOp d, mlir::OpBuilder &builder);
  void runOnPacketFlow(DeviceOp d, mlir::OpBuilder &builder);

  typedef std::pair<mlir::Operation *, Port> PhysPort;

  bool findPathToDest(SwitchSettings settings, TileLoc currTile,
                      StrmSwPortType currDestBundle, int currDestChannel,
                      TileLoc finalTile, StrmSwPortType finalDestBundle,
                      int finalDestChannel);

  mlir::Operation *getOrCreateTile(mlir::OpBuilder &builder, int col, int row);
  SwitchboxOp getOrCreateSwitchbox(mlir::OpBuilder &builder, TileOp tile);
  mlir::Pass::Option<bool> clRouteCircuit{
      *this, "route-circuit",
      llvm::cl::desc("Flag to enable aie.flow lowering."),
      llvm::cl::init(true)};
  mlir::Pass::Option<bool> clRoutePacket{
      *this, "route-packet",
      llvm::cl::desc("Flag to enable aie.packetflow lowering."),
      llvm::cl::init(true)};
};

void AIEPathfinderPass::runOnFlow(DeviceOp d, OpBuilder &builder) {
  // Apply rewrite rule to switchboxes to add assignments to every 'connect'
  // operation inside
  ConversionTarget target(getContext());
  target.addLegalOp<TileOp>();
  target.addLegalOp<ConnectOp>();
  target.addLegalOp<SwitchboxOp>();
  target.addLegalOp<ShimMuxOp>();
  target.addLegalOp<EndOp>();

  AMDAIEDeviceModel targetModel =
      getDeviceModel(static_cast<AMDAIEDevice>(d.getDevice()));

  RewritePatternSet patterns(&getContext());
  patterns.insert<ConvertFlowsToInterconnect>(d.getContext(), d, analyzer);
  if (failed(applyPartialConversion(d, target, std::move(patterns))))
    return signalPassFailure();

  // Populate wires between switchboxes and tiles.
  for (int col = 0; col <= analyzer.getMaxCol(); col++) {
    for (int row = 0; row <= analyzer.getMaxRow(); row++) {
      TileOp tile;
      if (analyzer.coordToTile.count({col, row}))
        tile = analyzer.coordToTile[{col, row}];
      else
        continue;
      SwitchboxOp sw;
      if (analyzer.coordToSwitchbox.count({col, row}))
        sw = analyzer.coordToSwitchbox[{col, row}];
      else
        continue;
      if (col > 0) {
        // connections east-west between stream switches
        if (analyzer.coordToSwitchbox.count({col - 1, row})) {
          auto westsw = analyzer.coordToSwitchbox[{col - 1, row}];
          builder.create<WireOp>(builder.getUnknownLoc(), westsw,
                                 toWireB(StrmSwPortType::EAST), sw,
                                 toWireB(StrmSwPortType::WEST));
        }
      }
      if (row > 0) {
        // connections between abstract 'core' of tile
        builder.create<WireOp>(builder.getUnknownLoc(), tile,
                               toWireB(StrmSwPortType::CORE), sw,
                               toWireB(StrmSwPortType::CORE));
        // connections between abstract 'dma' of tile
        builder.create<WireOp>(builder.getUnknownLoc(), tile,
                               toWireB(StrmSwPortType::DMA), sw,
                               toWireB(StrmSwPortType::DMA));
        // connections north-south inside array ( including connection to shim
        // row)
        if (analyzer.coordToSwitchbox.count({col, row - 1})) {
          auto southsw = analyzer.coordToSwitchbox[{col, row - 1}];
          builder.create<WireOp>(builder.getUnknownLoc(), southsw,
                                 toWireB(StrmSwPortType::NORTH), sw,
                                 toWireB(StrmSwPortType::SOUTH));
        }
      } else if (row == 0) {
        if (targetModel.isShimNOCTile(tile.getCol(), tile.getRow())) {
          if (analyzer.coordToShimMux.count({col, 0})) {
            auto shimsw = analyzer.coordToShimMux[{col, 0}];
            builder.create<WireOp>(
                builder.getUnknownLoc(), shimsw,
                toWireB(StrmSwPortType::NORTH),  // Changed to connect into the
                                                 // north
                sw, toWireB(StrmSwPortType::SOUTH));
            // PLIO is attached to shim mux
            if (analyzer.coordToPLIO.count(col)) {
              auto plio = analyzer.coordToPLIO[col];
              builder.create<WireOp>(builder.getUnknownLoc(), plio,
                                     toWireB(StrmSwPortType::NORTH), shimsw,
                                     toWireB(StrmSwPortType::SOUTH));
            }

            // abstract 'DMA' connection on tile is attached to shim mux ( in
            // row 0 )
            builder.create<WireOp>(builder.getUnknownLoc(), tile,
                                   toWireB(StrmSwPortType::DMA), shimsw,
                                   toWireB(StrmSwPortType::DMA));
          }
        } else if (targetModel.isShimPLTile(tile.getCol(), tile.getRow())) {
          // PLIO is attached directly to switch
          if (analyzer.coordToPLIO.count(col)) {
            auto plio = analyzer.coordToPLIO[col];
            builder.create<WireOp>(builder.getUnknownLoc(), plio,
                                   toWireB(StrmSwPortType::NORTH), sw,
                                   toWireB(StrmSwPortType::SOUTH));
          }
        }
      }
    }
  }
}

Operation *AIEPathfinderPass::getOrCreateTile(OpBuilder &builder, int col,
                                              int row) {
  TileLoc index = {col, row};
  Operation *tileOp = tiles[index];
  if (!tileOp) {
    auto tile = builder.create<TileOp>(builder.getUnknownLoc(), col, row);
    tileOp = tile.getOperation();
    tiles[index] = tileOp;
  }
  return tileOp;
}

SwitchboxOp AIEPathfinderPass::getOrCreateSwitchbox(OpBuilder &builder,
                                                    TileOp tile) {
  for (auto i : tile.getResult().getUsers()) {
    if (llvm::isa<SwitchboxOp>(*i)) {
      return llvm::cast<SwitchboxOp>(*i);
    }
  }
  return builder.create<SwitchboxOp>(builder.getUnknownLoc(), tile);
}

template <typename MyOp>
struct AIEOpRemoval : OpConversionPattern<MyOp> {
  using OpConversionPattern<MyOp>::OpConversionPattern;
  using OpAdaptor = typename MyOp::Adaptor;

  explicit AIEOpRemoval(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<MyOp>(context, benefit) {}

  LogicalResult matchAndRewrite(
      MyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();

    rewriter.eraseOp(Op);
    return success();
  }
};

bool AIEPathfinderPass::findPathToDest(SwitchSettings settings,
                                       TileLoc currTile,
                                       StrmSwPortType currDestBundle,
                                       int currDestChannel, TileLoc finalTile,
                                       StrmSwPortType finalDestBundle,
                                       int finalDestChannel) {
  if ((currTile == finalTile) && (currDestBundle == finalDestBundle) &&
      (currDestChannel == finalDestChannel)) {
    return true;
  }

  StrmSwPortType neighbourSourceBundle;
  TileLoc neighbourTile;
  if (currDestBundle == StrmSwPortType::EAST) {
    neighbourSourceBundle = StrmSwPortType::WEST;
    neighbourTile = {currTile.col + 1, currTile.row};
  } else if (currDestBundle == StrmSwPortType::WEST) {
    neighbourSourceBundle = StrmSwPortType::EAST;
    neighbourTile = {currTile.col - 1, currTile.row};
  } else if (currDestBundle == StrmSwPortType::NORTH) {
    neighbourSourceBundle = StrmSwPortType::SOUTH;
    neighbourTile = {currTile.col, currTile.row + 1};
  } else if (currDestBundle == StrmSwPortType::SOUTH) {
    neighbourSourceBundle = StrmSwPortType::NORTH;
    neighbourTile = {currTile.col, currTile.row - 1};
  } else {
    return false;
  }

  int neighbourSourceChannel = currDestChannel;
  for (const auto &[sbNode, setting] : settings) {
    TileLoc tile = {sbNode.col, sbNode.row};
    if ((tile == neighbourTile) &&
        (setting.src.bundle == neighbourSourceBundle) &&
        (setting.src.channel == neighbourSourceChannel)) {
      for (const auto &[bundle, channel] : setting.dsts) {
        if (findPathToDest(settings, neighbourTile, bundle, channel, finalTile,
                           finalDestBundle, finalDestChannel)) {
          return true;
        }
      }
    }
  }

  return false;
}

void AIEPathfinderPass::runOnPacketFlow(DeviceOp device, OpBuilder &builder) {
  ConversionTarget target(getContext());

  // Map from a port and flowID to
  DenseMap<std::pair<PhysPort, int>, SmallVector<PhysPort, 4>> packetFlows;
  SmallVector<std::pair<PhysPort, int>, 4> slavePorts;
  DenseMap<std::pair<PhysPort, int>, int> slaveAMSels;
  // Map from a port to
  DenseMap<PhysPort, Attribute> keepPktHeaderAttr;

  for (auto tileOp : device.getOps<TileOp>()) {
    int col = tileOp.colIndex();
    int row = tileOp.rowIndex();
    tiles[{col, row}] = tileOp;
  }

  // The logical model of all the switchboxes.
  DenseMap<TileLoc, SmallVector<std::pair<Connect, int>, 8>> switchboxes;
  for (PacketFlowOp pktFlowOp : device.getOps<PacketFlowOp>()) {
    Region &r = pktFlowOp.getPorts();
    Block &b = r.front();
    int flowID = pktFlowOp.IDInt();
    Port srcPort, destPort;
    TileOp srcTile, destTile;
    TileLoc srcCoords, destCoords;

    for (Operation &Op : b.getOperations()) {
      if (auto pktSource = dyn_cast<PacketSourceOp>(Op)) {
        srcTile = dyn_cast<TileOp>(pktSource.getTile().getDefiningOp());
        srcPort = {toStrmT(pktSource.port().bundle), pktSource.port().channel};
        srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
      } else if (auto pktDest = dyn_cast<PacketDestOp>(Op)) {
        destTile = dyn_cast<TileOp>(pktDest.getTile().getDefiningOp());
        destPort = {toStrmT(pktDest.port().bundle), pktDest.port().channel};
        destCoords = {destTile.colIndex(), destTile.rowIndex()};
        // Assign "keep_pkt_header flag"
        if (pktFlowOp->hasAttr("keep_pkt_header"))
          keepPktHeaderAttr[{destTile, destPort}] =
              StringAttr::get(Op.getContext(), "true");
        SwitchboxNode srcSB = analyzer.pathfinder.getSwitchboxNode(
            {srcCoords.col, srcCoords.row});
        PathEndPoint srcPoint = {srcSB, srcPort};
        SwitchSettings settings = analyzer.flowSolutions[srcPoint];
        // add connections for all the Switchboxes in SwitchSettings
        for (const auto &[curr, setting] : settings) {
          for (const auto &[bundle, channel] : setting.dsts) {
            TileLoc currTile = {curr.col, curr.row};
            // reject false broadcast
            if (!findPathToDest(settings, currTile, bundle, channel, destCoords,
                                destPort.bundle, destPort.channel))
              continue;
            Connect connect = {{setting.src.bundle, setting.src.channel},
                               {bundle, channel}};
            if (std::find(
                    switchboxes[currTile].begin(), switchboxes[currTile].end(),
                    std::pair{connect, flowID}) == switchboxes[currTile].end())
              switchboxes[currTile].push_back({connect, flowID});
          }
        }
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Check switchboxes\n");

  for (const auto &[tileId, connects] : switchboxes) {
    int col = tileId.col;
    int row = tileId.row;
    Operation *tileOp = getOrCreateTile(builder, col, row);
    LLVM_DEBUG(llvm::dbgs() << "***switchbox*** " << col << " " << row << '\n');
    for (const auto &[conn, flowID] : connects) {
      Port sourcePort = conn.src;
      Port destPort = conn.dst;
      auto sourceFlow =
          std::make_pair(std::make_pair(tileOp, sourcePort), flowID);
      packetFlows[sourceFlow].push_back({tileOp, destPort});
      slavePorts.push_back(sourceFlow);
      LLVM_DEBUG(llvm::dbgs()
                 << "flowID " << flowID << ':' << sourcePort.bundle << " "
                 << sourcePort.channel << " -> " << destPort.bundle << " "
                 << destPort.channel << "\n");
    }
  }

  // amsel()
  // masterset()
  // packetrules()
  // rule()

  // Compute arbiter assignments. Each arbiter has four msels.
  // Therefore, the number of "logical" arbiters is 6 x 4 = 24
  // A master port can only be associated with one arbiter

  // A map from Tile and master selectValue to the ports targetted by that
  // master select.
  DenseMap<std::pair<Operation *, int>, SmallVector<Port, 4>> masterAMSels;

  // Count of currently used logical arbiters for each tile.
  DenseMap<Operation *, int> amselValues;
  int numMsels = 4;
  int numArbiters = 6;

  std::vector<std::pair<std::pair<PhysPort, int>, SmallVector<PhysPort, 4>>>
      sortedPacketFlows(packetFlows.begin(), packetFlows.end());

  // To get determinsitic behaviour
  std::sort(sortedPacketFlows.begin(), sortedPacketFlows.end(),
            [](const auto &lhs, const auto &rhs) {
              auto lhsFlowID = lhs.first.second;
              auto rhsFlowID = rhs.first.second;
              return lhsFlowID < rhsFlowID;
            });

  // Check all multi-cast flows (same source, same ID). They should be
  // assigned the same arbiter and msel so that the flow can reach all the
  // destination ports at the same time For destination ports that appear in
  // different (multicast) flows, it should have a different <arbiterID, msel>
  // value pair for each flow
  for (const auto &packetFlow : sortedPacketFlows) {
    // The Source Tile of the flow
    Operation *tileOp = packetFlow.first.first.first;
    if (amselValues.count(tileOp) == 0) amselValues[tileOp] = 0;

    // arb0: 6*0,   6*1,   6*2,   6*3
    // arb1: 6*0+1, 6*1+1, 6*2+1, 6*3+1
    // arb2: 6*0+2, 6*1+2, 6*2+2, 6*3+2
    // arb3: 6*0+3, 6*1+3, 6*2+3, 6*3+3
    // arb4: 6*0+4, 6*1+4, 6*2+4, 6*3+4
    // arb5: 6*0+5, 6*1+5, 6*2+5, 6*3+5

    int amselValue = amselValues[tileOp];
    assert(amselValue < numArbiters && "Could not allocate new arbiter!");

    // Find existing arbiter assignment
    // If there is an assignment of an arbiter to a master port before, we
    // assign all the master ports here with the same arbiter but different
    // msel
    bool foundMatchedDest = false;
    for (const auto &map : masterAMSels) {
      if (map.first.first != tileOp) continue;
      amselValue = map.first.second;

      // check if same destinations
      SmallVector<Port, 4> ports(masterAMSels[{tileOp, amselValue}]);
      if (ports.size() != packetFlow.second.size()) continue;

      bool matched = true;
      for (auto dest : packetFlow.second) {
        if (Port port = dest.second;
            std::find(ports.begin(), ports.end(), port) == ports.end()) {
          matched = false;
          break;
        }
      }

      if (matched) {
        foundMatchedDest = true;
        break;
      }
    }

    if (!foundMatchedDest) {
      bool foundAMSelValue = false;
      for (int a = 0; a < numArbiters; a++) {
        for (int i = 0; i < numMsels; i++) {
          amselValue = a + i * numArbiters;
          if (masterAMSels.count({tileOp, amselValue}) == 0) {
            foundAMSelValue = true;
            break;
          }
        }

        if (foundAMSelValue) break;
      }

      for (auto dest : packetFlow.second) {
        Port port = dest.second;
        masterAMSels[{tileOp, amselValue}].push_back(port);
      }
    }

    slaveAMSels[packetFlow.first] = amselValue;
    amselValues[tileOp] = amselValue % numArbiters;
  }

  // Compute the master set IDs
  // A map from a switchbox output port to the number of that port.
  DenseMap<PhysPort, SmallVector<int, 4>> mastersets;
  for (const auto &[physPort, ports] : masterAMSels) {
    Operation *tileOp = physPort.first;
    assert(tileOp && "expected tileop");
    int amselValue = physPort.second;
    for (auto port : ports) {
      mastersets[{tileOp, port}].push_back(amselValue);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "CHECK mastersets\n");
#ifndef NDEBUG
  for (const auto &[physPort, values] : mastersets) {
    Operation *tileOp = physPort.first;
    StrmSwPortType bundle = physPort.second.bundle;
    int channel = physPort.second.channel;
    assert(tileOp);
    auto tile = dyn_cast<TileOp>(tileOp);
    LLVM_DEBUG(llvm::dbgs() << "master " << tile << " " << bundle << " : "
                            << channel << '\n');
    for (auto value : values)
      LLVM_DEBUG(llvm::dbgs() << "amsel: " << value << '\n');
  }
#endif

  // Compute mask values
  // Merging as many stream flows as possible
  // The flows must originate from the same source port and have different IDs
  // Two flows can be merged if they share the same destinations
  SmallVector<SmallVector<std::pair<PhysPort, int>, 4>, 4> slaveGroups;
  SmallVector<std::pair<PhysPort, int>, 4> workList(slavePorts);
  while (!workList.empty()) {
    auto slave1 = workList.pop_back_val();
    Port slavePort1 = slave1.first.second;

    bool foundgroup = false;
    for (auto &group : slaveGroups) {
      auto slave2 = group.front();
      if (Port slavePort2 = slave2.first.second; slavePort1 != slavePort2)
        continue;

      bool matched = true;
      auto dests1 = packetFlows[slave1];
      auto dests2 = packetFlows[slave2];
      if (dests1.size() != dests2.size()) continue;

      for (auto dest1 : dests1) {
        if (std::find(dests2.begin(), dests2.end(), dest1) == dests2.end()) {
          matched = false;
          break;
        }
      }

      if (matched) {
        group.push_back(slave1);
        foundgroup = true;
        break;
      }
    }

    if (!foundgroup) {
      SmallVector<std::pair<PhysPort, int>, 4> group({slave1});
      slaveGroups.push_back(group);
    }
  }

  DenseMap<std::pair<PhysPort, int>, int> slaveMasks;
  for (const auto &group : slaveGroups) {
    // Iterate over all the ID values in a group
    // If bit n-th (n <= 5) of an ID value differs from bit n-th of another ID
    // value, the bit position should be "don't care", and we will set the
    // mask bit of that position to 0
    int mask[5] = {-1, -1, -1, -1, -1};
    for (auto port : group) {
      int ID = port.second;
      for (int i = 0; i < 5; i++) {
        if (mask[i] == -1)
          mask[i] = ID >> i & 0x1;
        else if (mask[i] != (ID >> i & 0x1))
          mask[i] = 2;  // found bit difference --> mark as "don't care"
      }
    }

    int maskValue = 0;
    for (int i = 4; i >= 0; i--) {
      if (mask[i] == 2)  // don't care
        mask[i] = 0;
      else
        mask[i] = 1;
      maskValue = (maskValue << 1) + mask[i];
    }
    for (auto port : group) slaveMasks[port] = maskValue;
  }

#ifndef NDEBUG
  LLVM_DEBUG(llvm::dbgs() << "CHECK Slave Masks\n");
  for (auto map : slaveMasks) {
    auto port = map.first.first;
    auto tile = dyn_cast<TileOp>(port.first);
    StrmSwPortType bundle = port.second.bundle;
    int channel = port.second.channel;
    int ID = map.first.second;
    int mask = map.second;

    LLVM_DEBUG(llvm::dbgs()
               << "Port " << tile << " " << bundle << " " << channel << '\n');
    LLVM_DEBUG(llvm::dbgs()
               << "Mask " << "0x" << llvm::Twine::utohexstr(mask) << '\n');
    LLVM_DEBUG(llvm::dbgs()
               << "ID " << "0x" << llvm::Twine::utohexstr(ID) << '\n');
    for (int i = 0; i < 31; i++) {
      if ((i & mask) == (ID & mask))
        LLVM_DEBUG(llvm::dbgs() << "matches flow ID " << "0x"
                                << llvm::Twine::utohexstr(i) << '\n');
    }
  }
#endif

  // Realize the routes in MLIR
  for (auto map : tiles) {
    Operation *tileOp = map.second;
    auto tile = dyn_cast<TileOp>(tileOp);

    // Create a switchbox for the routes and insert inside it.
    builder.setInsertionPointAfter(tileOp);
    SwitchboxOp swbox = getOrCreateSwitchbox(builder, tile);
    SwitchboxOp::ensureTerminator(swbox.getConnections(), builder,
                                  builder.getUnknownLoc());
    Block &b = swbox.getConnections().front();
    builder.setInsertionPoint(b.getTerminator());

    std::vector<bool> amselOpNeededVector(32);
    for (const auto &masterset : mastersets) {
      if (tileOp != masterset.first.first) continue;
      for (auto value : masterset.second) {
        amselOpNeededVector[value] = true;
      }
    }
    // Create all the amsel Ops
    DenseMap<int, AMSelOp> amselOps;
    for (int i = 0; i < 32; i++) {
      if (amselOpNeededVector[i]) {
        int arbiterID = i % numArbiters;
        int msel = i / numArbiters;
        auto amsel =
            builder.create<AMSelOp>(builder.getUnknownLoc(), arbiterID, msel);
        amselOps[i] = amsel;
      }
    }
    // Create all the master set Ops
    // First collect the master sets for this tile.
    SmallVector<Port, 4> tileMasters;
    for (const auto &masterset : mastersets) {
      if (tileOp != masterset.first.first) continue;
      tileMasters.push_back(masterset.first.second);
    }
    // Sort them so we get a reasonable order
    std::sort(tileMasters.begin(), tileMasters.end());
    for (auto tileMaster : tileMasters) {
      StrmSwPortType bundle = tileMaster.bundle;
      int channel = tileMaster.channel;
      SmallVector<int, 4> msels = mastersets[{tileOp, tileMaster}];
      SmallVector<Value, 4> amsels;
      for (auto msel : msels) {
        assert(amselOps.count(msel) == 1);
        amsels.push_back(amselOps[msel]);
      }

      auto msOp = builder.create<MasterSetOp>(builder.getUnknownLoc(),
                                              builder.getIndexType(),
                                              toWireB(bundle), channel, amsels);
      if (auto pktFlowAttrs = keepPktHeaderAttr[{tileOp, tileMaster}])
        msOp->setAttr("keep_pkt_header", pktFlowAttrs);
    }

    // Generate the packet rules
    DenseMap<Port, PacketRulesOp> slaveRules;
    for (auto group : slaveGroups) {
      builder.setInsertionPoint(b.getTerminator());

      auto port = group.front().first;
      if (tileOp != port.first) continue;

      StrmSwPortType bundle = port.second.bundle;
      int channel = port.second.channel;
      auto slave = port.second;

      int mask = slaveMasks[group.front()];
      int ID = group.front().second & mask;

      // Verify that we actually map all the ID's correctly.
#ifndef NDEBUG
      for (auto _slave : group) assert((_slave.second & mask) == ID);
#endif
      Value amsel = amselOps[slaveAMSels[group.front()]];

      PacketRulesOp packetrules;
      if (slaveRules.count(slave) == 0) {
        packetrules = builder.create<PacketRulesOp>(builder.getUnknownLoc(),
                                                    toWireB(bundle), channel);
        PacketRulesOp::ensureTerminator(packetrules.getRules(), builder,
                                        builder.getUnknownLoc());
        slaveRules[slave] = packetrules;
      } else
        packetrules = slaveRules[slave];

      Block &rules = packetrules.getRules().front();
      builder.setInsertionPoint(rules.getTerminator());
      builder.create<PacketRuleOp>(builder.getUnknownLoc(), mask, ID, amsel);
    }
  }

  AMDAIEDeviceModel targetModel =
      getDeviceModel(static_cast<AMDAIEDevice>(device.getDevice()));

  // Add support for shimDMA
  // From shimDMA to BLI: 1) shimDMA 0 --> North 3
  //                      2) shimDMA 1 --> North 7
  // From BLI to shimDMA: 1) North   2 --> shimDMA 0
  //                      2) North   3 --> shimDMA 1

  for (auto switchbox : make_early_inc_range(device.getOps<SwitchboxOp>())) {
    auto retVal = switchbox->getOperand(0);
    auto tileOp = retVal.getDefiningOp<TileOp>();

    // Check if it is a shim Tile
    if (!targetModel.isShimNOCTile(tileOp.getCol(), tileOp.getRow())) continue;

    // Check if the switchbox is empty
    if (&switchbox.getBody()->front() == switchbox.getBody()->getTerminator())
      continue;

    Region &r = switchbox.getConnections();
    Block &b = r.front();

    // Find if the corresponding shimmux exsists or not
    int shimExist = 0;
    ShimMuxOp shimOp;
    for (auto shimmux : device.getOps<ShimMuxOp>()) {
      if (shimmux.getTile() == tileOp) {
        shimExist = 1;
        shimOp = shimmux;
        break;
      }
    }

    for (Operation &Op : b.getOperations()) {
      if (auto pktrules = dyn_cast<PacketRulesOp>(Op)) {
        // check if there is MM2S DMA in the switchbox of the 0th row
        if (toStrmT(pktrules.getSourceBundle()) == StrmSwPortType::DMA) {
          // If there is, then it should be put into the corresponding shimmux
          // If shimmux not defined then create shimmux
          if (!shimExist) {
            builder.setInsertionPointAfter(tileOp);
            shimOp = builder.create<ShimMuxOp>(builder.getUnknownLoc(), tileOp);
            Region &r1 = shimOp.getConnections();
            Block *b1 = builder.createBlock(&r1);
            builder.setInsertionPointToEnd(b1);
            builder.create<EndOp>(builder.getUnknownLoc());
            shimExist = 1;
          }

          Region &r0 = shimOp.getConnections();
          Block &b0 = r0.front();
          builder.setInsertionPointToStart(&b0);

          pktrules.setSourceBundle(toWireB(StrmSwPortType::SOUTH));
          if (pktrules.getSourceChannel() == 0) {
            pktrules.setSourceChannel(3);
            builder.create<ConnectOp>(builder.getUnknownLoc(),
                                      toWireB(StrmSwPortType::DMA), 0,
                                      toWireB(StrmSwPortType::NORTH), 3);
          }
          if (pktrules.getSourceChannel() == 1) {
            pktrules.setSourceChannel(7);
            builder.create<ConnectOp>(builder.getUnknownLoc(),
                                      toWireB(StrmSwPortType::DMA), 1,
                                      toWireB(StrmSwPortType::NORTH), 7);
          }
        }
      }

      if (auto mtset = dyn_cast<MasterSetOp>(Op)) {
        // check if there is S2MM DMA in the switchbox of the 0th row
        if (toStrmT(mtset.getDestBundle()) == StrmSwPortType::DMA) {
          // If there is, then it should be put into the corresponding shimmux
          // If shimmux not defined then create shimmux
          if (!shimExist) {
            builder.setInsertionPointAfter(tileOp);
            shimOp = builder.create<ShimMuxOp>(builder.getUnknownLoc(), tileOp);
            Region &r1 = shimOp.getConnections();
            Block *b1 = builder.createBlock(&r1);
            builder.setInsertionPointToEnd(b1);
            builder.create<EndOp>(builder.getUnknownLoc());
            shimExist = 1;
          }

          Region &r0 = shimOp.getConnections();
          Block &b0 = r0.front();
          builder.setInsertionPointToStart(&b0);

          mtset.setDestBundle(toWireB(StrmSwPortType::SOUTH));
          if (mtset.getDestChannel() == 0) {
            mtset.setDestChannel(2);
            builder.create<ConnectOp>(builder.getUnknownLoc(),
                                      toWireB(StrmSwPortType::NORTH), 2,
                                      toWireB(StrmSwPortType::DMA), 0);
          }
          if (mtset.getDestChannel() == 1) {
            mtset.setDestChannel(3);
            builder.create<ConnectOp>(builder.getUnknownLoc(),
                                      toWireB(StrmSwPortType::NORTH), 3,
                                      toWireB(StrmSwPortType::DMA), 1);
          }
        }
      }
    }
  }

  RewritePatternSet patterns(&getContext());

  if (failed(applyPartialConversion(device, target, std::move(patterns))))
    signalPassFailure();
}

void AIEPathfinderPass::runOnOperation() {
  // create analysis pass with routing graph for entire device
  LLVM_DEBUG(llvm::dbgs() << "---Begin AIEPathfinderPass---\n");

  DeviceOp d = getOperation();
  if (failed(analyzer.runAnalysis(d))) return signalPassFailure();
  OpBuilder builder = OpBuilder::atBlockEnd(d.getBody());

  if (clRouteCircuit) runOnFlow(d, builder);
  if (clRoutePacket) runOnPacketFlow(d, builder);
}

}  // namespace mlir::iree_compiler::AMDAIE

namespace mlir::iree_compiler::AMDAIE {
std::unique_ptr<OperationPass<DeviceOp>> createAMDAIEPathfinderPass() {
  return std::make_unique<AIEPathfinderPass>();
}

void registerAMDAIERoutePathfinderFlows() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createAMDAIEPathfinderPass();
  });
}
}  // namespace mlir::iree_compiler::AMDAIE