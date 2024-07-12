// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <list>
#include <set>

#include "Passes.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "iree-amd-aie/aie_runtime/iree_aie_router.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DirectedGraph.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_os_ostream.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "amdaie-create-pathfinder-flows"

using namespace mlir;

using mlir::iree_compiler::AMDAIE::AMDAIEDevice;
using mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel;
using mlir::iree_compiler::AMDAIE::Channel;
using mlir::iree_compiler::AMDAIE::getConnectingBundle;
using mlir::iree_compiler::AMDAIE::PathEndPoint;
using mlir::iree_compiler::AMDAIE::Port;
using mlir::iree_compiler::AMDAIE::Switchbox;
using mlir::iree_compiler::AMDAIE::SwitchSetting;
using mlir::iree_compiler::AMDAIE::TileLoc;
using xilinx::AIE::ConnectOp;
using xilinx::AIE::DeviceOp;
using xilinx::AIE::DMAChannelDir;
using xilinx::AIE::EndOp;
using xilinx::AIE::FlowOp;
using xilinx::AIE::Interconnect;
using xilinx::AIE::ShimMuxOp;
using xilinx::AIE::SwitchboxOp;
using xilinx::AIE::TileOp;
using xilinx::AIE::WireBundle;
using xilinx::AIE::WireOp;

namespace {
StrmSwPortType toStrmT(WireBundle w) {
  switch (w) {
    case WireBundle::Core:
      return StrmSwPortType::CORE;
    case WireBundle::DMA:
      return StrmSwPortType::DMA;
    case WireBundle::FIFO:
      return StrmSwPortType::FIFO;
    case WireBundle::South:
      return StrmSwPortType::SOUTH;
    case WireBundle::West:
      return StrmSwPortType::WEST;
    case WireBundle::North:
      return StrmSwPortType::NORTH;
    case WireBundle::East:
      return StrmSwPortType::EAST;
    case WireBundle::PLIO:
      llvm::report_fatal_error("unhandled PLIO");
    case WireBundle::NOC:
      llvm::report_fatal_error("unhandled NOC");
    case WireBundle::Trace:
      return StrmSwPortType::TRACE;
    case WireBundle::Ctrl:
      return StrmSwPortType::CTRL;
    default:
      llvm::report_fatal_error("unhandled WireBundle");
  }
}

WireBundle toWireB(StrmSwPortType w) {
  switch (w) {
    case StrmSwPortType::CORE:
      return WireBundle::Core;
    case StrmSwPortType::DMA:
      return WireBundle::DMA;
    case StrmSwPortType::FIFO:
      return WireBundle::FIFO;
    case StrmSwPortType::SOUTH:
      return WireBundle::South;
    case StrmSwPortType::WEST:
      return WireBundle::West;
    case StrmSwPortType::NORTH:
      return WireBundle::North;
    case StrmSwPortType::EAST:
      return WireBundle::East;
    case StrmSwPortType::TRACE:
      return WireBundle::Trace;
    case StrmSwPortType::CTRL:
      return WireBundle::Ctrl;
    default:
      llvm::report_fatal_error("unhandled WireBundle");
  }
}

}  // namespace

bool operator==(const StrmSwPortType &lhs, const WireBundle &rhs) {
  return lhs == toStrmT(rhs);
}

bool operator==(const WireBundle &lhs, const StrmSwPortType &rhs) {
  return rhs == lhs;
}

namespace mlir::iree_compiler::AMDAIE {
// DynamicTileAnalysis integrates the Pathfinder class into the MLIR
// environment. It passes flows to the Pathfinder as ordered pairs of ints.
// Detailed routing is received as SwitchboxSettings
// It then converts these settings to MLIR operations
class DynamicTileAnalysis {
 public:
  int maxCol, maxRow;
  std::shared_ptr<Router> pathfinder;
  std::map<PathEndPoint, SwitchSettings> flowSolutions;
  std::map<PathEndPoint, bool> processedFlows;
  llvm::DenseMap<TileLoc, TileOp> coordToTile;
  llvm::DenseMap<TileLoc, SwitchboxOp> coordToSwitchbox;
  llvm::DenseMap<TileLoc, ShimMuxOp> coordToShimMux;

  const int maxIterations = 1000;  // how long until declared unroutable

  DynamicTileAnalysis() : pathfinder(std::make_shared<Router>()) {}

  mlir::LogicalResult runAnalysis(DeviceOp &device);

  int getMaxCol() const { return maxCol; }
  int getMaxRow() const { return maxRow; }

  TileOp getTile(mlir::OpBuilder &builder, int col, int row);
  SwitchboxOp getSwitchbox(mlir::OpBuilder &builder, int col, int row);
  ShimMuxOp getShimMux(mlir::OpBuilder &builder, int col);
};

LogicalResult DynamicTileAnalysis::runAnalysis(DeviceOp &device) {
  // find the maxCol and maxRow
  maxCol = 0;
  maxRow = 0;
  for (TileOp tileOp : device.getOps<TileOp>()) {
    maxCol = std::max(maxCol, tileOp.colIndex());
    maxRow = std::max(maxRow, tileOp.rowIndex());
  }

  AMDAIEDeviceModel deviceModel =
      getDeviceModel(static_cast<AMDAIEDevice>(device.getDevice()));
  pathfinder->initialize(maxCol, maxRow, deviceModel);

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
    pathfinder->addFlow(srcCoords, srcPort, dstCoords, dstPort);
  }

  // add existing connections so Pathfinder knows which resources are
  // available search all existing SwitchBoxOps for exising connections
  for (SwitchboxOp switchboxOp : device.getOps<SwitchboxOp>()) {
    for (ConnectOp connectOp : switchboxOp.getOps<ConnectOp>()) {
      auto sb = connectOp->getParentOfType<SwitchboxOp>();
      // TODO: keep track of capacity?
      if (sb.getTileOp().isShimNOCTile()) continue;
      if (!pathfinder->addFixedConnection(
              {{sb.colIndex(), sb.rowIndex()},
               {toStrmT(connectOp.getSourceBundle()),
                connectOp.getSourceChannel()},
               {toStrmT(connectOp.getDestBundle()),
                connectOp.getDestChannel()}}))
        return switchboxOp.emitOpError() << "Couldn't connect " << connectOp;
    }
  }

  // all flows are now populated, call the congestion-aware pathfinder
  // algorithm
  // check whether the pathfinder algorithm creates a legal routing
  if (auto maybeFlowSolutions = pathfinder->findPaths(maxIterations))
    flowSolutions = maybeFlowSolutions.value();
  else
    return device.emitError("Unable to find a legal routing");

  // initialize all flows as unprocessed to prep for rewrite
  for (const auto &[pathEndPoint, switchSetting] : flowSolutions)
    processedFlows[pathEndPoint] = false;

  // fill in coords to TileOps, SwitchboxOps, and ShimMuxOps
  for (auto tileOp : device.getOps<TileOp>()) {
    int col, row;
    col = tileOp.colIndex();
    row = tileOp.rowIndex();
    maxCol = std::max(maxCol, col);
    maxRow = std::max(maxRow, row);
    assert(coordToTile.count({col, row}) == 0 &&
           "expected tile not in coordToTile yet");
    coordToTile[{col, row}] = tileOp;
  }
  for (auto switchboxOp : device.getOps<SwitchboxOp>()) {
    int col = switchboxOp.colIndex();
    int row = switchboxOp.rowIndex();
    assert(coordToSwitchbox.count({col, row}) == 0 &&
           "expected tile not in coordToSwitchbox");
    coordToSwitchbox[{col, row}] = switchboxOp;
  }
  for (auto shimmuxOp : device.getOps<ShimMuxOp>()) {
    int col = shimmuxOp.colIndex();
    int row = shimmuxOp.rowIndex();
    assert(coordToShimMux.count({col, row}) == 0 &&
           "expected tile not in coordToShimMux");
    coordToShimMux[{col, row}] = shimmuxOp;
  }

  return success();
}

TileOp DynamicTileAnalysis::getTile(OpBuilder &builder, int col, int row) {
  if (coordToTile.count({col, row})) return coordToTile[{col, row}];

  auto tileOp = builder.create<TileOp>(builder.getUnknownLoc(), col, row);
  coordToTile[{col, row}] = tileOp;
  maxCol = std::max(maxCol, col);
  maxRow = std::max(maxRow, row);
  return tileOp;
}

SwitchboxOp DynamicTileAnalysis::getSwitchbox(OpBuilder &builder, int col,
                                              int row) {
  assert(col >= 0 && "expected col >=0");
  assert(row >= 0 && "expected row >= 0");
  if (coordToSwitchbox.count({col, row})) return coordToSwitchbox[{col, row}];

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
  assert(col >= 0 && "expected col >= 0");
  int row = 0;
  if (coordToShimMux.count({col, row})) return coordToShimMux[{col, row}];

  assert(getTile(builder, col, row).isShimNOCTile() &&
         "expected tile is ShimNOC");
  auto switchboxOp = builder.create<ShimMuxOp>(builder.getUnknownLoc(),
                                               getTile(builder, col, row));
  SwitchboxOp::ensureTerminator(switchboxOp.getConnections(), builder,
                                builder.getUnknownLoc());
  coordToShimMux[{col, row}] = switchboxOp;
  maxCol = std::max(maxCol, col);
  maxRow = std::max(maxRow, row);
  return switchboxOp;
}

// allocates channels between switchboxes ( but does not assign them)
// instantiates shim-muxes AND allocates channels ( no need to rip these up in )
struct ConvertFlowsToInterconnect : OpConversionPattern<FlowOp> {
  using OpConversionPattern::OpConversionPattern;
  DynamicTileAnalysis &analyzer;
  ConvertFlowsToInterconnect(MLIRContext *context, DynamicTileAnalysis &a,
                             PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), analyzer(a) {}

  LogicalResult matchAndRewrite(
      FlowOp flowOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto srcTile = cast<TileOp>(flowOp.getSource().getDefiningOp());
    TileLoc srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
    auto srcBundle = flowOp.getSourceBundle();
    if (srcBundle == WireBundle::PLIO || srcBundle == WireBundle::NOC)
      return flowOp.emitOpError("unsupported PLIO/NOC srcBundle");
    auto srcChannel = flowOp.getSourceChannel();
    Port srcPort = {toStrmT(srcBundle), srcChannel};
    Switchbox srcSB = {srcCoords.col, srcCoords.row};
    PathEndPoint srcPoint = {srcSB, srcPort};
    if (srcTile.isShimPLTile())
      return srcTile.emitOpError("ShimPL not supported");
    if (analyzer.processedFlows[srcPoint]) {
      rewriter.eraseOp(flowOp);
      return success();
    }

    auto addConnection = [&rewriter](
                             // could be a shim-mux or a switchbox.
                             Interconnect op, StrmSwPortType inBundle,
                             int inIndex, StrmSwPortType outBundle,
                             int outIndex) {
      Region &r = op.getConnections();
      Block &b = r.front();
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(b.getTerminator());
      rewriter.create<ConnectOp>(rewriter.getUnknownLoc(), toWireB(inBundle),
                                 inIndex, toWireB(outBundle), outIndex);
    };

    // if the flow (aka "net") for this FlowOp hasn't been processed yet,
    // add all switchbox connections to implement the flow
    SwitchSettings settings = analyzer.flowSolutions[srcPoint];
    // add connections for all the Switchboxes in SwitchSettings
    for (const auto &[curr, setting] : settings) {
      TileOp currTile = analyzer.getTile(rewriter, curr.col, curr.row);
      // TODO(max): remove PL tests
      // if (currTile.isShimPLTile())
      //   llvm::report_fatal_error("ShimPL not supported");
      SwitchboxOp swOp = analyzer.getSwitchbox(rewriter, curr.col, curr.row);
      int shimCh = srcChannel;
      // shim DMAs at start of flows
      // TODO: must reserve N3, N7, S2, S3 for DMA connections
      if (curr == srcSB && srcTile.isShimNOCTile() &&
          srcBundle == StrmSwPortType::DMA) {
        // must be either DMA0 -> N3 or DMA1 -> N7
        // TODO(max): these are from AIE1 (Figure 6-33 BLI to ME streams
        // connectivity in NoC Tile)
        shimCh = srcChannel == 0 ? 3 : 7;
        ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, srcSB.col);
        addConnection(cast<Interconnect>(shimMuxOp.getOperation()),
                      toStrmT(srcBundle), srcChannel, StrmSwPortType::NORTH,
                      shimCh);
      }

      for (const auto &[bundle, channel] : setting.dsts) {
        // handle special shim connectivity
        if (curr == srcSB && srcTile.isShimNOCTile()) {
          addConnection(cast<Interconnect>(swOp.getOperation()),
                        StrmSwPortType::SOUTH, shimCh, bundle, channel);
        } else if (currTile.isShimNOCorPLTile() &&
                   bundle == StrmSwPortType::DMA) {
          shimCh = channel;
          if (currTile.isShimNOCTile() && bundle == StrmSwPortType::DMA) {
            // shim DMAs at end of flows
            // must be either N2 -> DMA0 or N3 -> DMA1
            // TODO(max): these are from AIE1 (Figure 6-32 ME to BLI streams
            // connectivity in NoC Tile)
            shimCh = channel == 0 ? 2 : 3;
            ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, curr.col);
            addConnection(cast<Interconnect>(shimMuxOp.getOperation()),
                          StrmSwPortType::NORTH, shimCh, bundle, channel);
          }
          addConnection(cast<Interconnect>(swOp.getOperation()),
                        setting.src.bundle, setting.src.channel,
                        StrmSwPortType::SOUTH, shimCh);
        } else {
          // otherwise, regular switchbox connection
          addConnection(cast<Interconnect>(swOp.getOperation()),
                        setting.src.bundle, setting.src.channel, bundle,
                        channel);
        }
      }
    }

    analyzer.processedFlows[srcPoint] = true;

    rewriter.eraseOp(flowOp);
    return success();
  }
};

/// Overall Flow:
/// rewrite switchboxes to assign unassigned connections, ensure this can be
/// done concurrently ( by different threads)
/// 1. Goal is to rewrite all flows in the device into switchboxes + shim-mux
/// 2. multiple passes of the rewrite pattern rewriting streamswitch
/// configurations to routes
/// 3. rewrite flows to stream-switches using 'weights' from analysis pass.
/// 4. check a region is legal
/// 5. rewrite stream-switches (within a bounding box) back to flows
struct AMDAIEPathfinderPass : mlir::OperationPass<DeviceOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AMDAIEPathfinderPass)

  AMDAIEPathfinderPass() : mlir::OperationPass<DeviceOp>(resolveTypeID()) {}

  llvm::StringRef getArgument() const override {
    return "amdaie-create-pathfinder-flows";
  }

  llvm::StringRef getName() const override { return "AMDAIEPathfinderPass"; }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    return std::make_unique<AMDAIEPathfinderPass>(
        *static_cast<const AMDAIEPathfinderPass *>(this));
  }

  DynamicTileAnalysis analyzer;
  AMDAIEPathfinderPass(DynamicTileAnalysis analyzer)
      : mlir::OperationPass<DeviceOp>(resolveTypeID()),
        analyzer(std::move(analyzer)) {}

  void runOnOperation() override;

  bool attemptFixupMemTileRouting(const mlir::OpBuilder &builder,
                                  SwitchboxOp northSwOp, SwitchboxOp southSwOp,
                                  ConnectOp &problemConnect);

  bool reconnectConnectOps(const mlir::OpBuilder &builder, SwitchboxOp sw,
                           ConnectOp problemConnect, bool isIncomingToSW,
                           StrmSwPortType problemBundle, int problemChan,
                           int emptyChan);

  ConnectOp replaceConnectOpWithNewDest(mlir::OpBuilder builder,
                                        ConnectOp connect,
                                        StrmSwPortType newBundle,
                                        int newChannel);
  ConnectOp replaceConnectOpWithNewSource(mlir::OpBuilder builder,
                                          ConnectOp connect,
                                          StrmSwPortType newBundle,
                                          int newChannel);

  SwitchboxOp getSwitchbox(DeviceOp &d, int col, int row);
};

void AMDAIEPathfinderPass::runOnOperation() {
  // Apply rewrite rule to switchboxes to add assignments to every 'connect'
  // operation inside
  ConversionTarget target(getContext());
  target.addLegalOp<TileOp>();
  target.addLegalOp<ConnectOp>();
  target.addLegalOp<SwitchboxOp>();
  target.addLegalOp<ShimMuxOp>();
  target.addLegalOp<EndOp>();

  DeviceOp d = getOperation();
  if (failed(analyzer.runAnalysis(d))) return signalPassFailure();
  OpBuilder builder = OpBuilder::atBlockEnd(d.getBody());

  RewritePatternSet patterns(&getContext());
  patterns.insert<ConvertFlowsToInterconnect>(d.getContext(), analyzer);
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

      if (col > 0 && analyzer.coordToSwitchbox.count({col - 1, row})) {
        // connections east-west between stream switches
        auto westsw = analyzer.coordToSwitchbox[{col - 1, row}];
        builder.create<WireOp>(builder.getUnknownLoc(), westsw,
                               toWireB(StrmSwPortType::EAST), sw,
                               toWireB(StrmSwPortType::WEST));
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
      } else if (row == 0 && tile.isShimNOCTile() &&
                 analyzer.coordToShimMux.count({col, 0})) {
        auto shimsw = analyzer.coordToShimMux[{col, 0}];
        builder.create<WireOp>(builder.getUnknownLoc(), shimsw,
                               // Changed to connect into the north
                               toWireB(StrmSwPortType::NORTH), sw,
                               toWireB(StrmSwPortType::SOUTH));
        // abstract 'DMA' connection on tile is attached to shim mux ( in
        // row 0 )
        builder.create<WireOp>(builder.getUnknownLoc(), tile,
                               toWireB(StrmSwPortType::DMA), shimsw,
                               toWireB(StrmSwPortType::DMA));
      }
    }
  }

  // If the routing violates architecture-specific routing constraints, then
  // attempt to partially reroute.
  AMDAIEDeviceModel deviceModel =
      getDeviceModel(static_cast<AMDAIEDevice>(d.getDevice()));
  std::vector<ConnectOp> problemConnects;
  d.walk([&](ConnectOp connect) {
    if (auto sw = connect->getParentOfType<SwitchboxOp>()) {
      // Constraint: memtile stream switch constraints
      if (auto tile = sw.getTileOp();
          tile.isMemTile() &&
          !deviceModel.isLegalMemtileConnection(
              tile.getCol(), tile.getRow(), toStrmT(connect.getSourceBundle()),
              connect.getSourceChannel(), toStrmT(connect.getDestBundle()),
              connect.getDestChannel()))
        problemConnects.push_back(connect);
    }
  });

  for (auto connect : problemConnects) {
    auto swBox = connect->getParentOfType<SwitchboxOp>();
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPoint(connect);
    auto northSw = getSwitchbox(d, swBox.colIndex(), swBox.rowIndex() + 1);
    if (auto southSw = getSwitchbox(d, swBox.colIndex(), swBox.rowIndex() - 1);
        !attemptFixupMemTileRouting(builder, northSw, southSw, connect))
      return signalPassFailure();
  }
}

bool AMDAIEPathfinderPass::attemptFixupMemTileRouting(
    const OpBuilder &builder, SwitchboxOp northSwOp, SwitchboxOp southSwOp,
    ConnectOp &problemConnect) {
  int problemNorthChannel;
  if (problemConnect.getSourceBundle() == StrmSwPortType::NORTH)
    problemNorthChannel = problemConnect.getSourceChannel();
  else if (problemConnect.getDestBundle() == StrmSwPortType::NORTH)
    problemNorthChannel = problemConnect.getDestChannel();
  else
    return false;  // Problem is not about n-s routing
  int problemSouthChannel;
  if (problemConnect.getSourceBundle() == StrmSwPortType::SOUTH)
    problemSouthChannel = problemConnect.getSourceChannel();
  else if (problemConnect.getDestBundle() == StrmSwPortType::SOUTH)
    problemSouthChannel = problemConnect.getDestChannel();
  else
    return false;  // Problem is not about n-s routing

  // Attempt to reroute northern neighbouring sw
  if (reconnectConnectOps(builder, northSwOp, problemConnect, true,
                          StrmSwPortType::SOUTH, problemNorthChannel,
                          problemSouthChannel))
    return true;
  if (reconnectConnectOps(builder, northSwOp, problemConnect, false,
                          StrmSwPortType::SOUTH, problemNorthChannel,
                          problemSouthChannel))
    return true;
  // Otherwise, attempt to reroute southern neighbouring sw
  if (reconnectConnectOps(builder, southSwOp, problemConnect, true,
                          StrmSwPortType::NORTH, problemSouthChannel,
                          problemNorthChannel))
    return true;
  if (reconnectConnectOps(builder, southSwOp, problemConnect, false,
                          StrmSwPortType::NORTH, problemSouthChannel,
                          problemNorthChannel))
    return true;
  return false;
}

bool AMDAIEPathfinderPass::reconnectConnectOps(const OpBuilder &builder,
                                               SwitchboxOp sw,
                                               ConnectOp problemConnect,
                                               bool isIncomingToSW,
                                               StrmSwPortType problemBundle,
                                               int problemChan, int emptyChan) {
  bool hasEmptyChannelSlot = true;
  bool foundCandidateForFixup = false;
  ConnectOp candidate;
  if (isIncomingToSW) {
    for (ConnectOp connect : sw.getOps<ConnectOp>()) {
      if (connect.getDestBundle() == problemBundle &&
          connect.getDestChannel() == problemChan) {
        candidate = connect;
        foundCandidateForFixup = true;
      }
      if (connect.getDestBundle() == problemBundle &&
          connect.getDestChannel() == emptyChan)
        hasEmptyChannelSlot = false;
    }
  } else {
    for (ConnectOp connect : sw.getOps<ConnectOp>()) {
      if (connect.getSourceBundle() == problemBundle &&
          connect.getSourceChannel() == problemChan) {
        candidate = connect;
        foundCandidateForFixup = true;
      }
      if (connect.getSourceBundle() == problemBundle &&
          connect.getSourceChannel() == emptyChan)
        hasEmptyChannelSlot = false;
    }
  }

  if (foundCandidateForFixup && hasEmptyChannelSlot) {
    StrmSwPortType problemBundleOpposite =
        problemBundle == StrmSwPortType::NORTH ? StrmSwPortType::SOUTH
                                               : StrmSwPortType::NORTH;
    // Found empty channel slot, perform reroute
    if (isIncomingToSW) {
      replaceConnectOpWithNewDest(builder, candidate, problemBundle, emptyChan);
      replaceConnectOpWithNewSource(builder, problemConnect,
                                    problemBundleOpposite, emptyChan);
    } else {
      replaceConnectOpWithNewSource(builder, candidate, problemBundle,
                                    emptyChan);
      replaceConnectOpWithNewDest(builder, problemConnect,
                                  problemBundleOpposite, emptyChan);
    }
    return true;
  }
  return false;
}

// Replace connect op
ConnectOp AMDAIEPathfinderPass::replaceConnectOpWithNewDest(
    OpBuilder builder, ConnectOp connect, StrmSwPortType newBundle,
    int newChannel) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(connect);
  auto newOp = builder.create<ConnectOp>(
      builder.getUnknownLoc(), connect.getSourceBundle(),
      connect.getSourceChannel(), toWireB(newBundle), newChannel);
  connect.erase();
  return newOp;
}

ConnectOp AMDAIEPathfinderPass::replaceConnectOpWithNewSource(
    OpBuilder builder, ConnectOp connect, StrmSwPortType newBundle,
    int newChannel) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(connect);
  auto newOp = builder.create<ConnectOp>(
      builder.getUnknownLoc(), toWireB(newBundle), newChannel,
      connect.getDestBundle(), connect.getDestChannel());
  connect.erase();
  return newOp;
}

SwitchboxOp AMDAIEPathfinderPass::getSwitchbox(DeviceOp &d, int col, int row) {
  SwitchboxOp output = nullptr;
  d.walk([&](SwitchboxOp swBox) {
    if (swBox.colIndex() == col && swBox.rowIndex() == row) {
      output = swBox;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return output;
}

std::unique_ptr<OperationPass<DeviceOp>> createAMDAIEPathfinderPass() {
  return std::make_unique<AMDAIEPathfinderPass>();
}

void registerAMDAIERoutePathfinderFlows() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createAMDAIEPathfinderPass();
  });
}

}  // namespace mlir::iree_compiler::AMDAIE
