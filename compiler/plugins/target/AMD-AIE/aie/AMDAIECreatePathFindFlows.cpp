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
#include "d_ary_heap.h"
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
using xilinx::AIE::WireBundle;
using xilinx::AIE::WireOp;

using xilinx::AIE::Connect;
using xilinx::AIE::Port;
using xilinx::AIE::TileID;

#include "aie/Dialect/AIE/IR/AIETargetModel.h"
using xilinx::AIE::AIETargetModel;

#define DEBUG_TYPE "amdaie-create-pathfinder-flows"
#define OVER_CAPACITY_COEFF 0.02
#define USED_CAPACITY_COEFF 0.02
#define DEMAND_COEFF 1.1

namespace mlir::iree_compiler::AMDAIE {

StrmSwPortType _toStrmT(WireBundle w) {
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
      return StrmSwPortType::PLIO;
    case WireBundle::NOC:
      return StrmSwPortType::NOC;
    case WireBundle::Trace:
      return StrmSwPortType::TRACE;
    case WireBundle::Ctrl:
      return StrmSwPortType::CTRL;
    default:
      llvm::report_fatal_error("unhandled WireBundle");
  }
}

WireBundle toStrmT(WireBundle w) { return w; }

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
    case StrmSwPortType::PLIO:
      return WireBundle::PLIO;
    case StrmSwPortType::NOC:
      return WireBundle::NOC;
    case StrmSwPortType::CTRL:
      return WireBundle::Ctrl;
    default:
      llvm::report_fatal_error("unhandled WireBundle");
  }
}

}  // namespace mlir::iree_compiler::AMDAIE

namespace mlir::iree_compiler::AMDAIE {
enum class Connectivity { INVALID = -1, AVAILABLE = 0, OCCUPIED = 1 };

using SwitchboxNode = struct SwitchboxNode {
  SwitchboxNode(int col, int row, int id, int maxCol, int maxRow,
                const AIETargetModel &targetModel)
      : col{col}, row{row}, id{id} {
    std::vector<WireBundle> bundles = {
        WireBundle::Core,  WireBundle::DMA,  WireBundle::FIFO,
        WireBundle::South, WireBundle::West, WireBundle::North,
        WireBundle::East,  WireBundle::PLIO, WireBundle::NOC,
        WireBundle::Trace, WireBundle::Ctrl};

    for (WireBundle bundle : bundles) {
      int maxCapacity = targetModel.getNumSourceSwitchboxConnections(
          col, row, toStrmT(bundle));
      if (targetModel.isShimNOCorPLTile(col, row) && maxCapacity == 0) {
        // wordaround for shimMux, todo: integrate shimMux into routable grid
        maxCapacity = targetModel.getNumSourceShimMuxConnections(
            col, row, toStrmT(bundle));
      }

      for (int channel = 0; channel < maxCapacity; channel++) {
        Port inPort = {bundle, channel};
        inPortToId[inPort] = inPortId;
        inPortId++;
      }

      maxCapacity =
          targetModel.getNumDestSwitchboxConnections(col, row, toStrmT(bundle));
      if (targetModel.isShimNOCorPLTile(col, row) && maxCapacity == 0) {
        // wordaround for shimMux, todo: integrate shimMux into routable grid
        maxCapacity =
            targetModel.getNumDestShimMuxConnections(col, row, toStrmT(bundle));
      }
      for (int channel = 0; channel < maxCapacity; channel++) {
        Port outPort = {bundle, channel};
        outPortToId[outPort] = outPortId;
        outPortId++;
      }
    }

    connectionMatrix.resize(inPortId, std::vector<Connectivity>(
                                          outPortId, Connectivity::AVAILABLE));

    // illegal connection
    for (const auto &[inPort, inId] : inPortToId) {
      for (const auto &[outPort, outId] : outPortToId) {
        if (!targetModel.isLegalTileConnection(
                col, row, toStrmT(inPort.bundle), inPort.channel,
                toStrmT(outPort.bundle), outPort.channel))
          connectionMatrix[inId][outId] = Connectivity::INVALID;

        if (targetModel.isShimNOCorPLTile(col, row)) {
          // wordaround for shimMux, todo: integrate shimMux into routable grid
          auto isBundleInList = [](WireBundle bundle,
                                   std::vector<WireBundle> bundles) {
            return std::find(bundles.begin(), bundles.end(), bundle) !=
                   bundles.end();
          };
          std::vector<WireBundle> bundles = {WireBundle::DMA, WireBundle::NOC,
                                             WireBundle::PLIO};
          if (isBundleInList(inPort.bundle, bundles) ||
              isBundleInList(outPort.bundle, bundles))
            connectionMatrix[inId][outId] = Connectivity::AVAILABLE;
        }
      }
    }
  }

  // given a outPort, find availble input channel
  std::vector<int> findAvailableChannelIn(WireBundle inBundle, Port outPort,
                                          bool isPkt) {
    std::vector<int> availableChannels;
    if (outPortToId.count(outPort) > 0) {
      int outId = outPortToId[outPort];
      if (isPkt) {
        for (const auto &[inPort, inId] : inPortToId) {
          if (inPort.bundle == inBundle &&
              connectionMatrix[inId][outId] != Connectivity::INVALID) {
            bool available = true;
            if (inPortPktCount.count(inPort) == 0) {
              for (const auto &[outPort, outId] : outPortToId) {
                if (connectionMatrix[inId][outId] == Connectivity::OCCUPIED) {
                  // occupied by others as circuit-switched
                  available = false;
                  break;
                }
              }
            } else {
              if (inPortPktCount[inPort] >= maxPktStream) {
                // occupied by others as packet-switched but exceed max packet
                // stream capacity
                available = false;
              }
            }
            if (available) availableChannels.push_back(inPort.channel);
          }
        }
      } else {
        for (const auto &[inPort, inId] : inPortToId) {
          if (inPort.bundle == inBundle &&
              connectionMatrix[inId][outId] == Connectivity::AVAILABLE) {
            bool available = true;
            for (const auto &[outPort, outId] : outPortToId) {
              if (connectionMatrix[inId][outId] == Connectivity::OCCUPIED) {
                available = false;
                break;
              }
            }
            if (available) availableChannels.push_back(inPort.channel);
          }
        }
      }
    }
    return availableChannels;
  }

  bool allocate(Port inPort, Port outPort, bool isPkt) {
    // invalid port
    if (outPortToId.count(outPort) == 0 || inPortToId.count(inPort) == 0)
      return false;

    int inId = inPortToId[inPort];
    int outId = outPortToId[outPort];

    // invalid connection
    if (connectionMatrix[inId][outId] == Connectivity::INVALID) return false;

    if (isPkt) {
      // a packet-switched stream to be allocated
      if (inPortPktCount.count(inPort) == 0) {
        for (const auto &[outPort, outId] : outPortToId) {
          if (connectionMatrix[inId][outId] == Connectivity::OCCUPIED) {
            // occupied by others as circuit-switched, allocation fail!
            return false;
          }
        }
        // empty channel, allocation succeed!
        inPortPktCount[inPort] = 1;
        connectionMatrix[inId][outId] = Connectivity::OCCUPIED;
        return true;
      } else {
        if (inPortPktCount[inPort] >= maxPktStream) {
          // occupied by others as packet-switched but exceed max packet stream
          // capacity, allocation fail!
          return false;
        } else {
          // valid packet-switched, allocation succeed!
          inPortPktCount[inPort]++;
          return true;
        }
      }
    } else {
      // a circuit-switched stream to be allocated
      if (connectionMatrix[inId][outId] == Connectivity::AVAILABLE) {
        // empty channel, allocation succeed!
        connectionMatrix[inId][outId] = Connectivity::OCCUPIED;
        return true;
      } else {
        // occupied by others, allocation fail!
        return false;
      }
    }
  }

  void clearAllocation() {
    for (int inId = 0; inId < inPortId; inId++) {
      for (int outId = 0; outId < outPortId; outId++) {
        if (connectionMatrix[inId][outId] != Connectivity::INVALID) {
          connectionMatrix[inId][outId] = Connectivity::AVAILABLE;
        }
      }
    }
    inPortPktCount.clear();
  }

  friend std::ostream &operator<<(std::ostream &os, const SwitchboxNode &s) {
    os << "Switchbox(" << s.col << ", " << s.row << ")";
    return os;
  }

  GENERATE_TO_STRING(SwitchboxNode);

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const SwitchboxNode &s) {
    os << to_string(s);
    return os;
  }

  bool operator<(const SwitchboxNode &rhs) const {
    return std::tie(col, row) < std::tie(rhs.col, rhs.row);
  }

  bool operator==(const SwitchboxNode &rhs) const {
    return std::tie(col, row) == std::tie(rhs.col, rhs.row);
  }

  int col, row, id;
  int inPortId = 0, outPortId = 0;
  std::map<Port, int> inPortToId, outPortToId;

  // tenary representation of switchbox connectivity
  // -1: invalid in arch, 0: empty and available, 1: occupued
  std::vector<std::vector<Connectivity>> connectionMatrix;

  // input ports with incoming packet-switched streams
  std::map<Port, int> inPortPktCount;

  // up to 32 packet-switched stram through a port
  const int maxPktStream = 32;
};

using ChannelEdge = struct ChannelEdge {
  ChannelEdge(SwitchboxNode *src, SwitchboxNode *target)
      : src(src), target(target) {
    // get bundle from src to target coordinates
    if (src->col == target->col) {
      if (src->row > target->row)
        bundle = WireBundle::South;
      else
        bundle = WireBundle::North;
    } else {
      if (src->col > target->col)
        bundle = WireBundle::West;
      else
        bundle = WireBundle::East;
    }

    // maximum number of routing resources
    maxCapacity = 0;
    for (auto &[outPort, _] : src->outPortToId) {
      if (outPort.bundle == bundle) {
        maxCapacity++;
      }
    }
  }

  friend std::ostream &operator<<(std::ostream &os, const ChannelEdge &c) {
    os << "Channel(src=" << c.src << ", dst=" << c.target << ")";
    return os;
  }

  GENERATE_TO_STRING(ChannelEdge)

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const ChannelEdge &c) {
    os << to_string(c);
    return os;
  }

  SwitchboxNode *src;
  SwitchboxNode *target;

  int maxCapacity;
  WireBundle bundle;
};

// A SwitchSetting defines the required settings for a SwitchboxNode for a flow
// SwitchSetting.src is the incoming signal
// SwitchSetting.dsts is the fanout
using SwitchSetting = struct SwitchSetting {
  SwitchSetting() = default;
  SwitchSetting(Port src) : src(src) {}
  SwitchSetting(Port src, std::set<Port> dsts)
      : src(src), dsts(std::move(dsts)) {}
  Port src;
  std::set<Port> dsts;

  // friend definition (will define the function as a non-member function of the
  // namespace surrounding the class).
  friend std::ostream &operator<<(std::ostream &os,
                                  const SwitchSetting &setting) {
    os << setting.src << " -> "
       << "{"
       << join(llvm::map_range(setting.dsts,
                               [](const Port &port) {
                                 std::ostringstream ss;
                                 ss << port;
                                 return ss.str();
                               }),
               ", ")
       << "}";
    return os;
  }

  GENERATE_TO_STRING(SwitchSetting)

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const SwitchSetting &s) {
    os << to_string(s);
    return os;
  }

  bool operator<(const SwitchSetting &rhs) const { return src < rhs.src; }
};

using SwitchSettings = std::map<SwitchboxNode, SwitchSetting>;

// A Flow defines source and destination vertices
// Only one source, but any number of destinations (fanout)
using PathEndPoint = struct PathEndPoint {
  SwitchboxNode sb;
  Port port;

  friend std::ostream &operator<<(std::ostream &os, const PathEndPoint &s) {
    os << "PathEndPoint(" << s.sb << ": " << s.port << ")";
    return os;
  }

  GENERATE_TO_STRING(PathEndPoint)

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const PathEndPoint &s) {
    os << to_string(s);
    return os;
  }

  // Needed for the std::maps that store PathEndPoint.
  bool operator<(const PathEndPoint &rhs) const {
    return std::tie(sb, port) < std::tie(rhs.sb, rhs.port);
  }

  bool operator==(const PathEndPoint &rhs) const {
    return std::tie(sb, port) == std::tie(rhs.sb, rhs.port);
  }
};

// A Flow defines source and destination vertices
// Only one source, but any number of destinations (fanout)
using PathEndPointNode = struct PathEndPointNode : PathEndPoint {
  PathEndPointNode(SwitchboxNode *sb, Port port)
      : PathEndPoint{*sb, port}, sb(sb) {}
  SwitchboxNode *sb;
};

using FlowNode = struct FlowNode {
  bool isPacketFlow;
  PathEndPointNode src;
  std::vector<PathEndPointNode> dsts;
};

}  // namespace mlir::iree_compiler::AMDAIE

namespace llvm {

inline raw_ostream &operator<<(
    raw_ostream &os, const mlir::iree_compiler::AMDAIE::SwitchSettings &ss) {
  std::stringstream s;
  s << "\tSwitchSettings: ";
  for (const auto &[sb, setting] : ss) {
    s << sb << ": " << setting << " | ";
  }
  s << "\n";
  os << s.str();
  return os;
}

}  // namespace llvm

template <>
struct std::hash<mlir::iree_compiler::AMDAIE::SwitchboxNode> {
  std::size_t operator()(
      const mlir::iree_compiler::AMDAIE::SwitchboxNode &s) const noexcept {
    return std::hash<xilinx::AIE::TileID>{}({s.col, s.row});
  }
};

template <>
struct std::hash<mlir::iree_compiler::AMDAIE::PathEndPoint> {
  std::size_t operator()(
      const mlir::iree_compiler::AMDAIE::PathEndPoint &pe) const noexcept {
    std::size_t h1 = std::hash<xilinx::AIE::Port>{}(pe.port);
    std::size_t h2 =
        std::hash<mlir::iree_compiler::AMDAIE::SwitchboxNode>{}(pe.sb);
    return h1 ^ (h2 << 1);
  }
};

namespace mlir::iree_compiler::AMDAIE {
class Pathfinder {
 public:
  Pathfinder() = default;
  void initialize(int maxCol, int maxRow, const AIETargetModel &targetModel);
  void addFlow(TileID srcCoords, Port srcPort, TileID dstCoords, Port dstPort,
               bool isPacketFlow);
  bool addFixedConnection(SwitchboxOp switchboxOp);
  std::optional<std::map<PathEndPoint, SwitchSettings>> findPaths(
      int maxIterations);

  std::map<SwitchboxNode *, SwitchboxNode *> dijkstraShortestPaths(
      SwitchboxNode *src);

  SwitchboxNode getSwitchboxNode(TileID coords) { return grid.at(coords); }

 private:
  // Flows to be routed
  std::vector<FlowNode> flows;

  // Grid of switchboxes available
  std::map<TileID, SwitchboxNode> grid;

  // Use a list instead of a vector because nodes have an edge list of raw
  // pointers to edges (so growing a vector would invalidate the pointers).
  std::list<ChannelEdge> edges;

  // Use Dijkstra's shortest path to find routes, and use "demand" as the
  // weights.
  std::map<ChannelEdge *, double> demand;

  // History of Channel being over capacity
  std::map<ChannelEdge *, int> overCapacity;

  // how many flows are actually using this Channel
  std::map<ChannelEdge *, int> usedCapacity;
};

// DynamicTileAnalysis integrates the Pathfinder class into the MLIR
// environment. It passes flows to the Pathfinder as ordered pairs of ints.
// Detailed routing is received as SwitchboxSettings
// It then converts these settings to MLIR operations
class DynamicTileAnalysis {
 public:
  int maxCol, maxRow;
  std::shared_ptr<Pathfinder> pathfinder;
  std::map<PathEndPoint, SwitchSettings> flowSolutions;
  std::map<PathEndPoint, bool> processedFlows;

  llvm::DenseMap<TileID, TileOp> coordToTile;
  llvm::DenseMap<TileID, SwitchboxOp> coordToSwitchbox;
  llvm::DenseMap<TileID, ShimMuxOp> coordToShimMux;
  llvm::DenseMap<int, PLIOOp> coordToPLIO;

  const int maxIterations = 1000;  // how long until declared unroutable

  DynamicTileAnalysis() : pathfinder(std::make_shared<Pathfinder>()) {}
  DynamicTileAnalysis(std::shared_ptr<Pathfinder> p)
      : pathfinder(std::move(p)) {}

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

  //  AMDAIEDeviceModel targetModel =
  //      getDeviceModel(static_cast<AMDAIEDevice>(device.getDevice()));
  //  pathfinder->initialize(maxCol, maxRow, targetModel);
  pathfinder->initialize(maxCol, maxRow, device.getTargetModel());

  // for each flow in the device, add it to pathfinder
  // each source can map to multiple different destinations (fanout)
  for (FlowOp flowOp : device.getOps<FlowOp>()) {
    TileOp srcTile = cast<TileOp>(flowOp.getSource().getDefiningOp());
    TileOp dstTile = cast<TileOp>(flowOp.getDest().getDefiningOp());
    TileID srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
    TileID dstCoords = {dstTile.colIndex(), dstTile.rowIndex()};
    Port srcPort = {flowOp.getSourceBundle(), flowOp.getSourceChannel()};
    Port dstPort = {flowOp.getDestBundle(), flowOp.getDestChannel()};
    LLVM_DEBUG(llvm::dbgs()
               << "\tAdding Flow: (" << srcCoords.col << ", " << srcCoords.row
               << ")" << stringifyWireBundle(srcPort.bundle) << srcPort.channel
               << " -> (" << dstCoords.col << ", " << dstCoords.row << ")"
               << stringifyWireBundle(dstPort.bundle) << dstPort.channel
               << "\n");
    pathfinder->addFlow(srcCoords, srcPort, dstCoords, dstPort, false);
  }

  for (PacketFlowOp pktFlowOp : device.getOps<PacketFlowOp>()) {
    Region &r = pktFlowOp.getPorts();
    Block &b = r.front();
    Port srcPort, dstPort;
    TileOp srcTile, dstTile;
    TileID srcCoords, dstCoords;
    for (Operation &Op : b.getOperations()) {
      if (auto pktSource = dyn_cast<PacketSourceOp>(Op)) {
        srcTile = dyn_cast<TileOp>(pktSource.getTile().getDefiningOp());
        srcPort = pktSource.port();
        srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
      } else if (auto pktDest = dyn_cast<PacketDestOp>(Op)) {
        dstTile = dyn_cast<TileOp>(pktDest.getTile().getDefiningOp());
        dstPort = pktDest.port();
        dstCoords = {dstTile.colIndex(), dstTile.rowIndex()};
        LLVM_DEBUG(llvm::dbgs()
                   << "\tAdding Packet Flow: (" << srcCoords.col << ", "
                   << srcCoords.row << ")"
                   << stringifyWireBundle(srcPort.bundle) << srcPort.channel
                   << " -> (" << dstCoords.col << ", " << dstCoords.row << ")"
                   << stringifyWireBundle(dstPort.bundle) << dstPort.channel
                   << "\n");
        // todo: support many-to-one & many-to-many?
        pathfinder->addFlow(srcCoords, srcPort, dstCoords, dstPort, true);
      }
    }
  }

  // add existing connections so Pathfinder knows which resources are
  // available search all existing SwitchBoxOps for exising connections
  for (SwitchboxOp switchboxOp : device.getOps<SwitchboxOp>()) {
    if (!pathfinder->addFixedConnection(switchboxOp))
      return switchboxOp.emitOpError() << "Unable to add fixed connections";
  }

  // all flows are now populated, call the congestion-aware pathfinder
  // algorithm
  // check whether the pathfinder algorithm creates a legal routing
  if (auto maybeFlowSolutions = pathfinder->findPaths(maxIterations))
    flowSolutions = maybeFlowSolutions.value();
  else
    return device.emitError("Unable to find a legal routing");

  // initialize all flows as unprocessed to prep for rewrite
  for (const auto &[pathEndPoint, switchSetting] : flowSolutions) {
    processedFlows[pathEndPoint] = false;
    LLVM_DEBUG(llvm::dbgs() << "Flow starting at (" << pathEndPoint.sb.col
                            << "," << pathEndPoint.sb.row << "):\t");
    LLVM_DEBUG(llvm::dbgs() << switchSetting);
  }

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
  assert(getTile(builder, col, row).isShimNOCTile());
  auto switchboxOp = builder.create<ShimMuxOp>(builder.getUnknownLoc(),
                                               getTile(builder, col, row));
  SwitchboxOp::ensureTerminator(switchboxOp.getConnections(), builder,
                                builder.getUnknownLoc());
  coordToShimMux[{col, row}] = switchboxOp;
  maxCol = std::max(maxCol, col);
  maxRow = std::max(maxRow, row);
  return switchboxOp;
}

void Pathfinder::initialize(int maxCol, int maxRow,
                            const AIETargetModel &targetModel) {
  // make grid of switchboxes
  int id = 0;
  for (int row = 0; row <= maxRow; row++) {
    for (int col = 0; col <= maxCol; col++) {
      grid.insert({{col, row},
                   SwitchboxNode{col, row, id++, maxCol, maxRow, targetModel}});
      SwitchboxNode &thisNode = grid.at({col, row});
      if (row > 0) {  // if not in row 0 add channel to North/South
        SwitchboxNode &southernNeighbor = grid.at({col, row - 1});
        // get the number of outgoing connections on the south side - outgoing
        // because these correspond to rhs of a connect op
        if (targetModel.getNumDestSwitchboxConnections(
                col, row, toStrmT(WireBundle::South))) {
          edges.emplace_back(&thisNode, &southernNeighbor);
        }
        // get the number of incoming connections on the south side - incoming
        // because they correspond to connections on the southside that are then
        // routed using internal connect ops through the switchbox (i.e., lhs of
        // connect ops)
        if (targetModel.getNumSourceSwitchboxConnections(
                col, row, toStrmT(WireBundle::South))) {
          edges.emplace_back(&southernNeighbor, &thisNode);
        }
      }

      if (col > 0) {  // if not in col 0 add channel to East/West
        SwitchboxNode &westernNeighbor = grid.at({col - 1, row});
        if (targetModel.getNumDestSwitchboxConnections(
                col, row, toStrmT(WireBundle::West))) {
          edges.emplace_back(&thisNode, &westernNeighbor);
        }
        if (targetModel.getNumSourceSwitchboxConnections(
                col, row, toStrmT(WireBundle::West))) {
          edges.emplace_back(&westernNeighbor, &thisNode);
        }
      }
    }
  }
}

// Add a flow from src to dst can have an arbitrary number of dst locations due
// to fanout.
void Pathfinder::addFlow(TileID srcCoords, Port srcPort, TileID dstCoords,
                         Port dstPort, bool isPacketFlow) {
  // check if a flow with this source already exists
  for (auto &[isPkt, src, dsts] : flows) {
    SwitchboxNode *existingSrcPtr = src.sb;
    assert(existingSrcPtr && "nullptr flow source");
    if (Port existingPort = src.port; existingSrcPtr->col == srcCoords.col &&
                                      existingSrcPtr->row == srcCoords.row &&
                                      existingPort == srcPort) {
      // find the vertex corresponding to the destination
      SwitchboxNode *matchingDstSbPtr = &grid.at(dstCoords);
      dsts.emplace_back(matchingDstSbPtr, dstPort);
      return;
    }
  }

  // If no existing flow was found with this source, create a new flow.
  SwitchboxNode *matchingSrcSbPtr = &grid.at(srcCoords);
  SwitchboxNode *matchingDstSbPtr = &grid.at(dstCoords);
  flows.push_back({isPacketFlow, PathEndPointNode{matchingSrcSbPtr, srcPort},
                   std::vector<PathEndPointNode>{{matchingDstSbPtr, dstPort}}});
}

// Keep track of connections already used in the AIE; Pathfinder algorithm will
// avoid using these.
bool Pathfinder::addFixedConnection(SwitchboxOp switchboxOp) {
  int col = switchboxOp.colIndex();
  int row = switchboxOp.rowIndex();
  SwitchboxNode &sb = grid.at({col, row});
  std::set<int> invalidInId, invalidOutId;

  for (ConnectOp connectOp : switchboxOp.getOps<ConnectOp>()) {
    Port srcPort = connectOp.sourcePort();
    Port destPort = connectOp.destPort();
    if (sb.inPortToId.count(srcPort) == 0 ||
        sb.outPortToId.count(destPort) == 0)
      return false;
    int inId = sb.inPortToId.at(srcPort);
    int outId = sb.outPortToId.at(destPort);
    if (sb.connectionMatrix[inId][outId] != Connectivity::AVAILABLE)
      return false;
    invalidInId.insert(inId);
    invalidOutId.insert(outId);
  }

  for (const auto &[inPort, inId] : sb.inPortToId) {
    for (const auto &[outPort, outId] : sb.outPortToId) {
      if (invalidInId.find(inId) != invalidInId.end() ||
          invalidOutId.find(outId) != invalidOutId.end()) {
        // mark as invalid
        sb.connectionMatrix[inId][outId] = Connectivity::INVALID;
      }
    }
  }
  return true;
}

static constexpr double INF = std::numeric_limits<double>::max();

std::map<SwitchboxNode *, SwitchboxNode *> Pathfinder::dijkstraShortestPaths(
    SwitchboxNode *src) {
  // Use std::map instead of DenseMap because DenseMap doesn't let you overwrite
  // tombstones.
  auto distance = std::map<SwitchboxNode *, double>();
  auto preds = std::map<SwitchboxNode *, SwitchboxNode *>();
  std::map<SwitchboxNode *, uint64_t> indexInHeap;
  typedef d_ary_heap_indirect<
      /*Value=*/SwitchboxNode *, /*Arity=*/4,
      /*IndexInHeapPropertyMap=*/std::map<SwitchboxNode *, uint64_t>,
      /*DistanceMap=*/std::map<SwitchboxNode *, double> &,
      /*Compare=*/std::less<>>
      MutableQueue;
  MutableQueue Q(distance, indexInHeap);

  for (auto &[_, sb] : grid) distance.emplace(&sb, INF);
  distance[src] = 0.0;

  std::map<SwitchboxNode *, std::vector<ChannelEdge *>> channels;

  enum Color { WHITE, GRAY, BLACK };
  std::map<SwitchboxNode *, Color> colors;
  for (auto &[_, sb] : grid) {
    SwitchboxNode *sbPtr = &sb;
    colors[sbPtr] = WHITE;
    for (auto &e : edges) {
      if (e.src == sbPtr) {
        channels[sbPtr].push_back(&e);
      }
    }
    std::sort(channels[sbPtr].begin(), channels[sbPtr].end(),
              [](const ChannelEdge *c1, ChannelEdge *c2) {
                return c1->target->id < c2->target->id;
              });
  }

  Q.push(src);
  while (!Q.empty()) {
    src = Q.top();
    Q.pop();
    for (ChannelEdge *e : channels[src]) {
      SwitchboxNode *dest = e->target;
      bool relax = distance[src] + demand[e] < distance[dest];
      if (colors[dest] == WHITE) {
        if (relax) {
          distance[dest] = distance[src] + demand[e];
          preds[dest] = src;
          colors[dest] = GRAY;
        }
        Q.push(dest);
      } else if (colors[dest] == GRAY && relax) {
        distance[dest] = distance[src] + demand[e];
        preds[dest] = src;
      }
    }
    colors[src] = BLACK;
  }
  return preds;
}

// Perform congestion-aware routing for all flows which have been added.
// Use Dijkstra's shortest path to find routes, and use "demand" as the weights.
// If the routing finds too much congestion, update the demand weights
// and repeat the process until a valid solution is found.
// Returns a map specifying switchbox settings for all flows.
// If no legal routing can be found after maxIterations, returns empty vector.
std::optional<std::map<PathEndPoint, SwitchSettings>> Pathfinder::findPaths(
    const int maxIterations) {
  LLVM_DEBUG(llvm::dbgs() << "Begin Pathfinder::findPaths\n");
  int iterationCount = 0;
  std::map<PathEndPoint, SwitchSettings> routingSolution;

  // initialize all Channel histories to 0
  for (auto &ch : edges) {
    overCapacity[&ch] = 0;
    usedCapacity[&ch] = 0;
  }
  // assume legal until found otherwise
  bool isLegal = true;

  do {
    LLVM_DEBUG(llvm::dbgs()
               << "Begin findPaths iteration #" << iterationCount << "\n");
    // update demand on all channels
    for (auto &ch : edges) {
      double history = 1.0 + OVER_CAPACITY_COEFF * overCapacity[&ch];
      double congestion = 1.0 + USED_CAPACITY_COEFF * usedCapacity[&ch];
      demand[&ch] = history * congestion;
    }
    // if reach maxIterations, throw an error since no routing can be found
    if (++iterationCount > maxIterations) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Pathfinder: maxIterations has been exceeded ("
                 << maxIterations
                 << " iterations)...unable to find routing for flows.\n");
      return std::nullopt;
    }

    // "rip up" all routes
    routingSolution.clear();
    for (auto &[tileID, node] : grid) node.clearAllocation();
    for (auto &ch : edges) usedCapacity[&ch] = 0;
    isLegal = true;

    // for each flow, find the shortest path from source to destination
    // update used_capacity for the path between them
    for (const auto &[isPkt, src, dsts] : flows) {
      // Use dijkstra to find path given current demand from the start
      // switchbox; find the shortest paths to each other switchbox. Output is
      // in the predecessor map, which must then be processed to get individual
      // switchbox settings
      assert(src.sb && "nonexistent flow source");
      std::set<SwitchboxNode *> processed;
      std::map<SwitchboxNode *, SwitchboxNode *> preds =
          dijkstraShortestPaths(src.sb);

      auto findIncomingEdge = [&](SwitchboxNode *sb) -> ChannelEdge * {
        for (auto &e : edges) {
          if (e.src == preds[sb] && e.target == sb) {
            return &e;
          }
        }
        return nullptr;
      };

      // trace the path of the flow backwards via predecessors
      // increment used_capacity for the associated channels
      SwitchSettings switchSettings;
      // set the input bundle for the source endpoint
      switchSettings[*src.sb].src = src.port;
      processed.insert(src.sb);
      // track destination ports used by src.sb
      std::vector<Port> srcDestPorts;
      for (const PathEndPointNode &endPoint : dsts) {
        SwitchboxNode *curr = endPoint.sb;
        assert(curr && "endpoint has no source switchbox");
        // set the output bundle for this destination endpoint
        switchSettings[*curr].dsts.insert(endPoint.port);
        Port lastDestPort = endPoint.port;
        // trace backwards until a vertex already processed is reached
        while (!processed.count(curr)) {
          // find the incoming edge from the pred to curr
          ChannelEdge *ch = findIncomingEdge(curr);
          assert(ch != nullptr && "couldn't find ch");
          int channel;
          // find all available channels in
          std::vector<int> availableChannels = curr->findAvailableChannelIn(
              getConnectingBundle(ch->bundle), lastDestPort, isPkt);
          if (availableChannels.size() > 0) {
            // if possible, choose the channel that predecessor can also use
            // todo: consider all predecessors?
            int bFound = false;
            auto &pred = preds[curr];
            if (!processed.count(pred) && pred != src.sb) {
              ChannelEdge *predCh = findIncomingEdge(pred);
              assert(predCh != nullptr && "couldn't find ch");
              for (int availableCh : availableChannels) {
                channel = availableCh;
                std::vector<int> availablePredChannels =
                    pred->findAvailableChannelIn(
                        getConnectingBundle(predCh->bundle),
                        {ch->bundle, channel}, isPkt);
                if (availablePredChannels.size() > 0) {
                  bFound = true;
                  break;
                }
              }
            }
            if (!bFound) channel = availableChannels[0];
            bool succeed =
                curr->allocate({getConnectingBundle(ch->bundle), channel},
                               lastDestPort, isPkt);
            if (!succeed) assert(false && "invalid allocation");
            LLVM_DEBUG(llvm::dbgs()
                       << *curr << ", connecting: "
                       << stringifyWireBundle(getConnectingBundle(ch->bundle))
                       << channel << " -> "
                       << stringifyWireBundle(lastDestPort.bundle)
                       << lastDestPort.channel << "\n");
          } else {
            // if no channel available, use a virtual channel id and mark
            // routing as being invalid
            channel = usedCapacity[ch];
            if (isLegal) {
              overCapacity[ch]++;
              LLVM_DEBUG(llvm::dbgs()
                         << *curr << ", congestion: "
                         << stringifyWireBundle(getConnectingBundle(ch->bundle))
                         << ", used_capacity = " << usedCapacity[ch]
                         << ", over_capacity_count = " << overCapacity[ch]
                         << "\n");
            }
            isLegal = false;
          }
          usedCapacity[ch]++;

          // add the entrance port for this Switchbox
          Port currSourcePort = {getConnectingBundle(ch->bundle), channel};
          switchSettings[*curr].src = {currSourcePort};

          // add the current Switchbox to the map of the predecessor
          Port PredDestPort = {ch->bundle, channel};
          switchSettings[*preds[curr]].dsts.insert(PredDestPort);
          lastDestPort = PredDestPort;

          // if at capacity, bump demand to discourage using this Channel
          if (usedCapacity[ch] >= ch->maxCapacity) {
            // this means the order matters!
            demand[ch] *= DEMAND_COEFF;
            LLVM_DEBUG(llvm::dbgs()
                       << *curr << ", bump demand: "
                       << stringifyWireBundle(getConnectingBundle(ch->bundle))
                       << ", demand = " << demand[ch] << "\n");
          }

          processed.insert(curr);
          curr = preds[curr];

          // allocation may fail, as we start from the dest of flow while
          // src.port is not chosen by router
          if (curr == src.sb &&
              std::find(srcDestPorts.begin(), srcDestPorts.end(),
                        lastDestPort) == srcDestPorts.end()) {
            bool succeed = src.sb->allocate(src.port, lastDestPort, isPkt);
            if (!succeed) {
              isLegal = false;
              overCapacity[ch]++;
              LLVM_DEBUG(llvm::dbgs()
                         << *curr << ", unable to connect: "
                         << stringifyWireBundle(src.port.bundle)
                         << src.port.channel << " -> "
                         << stringifyWireBundle(lastDestPort.bundle)
                         << lastDestPort.channel << "\n");
            }
            srcDestPorts.push_back(lastDestPort);
          }
        }
      }
      // add this flow to the proposed solution
      routingSolution[src] = switchSettings;
    }

  } while (!isLegal);  // continue iterations until a legal routing is found

  return routingSolution;
}

}  // namespace mlir::iree_compiler::AMDAIE

static std::vector<Operation *> flowOps;

namespace mlir::iree_compiler::AMDAIE {

struct ConvertFlowsToInterconnect : OpConversionPattern<FlowOp> {
  using OpConversionPattern::OpConversionPattern;
  DeviceOp &device;
  DynamicTileAnalysis &analyzer;
  bool keepFlowOp;
  ConvertFlowsToInterconnect(MLIRContext *context, DeviceOp &d,
                             DynamicTileAnalysis &a, bool keepFlowOp,
                             PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit),
        device(d),
        analyzer(a),
        keepFlowOp(keepFlowOp) {}

  LogicalResult match(FlowOp op) const override { return success(); }

  void addConnection(ConversionPatternRewriter &rewriter,
                     // could be a shim-mux or a switchbox.
                     Interconnect op, FlowOp flowOp, WireBundle inBundle,
                     int inIndex, WireBundle outBundle, int outIndex) const {
    Region &r = op.getConnections();
    Block &b = r.front();
    auto point = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(b.getTerminator());

    rewriter.create<ConnectOp>(rewriter.getUnknownLoc(), inBundle, inIndex,
                               outBundle, outIndex);

    rewriter.restoreInsertionPoint(point);

    LLVM_DEBUG(llvm::dbgs()
               << "\t\taddConnection() (" << op.colIndex() << ","
               << op.rowIndex() << ") " << stringifyWireBundle(inBundle)
               << inIndex << " -> " << stringifyWireBundle(outBundle)
               << outIndex << "\n");
  }

  void rewrite(FlowOp flowOp, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    Operation *Op = flowOp.getOperation();

    auto srcTile = cast<TileOp>(flowOp.getSource().getDefiningOp());
    TileID srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
    auto srcBundle = flowOp.getSourceBundle();
    auto srcChannel = flowOp.getSourceChannel();
    Port srcPort = {srcBundle, srcChannel};

    if (keepFlowOp) {
      auto *clonedOp = Op->clone();
      flowOps.push_back(clonedOp);
    }

#ifndef NDEBUG
    auto dstTile = cast<TileOp>(flowOp.getDest().getDefiningOp());
    TileID dstCoords = {dstTile.colIndex(), dstTile.rowIndex()};
    auto dstBundle = flowOp.getDestBundle();
    auto dstChannel = flowOp.getDestChannel();
    LLVM_DEBUG(llvm::dbgs()
               << "\n\t---Begin rewrite() for flowOp: (" << srcCoords.col
               << ", " << srcCoords.row << ")" << stringifyWireBundle(srcBundle)
               << srcChannel << " -> (" << dstCoords.col << ", "
               << dstCoords.row << ")" << stringifyWireBundle(dstBundle)
               << dstChannel << "\n\t");
#endif

    // if the flow (aka "net") for this FlowOp hasn't been processed yet,
    // add all switchbox connections to implement the flow
    SwitchboxNode srcSB =
        analyzer.pathfinder->getSwitchboxNode({srcCoords.col, srcCoords.row});
    if (PathEndPoint srcPoint = {srcSB, srcPort};
        !analyzer.processedFlows[srcPoint]) {
      SwitchSettings settings = analyzer.flowSolutions[srcPoint];
      // add connections for all the Switchboxes in SwitchSettings
      for (const auto &[curr, setting] : settings) {
        SwitchboxOp swOp = analyzer.getSwitchbox(rewriter, curr.col, curr.row);
        int shimCh = srcChannel;
        // TODO: must reserve N3, N7, S2, S3 for DMA connections
        if (curr == srcSB &&
            analyzer.getTile(rewriter, srcSB.col, srcSB.row).isShimNOCTile()) {
          // shim DMAs at start of flows
          if (srcBundle == WireBundle::DMA) {
            shimCh = srcChannel == 0
                         ? 3
                         : 7;  // must be either DMA0 -> N3 or DMA1 -> N7
            ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, srcSB.col);
            addConnection(rewriter,
                          cast<Interconnect>(shimMuxOp.getOperation()), flowOp,
                          srcBundle, srcChannel, WireBundle::North, shimCh);
          } else if (srcBundle ==
                     WireBundle::NOC) {  // must be NOC0/NOC1 -> N2/N3 or
                                         // NOC2/NOC3 -> N6/N7
            shimCh = srcChannel >= 2 ? srcChannel + 4 : srcChannel + 2;
            ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, srcSB.col);
            addConnection(rewriter,
                          cast<Interconnect>(shimMuxOp.getOperation()), flowOp,
                          srcBundle, srcChannel, WireBundle::North, shimCh);
          } else if (srcBundle ==
                     WireBundle::PLIO) {  // PLIO at start of flows with mux
            if (srcChannel == 2 || srcChannel == 3 || srcChannel == 6 ||
                srcChannel == 7) {  // Only some PLIO requrie mux
              ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, srcSB.col);
              addConnection(
                  rewriter, cast<Interconnect>(shimMuxOp.getOperation()),
                  flowOp, srcBundle, srcChannel, WireBundle::North, shimCh);
            }
          }
        }
        for (const auto &[bundle, channel] : setting.dsts) {
          // handle special shim connectivity
          if (curr == srcSB && analyzer.getTile(rewriter, srcSB.col, srcSB.row)
                                   .isShimNOCorPLTile()) {
            addConnection(rewriter, cast<Interconnect>(swOp.getOperation()),
                          flowOp, WireBundle::South, shimCh, bundle, channel);
          } else if (analyzer.getTile(rewriter, curr.col, curr.row)
                         .isShimNOCorPLTile() &&
                     (bundle == WireBundle::DMA || bundle == WireBundle::PLIO ||
                      bundle == WireBundle::NOC)) {
            shimCh = channel;
            if (analyzer.getTile(rewriter, curr.col, curr.row)
                    .isShimNOCTile()) {
              // shim DMAs at end of flows
              if (bundle == WireBundle::DMA) {
                shimCh = channel == 0
                             ? 2
                             : 3;  // must be either N2 -> DMA0 or N3 -> DMA1
                ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, curr.col);
                addConnection(
                    rewriter, cast<Interconnect>(shimMuxOp.getOperation()),
                    flowOp, WireBundle::North, shimCh, bundle, channel);
              } else if (bundle == WireBundle::NOC) {
                shimCh = channel + 2;  // must be either N2/3/4/5 -> NOC0/1/2/3
                ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, curr.col);
                addConnection(
                    rewriter, cast<Interconnect>(shimMuxOp.getOperation()),
                    flowOp, WireBundle::North, shimCh, bundle, channel);
              } else if (channel >=
                         2) {  // must be PLIO...only PLIO >= 2 require mux
                ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, curr.col);
                addConnection(
                    rewriter, cast<Interconnect>(shimMuxOp.getOperation()),
                    flowOp, WireBundle::North, shimCh, bundle, channel);
              }
            }
            addConnection(rewriter, cast<Interconnect>(swOp.getOperation()),
                          flowOp, setting.src.bundle, setting.src.channel,
                          WireBundle::South, shimCh);
          } else {
            // otherwise, regular switchbox connection
            addConnection(rewriter, cast<Interconnect>(swOp.getOperation()),
                          flowOp, setting.src.bundle, setting.src.channel,
                          bundle, channel);
          }
        }

        LLVM_DEBUG(llvm::dbgs() << curr << ": " << setting << " | "
                                << "\n");
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "\n\t\tFinished adding ConnectOps to implement flowOp.\n");
      analyzer.processedFlows[srcPoint] = true;
    } else
      LLVM_DEBUG(llvm::dbgs() << "Flow already processed!\n");

    rewriter.eraseOp(Op);
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
// struct AMDAIEPathfinderPass : mlir::OperationPass<DeviceOp> {
//   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AMDAIEPathfinderPass)
//
//   AMDAIEPathfinderPass() : mlir::OperationPass<DeviceOp>(resolveTypeID()) {}
//   AMDAIEPathfinderPass(const AMDAIEPathfinderPass &other)
//       : ::mlir::OperationPass<DeviceOp>(other) {}
//   AMDAIEPathfinderPass(const AIERoutePathfinderFlowsOptions &options)
//       : AMDAIEPathfinderPass() {
//     clRouteCircuit = options.clRouteCircuit;
//     clRoutePacket = options.clRoutePacket;
//     clKeepFlowOp = options.clKeepFlowOp;
//   }
//
//   llvm::StringRef getArgument() const override {
//     return "amdaie-create-pathfinder-flows";
//   }
//
//   llvm::StringRef getName() const override { return "AMDAIEPathfinderPass"; }
//
//   std::unique_ptr<mlir::Pass> clonePass() const override {
//     return std::make_unique<AMDAIEPathfinderPass>(
//         *static_cast<const AMDAIEPathfinderPass *>(this));
//   }
//
//   DynamicTileAnalysis analyzer;
//   mlir::DenseMap<TileID, mlir::Operation *> tiles;
//
//   void runOnOperation() override;
//   void runOnFlow(DeviceOp d, mlir::OpBuilder &builder);
//   void runOnPacketFlow(DeviceOp d, mlir::OpBuilder &builder);
//
//   typedef std::pair<mlir::Operation *, Port> PhysPort;
//
//   typedef struct {
//     SwitchboxOp sw;
//     Port sourcePort;
//     Port destPort;
//   } SwConnection;
//
//   bool findPathToDest(SwitchSettings settings, TileID currTile,
//                       WireBundle currDestBundle, int currDestChannel,
//                       TileID finalTile, WireBundle finalDestBundle,
//                       int finalDestChannel);
//
//   SwitchboxOp getSwitchbox(DeviceOp &d, int col, int row);
//
//   mlir::Operation *getOrCreateTile(mlir::OpBuilder &builder, int col, int
//   row); SwitchboxOp getOrCreateSwitchbox(mlir::OpBuilder &builder, TileOp
//   tile);
//
//   ::mlir::Pass::Option<bool> clRouteCircuit{
//       *this, "route-circuit",
//       ::llvm::cl::desc("Flag to enable aie.flow lowering."),
//       ::llvm::cl::init(true)};
//   ::mlir::Pass::Option<bool> clRoutePacket{
//       *this, "route-packet",
//       ::llvm::cl::desc("Flag to enable aie.packetflow lowering."),
//       ::llvm::cl::init(true)};
//   ::mlir::Pass::Option<bool> clKeepFlowOp{
//       *this, "keep-flow-op",
//       ::llvm::cl::desc("Flag to not erase aie.flow/packetflow after its "
//                        "lowering,used for routing visualization."),
//       ::llvm::cl::init(false)};
// };
//
// void AMDAIEPathfinderPass::runOnFlow(DeviceOp d, OpBuilder &builder) {
//   // Apply rewrite rule to switchboxes to add assignments to every 'connect'
//   // operation inside
//   ConversionTarget target(getContext());
//   target.addLegalOp<TileOp>();
//   target.addLegalOp<ConnectOp>();
//   target.addLegalOp<SwitchboxOp>();
//   target.addLegalOp<ShimMuxOp>();
//   target.addLegalOp<EndOp>();
//
//   RewritePatternSet patterns(&getContext());
//   patterns.insert<ConvertFlowsToInterconnect>(d.getContext(), d, analyzer,
//                                               clKeepFlowOp);
//   if (failed(applyPartialConversion(d, target, std::move(patterns))))
//     return signalPassFailure();
//
//   // Keep for visualization
//   if (clKeepFlowOp)
//     for (auto op : flowOps) builder.insert(op);
//
//   // Populate wires between switchboxes and tiles.
//   for (int col = 0; col <= analyzer.getMaxCol(); col++) {
//     for (int row = 0; row <= analyzer.getMaxRow(); row++) {
//       TileOp tile;
//       if (analyzer.coordToTile.count({col, row}))
//         tile = analyzer.coordToTile[{col, row}];
//       else
//         continue;
//       SwitchboxOp sw;
//       if (analyzer.coordToSwitchbox.count({col, row}))
//         sw = analyzer.coordToSwitchbox[{col, row}];
//       else
//         continue;
//       if (col > 0) {
//         // connections east-west between stream switches
//         if (analyzer.coordToSwitchbox.count({col - 1, row})) {
//           auto westsw = analyzer.coordToSwitchbox[{col - 1, row}];
//           builder.create<WireOp>(builder.getUnknownLoc(), westsw,
//                                  WireBundle::East, sw, WireBundle::West);
//         }
//       }
//       if (row > 0) {
//         // connections between abstract 'core' of tile
//         builder.create<WireOp>(builder.getUnknownLoc(), tile,
//         WireBundle::Core,
//                                sw, WireBundle::Core);
//         // connections between abstract 'dma' of tile
//         builder.create<WireOp>(builder.getUnknownLoc(), tile,
//         WireBundle::DMA,
//                                sw, WireBundle::DMA);
//         // connections north-south inside array ( including connection to
//         shim
//         // row)
//         if (analyzer.coordToSwitchbox.count({col, row - 1})) {
//           auto southsw = analyzer.coordToSwitchbox[{col, row - 1}];
//           builder.create<WireOp>(builder.getUnknownLoc(), southsw,
//                                  WireBundle::North, sw, WireBundle::South);
//         }
//       } else if (row == 0) {
//         if (tile.isShimNOCTile()) {
//           if (analyzer.coordToShimMux.count({col, 0})) {
//             auto shimsw = analyzer.coordToShimMux[{col, 0}];
//             builder.create<WireOp>(
//                 builder.getUnknownLoc(), shimsw,
//                 WireBundle::North,  // Changed to connect into the north
//                 sw, WireBundle::South);
//             // PLIO is attached to shim mux
//             if (analyzer.coordToPLIO.count(col)) {
//               auto plio = analyzer.coordToPLIO[col];
//               builder.create<WireOp>(builder.getUnknownLoc(), plio,
//                                      WireBundle::North, shimsw,
//                                      WireBundle::South);
//             }
//
//             // abstract 'DMA' connection on tile is attached to shim mux ( in
//             // row 0 )
//             builder.create<WireOp>(builder.getUnknownLoc(), tile,
//                                    WireBundle::DMA, shimsw, WireBundle::DMA);
//           }
//         } else if (tile.isShimPLTile()) {
//           // PLIO is attached directly to switch
//           if (analyzer.coordToPLIO.count(col)) {
//             auto plio = analyzer.coordToPLIO[col];
//             builder.create<WireOp>(builder.getUnknownLoc(), plio,
//                                    WireBundle::North, sw, WireBundle::South);
//           }
//         }
//       }
//     }
//   }
// }
//
// Operation *AMDAIEPathfinderPass::getOrCreateTile(OpBuilder &builder, int col,
//                                                  int row) {
//   TileID index = {col, row};
//   Operation *tileOp = tiles[index];
//   if (!tileOp) {
//     auto tile = builder.create<TileOp>(builder.getUnknownLoc(), col, row);
//     tileOp = tile.getOperation();
//     tiles[index] = tileOp;
//   }
//   return tileOp;
// }
//
// SwitchboxOp AMDAIEPathfinderPass::getOrCreateSwitchbox(OpBuilder &builder,
//                                                        TileOp tile) {
//   for (auto i : tile.getResult().getUsers()) {
//     if (llvm::isa<SwitchboxOp>(*i)) {
//       return llvm::cast<SwitchboxOp>(*i);
//     }
//   }
//   return builder.create<SwitchboxOp>(builder.getUnknownLoc(), tile);
// }
//
// template <typename MyOp>
// struct AIEOpRemoval : OpConversionPattern<MyOp> {
//   using OpConversionPattern<MyOp>::OpConversionPattern;
//   using OpAdaptor = typename MyOp::Adaptor;
//
//   explicit AIEOpRemoval(MLIRContext *context, PatternBenefit benefit = 1)
//       : OpConversionPattern<MyOp>(context, benefit) {}
//
//   LogicalResult matchAndRewrite(
//       MyOp op, OpAdaptor adaptor,
//       ConversionPatternRewriter &rewriter) const override {
//     Operation *Op = op.getOperation();
//
//     rewriter.eraseOp(Op);
//     return success();
//   }
// };
//
// bool AMDAIEPathfinderPass::findPathToDest(SwitchSettings settings,
//                                           TileID currTile,
//                                           WireBundle currDestBundle,
//                                           int currDestChannel, TileID
//                                           finalTile, WireBundle
//                                           finalDestBundle, int
//                                           finalDestChannel) {
//   if ((currTile == finalTile) && (currDestBundle == finalDestBundle) &&
//       (currDestChannel == finalDestChannel)) {
//     return true;
//   }
//
//   WireBundle neighbourSourceBundle;
//   TileID neighbourTile;
//   if (currDestBundle == WireBundle::East) {
//     neighbourSourceBundle = WireBundle::West;
//     neighbourTile = {currTile.col + 1, currTile.row};
//   } else if (currDestBundle == WireBundle::West) {
//     neighbourSourceBundle = WireBundle::East;
//     neighbourTile = {currTile.col - 1, currTile.row};
//   } else if (currDestBundle == WireBundle::North) {
//     neighbourSourceBundle = WireBundle::South;
//     neighbourTile = {currTile.col, currTile.row + 1};
//   } else if (currDestBundle == WireBundle::South) {
//     neighbourSourceBundle = WireBundle::North;
//     neighbourTile = {currTile.col, currTile.row - 1};
//   } else {
//     return false;
//   }
//
//   int neighbourSourceChannel = currDestChannel;
//   for (const auto &[sbNode, setting] : settings) {
//     TileID tile = {sbNode.col, sbNode.row};
//     if ((tile == neighbourTile) &&
//         (setting.src.bundle == neighbourSourceBundle) &&
//         (setting.src.channel == neighbourSourceChannel)) {
//       for (const auto &[bundle, channel] : setting.dsts) {
//         if (findPathToDest(settings, neighbourTile, bundle, channel,
//         finalTile,
//                            finalDestBundle, finalDestChannel)) {
//           return true;
//         }
//       }
//     }
//   }
//
//   return false;
// }
//
// void AMDAIEPathfinderPass::runOnPacketFlow(DeviceOp device,
//                                            OpBuilder &builder) {
//   ConversionTarget target(getContext());
//
//   // Map from a port and flowID to
//   DenseMap<std::pair<PhysPort, int>, SmallVector<PhysPort, 4>> packetFlows;
//   SmallVector<std::pair<PhysPort, int>, 4> slavePorts;
//   DenseMap<std::pair<PhysPort, int>, int> slaveAMSels;
//   // Map from a port to
//   DenseMap<PhysPort, Attribute> keepPktHeaderAttr;
//
//   for (auto tileOp : device.getOps<TileOp>()) {
//     int col = tileOp.colIndex();
//     int row = tileOp.rowIndex();
//     tiles[{col, row}] = tileOp;
//   }
//
//   // The logical model of all the switchboxes.
//   DenseMap<TileID, SmallVector<std::pair<Connect, int>, 8>> switchboxes;
//   for (PacketFlowOp pktFlowOp : device.getOps<PacketFlowOp>()) {
//     Region &r = pktFlowOp.getPorts();
//     Block &b = r.front();
//     int flowID = pktFlowOp.IDInt();
//     Port srcPort, destPort;
//     TileOp srcTile, destTile;
//     TileID srcCoords, destCoords;
//
//     for (Operation &Op : b.getOperations()) {
//       if (auto pktSource = dyn_cast<PacketSourceOp>(Op)) {
//         srcTile = dyn_cast<TileOp>(pktSource.getTile().getDefiningOp());
//         srcPort = pktSource.port();
//         srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
//       } else if (auto pktDest = dyn_cast<PacketDestOp>(Op)) {
//         destTile = dyn_cast<TileOp>(pktDest.getTile().getDefiningOp());
//         destPort = pktDest.port();
//         destCoords = {destTile.colIndex(), destTile.rowIndex()};
//         // Assign "keep_pkt_header flag"
//         if (pktFlowOp->hasAttr("keep_pkt_header"))
//           keepPktHeaderAttr[{destTile, destPort}] =
//               StringAttr::get(Op.getContext(), "true");
//         SwitchboxNode srcSB = analyzer.pathfinder->getSwitchboxNode(
//             {srcCoords.col, srcCoords.row});
//         if (PathEndPoint srcPoint = {srcSB, srcPort};
//             !analyzer.processedFlows[srcPoint]) {
//           SwitchSettings settings = analyzer.flowSolutions[srcPoint];
//           // add connections for all the Switchboxes in SwitchSettings
//           for (const auto &[curr, setting] : settings) {
//             for (const auto &[bundle, channel] : setting.dsts) {
//               TileID currTile = {curr.col, curr.row};
//               // reject false broadcast
//               if (!findPathToDest(settings, currTile, bundle, channel,
//                                   destCoords, destPort.bundle,
//                                   destPort.channel))
//                 continue;
//               Connect connect = {{setting.src.bundle, setting.src.channel},
//                                  {bundle, channel}};
//               if (std::find(switchboxes[currTile].begin(),
//                             switchboxes[currTile].end(),
//                             std::pair{connect, flowID}) ==
//                   switchboxes[currTile].end())
//                 switchboxes[currTile].push_back({connect, flowID});
//             }
//           }
//         }
//       }
//     }
//   }
//
//   LLVM_DEBUG(llvm::dbgs() << "Check switchboxes\n");
//
//   for (const auto &[tileId, connects] : switchboxes) {
//     int col = tileId.col;
//     int row = tileId.row;
//     Operation *tileOp = getOrCreateTile(builder, col, row);
//     LLVM_DEBUG(llvm::dbgs() << "***switchbox*** " << col << " " << row <<
//     '\n'); for (const auto &[conn, flowID] : connects) {
//       Port sourcePort = conn.src;
//       Port destPort = conn.dst;
//       auto sourceFlow =
//           std::make_pair(std::make_pair(tileOp, sourcePort), flowID);
//       packetFlows[sourceFlow].push_back({tileOp, destPort});
//       slavePorts.push_back(sourceFlow);
//       LLVM_DEBUG(llvm::dbgs() << "flowID " << flowID << ':'
//                               << stringifyWireBundle(sourcePort.bundle) << "
//                               "
//                               << sourcePort.channel << " -> "
//                               << stringifyWireBundle(destPort.bundle) << " "
//                               << destPort.channel << "\n");
//     }
//   }
//
//   // amsel()
//   // masterset()
//   // packetrules()
//   // rule()
//
//   // Compute arbiter assignments. Each arbiter has four msels.
//   // Therefore, the number of "logical" arbiters is 6 x 4 = 24
//   // A master port can only be associated with one arbiter
//
//   // A map from Tile and master selectValue to the ports targetted by that
//   // master select.
//   DenseMap<std::pair<Operation *, int>, SmallVector<Port, 4>> masterAMSels;
//
//   // Count of currently used logical arbiters for each tile.
//   DenseMap<Operation *, int> amselValues;
//   int numMsels = 4;
//   int numArbiters = 6;
//
//   std::vector<std::pair<std::pair<PhysPort, int>, SmallVector<PhysPort, 4>>>
//       sortedPacketFlows(packetFlows.begin(), packetFlows.end());
//
//   // To get determinsitic behaviour
//   std::sort(sortedPacketFlows.begin(), sortedPacketFlows.end(),
//             [](const auto &lhs, const auto &rhs) {
//               auto lhsFlowID = lhs.first.second;
//               auto rhsFlowID = rhs.first.second;
//               return lhsFlowID < rhsFlowID;
//             });
//
//   // Check all multi-cast flows (same source, same ID). They should be
//   // assigned the same arbiter and msel so that the flow can reach all the
//   // destination ports at the same time For destination ports that appear in
//   // different (multicast) flows, it should have a different <arbiterID,
//   msel>
//   // value pair for each flow
//   for (const auto &packetFlow : sortedPacketFlows) {
//     // The Source Tile of the flow
//     Operation *tileOp = packetFlow.first.first.first;
//     if (amselValues.count(tileOp) == 0) amselValues[tileOp] = 0;
//
//     // arb0: 6*0,   6*1,   6*2,   6*3
//     // arb1: 6*0+1, 6*1+1, 6*2+1, 6*3+1
//     // arb2: 6*0+2, 6*1+2, 6*2+2, 6*3+2
//     // arb3: 6*0+3, 6*1+3, 6*2+3, 6*3+3
//     // arb4: 6*0+4, 6*1+4, 6*2+4, 6*3+4
//     // arb5: 6*0+5, 6*1+5, 6*2+5, 6*3+5
//
//     int amselValue = amselValues[tileOp];
//     assert(amselValue < numArbiters && "Could not allocate new arbiter!");
//
//     // Find existing arbiter assignment
//     // If there is an assignment of an arbiter to a master port before, we
//     // assign all the master ports here with the same arbiter but different
//     // msel
//     bool foundMatchedDest = false;
//     for (const auto &map : masterAMSels) {
//       if (map.first.first != tileOp) continue;
//       amselValue = map.first.second;
//
//       // check if same destinations
//       SmallVector<Port, 4> ports(masterAMSels[{tileOp, amselValue}]);
//       if (ports.size() != packetFlow.second.size()) continue;
//
//       bool matched = true;
//       for (auto dest : packetFlow.second) {
//         if (Port port = dest.second;
//             std::find(ports.begin(), ports.end(), port) == ports.end()) {
//           matched = false;
//           break;
//         }
//       }
//
//       if (matched) {
//         foundMatchedDest = true;
//         break;
//       }
//     }
//
//     if (!foundMatchedDest) {
//       bool foundAMSelValue = false;
//       for (int a = 0; a < numArbiters; a++) {
//         for (int i = 0; i < numMsels; i++) {
//           amselValue = a + i * numArbiters;
//           if (masterAMSels.count({tileOp, amselValue}) == 0) {
//             foundAMSelValue = true;
//             break;
//           }
//         }
//
//         if (foundAMSelValue) break;
//       }
//
//       for (auto dest : packetFlow.second) {
//         Port port = dest.second;
//         masterAMSels[{tileOp, amselValue}].push_back(port);
//       }
//     }
//
//     slaveAMSels[packetFlow.first] = amselValue;
//     amselValues[tileOp] = amselValue % numArbiters;
//   }
//
//   // Compute the master set IDs
//   // A map from a switchbox output port to the number of that port.
//   DenseMap<PhysPort, SmallVector<int, 4>> mastersets;
//   for (const auto &[physPort, ports] : masterAMSels) {
//     Operation *tileOp = physPort.first;
//     assert(tileOp);
//     int amselValue = physPort.second;
//     for (auto port : ports) {
//       PhysPort physPort = {tileOp, port};
//       mastersets[physPort].push_back(amselValue);
//     }
//   }
//
//   LLVM_DEBUG(llvm::dbgs() << "CHECK mastersets\n");
// #ifndef NDEBUG
//   for (const auto &[physPort, values] : mastersets) {
//     Operation *tileOp = physPort.first;
//     WireBundle bundle = physPort.second.bundle;
//     int channel = physPort.second.channel;
//     assert(tileOp);
//     auto tile = dyn_cast<TileOp>(tileOp);
//     LLVM_DEBUG(llvm::dbgs()
//                << "master " << tile << " " << stringifyWireBundle(bundle)
//                << " : " << channel << '\n');
//     for (auto value : values)
//       LLVM_DEBUG(llvm::dbgs() << "amsel: " << value << '\n');
//   }
// #endif
//
//   // Compute mask values
//   // Merging as many stream flows as possible
//   // The flows must originate from the same source port and have different
//   IDs
//   // Two flows can be merged if they share the same destinations
//   SmallVector<SmallVector<std::pair<PhysPort, int>, 4>, 4> slaveGroups;
//   SmallVector<std::pair<PhysPort, int>, 4> workList(slavePorts);
//   while (!workList.empty()) {
//     auto slave1 = workList.pop_back_val();
//     Port slavePort1 = slave1.first.second;
//
//     bool foundgroup = false;
//     for (auto &group : slaveGroups) {
//       auto slave2 = group.front();
//       if (Port slavePort2 = slave2.first.second; slavePort1 != slavePort2)
//         continue;
//
//       bool matched = true;
//       auto dests1 = packetFlows[slave1];
//       auto dests2 = packetFlows[slave2];
//       if (dests1.size() != dests2.size()) continue;
//
//       for (auto dest1 : dests1) {
//         if (std::find(dests2.begin(), dests2.end(), dest1) == dests2.end()) {
//           matched = false;
//           break;
//         }
//       }
//
//       if (matched) {
//         group.push_back(slave1);
//         foundgroup = true;
//         break;
//       }
//     }
//
//     if (!foundgroup) {
//       SmallVector<std::pair<PhysPort, int>, 4> group({slave1});
//       slaveGroups.push_back(group);
//     }
//   }
//
//   DenseMap<std::pair<PhysPort, int>, int> slaveMasks;
//   for (const auto &group : slaveGroups) {
//     // Iterate over all the ID values in a group
//     // If bit n-th (n <= 5) of an ID value differs from bit n-th of another
//     ID
//     // value, the bit position should be "don't care", and we will set the
//     // mask bit of that position to 0
//     int mask[5] = {-1, -1, -1, -1, -1};
//     for (auto port : group) {
//       int ID = port.second;
//       for (int i = 0; i < 5; i++) {
//         if (mask[i] == -1)
//           mask[i] = ID >> i & 0x1;
//         else if (mask[i] != (ID >> i & 0x1))
//           mask[i] = 2;  // found bit difference --> mark as "don't care"
//       }
//     }
//
//     int maskValue = 0;
//     for (int i = 4; i >= 0; i--) {
//       if (mask[i] == 2)  // don't care
//         mask[i] = 0;
//       else
//         mask[i] = 1;
//       maskValue = (maskValue << 1) + mask[i];
//     }
//     for (auto port : group) slaveMasks[port] = maskValue;
//   }
//
// #ifndef NDEBUG
//   LLVM_DEBUG(llvm::dbgs() << "CHECK Slave Masks\n");
//   for (auto map : slaveMasks) {
//     auto port = map.first.first;
//     auto tile = dyn_cast<TileOp>(port.first);
//     WireBundle bundle = port.second.bundle;
//     int channel = port.second.channel;
//     int ID = map.first.second;
//     int mask = map.second;
//
//     LLVM_DEBUG(llvm::dbgs()
//                << "Port " << tile << " " << stringifyWireBundle(bundle) << "
//                "
//                << channel << '\n');
//     LLVM_DEBUG(llvm::dbgs() << "Mask "
//                             << "0x" << llvm::Twine::utohexstr(mask) << '\n');
//     LLVM_DEBUG(llvm::dbgs() << "ID "
//                             << "0x" << llvm::Twine::utohexstr(ID) << '\n');
//     for (int i = 0; i < 31; i++) {
//       if ((i & mask) == (ID & mask))
//         LLVM_DEBUG(llvm::dbgs() << "matches flow ID "
//                                 << "0x" << llvm::Twine::utohexstr(i) <<
//                                 '\n');
//     }
//   }
// #endif
//
//   // Realize the routes in MLIR
//   for (auto map : tiles) {
//     Operation *tileOp = map.second;
//     auto tile = dyn_cast<TileOp>(tileOp);
//
//     // Create a switchbox for the routes and insert inside it.
//     builder.setInsertionPointAfter(tileOp);
//     SwitchboxOp swbox = getOrCreateSwitchbox(builder, tile);
//     SwitchboxOp::ensureTerminator(swbox.getConnections(), builder,
//                                   builder.getUnknownLoc());
//     Block &b = swbox.getConnections().front();
//     builder.setInsertionPoint(b.getTerminator());
//
//     std::vector<bool> amselOpNeededVector(32);
//     for (const auto &map : mastersets) {
//       if (tileOp != map.first.first) continue;
//
//       for (auto value : map.second) {
//         amselOpNeededVector[value] = true;
//       }
//     }
//     // Create all the amsel Ops
//     DenseMap<int, AMSelOp> amselOps;
//     for (int i = 0; i < 32; i++) {
//       if (amselOpNeededVector[i]) {
//         int arbiterID = i % numArbiters;
//         int msel = i / numArbiters;
//         auto amsel =
//             builder.create<AMSelOp>(builder.getUnknownLoc(), arbiterID,
//             msel);
//         amselOps[i] = amsel;
//       }
//     }
//     // Create all the master set Ops
//     // First collect the master sets for this tile.
//     SmallVector<Port, 4> tileMasters;
//     for (const auto &map : mastersets) {
//       if (tileOp != map.first.first) continue;
//       tileMasters.push_back(map.first.second);
//     }
//     // Sort them so we get a reasonable order
//     std::sort(tileMasters.begin(), tileMasters.end());
//     for (auto tileMaster : tileMasters) {
//       WireBundle bundle = tileMaster.bundle;
//       int channel = tileMaster.channel;
//       SmallVector<int, 4> msels = mastersets[{tileOp, tileMaster}];
//       SmallVector<Value, 4> amsels;
//       for (auto msel : msels) {
//         assert(amselOps.count(msel) == 1);
//         amsels.push_back(amselOps[msel]);
//       }
//
//       auto msOp = builder.create<MasterSetOp>(builder.getUnknownLoc(),
//                                               builder.getIndexType(), bundle,
//                                               channel, amsels);
//       if (auto pktFlowAttrs = keepPktHeaderAttr[{tileOp, tileMaster}])
//         msOp->setAttr("keep_pkt_header", pktFlowAttrs);
//     }
//
//     // Generate the packet rules
//     DenseMap<Port, PacketRulesOp> slaveRules;
//     for (auto group : slaveGroups) {
//       builder.setInsertionPoint(b.getTerminator());
//
//       auto port = group.front().first;
//       if (tileOp != port.first) continue;
//
//       WireBundle bundle = port.second.bundle;
//       int channel = port.second.channel;
//       auto slave = port.second;
//
//       int mask = slaveMasks[group.front()];
//       int ID = group.front().second & mask;
//
//       // Verify that we actually map all the ID's correctly.
// #ifndef NDEBUG
//       for (auto slave : group) assert((slave.second & mask) == ID);
// #endif
//       Value amsel = amselOps[slaveAMSels[group.front()]];
//
//       PacketRulesOp packetrules;
//       if (slaveRules.count(slave) == 0) {
//         packetrules = builder.create<PacketRulesOp>(builder.getUnknownLoc(),
//                                                     bundle, channel);
//         PacketRulesOp::ensureTerminator(packetrules.getRules(), builder,
//                                         builder.getUnknownLoc());
//         slaveRules[slave] = packetrules;
//       } else
//         packetrules = slaveRules[slave];
//
//       Block &rules = packetrules.getRules().front();
//       builder.setInsertionPoint(rules.getTerminator());
//       builder.create<PacketRuleOp>(builder.getUnknownLoc(), mask, ID, amsel);
//     }
//   }
//
//   // Add support for shimDMA
//   // From shimDMA to BLI: 1) shimDMA 0 --> North 3
//   //                      2) shimDMA 1 --> North 7
//   // From BLI to shimDMA: 1) North   2 --> shimDMA 0
//   //                      2) North   3 --> shimDMA 1
//
//   for (auto switchbox : make_early_inc_range(device.getOps<SwitchboxOp>())) {
//     auto retVal = switchbox->getOperand(0);
//     auto tileOp = retVal.getDefiningOp<TileOp>();
//
//     // Check if it is a shim Tile
//     if (!tileOp.isShimNOCTile()) continue;
//
//     // Check if the switchbox is empty
//     if (&switchbox.getBody()->front() ==
//     switchbox.getBody()->getTerminator())
//       continue;
//
//     Region &r = switchbox.getConnections();
//     Block &b = r.front();
//
//     // Find if the corresponding shimmux exsists or not
//     int shimExist = 0;
//     ShimMuxOp shimOp;
//     for (auto shimmux : device.getOps<ShimMuxOp>()) {
//       if (shimmux.getTile() == tileOp) {
//         shimExist = 1;
//         shimOp = shimmux;
//         break;
//       }
//     }
//
//     for (Operation &Op : b.getOperations()) {
//       if (auto pktrules = dyn_cast<PacketRulesOp>(Op)) {
//         // check if there is MM2S DMA in the switchbox of the 0th row
//         if (pktrules.getSourceBundle() == WireBundle::DMA) {
//           // If there is, then it should be put into the corresponding
//           shimmux
//           // If shimmux not defined then create shimmux
//           if (!shimExist) {
//             builder.setInsertionPointAfter(tileOp);
//             shimOp = builder.create<ShimMuxOp>(builder.getUnknownLoc(),
//             tileOp); Region &r1 = shimOp.getConnections(); Block *b1 =
//             builder.createBlock(&r1); builder.setInsertionPointToEnd(b1);
//             builder.create<EndOp>(builder.getUnknownLoc());
//             shimExist = 1;
//           }
//
//           Region &r0 = shimOp.getConnections();
//           Block &b0 = r0.front();
//           builder.setInsertionPointToStart(&b0);
//
//           pktrules.setSourceBundle(WireBundle::South);
//           if (pktrules.getSourceChannel() == 0) {
//             pktrules.setSourceChannel(3);
//             builder.create<ConnectOp>(builder.getUnknownLoc(),
//             WireBundle::DMA,
//                                       0, WireBundle::North, 3);
//           }
//           if (pktrules.getSourceChannel() == 1) {
//             pktrules.setSourceChannel(7);
//             builder.create<ConnectOp>(builder.getUnknownLoc(),
//             WireBundle::DMA,
//                                       1, WireBundle::North, 7);
//           }
//         }
//       }
//
//       if (auto mtset = dyn_cast<MasterSetOp>(Op)) {
//         // check if there is S2MM DMA in the switchbox of the 0th row
//         if (mtset.getDestBundle() == WireBundle::DMA) {
//           // If there is, then it should be put into the corresponding
//           shimmux
//           // If shimmux not defined then create shimmux
//           if (!shimExist) {
//             builder.setInsertionPointAfter(tileOp);
//             shimOp = builder.create<ShimMuxOp>(builder.getUnknownLoc(),
//             tileOp); Region &r1 = shimOp.getConnections(); Block *b1 =
//             builder.createBlock(&r1); builder.setInsertionPointToEnd(b1);
//             builder.create<EndOp>(builder.getUnknownLoc());
//             shimExist = 1;
//           }
//
//           Region &r0 = shimOp.getConnections();
//           Block &b0 = r0.front();
//           builder.setInsertionPointToStart(&b0);
//
//           mtset.setDestBundle(WireBundle::South);
//           if (mtset.getDestChannel() == 0) {
//             mtset.setDestChannel(2);
//             builder.create<ConnectOp>(builder.getUnknownLoc(),
//                                       WireBundle::North, 2, WireBundle::DMA,
//                                       0);
//           }
//           if (mtset.getDestChannel() == 1) {
//             mtset.setDestChannel(3);
//             builder.create<ConnectOp>(builder.getUnknownLoc(),
//                                       WireBundle::North, 3, WireBundle::DMA,
//                                       1);
//           }
//         }
//       }
//     }
//   }
//
//   RewritePatternSet patterns(&getContext());
//
//   if (!clKeepFlowOp)
//     patterns.add<AIEOpRemoval<PacketFlowOp>>(device.getContext());
//
//   if (failed(applyPartialConversion(device, target, std::move(patterns))))
//     signalPassFailure();
// }
//
// void AMDAIEPathfinderPass::runOnOperation() {
//   flowOps.clear();
//   // create analysis pass with routing graph for entire device
//   LLVM_DEBUG(llvm::dbgs() << "---Begin AMDAIEPathfinderPass---\n");
//
//   DeviceOp d = getOperation();
//   if (failed(analyzer.runAnalysis(d))) return signalPassFailure();
//   OpBuilder builder = OpBuilder::atBlockEnd(d.getBody());
//
//   if (clRouteCircuit) runOnFlow(d, builder);
//   if (clRoutePacket) runOnPacketFlow(d, builder);
// }
//
// SwitchboxOp AMDAIEPathfinderPass::getSwitchbox(DeviceOp &d, int col, int row)
// {
//   SwitchboxOp output = nullptr;
//   d.walk([&](SwitchboxOp swBox) {
//     if (swBox.colIndex() == col && swBox.rowIndex() == row) {
//       output = swBox;
//     }
//   });
//   return output;
// }
//
}  // namespace mlir::iree_compiler::AMDAIE

namespace mlir::iree_compiler::AMDAIE {

template <typename DerivedT>
class AIERoutePathfinderFlowsBase : public ::mlir::OperationPass<DeviceOp> {
 public:
  using Base = AIERoutePathfinderFlowsBase;

  AIERoutePathfinderFlowsBase()
      : ::mlir::OperationPass<DeviceOp>(::mlir::TypeID::get<DerivedT>()) {}
  AIERoutePathfinderFlowsBase(const AIERoutePathfinderFlowsBase &other)
      : ::mlir::OperationPass<DeviceOp>(other) {}
  AIERoutePathfinderFlowsBase &operator=(const AIERoutePathfinderFlowsBase &) =
      delete;
  AIERoutePathfinderFlowsBase(AIERoutePathfinderFlowsBase &&) = delete;
  AIERoutePathfinderFlowsBase &operator=(AIERoutePathfinderFlowsBase &&) =
      delete;
  ~AIERoutePathfinderFlowsBase() = default;

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("amdaie-create-pathfinder-flows");
  }
  ::llvm::StringRef getArgument() const override {
    return "amdaie-create-pathfinder-flows";
  }

  ::llvm::StringRef getDescription() const override {
    return "Route aie.flow and aie.packetflow operations through switchboxes";
  }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("AIERoutePathfinderFlows");
  }
  ::llvm::StringRef getName() const override {
    return "AIERoutePathfinderFlows";
  }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<DerivedT>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<xilinx::AIE::AIEDialect>();
  }

  /// Explicitly declare the TypeID for this class. We declare an explicit
  /// private instantiation because Pass classes should only be visible by the
  /// current library.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      AIERoutePathfinderFlowsBase<DerivedT>)

  AIERoutePathfinderFlowsBase(const AIERoutePathfinderFlowsOptions &options)
      : AIERoutePathfinderFlowsBase() {
    clRouteCircuit = options.clRouteCircuit;
    clRoutePacket = options.clRoutePacket;
    clKeepFlowOp = options.clKeepFlowOp;
  }

 protected:
  ::mlir::Pass::Option<bool> clRouteCircuit{
      *this, "route-circuit",
      ::llvm::cl::desc("Flag to enable aie.flow lowering."),
      ::llvm::cl::init(true)};
  ::mlir::Pass::Option<bool> clRoutePacket{
      *this, "route-packet",
      ::llvm::cl::desc("Flag to enable aie.packetflow lowering."),
      ::llvm::cl::init(true)};
  ::mlir::Pass::Option<bool> clKeepFlowOp{
      *this, "keep-flow-op",
      ::llvm::cl::desc("Flag to not erase aie.flow/packetflow after its "
                       "lowering,used for routing visualization."),
      ::llvm::cl::init(false)};

 private:
};

struct AIEPathfinderPass : AIERoutePathfinderFlowsBase<AIEPathfinderPass> {
  DynamicTileAnalysis analyzer;
  mlir::DenseMap<TileID, mlir::Operation *> tiles;

  AIEPathfinderPass() = default;
  AIEPathfinderPass(DynamicTileAnalysis analyzer)
      : analyzer(std::move(analyzer)) {}

  void runOnOperation() override;
  void runOnFlow(DeviceOp d, mlir::OpBuilder &builder);
  void runOnPacketFlow(DeviceOp d, mlir::OpBuilder &builder);

  typedef std::pair<mlir::Operation *, Port> PhysPort;

  typedef struct {
    SwitchboxOp sw;
    Port sourcePort;
    Port destPort;
  } SwConnection;

  bool findPathToDest(SwitchSettings settings, TileID currTile,
                      WireBundle currDestBundle, int currDestChannel,
                      TileID finalTile, WireBundle finalDestBundle,
                      int finalDestChannel);

  SwitchboxOp getSwitchbox(DeviceOp &d, int col, int row);

  mlir::Operation *getOrCreateTile(mlir::OpBuilder &builder, int col, int row);
  SwitchboxOp getOrCreateSwitchbox(mlir::OpBuilder &builder, TileOp tile);
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

  RewritePatternSet patterns(&getContext());
  patterns.insert<ConvertFlowsToInterconnect>(d.getContext(), d, analyzer,
                                              clKeepFlowOp);
  if (failed(applyPartialConversion(d, target, std::move(patterns))))
    return signalPassFailure();

  // Keep for visualization
  if (clKeepFlowOp)
    for (auto op : flowOps) builder.insert(op);

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
                                 WireBundle::East, sw, WireBundle::West);
        }
      }
      if (row > 0) {
        // connections between abstract 'core' of tile
        builder.create<WireOp>(builder.getUnknownLoc(), tile, WireBundle::Core,
                               sw, WireBundle::Core);
        // connections between abstract 'dma' of tile
        builder.create<WireOp>(builder.getUnknownLoc(), tile, WireBundle::DMA,
                               sw, WireBundle::DMA);
        // connections north-south inside array ( including connection to shim
        // row)
        if (analyzer.coordToSwitchbox.count({col, row - 1})) {
          auto southsw = analyzer.coordToSwitchbox[{col, row - 1}];
          builder.create<WireOp>(builder.getUnknownLoc(), southsw,
                                 WireBundle::North, sw, WireBundle::South);
        }
      } else if (row == 0) {
        if (tile.isShimNOCTile()) {
          if (analyzer.coordToShimMux.count({col, 0})) {
            auto shimsw = analyzer.coordToShimMux[{col, 0}];
            builder.create<WireOp>(
                builder.getUnknownLoc(), shimsw,
                WireBundle::North,  // Changed to connect into the north
                sw, WireBundle::South);
            // PLIO is attached to shim mux
            if (analyzer.coordToPLIO.count(col)) {
              auto plio = analyzer.coordToPLIO[col];
              builder.create<WireOp>(builder.getUnknownLoc(), plio,
                                     WireBundle::North, shimsw,
                                     WireBundle::South);
            }

            // abstract 'DMA' connection on tile is attached to shim mux ( in
            // row 0 )
            builder.create<WireOp>(builder.getUnknownLoc(), tile,
                                   WireBundle::DMA, shimsw, WireBundle::DMA);
          }
        } else if (tile.isShimPLTile()) {
          // PLIO is attached directly to switch
          if (analyzer.coordToPLIO.count(col)) {
            auto plio = analyzer.coordToPLIO[col];
            builder.create<WireOp>(builder.getUnknownLoc(), plio,
                                   WireBundle::North, sw, WireBundle::South);
          }
        }
      }
    }
  }
}

Operation *AIEPathfinderPass::getOrCreateTile(OpBuilder &builder, int col,
                                              int row) {
  TileID index = {col, row};
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

bool AIEPathfinderPass::findPathToDest(SwitchSettings settings, TileID currTile,
                                       WireBundle currDestBundle,
                                       int currDestChannel, TileID finalTile,
                                       WireBundle finalDestBundle,
                                       int finalDestChannel) {
  if ((currTile == finalTile) && (currDestBundle == finalDestBundle) &&
      (currDestChannel == finalDestChannel)) {
    return true;
  }

  WireBundle neighbourSourceBundle;
  TileID neighbourTile;
  if (currDestBundle == WireBundle::East) {
    neighbourSourceBundle = WireBundle::West;
    neighbourTile = {currTile.col + 1, currTile.row};
  } else if (currDestBundle == WireBundle::West) {
    neighbourSourceBundle = WireBundle::East;
    neighbourTile = {currTile.col - 1, currTile.row};
  } else if (currDestBundle == WireBundle::North) {
    neighbourSourceBundle = WireBundle::South;
    neighbourTile = {currTile.col, currTile.row + 1};
  } else if (currDestBundle == WireBundle::South) {
    neighbourSourceBundle = WireBundle::North;
    neighbourTile = {currTile.col, currTile.row - 1};
  } else {
    return false;
  }

  int neighbourSourceChannel = currDestChannel;
  for (const auto &[sbNode, setting] : settings) {
    TileID tile = {sbNode.col, sbNode.row};
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
  DenseMap<TileID, SmallVector<std::pair<Connect, int>, 8>> switchboxes;
  for (PacketFlowOp pktFlowOp : device.getOps<PacketFlowOp>()) {
    Region &r = pktFlowOp.getPorts();
    Block &b = r.front();
    int flowID = pktFlowOp.IDInt();
    Port srcPort, destPort;
    TileOp srcTile, destTile;
    TileID srcCoords, destCoords;

    for (Operation &Op : b.getOperations()) {
      if (auto pktSource = dyn_cast<PacketSourceOp>(Op)) {
        srcTile = dyn_cast<TileOp>(pktSource.getTile().getDefiningOp());
        srcPort = pktSource.port();
        srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
      } else if (auto pktDest = dyn_cast<PacketDestOp>(Op)) {
        destTile = dyn_cast<TileOp>(pktDest.getTile().getDefiningOp());
        destPort = pktDest.port();
        destCoords = {destTile.colIndex(), destTile.rowIndex()};
        // Assign "keep_pkt_header flag"
        if (pktFlowOp->hasAttr("keep_pkt_header"))
          keepPktHeaderAttr[{destTile, destPort}] =
              StringAttr::get(Op.getContext(), "true");
        SwitchboxNode srcSB = analyzer.pathfinder->getSwitchboxNode(
            {srcCoords.col, srcCoords.row});
        if (PathEndPoint srcPoint = {srcSB, srcPort};
            !analyzer.processedFlows[srcPoint]) {
          SwitchSettings settings = analyzer.flowSolutions[srcPoint];
          // add connections for all the Switchboxes in SwitchSettings
          for (const auto &[curr, setting] : settings) {
            for (const auto &[bundle, channel] : setting.dsts) {
              TileID currTile = {curr.col, curr.row};
              // reject false broadcast
              if (!findPathToDest(settings, currTile, bundle, channel,
                                  destCoords, destPort.bundle,
                                  destPort.channel))
                continue;
              Connect connect = {{setting.src.bundle, setting.src.channel},
                                 {bundle, channel}};
              if (std::find(switchboxes[currTile].begin(),
                            switchboxes[currTile].end(),
                            std::pair{connect, flowID}) ==
                  switchboxes[currTile].end())
                switchboxes[currTile].push_back({connect, flowID});
            }
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
      LLVM_DEBUG(llvm::dbgs() << "flowID " << flowID << ':'
                              << stringifyWireBundle(sourcePort.bundle) << " "
                              << sourcePort.channel << " -> "
                              << stringifyWireBundle(destPort.bundle) << " "
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
    assert(tileOp);
    int amselValue = physPort.second;
    for (auto port : ports) {
      PhysPort physPort = {tileOp, port};
      mastersets[physPort].push_back(amselValue);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "CHECK mastersets\n");
#ifndef NDEBUG
  for (const auto &[physPort, values] : mastersets) {
    Operation *tileOp = physPort.first;
    WireBundle bundle = physPort.second.bundle;
    int channel = physPort.second.channel;
    assert(tileOp);
    auto tile = dyn_cast<TileOp>(tileOp);
    LLVM_DEBUG(llvm::dbgs()
               << "master " << tile << " " << stringifyWireBundle(bundle)
               << " : " << channel << '\n');
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
    WireBundle bundle = port.second.bundle;
    int channel = port.second.channel;
    int ID = map.first.second;
    int mask = map.second;

    LLVM_DEBUG(llvm::dbgs()
               << "Port " << tile << " " << stringifyWireBundle(bundle) << " "
               << channel << '\n');
    LLVM_DEBUG(llvm::dbgs() << "Mask "
                            << "0x" << llvm::Twine::utohexstr(mask) << '\n');
    LLVM_DEBUG(llvm::dbgs() << "ID "
                            << "0x" << llvm::Twine::utohexstr(ID) << '\n');
    for (int i = 0; i < 31; i++) {
      if ((i & mask) == (ID & mask))
        LLVM_DEBUG(llvm::dbgs() << "matches flow ID "
                                << "0x" << llvm::Twine::utohexstr(i) << '\n');
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
    for (const auto &map : mastersets) {
      if (tileOp != map.first.first) continue;

      for (auto value : map.second) {
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
    for (const auto &map : mastersets) {
      if (tileOp != map.first.first) continue;
      tileMasters.push_back(map.first.second);
    }
    // Sort them so we get a reasonable order
    std::sort(tileMasters.begin(), tileMasters.end());
    for (auto tileMaster : tileMasters) {
      WireBundle bundle = tileMaster.bundle;
      int channel = tileMaster.channel;
      SmallVector<int, 4> msels = mastersets[{tileOp, tileMaster}];
      SmallVector<Value, 4> amsels;
      for (auto msel : msels) {
        assert(amselOps.count(msel) == 1);
        amsels.push_back(amselOps[msel]);
      }

      auto msOp = builder.create<MasterSetOp>(builder.getUnknownLoc(),
                                              builder.getIndexType(), bundle,
                                              channel, amsels);
      if (auto pktFlowAttrs = keepPktHeaderAttr[{tileOp, tileMaster}])
        msOp->setAttr("keep_pkt_header", pktFlowAttrs);
    }

    // Generate the packet rules
    DenseMap<Port, PacketRulesOp> slaveRules;
    for (auto group : slaveGroups) {
      builder.setInsertionPoint(b.getTerminator());

      auto port = group.front().first;
      if (tileOp != port.first) continue;

      WireBundle bundle = port.second.bundle;
      int channel = port.second.channel;
      auto slave = port.second;

      int mask = slaveMasks[group.front()];
      int ID = group.front().second & mask;

      // Verify that we actually map all the ID's correctly.
#ifndef NDEBUG
      for (auto slave : group) assert((slave.second & mask) == ID);
#endif
      Value amsel = amselOps[slaveAMSels[group.front()]];

      PacketRulesOp packetrules;
      if (slaveRules.count(slave) == 0) {
        packetrules = builder.create<PacketRulesOp>(builder.getUnknownLoc(),
                                                    bundle, channel);
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

  // Add support for shimDMA
  // From shimDMA to BLI: 1) shimDMA 0 --> North 3
  //                      2) shimDMA 1 --> North 7
  // From BLI to shimDMA: 1) North   2 --> shimDMA 0
  //                      2) North   3 --> shimDMA 1

  for (auto switchbox : make_early_inc_range(device.getOps<SwitchboxOp>())) {
    auto retVal = switchbox->getOperand(0);
    auto tileOp = retVal.getDefiningOp<TileOp>();

    // Check if it is a shim Tile
    if (!tileOp.isShimNOCTile()) continue;

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
        if (pktrules.getSourceBundle() == WireBundle::DMA) {
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

          pktrules.setSourceBundle(WireBundle::South);
          if (pktrules.getSourceChannel() == 0) {
            pktrules.setSourceChannel(3);
            builder.create<ConnectOp>(builder.getUnknownLoc(), WireBundle::DMA,
                                      0, WireBundle::North, 3);
          }
          if (pktrules.getSourceChannel() == 1) {
            pktrules.setSourceChannel(7);
            builder.create<ConnectOp>(builder.getUnknownLoc(), WireBundle::DMA,
                                      1, WireBundle::North, 7);
          }
        }
      }

      if (auto mtset = dyn_cast<MasterSetOp>(Op)) {
        // check if there is S2MM DMA in the switchbox of the 0th row
        if (mtset.getDestBundle() == WireBundle::DMA) {
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

          mtset.setDestBundle(WireBundle::South);
          if (mtset.getDestChannel() == 0) {
            mtset.setDestChannel(2);
            builder.create<ConnectOp>(builder.getUnknownLoc(),
                                      WireBundle::North, 2, WireBundle::DMA, 0);
          }
          if (mtset.getDestChannel() == 1) {
            mtset.setDestChannel(3);
            builder.create<ConnectOp>(builder.getUnknownLoc(),
                                      WireBundle::North, 3, WireBundle::DMA, 1);
          }
        }
      }
    }
  }

  RewritePatternSet patterns(&getContext());

  if (!clKeepFlowOp)
    patterns.add<AIEOpRemoval<PacketFlowOp>>(device.getContext());

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

SwitchboxOp AIEPathfinderPass::getSwitchbox(DeviceOp &d, int col, int row) {
  SwitchboxOp output = nullptr;
  d.walk([&](SwitchboxOp swBox) {
    if (swBox.colIndex() == col && swBox.rowIndex() == row) {
      output = swBox;
    }
  });
  return output;
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
