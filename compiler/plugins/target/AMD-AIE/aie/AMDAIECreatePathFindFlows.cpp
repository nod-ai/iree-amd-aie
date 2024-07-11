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
#include "llvm/ADT/DirectedGraph.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_os_ostream.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "amdaie-create-pathfinder-flows"

using namespace mlir;

using mlir::iree_compiler::AMDAIE::AMDAIEDevice;
using mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel;
using mlir::iree_compiler::AMDAIE::TileLoc;
using xilinx::AIE::ConnectOp;
using xilinx::AIE::DeviceOp;
using xilinx::AIE::DMAChannelDir;
using xilinx::AIE::EndOp;
using xilinx::AIE::FlowOp;
using xilinx::AIE::getConnectingBundle;
using xilinx::AIE::Interconnect;
using xilinx::AIE::PLIOOp;
using xilinx::AIE::ShimMuxOp;
using xilinx::AIE::SwitchboxOp;
using xilinx::AIE::TileOp;
using xilinx::AIE::WireBundle;
using xilinx::AIE::WireOp;

#define OVER_CAPACITY_COEFF 0.02
#define USED_CAPACITY_COEFF 0.02
#define DEMAND_COEFF 1.1

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

bool operator==(const StrmSwPortType &lhs, const WireBundle &rhs) {
  return lhs == toStrmT(rhs);
}
}  // namespace

namespace {
struct Port {
  WireBundle bundle;
  int channel;

  bool operator==(const Port &rhs) const {
    return std::tie(bundle, channel) == std::tie(rhs.bundle, rhs.channel);
  }

  bool operator!=(const Port &rhs) const { return !(*this == rhs); }

  bool operator<(const Port &rhs) const {
    return std::tie(bundle, channel) < std::tie(rhs.bundle, rhs.channel);
  }

  friend std::ostream &operator<<(std::ostream &os, const Port &port) {
    os << "(";
    switch (port.bundle) {
      case WireBundle::Core:
        os << "Core";
        break;
      case WireBundle::DMA:
        os << "DMA";
        break;
      case WireBundle::North:
        os << "N";
        break;
      case WireBundle::East:
        os << "E";
        break;
      case WireBundle::South:
        os << "S";
        break;
      case WireBundle::West:
        os << "W";
        break;
      default:
        os << "X";
        break;
    }
    os << ": " << std::to_string(port.channel) << ")";
    return os;
  }

  GENERATE_TO_STRING(Port)

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const Port &port) {
    os << to_string(port);
    return os;
  }
};
}  // namespace

namespace std {
template <>
struct less<Port> {
  bool operator()(const Port &a, const Port &b) const {
    return a.bundle == b.bundle ? a.channel < b.channel : a.bundle < b.bundle;
  }
};

template <>
struct hash<Port> {
  size_t operator()(const Port &p) const noexcept {
    size_t h1 = hash<WireBundle>{}(p.bundle);
    size_t h2 = hash<int>{}(p.channel);
    return h1 ^ h2 << 1;
  }
};
}  // namespace std

namespace {

#define GENERATE_TO_STRING(TYPE_WITH_INSERTION_OP)                \
  friend std::string to_string(const TYPE_WITH_INSERTION_OP &s) { \
    std::ostringstream ss;                                        \
    ss << s;                                                      \
    return ss.str();                                              \
  }

typedef struct Connect {
  Port src;
  Port dst;

  bool operator==(const Connect &rhs) const {
    return std::tie(src, dst) == std::tie(rhs.src, rhs.dst);
  }
} Connect;

typedef struct DMAChannel {
  DMAChannelDir direction;
  int channel;

  bool operator==(const DMAChannel &rhs) const {
    return std::tie(direction, channel) == std::tie(rhs.direction, rhs.channel);
  }
} DMAChannel;

struct Switchbox : TileLoc {
  // Necessary for initializer construction?
  Switchbox(TileLoc t) : TileLoc(t) {}
  Switchbox(int col, int row) : TileLoc{col, row} {}
  friend std::ostream &operator<<(std::ostream &os, const Switchbox &s) {
    os << "Switchbox(" << s.col << ", " << s.row << ")";
    return os;
  }

  GENERATE_TO_STRING(Switchbox);

  bool operator==(const Switchbox &rhs) const {
    return static_cast<TileLoc>(*this) == rhs;
  }
};

struct Channel {
  Channel(Switchbox &src, Switchbox &target, WireBundle bundle, int maxCapacity)
      : src(src), target(target), bundle(bundle), maxCapacity(maxCapacity) {}

  friend std::ostream &operator<<(std::ostream &os, const Channel &c) {
    os << "Channel(src=" << c.src << ", dst=" << c.target << ")";
    return os;
  }

  GENERATE_TO_STRING(Channel)

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const Channel &c) {
    os << to_string(c);
    return os;
  }

  Switchbox &src;
  Switchbox &target;
  WireBundle bundle;
  int maxCapacity = 0;   // maximum number of routing resources
  double demand = 0.0;   // indicates how many flows want to use this Channel
  int usedCapacity = 0;  // how many flows are actually using this Channel
  std::set<int> fixedCapacity;  // channels not available to the algorithm
  int overCapacityCount = 0;    // history of Channel being over capacity
};

// A SwitchSetting defines the required settings for a Switchbox for a flow
// SwitchSetting.src is the incoming signal
// SwitchSetting.dsts is the fanout
struct SwitchSetting {
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
    os << setting.src << " -> " << "{"
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

// A Flow defines source and destination vertices
// Only one source, but any number of destinations (fanout)
struct PathEndPoint {
  Switchbox sb;
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

}  // namespace

namespace std {
template <>
struct hash<TileLoc> {
  size_t operator()(const TileLoc &s) const noexcept {
    size_t h1 = hash<int>{}(s.col);
    size_t h2 = hash<int>{}(s.row);
    return h1 ^ (h2 << 1);
  }
};
// For some mysterious reason, the only way to get the priorityQueue(cmp)
// comparison in dijkstraShortestPaths to work correctly is to define
// this template specialization for the pointers. Overloading operator
// will not work. Furthermore, if  you try to move this into AIEPathFinder.cpp
// you'll get a compile error about
// "specialization of ‘std::less<Switchbox*>’ after
// instantiation" because one of the graph traits below is doing the comparison
// internally (try moving this below the llvm namespace...)
template <>
struct less<Switchbox *> {
  bool operator()(const Switchbox *a, const Switchbox *b) const {
    return *a < *b;
  }
};

template <>
struct hash<Switchbox> {
  size_t operator()(const Switchbox &s) const noexcept {
    return hash<TileLoc>{}(s);
  }
};

template <>
struct hash<PathEndPoint> {
  size_t operator()(const PathEndPoint &pe) const noexcept {
    size_t h1 = hash<Port>{}(pe.port);
    size_t h2 = hash<Switchbox>{}(pe.sb);
    return h1 ^ (h2 << 1);
  }
};

}  // namespace std

namespace {
struct SwitchboxNode;
struct ChannelEdge;
using SwitchboxNodeBase = llvm::DGNode<SwitchboxNode, ChannelEdge>;
using ChannelEdgeBase = llvm::DGEdge<SwitchboxNode, ChannelEdge>;
using SwitchboxGraphBase = llvm::DirectedGraph<SwitchboxNode, ChannelEdge>;

struct SwitchboxNode : SwitchboxNodeBase, Switchbox {
  using Switchbox::Switchbox;
  SwitchboxNode(int col, int row, int id) : Switchbox{col, row}, id{id} {}
  int id;
};

// warning: 'mlir::iree_compiler::AMDAIE::ChannelEdge::src' will be initialized
// after SwitchboxNode &src; [-Wreorder]
struct ChannelEdge : ChannelEdgeBase, Channel {
  using Channel::Channel;

  explicit ChannelEdge(SwitchboxNode &target) = delete;
  ChannelEdge(SwitchboxNode &src, SwitchboxNode &target, WireBundle bundle,
              int maxCapacity)
      : ChannelEdgeBase(target),
        Channel(src, target, bundle, maxCapacity),
        src(src) {}

  // This class isn't designed to copied or moved.
  ChannelEdge(const ChannelEdge &E) = delete;
  ChannelEdge &operator=(ChannelEdge &&E) = delete;

  SwitchboxNode &src;
};

class SwitchboxGraph : public SwitchboxGraphBase {
 public:
  SwitchboxGraph() = default;
  ~SwitchboxGraph() = default;
};

using SwitchSettings = std::map<Switchbox, SwitchSetting>;

// A Flow defines source and destination vertices
// Only one source, but any number of destinations (fanout)
struct PathEndPointNode : PathEndPoint {
  PathEndPointNode(SwitchboxNode *sb, Port port)
      : PathEndPoint{*sb, port}, sb(sb) {}
  SwitchboxNode *sb;
};

struct FlowNode {
  PathEndPointNode src;
  std::vector<PathEndPointNode> dsts;
};

class Pathfinder {
 public:
  Pathfinder() = default;
  void initialize(int maxCol, int maxRow, AMDAIEDeviceModel &deviceModel);
  void addFlow(TileLoc srcCoords, Port srcPort, TileLoc dstCoords,
               Port dstPort);
  bool addFixedConnection(ConnectOp connectOp);
  std::optional<std::map<PathEndPoint, SwitchSettings>> findPaths(
      int maxIterations);

  Switchbox *getSwitchbox(TileLoc coords) {
    auto *sb = std::find_if(graph.begin(), graph.end(), [&](SwitchboxNode *sb) {
      return sb->col == coords.col && sb->row == coords.row;
    });
    assert(sb != graph.end() && "couldn't find sb");
    return *sb;
  }

 private:
  SwitchboxGraph graph;
  std::vector<FlowNode> flows;
  std::map<TileLoc, SwitchboxNode> grid;
  // Use a list instead of a vector because nodes have an edge list of raw
  // pointers to edges (so growing a vector would invalidate the pointers).
  std::list<ChannelEdge> edges;
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

  llvm::DenseMap<TileLoc, TileOp> coordToTile;
  llvm::DenseMap<TileLoc, SwitchboxOp> coordToSwitchbox;
  llvm::DenseMap<TileLoc, ShimMuxOp> coordToShimMux;
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

}  // namespace

namespace llvm {
template <>
struct DenseMapInfo<DMAChannel> {
  using FirstInfo = DenseMapInfo<DMAChannelDir>;
  using SecondInfo = DenseMapInfo<int>;

  static DMAChannel getEmptyKey() {
    return {FirstInfo::getEmptyKey(), SecondInfo::getEmptyKey()};
  }

  static DMAChannel getTombstoneKey() {
    return {FirstInfo::getTombstoneKey(), SecondInfo::getTombstoneKey()};
  }

  static unsigned getHashValue(const DMAChannel &d) {
    return detail::combineHashValue(FirstInfo::getHashValue(d.direction),
                                    SecondInfo::getHashValue(d.channel));
  }

  static bool isEqual(const DMAChannel &lhs, const DMAChannel &rhs) {
    return lhs == rhs;
  }
};

template <>
struct DenseMapInfo<Port> {
  using FirstInfo = DenseMapInfo<WireBundle>;
  using SecondInfo = DenseMapInfo<int>;

  static Port getEmptyKey() {
    return {FirstInfo::getEmptyKey(), SecondInfo::getEmptyKey()};
  }

  static Port getTombstoneKey() {
    return {FirstInfo::getTombstoneKey(), SecondInfo::getTombstoneKey()};
  }

  static unsigned getHashValue(const Port &d) {
    return detail::combineHashValue(FirstInfo::getHashValue(d.bundle),
                                    SecondInfo::getHashValue(d.channel));
  }

  static bool isEqual(const Port &lhs, const Port &rhs) { return lhs == rhs; }
};

template <>
struct GraphTraits<SwitchboxNode *> {
  using NodeRef = SwitchboxNode *;

  static SwitchboxNode *SwitchboxGraphGetSwitchbox(
      DGEdge<SwitchboxNode, ChannelEdge> *P) {
    return &P->getTargetNode();
  }

  // Provide a mapped iterator so that the GraphTrait-based implementations
  // can find the target nodes without having to explicitly go through the
  // edges.
  using ChildIteratorType =
      mapped_iterator<SwitchboxNode::iterator,
                      decltype(&SwitchboxGraphGetSwitchbox)>;
  using ChildEdgeIteratorType = SwitchboxNode::iterator;

  static NodeRef getEntryNode(NodeRef N) { return N; }
  static ChildIteratorType child_begin(NodeRef N) {
    return {N->begin(), &SwitchboxGraphGetSwitchbox};
  }
  static ChildIteratorType child_end(NodeRef N) {
    return {N->end(), &SwitchboxGraphGetSwitchbox};
  }

  static ChildEdgeIteratorType child_edge_begin(NodeRef N) {
    return N->begin();
  }
  static ChildEdgeIteratorType child_edge_end(NodeRef N) { return N->end(); }
};

template <>
struct GraphTraits<SwitchboxGraph *> : GraphTraits<SwitchboxNode *> {
  using nodes_iterator = SwitchboxGraph::iterator;
  static NodeRef getEntryNode(SwitchboxGraph *DG) { return *DG->begin(); }
  static nodes_iterator nodes_begin(SwitchboxGraph *DG) { return DG->begin(); }
  static nodes_iterator nodes_end(SwitchboxGraph *DG) { return DG->end(); }
};

inline raw_ostream &operator<<(raw_ostream &os, const SwitchSettings &ss) {
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

namespace {

LogicalResult DynamicTileAnalysis::runAnalysis(DeviceOp &device) {
  LLVM_DEBUG(llvm::dbgs() << "\t---Begin DynamicTileAnalysis Constructor---\n");
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
    Port srcPort = {flowOp.getSourceBundle(), flowOp.getSourceChannel()};
    Port dstPort = {flowOp.getDestBundle(), flowOp.getDestChannel()};
    LLVM_DEBUG(llvm::dbgs()
               << "\tAdding Flow: (" << srcCoords.col << ", " << srcCoords.row
               << ")" << stringifyWireBundle(srcPort.bundle) << srcPort.channel
               << " -> (" << dstCoords.col << ", " << dstCoords.row << ")"
               << stringifyWireBundle(dstPort.bundle) << dstPort.channel
               << "\n");
    pathfinder->addFlow(srcCoords, srcPort, dstCoords, dstPort);
  }

  // add existing connections so Pathfinder knows which resources are
  // available search all existing SwitchBoxOps for exising connections
  for (SwitchboxOp switchboxOp : device.getOps<SwitchboxOp>()) {
    for (ConnectOp connectOp : switchboxOp.getOps<ConnectOp>()) {
      if (!pathfinder->addFixedConnection(connectOp))
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
    assert(coordToShimMux.count(TileLoc{col, row}) == 0);
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
                            AMDAIEDeviceModel &deviceModel) {
  // make grid of switchboxes
  int id = 0;
  for (int row = 0; row <= maxRow; row++) {
    for (int col = 0; col <= maxCol; col++) {
      auto [it, _] = grid.insert({{col, row}, SwitchboxNode{col, row, id++}});
      (void)graph.addNode(it->second);
      SwitchboxNode &thisNode = grid.at({col, row});
      if (row > 0) {  // if not in row 0 add channel to North/South
        SwitchboxNode &southernNeighbor = grid.at({col, row - 1});
        // get the number of outgoing connections on the south side - outgoing
        // because these correspond to rhs of a connect op
        if (uint32_t maxCapacity = deviceModel.getNumDestSwitchboxConnections(
                col, row, toStrmT(WireBundle::South))) {
          edges.emplace_back(thisNode, southernNeighbor, WireBundle::South,
                             maxCapacity);
          (void)graph.connect(thisNode, southernNeighbor, edges.back());
        }
        // get the number of incoming connections on the south side - incoming
        // because they correspond to connections on the southside that are then
        // routed using internal connect ops through the switchbox (i.e., lhs of
        // connect ops)
        if (uint32_t maxCapacity = deviceModel.getNumSourceSwitchboxConnections(
                col, row, toStrmT(WireBundle::South))) {
          edges.emplace_back(southernNeighbor, thisNode, WireBundle::North,
                             maxCapacity);
          (void)graph.connect(southernNeighbor, thisNode, edges.back());
        }
      }

      if (col > 0) {  // if not in col 0 add channel to East/West
        SwitchboxNode &westernNeighbor = grid.at({col - 1, row});
        if (uint32_t maxCapacity = deviceModel.getNumDestSwitchboxConnections(
                col, row, toStrmT(WireBundle::West))) {
          edges.emplace_back(thisNode, westernNeighbor, WireBundle::West,
                             maxCapacity);
          (void)graph.connect(thisNode, westernNeighbor, edges.back());
        }
        if (uint32_t maxCapacity = deviceModel.getNumSourceSwitchboxConnections(
                col, row, toStrmT(WireBundle::West))) {
          edges.emplace_back(westernNeighbor, thisNode, WireBundle::East,
                             maxCapacity);
          (void)graph.connect(westernNeighbor, thisNode, edges.back());
        }
      }
    }
  }
}

// Add a flow from src to dst can have an arbitrary number of dst locations due
// to fanout.
void Pathfinder::addFlow(TileLoc srcCoords, Port srcPort, TileLoc dstCoords,
                         Port dstPort) {
  // check if a flow with this source already exists
  for (auto &[src, dsts] : flows) {
    SwitchboxNode *existingSrc = src.sb;
    assert(existingSrc && "nullptr flow source");
    if (Port existingPort = src.port; existingSrc->col == srcCoords.col &&
                                      existingSrc->row == srcCoords.row &&
                                      existingPort == srcPort) {
      // find the vertex corresponding to the destination
      auto *matchingSb = std::find_if(
          graph.begin(), graph.end(), [&](const SwitchboxNode *sb) {
            return sb->col == dstCoords.col && sb->row == dstCoords.row;
          });
      assert(matchingSb != graph.end() && "didn't find flow dest");
      dsts.emplace_back(*matchingSb, dstPort);
      return;
    }
  }

  // If no existing flow was found with this source, create a new flow.
  auto *matchingSrcSb =
      std::find_if(graph.begin(), graph.end(), [&](const SwitchboxNode *sb) {
        return sb->col == srcCoords.col && sb->row == srcCoords.row;
      });
  assert(matchingSrcSb != graph.end() && "didn't find flow source");
  auto *matchingDstSb =
      std::find_if(graph.begin(), graph.end(), [&](const SwitchboxNode *sb) {
        return sb->col == dstCoords.col && sb->row == dstCoords.row;
      });
  assert(matchingDstSb != graph.end() && "didn't add flow destinations");
  flows.push_back({PathEndPointNode{*matchingSrcSb, srcPort},
                   std::vector<PathEndPointNode>{{*matchingDstSb, dstPort}}});
}

// Keep track of connections already used in the AIE; Pathfinder algorithm will
// avoid using these.
bool Pathfinder::addFixedConnection(ConnectOp connectOp) {
  auto sb = connectOp->getParentOfType<SwitchboxOp>();
  // TODO: keep track of capacity?
  if (sb.getTileOp().isShimNOCTile()) return true;

  TileLoc sbTile(sb.getTileID().col, sb.getTileID().row);
  WireBundle sourceBundle = connectOp.getSourceBundle();
  WireBundle destBundle = connectOp.getDestBundle();

  // find the correct Channel and indicate the fixed direction
  // outgoing connection
  auto matchingCh =
      std::find_if(edges.begin(), edges.end(), [&](ChannelEdge &ch) {
        return static_cast<TileLoc>(ch.src) == sbTile &&
               ch.bundle == destBundle;
      });
  if (matchingCh != edges.end())
    return matchingCh->fixedCapacity.insert(connectOp.getDestChannel())
               .second ||
           true;

  // incoming connection
  matchingCh = std::find_if(edges.begin(), edges.end(), [&](ChannelEdge &ch) {
    return static_cast<TileLoc>(ch.target) == sbTile &&
           ch.bundle == getConnectingBundle(sourceBundle);
  });
  if (matchingCh != edges.end())
    return matchingCh->fixedCapacity.insert(connectOp.getSourceChannel())
               .second ||
           true;

  return false;
}

static constexpr double INF = std::numeric_limits<double>::max();

std::map<SwitchboxNode *, SwitchboxNode *> dijkstraShortestPaths(
    const SwitchboxGraph &graph, SwitchboxNode *src) {
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

  for (SwitchboxNode *sb : graph) distance.emplace(sb, INF);
  distance[src] = 0.0;

  std::map<SwitchboxNode *, std::vector<ChannelEdge *>> edges;

  enum Color { WHITE, GRAY, BLACK };
  std::map<SwitchboxNode *, Color> colors;
  for (SwitchboxNode *sb : graph) {
    colors[sb] = WHITE;
    edges[sb] = {sb->getEdges().begin(), sb->getEdges().end()};
    std::sort(edges[sb].begin(), edges[sb].end(),
              [](const ChannelEdge *c1, ChannelEdge *c2) {
                return c1->getTargetNode().id < c2->getTargetNode().id;
              });
  }

  Q.push(src);
  while (!Q.empty()) {
    src = Q.top();
    Q.pop();
    for (ChannelEdge *e : edges[src]) {
      SwitchboxNode *dest = &e->getTargetNode();
      bool relax = distance[src] + e->demand < distance[dest];
      if (colors[dest] == WHITE) {
        if (relax) {
          distance[dest] = distance[src] + e->demand;
          preds[dest] = src;
          colors[dest] = GRAY;
        }
        Q.push(dest);
      } else if (colors[dest] == GRAY && relax) {
        distance[dest] = distance[src] + e->demand;
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
  for (auto &ch : edges) ch.overCapacityCount = 0;

  // Check that every channel does not exceed max capacity.
  auto isLegal = [&] {
    bool legal = true;  // assume legal until found otherwise
    for (auto &e : edges) {
      if (e.usedCapacity > e.maxCapacity) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Too much capacity on Edge (" << e.getTargetNode().col
                   << ", " << e.getTargetNode().row << ") . "
                   << stringifyWireBundle(e.bundle) << "\t: used_capacity = "
                   << e.usedCapacity << "\t: Demand = " << e.demand << "\n");
        e.overCapacityCount++;
        LLVM_DEBUG(llvm::dbgs()
                   << "over_capacity_count = " << e.overCapacityCount << "\n");
        legal = false;
        break;
      }
    }

    return legal;
  };

  do {
    LLVM_DEBUG(llvm::dbgs()
               << "Begin findPaths iteration #" << iterationCount << "\n");
    // update demand on all channels
    for (auto &ch : edges) {
      if (ch.fixedCapacity.size() >=
          static_cast<std::set<int>::size_type>(ch.maxCapacity)) {
        ch.demand = INF;
      } else {
        double history = 1.0 + OVER_CAPACITY_COEFF * ch.overCapacityCount;
        double congestion = 1.0 + USED_CAPACITY_COEFF * ch.usedCapacity;
        ch.demand = history * congestion;
      }
    }
    // if reach maxIterations, throw an error since no routing can be found
    if (++iterationCount > maxIterations) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Pathfinder: maxIterations has been exceeded ("
                 << maxIterations
                 << " iterations)...unable to find routing for flows.\n");
      return std::nullopt;
    }

    // "rip up" all routes, i.e. set used capacity in each Channel to 0
    routingSolution.clear();
    for (auto &ch : edges) ch.usedCapacity = 0;

    // for each flow, find the shortest path from source to destination
    // update used_capacity for the path between them
    for (const auto &[src, dsts] : flows) {
      // Use dijkstra to find path given current demand from the start
      // switchbox; find the shortest paths to each other switchbox. Output is
      // in the predecessor map, which must then be processed to get individual
      // switchbox settings
      assert(src.sb && "nonexistent flow source");
      std::set<SwitchboxNode *> processed;
      std::map<SwitchboxNode *, SwitchboxNode *> preds =
          dijkstraShortestPaths(graph, src.sb);

      // trace the path of the flow backwards via predecessors
      // increment used_capacity for the associated channels
      SwitchSettings switchSettings;
      // set the input bundle for the source endpoint
      switchSettings[*src.sb].src = src.port;
      processed.insert(src.sb);
      for (const PathEndPointNode &endPoint : dsts) {
        SwitchboxNode *curr = endPoint.sb;
        assert(curr && "endpoint has no source switchbox");
        // set the output bundle for this destination endpoint
        switchSettings[*curr].dsts.insert(endPoint.port);

        // trace backwards until a vertex already processed is reached
        while (!processed.count(curr)) {
          // find the edge from the pred to curr by searching incident edges
          SmallVector<ChannelEdge *, 10> channels;
          graph.findIncomingEdgesToNode(*curr, channels);
          auto *matchingCh = std::find_if(
              channels.begin(), channels.end(),
              [&](ChannelEdge *ch) { return ch->src == *preds[curr]; });
          assert(matchingCh != channels.end() && "couldn't find ch");
          // incoming edge
          ChannelEdge *ch = *matchingCh;

          // don't use fixed channels
          while (ch->fixedCapacity.count(ch->usedCapacity)) ch->usedCapacity++;

          // add the entrance port for this Switchbox
          switchSettings[*curr].src = {getConnectingBundle(ch->bundle),
                                       ch->usedCapacity};
          // add the current Switchbox to the map of the predecessor
          switchSettings[*preds[curr]].dsts.insert(
              {ch->bundle, ch->usedCapacity});

          ch->usedCapacity++;
          // if at capacity, bump demand to discourage using this Channel
          if (ch->usedCapacity >= ch->maxCapacity) {
            LLVM_DEBUG(llvm::dbgs() << "ch over capacity: " << ch << "\n");
            // this means the order matters!
            ch->demand *= DEMAND_COEFF;
          }

          processed.insert(curr);
          curr = preds[curr];
        }
      }
      // add this flow to the proposed solution
      routingSolution[src] = switchSettings;
    }
  } while (!isLegal());  // continue iterations until a legal routing is found

  return routingSolution;
}
// allocates channels between switchboxes ( but does not assign them)
// instantiates shim-muxes AND allocates channels ( no need to rip these up in )
struct ConvertFlowsToInterconnect : OpConversionPattern<FlowOp> {
  using OpConversionPattern::OpConversionPattern;
  DeviceOp &device;
  DynamicTileAnalysis &analyzer;
  ConvertFlowsToInterconnect(MLIRContext *context, DeviceOp &d,
                             DynamicTileAnalysis &a, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), device(d), analyzer(a) {}

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
    TileLoc srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
    auto srcBundle = flowOp.getSourceBundle();
    auto srcChannel = flowOp.getSourceChannel();
    Port srcPort = {srcBundle, srcChannel};

#ifndef NDEBUG
    auto dstTile = cast<TileOp>(flowOp.getDest().getDefiningOp());
    TileLoc dstCoords = {dstTile.colIndex(), dstTile.rowIndex()};
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
    Switchbox srcSB = {srcCoords.col, srcCoords.row};
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

        LLVM_DEBUG(llvm::dbgs() << curr << ": " << setting << " | " << "\n");
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "\n\t\tFinished adding ConnectOps to implement flowOp.\n");
      analyzer.processedFlows[srcPoint] = true;
    } else
      LLVM_DEBUG(llvm::dbgs() << "Flow already processed!\n");

    rewriter.eraseOp(Op);
  }
};
}  // namespace

namespace mlir::iree_compiler::AMDAIE {
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
                           WireBundle problemBundle, int problemChan,
                           int emptyChan);

  ConnectOp replaceConnectOpWithNewDest(mlir::OpBuilder builder,
                                        ConnectOp connect, WireBundle newBundle,
                                        int newChannel);
  ConnectOp replaceConnectOpWithNewSource(mlir::OpBuilder builder,
                                          ConnectOp connect,
                                          WireBundle newBundle, int newChannel);

  SwitchboxOp getSwitchbox(DeviceOp &d, int col, int row);
};

void AMDAIEPathfinderPass::runOnOperation() {
  // create analysis pass with routing graph for entire device
  LLVM_DEBUG(llvm::dbgs() << "---Begin AMDAIEPathfinderPass---\n");

  DeviceOp d = getOperation();
  if (failed(analyzer.runAnalysis(d))) return signalPassFailure();
  OpBuilder builder = OpBuilder::atBlockEnd(d.getBody());

  // Apply rewrite rule to switchboxes to add assignments to every 'connect'
  // operation inside
  ConversionTarget target(getContext());
  target.addLegalOp<TileOp>();
  target.addLegalOp<ConnectOp>();
  target.addLegalOp<SwitchboxOp>();
  target.addLegalOp<ShimMuxOp>();
  target.addLegalOp<EndOp>();

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
              connect.getDestChannel())) {
        problemConnects.push_back(connect);
      }
    }
  });

  for (auto connect : problemConnects) {
    auto swBox = connect->getParentOfType<SwitchboxOp>();
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
  if (problemConnect.getSourceBundle() == WireBundle::North) {
    problemNorthChannel = problemConnect.getSourceChannel();
  } else if (problemConnect.getDestBundle() == WireBundle::North) {
    problemNorthChannel = problemConnect.getDestChannel();
  } else
    return false;  // Problem is not about n-s routing
  int problemSouthChannel;
  if (problemConnect.getSourceBundle() == WireBundle::South) {
    problemSouthChannel = problemConnect.getSourceChannel();
  } else if (problemConnect.getDestBundle() == WireBundle::South) {
    problemSouthChannel = problemConnect.getDestChannel();
  } else
    return false;  // Problem is not about n-s routing

  // Attempt to reroute northern neighbouring sw
  if (reconnectConnectOps(builder, northSwOp, problemConnect, true,
                          WireBundle::South, problemNorthChannel,
                          problemSouthChannel))
    return true;
  if (reconnectConnectOps(builder, northSwOp, problemConnect, false,
                          WireBundle::South, problemNorthChannel,
                          problemSouthChannel))
    return true;
  // Otherwise, attempt to reroute southern neighbouring sw
  if (reconnectConnectOps(builder, southSwOp, problemConnect, true,
                          WireBundle::North, problemSouthChannel,
                          problemNorthChannel))
    return true;
  if (reconnectConnectOps(builder, southSwOp, problemConnect, false,
                          WireBundle::North, problemSouthChannel,
                          problemNorthChannel))
    return true;
  return false;
}

bool AMDAIEPathfinderPass::reconnectConnectOps(const OpBuilder &builder,
                                               SwitchboxOp sw,
                                               ConnectOp problemConnect,
                                               bool isIncomingToSW,
                                               WireBundle problemBundle,
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
          connect.getDestChannel() == emptyChan) {
        hasEmptyChannelSlot = false;
      }
    }
  } else {
    for (ConnectOp connect : sw.getOps<ConnectOp>()) {
      if (connect.getSourceBundle() == problemBundle &&
          connect.getSourceChannel() == problemChan) {
        candidate = connect;
        foundCandidateForFixup = true;
      }
      if (connect.getSourceBundle() == problemBundle &&
          connect.getSourceChannel() == emptyChan) {
        hasEmptyChannelSlot = false;
      }
    }
  }
  if (foundCandidateForFixup && hasEmptyChannelSlot) {
    WireBundle problemBundleOpposite = problemBundle == WireBundle::North
                                           ? WireBundle::South
                                           : WireBundle::North;
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
    OpBuilder builder, ConnectOp connect, WireBundle newBundle,
    int newChannel) {
  builder.setInsertionPoint(connect);
  auto newOp = builder.create<ConnectOp>(
      builder.getUnknownLoc(), connect.getSourceBundle(),
      connect.getSourceChannel(), newBundle, newChannel);
  connect.erase();
  return newOp;
}

ConnectOp AMDAIEPathfinderPass::replaceConnectOpWithNewSource(
    OpBuilder builder, ConnectOp connect, WireBundle newBundle,
    int newChannel) {
  builder.setInsertionPoint(connect);
  auto newOp = builder.create<ConnectOp>(builder.getUnknownLoc(), newBundle,
                                         newChannel, connect.getDestBundle(),
                                         connect.getDestChannel());
  connect.erase();
  return newOp;
}

SwitchboxOp AMDAIEPathfinderPass::getSwitchbox(DeviceOp &d, int col, int row) {
  SwitchboxOp output = nullptr;
  d.walk([&](SwitchboxOp swBox) {
    if (swBox.colIndex() == col && swBox.rowIndex() == row) {
      output = swBox;
    }
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
