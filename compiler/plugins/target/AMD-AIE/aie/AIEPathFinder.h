//===- AIEPathfinder.h ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_PATHFINDER_H
#define AIE_PATHFINDER_H

#include <algorithm>
#include <iostream>
#include <list>
#include <set>

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "llvm/ADT/DirectedGraph.h"
#include "llvm/ADT/GraphTraits.h"

namespace llvm {
template <>
struct DenseMapInfo<TileLoc> {
  using FirstInfo = DenseMapInfo<int>;
  using SecondInfo = DenseMapInfo<int>;

  static TileLoc getEmptyKey() {
    return {FirstInfo::getEmptyKey(), SecondInfo::getEmptyKey()};
  }

  static TileLoc getTombstoneKey() {
    return {FirstInfo::getTombstoneKey(), SecondInfo::getTombstoneKey()};
  }

  static unsigned getHashValue(const TileLoc &t) {
    return detail::combineHashValue(FirstInfo::getHashValue(t.col),
                                    SecondInfo::getHashValue(t.row));
  }

  static bool isEqual(const TileLoc &lhs, const TileLoc &rhs) {
    return lhs == rhs;
  }
};
}  // namespace llvm

template <>
struct std::hash<TileLoc> {
  std::size_t operator()(const TileLoc &s) const noexcept {
    std::size_t h1 = std::hash<int>{}(s.col);
    std::size_t h2 = std::hash<int>{}(s.row);
    return h1 ^ (h2 << 1);
  }
};

using Switchbox = struct Switchbox : TileLoc {
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

using Channel = struct Channel {
  Channel(Switchbox &src, Switchbox &target, StrmSwPortType bundle,
          int maxCapacity)
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
  StrmSwPortType bundle;
  int maxCapacity = 0;   // maximum number of routing resources
  double demand = 0.0;   // indicates how many flows want to use this Channel
  int usedCapacity = 0;  // how many flows are actually using this Channel
  std::set<int> fixedCapacity;  // channels not available to the algorithm
  int overCapacityCount = 0;    // history of Channel being over capacity
};

#define GENERATE_TO_STRING(TYPE_WITH_INSERTION_OP)                \
  friend std::string to_string(const TYPE_WITH_INSERTION_OP &s) { \
    std::ostringstream ss;                                        \
    ss << s;                                                      \
    return ss.str();                                              \
  }

typedef struct Port {
  StrmSwPortType bundle;
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
      case StrmSwPortType::CORE:
        os << "Core";
        break;
      case StrmSwPortType::DMA:
        os << "DMA";
        break;
      case StrmSwPortType::NORTH:
        os << "N";
        break;
      case StrmSwPortType::EAST:
        os << "E";
        break;
      case StrmSwPortType::SOUTH:
        os << "S";
        break;
      case StrmSwPortType::WEST:
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

} Port;

template <>
struct std::less<Port> {
  bool operator()(const Port &a, const Port &b) const {
    return a.bundle == b.bundle ? a.channel < b.channel : a.bundle < b.bundle;
  }
};

struct SwitchboxNode;
struct ChannelEdge;
using SwitchboxNodeBase = llvm::DGNode<SwitchboxNode, ChannelEdge>;
using ChannelEdgeBase = llvm::DGEdge<SwitchboxNode, ChannelEdge>;
using SwitchboxGraphBase = llvm::DirectedGraph<SwitchboxNode, ChannelEdge>;

using SwitchboxNode = struct SwitchboxNode : SwitchboxNodeBase, Switchbox {
  using Switchbox::Switchbox;
  SwitchboxNode(int col, int row, int id) : Switchbox{col, row}, id{id} {}
  int id;
};

using ChannelEdge = struct ChannelEdge : ChannelEdgeBase, Channel {
  using Channel::Channel;

  explicit ChannelEdge(SwitchboxNode &target) = delete;
  ChannelEdge(SwitchboxNode &src, SwitchboxNode &target, StrmSwPortType bundle,
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

// A SwitchSetting defines the required settings for a Switchbox for a flow
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

using SwitchSettings = std::map<Switchbox, SwitchSetting>;

// A Flow defines source and destination vertices
// Only one source, but any number of destinations (fanout)
using PathEndPoint = struct PathEndPoint {
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

// A Flow defines source and destination vertices
// Only one source, but any number of destinations (fanout)
using PathEndPointNode = struct PathEndPointNode : PathEndPoint {
  PathEndPointNode(SwitchboxNode *sb, Port port)
      : PathEndPoint{*sb, port}, sb(sb) {}
  SwitchboxNode *sb;
};

using FlowNode = struct FlowNode {
  PathEndPointNode src;
  std::vector<PathEndPointNode> dsts;
};

class Router {
 public:
  Router() = default;
  // This has to go first so it can serve as a key function.
  // https://lld.llvm.org/missingkeyfunction
  virtual ~Router() = default;
  virtual void initialize(int maxCol, int maxRow,
                          AMDAIENPUDeviceModel targetModel) = 0;
  virtual void addFlow(TileLoc srcCoords, Port srcPort, TileLoc dstCoords,
                       Port dstPort) = 0;
  virtual bool addFixedConnection(xilinx::AIE::ConnectOp connectOp) = 0;
  virtual std::optional<std::map<PathEndPoint, SwitchSettings>> findPaths(
      int maxIterations) = 0;
  virtual Switchbox *getSwitchbox(TileLoc coords) = 0;
};

class Pathfinder : public Router {
 public:
  Pathfinder() = default;
  void initialize(int maxCol, int maxRow,
                  AMDAIENPUDeviceModel targetModel) override;
  void addFlow(TileLoc srcCoords, Port srcPort, TileLoc dstCoords,
               Port dstPort) override;
  bool addFixedConnection(xilinx::AIE::ConnectOp connectOp) override;
  std::optional<std::map<PathEndPoint, SwitchSettings>> findPaths(
      int maxIterations) override;

  Switchbox *getSwitchbox(TileLoc coords) override {
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
  std::shared_ptr<Router> pathfinder;
  std::map<PathEndPoint, SwitchSettings> flowSolutions;
  std::map<PathEndPoint, bool> processedFlows;

  llvm::DenseMap<TileLoc, xilinx::AIE::TileOp> coordToTile;
  llvm::DenseMap<TileLoc, xilinx::AIE::SwitchboxOp> coordToSwitchbox;
  llvm::DenseMap<TileLoc, xilinx::AIE::ShimMuxOp> coordToShimMux;
  llvm::DenseMap<int, xilinx::AIE::PLIOOp> coordToPLIO;

  const int maxIterations = 1000;  // how long until declared unroutable

  DynamicTileAnalysis() : pathfinder(std::make_shared<Pathfinder>()) {}
  DynamicTileAnalysis(std::shared_ptr<Router> p) : pathfinder(std::move(p)) {}

  mlir::LogicalResult runAnalysis(xilinx::AIE::DeviceOp &device);

  int getMaxCol() const { return maxCol; }
  int getMaxRow() const { return maxRow; }

  xilinx::AIE::TileOp getTile(mlir::OpBuilder &builder, int col, int row);

  xilinx::AIE::SwitchboxOp getSwitchbox(mlir::OpBuilder &builder, int col,
                                        int row);

  xilinx::AIE::ShimMuxOp getShimMux(mlir::OpBuilder &builder, int col);
};

// For some mysterious reason, the only way to get the priorityQueue(cmp)
// comparison in dijkstraShortestPaths to work correctly is to define
// this template specialization for the pointers. Overloading operator
// will not work. Furthermore, if  you try to move this into AIEPathFinder.cpp
// you'll get a compile error about
// "specialization of ‘std::less<xilinx::AIE::Switchbox*>’ after instantiation"
// because one of the graph traits below is doing the comparison internally
// (try moving this below the llvm namespace...)
namespace std {
template <>
struct less<Switchbox *> {
  bool operator()(const Switchbox *a, const Switchbox *b) const {
    return *a < *b;
  }
};
}  // namespace std

namespace llvm {

template <>
struct GraphTraits<SwitchboxNode *> {
  using NodeRef = SwitchboxNode *;

  static SwitchboxNode *SwitchboxGraphGetSwitchbox(
      DGEdge<SwitchboxNode, ChannelEdge> *P) {
    return &P->getTargetNode();
  }

  // Provide a mapped iterator so that the GraphTrait-based implementations can
  // find the target nodes without having to explicitly go through the edges.
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

template <>
struct DenseMapInfo<StrmSwPortType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline StrmSwPortType getEmptyKey() {
    return static_cast<StrmSwPortType>(StorageInfo::getEmptyKey());
  }

  static inline StrmSwPortType getTombstoneKey() {
    return static_cast<StrmSwPortType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const StrmSwPortType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const StrmSwPortType &lhs, const StrmSwPortType &rhs) {
    return lhs == rhs;
  }
};

template <>
struct DenseMapInfo<Port> {
  using FirstInfo = DenseMapInfo<StrmSwPortType>;
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

}  // namespace llvm

template <>
struct std::hash<Port> {
  std::size_t operator()(const Port &p) const noexcept {
    std::size_t h1 = std::hash<StrmSwPortType>{}(p.bundle);
    std::size_t h2 = std::hash<int>{}(p.channel);
    return h1 ^ h2 << 1;
  }
};

template <>
struct std::hash<Switchbox> {
  std::size_t operator()(const Switchbox &s) const noexcept {
    return std::hash<TileLoc>{}(s);
  }
};

template <>
struct std::hash<PathEndPoint> {
  std::size_t operator()(const PathEndPoint &pe) const noexcept {
    std::size_t h1 = std::hash<Port>{}(pe.port);
    std::size_t h2 = std::hash<Switchbox>{}(pe.sb);
    return h1 ^ (h2 << 1);
  }
};

#endif
