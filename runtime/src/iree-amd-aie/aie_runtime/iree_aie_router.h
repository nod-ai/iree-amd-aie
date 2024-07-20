// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions. See
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: # Apache-2.0 WITH LLVM-exception

#ifndef IREE_AIE_ROUTER_H
#define IREE_AIE_ROUTER_H

#include <list>
#include <map>
#include <set>

#include "iree_aie_runtime.h"
#include "llvm/ADT/DenseMapInfo.h"

namespace mlir::iree_compiler::AMDAIE {
struct Port {
  StrmSwPortType bundle;
  int channel;

  bool operator==(const Port &rhs) const {
    return std::tie(bundle, channel) == std::tie(rhs.bundle, rhs.channel);
  }

  bool operator!=(const Port &rhs) const { return !(*this == rhs); }

  bool operator<(const Port &rhs) const {
    return std::tie(bundle, channel) < std::tie(rhs.bundle, rhs.channel);
  }
};
}  // namespace mlir::iree_compiler::AMDAIE

template <>
struct std::less<mlir::iree_compiler::AMDAIE::Port> {
  bool operator()(const mlir::iree_compiler::AMDAIE::Port &a,
                  const mlir::iree_compiler::AMDAIE::Port &b) const {
    return a.bundle == b.bundle ? a.channel < b.channel : a.bundle < b.bundle;
  }
};

namespace mlir::iree_compiler::AMDAIE {

struct Connect {
  Port src;
  Port dst;

  bool operator==(const Connect &rhs) const {
    return std::tie(src, dst) == std::tie(rhs.src, rhs.dst);
  }
};

enum class Connectivity : int8_t { INVALID = -1, AVAILABLE = 0, OCCUPIED = 1 };
struct SwitchboxNode {
  SwitchboxNode(int col, int row, int id, const AMDAIEDeviceModel &targetModel)
      : col{col}, row{row}, id{id} {
    std::vector<StrmSwPortType> bundles = {
        StrmSwPortType::CORE,  StrmSwPortType::DMA,  StrmSwPortType::FIFO,
        StrmSwPortType::SOUTH, StrmSwPortType::WEST, StrmSwPortType::NORTH,
        StrmSwPortType::EAST,  StrmSwPortType::PLIO, StrmSwPortType::NOC,
        StrmSwPortType::TRACE, StrmSwPortType::CTRL};

    for (StrmSwPortType bundle : bundles) {
      int maxCapacity =
          targetModel.getNumSourceSwitchboxConnections(col, row, bundle);
      if (targetModel.isShimNOCorPLTile(col, row) && maxCapacity == 0) {
        // wordaround for shimMux, todo: integrate shimMux into routable grid
        maxCapacity =
            targetModel.getNumSourceShimMuxConnections(col, row, bundle);
      }

      for (int channel = 0; channel < maxCapacity; channel++) {
        Port inPort = {bundle, channel};
        inPortToId[inPort] = inPortId;
        inPortId++;
      }

      maxCapacity =
          targetModel.getNumDestSwitchboxConnections(col, row, bundle);
      if (targetModel.isShimNOCorPLTile(col, row) && maxCapacity == 0) {
        // wordaround for shimMux, todo: integrate shimMux into routable grid
        maxCapacity =
            targetModel.getNumDestShimMuxConnections(col, row, bundle);
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
        if (!targetModel.isLegalTileConnection(col, row, inPort.bundle,
                                               inPort.channel, outPort.bundle,
                                               outPort.channel))
          connectionMatrix[inId][outId] = Connectivity::INVALID;

        if (targetModel.isShimNOCorPLTile(col, row)) {
          // wordaround for shimMux, todo: integrate shimMux into routable grid
          auto isBundleInList = [](StrmSwPortType bundle,
                                   std::vector<StrmSwPortType> bundles) {
            return std::find(bundles.begin(), bundles.end(), bundle) !=
                   bundles.end();
          };
          std::vector<StrmSwPortType> bundles = {
              StrmSwPortType::DMA, StrmSwPortType::NOC, StrmSwPortType::PLIO};
          if (isBundleInList(inPort.bundle, bundles) ||
              isBundleInList(outPort.bundle, bundles))
            connectionMatrix[inId][outId] = Connectivity::AVAILABLE;
        }
      }
    }
  }

  // given a outPort, find availble input channel
  std::vector<int> findAvailableChannelIn(StrmSwPortType inBundle, Port outPort,
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

struct ChannelEdge {
  ChannelEdge(SwitchboxNode *src, SwitchboxNode *target)
      : src(src), target(target) {
    // get bundle from src to target coordinates
    if (src->col == target->col) {
      if (src->row > target->row)
        bundle = StrmSwPortType::SOUTH;
      else
        bundle = StrmSwPortType::NORTH;
    } else {
      if (src->col > target->col)
        bundle = StrmSwPortType::WEST;
      else
        bundle = StrmSwPortType::EAST;
    }

    // maximum number of routing resources
    maxCapacity = 0;
    for (auto &[outPort, _] : src->outPortToId) {
      if (outPort.bundle == bundle) {
        maxCapacity++;
      }
    }
  }

  SwitchboxNode *src;
  SwitchboxNode *target;

  int maxCapacity;
  StrmSwPortType bundle;
};

// A SwitchSetting defines the required settings for a SwitchboxNode for a flow
// SwitchSetting.src is the incoming signal
// SwitchSetting.dsts is the fanout
struct SwitchSetting {
  SwitchSetting() = default;
  SwitchSetting(Port src) : src(src) {}
  SwitchSetting(Port src, std::set<Port> dsts)
      : src(src), dsts(std::move(dsts)) {}
  Port src;
  std::set<Port> dsts;
  bool operator<(const SwitchSetting &rhs) const { return src < rhs.src; }
};

using SwitchSettings = std::map<SwitchboxNode, SwitchSetting>;

// A Flow defines source and destination vertices
// Only one source, but any number of destinations (fanout)
struct PathEndPoint {
  SwitchboxNode sb;
  Port port;
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
struct PathEndPointNode : PathEndPoint {
  PathEndPointNode(SwitchboxNode *sb, Port port)
      : PathEndPoint{*sb, port}, sb(sb) {}
  SwitchboxNode *sb;
};

struct FlowNode {
  bool isPacketFlow;
  PathEndPointNode src;
  std::vector<PathEndPointNode> dsts;
};

class Pathfinder {
 public:
  Pathfinder() = default;
  void initialize(int maxCol, int maxRow, const AMDAIEDeviceModel &targetModel);
  void addFlow(TileLoc srcCoords, Port srcPort, TileLoc dstCoords, Port dstPort,
               bool isPacketFlow);
  bool addFixedConnection(
      int col, int row,
      const std::vector<std::tuple<StrmSwPortType, int, StrmSwPortType, int>>
          &connects);
  std::optional<std::map<PathEndPoint, SwitchSettings>> findPaths(
      int maxIterations);

  std::map<SwitchboxNode *, SwitchboxNode *> dijkstraShortestPaths(
      SwitchboxNode *src);

  SwitchboxNode getSwitchboxNode(TileLoc coords) { return grid.at(coords); }

 private:
  // Flows to be routed
  std::vector<FlowNode> flows;

  // Grid of switchboxes available
  std::map<TileLoc, SwitchboxNode> grid;

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

#define TO_STRINGS(_) \
  _(Connect)          \
  _(Connectivity)     \
  _(PathEndPoint)     \
  _(Port)             \
  _(SwitchSetting)    \
  _(SwitchboxNode)

TO_STRINGS(TO_STRING_DECL)
#undef TO_STRINGS

#define BOTH_OSTREAM_OPS_FORALL_ROUTER_TYPES(OSTREAM_OP_, _) \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::Connect)       \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::Connectivity)  \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::PathEndPoint)  \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::Port)          \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::SwitchSetting) \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::SwitchboxNode)

BOTH_OSTREAM_OPS_FORALL_ROUTER_TYPES(OSTREAM_OP_DECL, BOTH_OSTREAM_OP)

}  // namespace mlir::iree_compiler::AMDAIE

template <>
struct std::hash<mlir::iree_compiler::AMDAIE::Port> {
  std::size_t operator()(
      const mlir::iree_compiler::AMDAIE::Port &p) const noexcept {
    std::size_t h1 =
        std::hash<mlir::iree_compiler::AMDAIE::StrmSwPortType>{}(p.bundle);
    std::size_t h2 = std::hash<int>{}(p.channel);
    return h1 ^ h2 << 1;
  }
};

namespace llvm {
template <>
struct DenseMapInfo<mlir::iree_compiler::AMDAIE::Port> {
  using FirstInfo = DenseMapInfo<mlir::iree_compiler::AMDAIE::StrmSwPortType>;
  using SecondInfo = DenseMapInfo<int>;

  static mlir::iree_compiler::AMDAIE::Port getEmptyKey() {
    return {FirstInfo::getEmptyKey(), SecondInfo::getEmptyKey()};
  }

  static mlir::iree_compiler::AMDAIE::Port getTombstoneKey() {
    return {FirstInfo::getTombstoneKey(), SecondInfo::getTombstoneKey()};
  }

  static unsigned getHashValue(const mlir::iree_compiler::AMDAIE::Port &d) {
    return detail::combineHashValue(FirstInfo::getHashValue(d.bundle),
                                    SecondInfo::getHashValue(d.channel));
  }

  static bool isEqual(const mlir::iree_compiler::AMDAIE::Port &lhs,
                      const mlir::iree_compiler::AMDAIE::Port &rhs) {
    return lhs == rhs;
  }
};

}  // namespace llvm

template <>
struct std::hash<mlir::iree_compiler::AMDAIE::SwitchboxNode> {
  std::size_t operator()(
      const mlir::iree_compiler::AMDAIE::SwitchboxNode &s) const noexcept {
    return std::hash<mlir::iree_compiler::AMDAIE::TileLoc>{}({s.col, s.row});
  }
};

template <>
struct std::hash<mlir::iree_compiler::AMDAIE::PathEndPoint> {
  std::size_t operator()(
      const mlir::iree_compiler::AMDAIE::PathEndPoint &pe) const noexcept {
    std::size_t h1 = std::hash<mlir::iree_compiler::AMDAIE::Port>{}(pe.port);
    std::size_t h2 =
        std::hash<mlir::iree_compiler::AMDAIE::SwitchboxNode>{}(pe.sb);
    return h1 ^ (h2 << 1);
  }
};

#endif  // IREE_AIE_ROUTER_H
