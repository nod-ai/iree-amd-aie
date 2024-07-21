// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions. See
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: # Apache-2.0 WITH LLVM-exception

#include "iree_aie_router.h"

#include <tuple>

#include "d_ary_heap.h"
#include "iree_aie_runtime.h"
#include "llvm/ADT/StringExtras.h"

#define DEBUG_TYPE "iree-aie-runtime-router"

#define OVER_CAPACITY_COEFF 0.02
#define USED_CAPACITY_COEFF 0.02
#define DEMAND_COEFF 1.1
static constexpr double INF = std::numeric_limits<double>::max();

namespace mlir::iree_compiler::AMDAIE {
std::string to_string(const SwitchSetting &setting) {
  return "SwitchSetting(" + to_string(setting.src) + " -> " + "{" +
         llvm::join(
             llvm::map_range(setting.dsts,
                             [](const Port &port) { return to_string(port); }),
             ", ") +
         "})";
}

std::string to_string(const SwitchSettings &settings) {
  return "SwitchSettings(" +
         llvm::join(llvm::map_range(
                        llvm::make_range(settings.begin(), settings.end()),
                        [](const llvm::detail::DenseMapPair<Switchbox,
                                                            SwitchSetting> &p) {
                          return to_string(p.getFirst()) + ": " +
                                 to_string(p.getSecond());
                        }),
                    ", ") +
         ")";
}

STRINGIFY_2TUPLE_STRUCT(Port, bundle, channel)
STRINGIFY_2TUPLE_STRUCT(Connect, src, dst)
STRINGIFY_2TUPLE_STRUCT(Switchbox, col, row)
STRINGIFY_2TUPLE_STRUCT(PathEndPoint, sb, port)

BOTH_OSTREAM_OPS_FORALL_ROUTER_TYPES(OSTREAM_OP_DEFN, BOTH_OSTREAM_OP)

struct SwitchboxNode : Switchbox {
  enum class Connectivity : int8_t {
    INVALID = -1,
    AVAILABLE = 0,
    OCCUPIED = 1
  };
  int id;
  int inPortId = 0, outPortId = 0;
  std::map<Port, int> inPortToId, outPortToId;
  std::vector<std::vector<Connectivity>> connectionMatrix;
  // input ports with incoming packet-switched streams
  std::map<Port, int> inPortPktCount;
  // up to 32 packet-switched stram through a port
  const int maxPktStream = 32;

  SwitchboxNode(int col, int row, int id, const AMDAIEDeviceModel &targetModel);
  std::vector<int> findAvailableChannelIn(StrmSwPortType inBundle, Port outPort,
                                          bool isPkt);
  bool allocate(Port inPort, Port outPort, bool isPkt);
  void clearAllocation();

  // TODO(max): do i really need to write this all out by hand?
  bool operator==(const SwitchboxNode &rhs) const {
    return col == rhs.col && row == rhs.row && id == rhs.id &&
           inPortId == rhs.inPortId && outPortId == rhs.outPortId &&
           inPortToId == rhs.inPortToId && outPortToId == rhs.outPortToId &&
           connectionMatrix == rhs.connectionMatrix &&
           inPortPktCount == rhs.inPortPktCount &&
           maxPktStream == rhs.maxPktStream;
  }
};

std::string to_string(const SwitchboxNode::Connectivity &value) {
  switch (value) {
    STRINGIFY_ENUM_CASE(SwitchboxNode::Connectivity::INVALID)
    STRINGIFY_ENUM_CASE(SwitchboxNode::Connectivity::AVAILABLE)
    STRINGIFY_ENUM_CASE(SwitchboxNode::Connectivity::OCCUPIED)
  }

  llvm::report_fatal_error("Unhandled Connectivity case");
}
STRINGIFY_3TUPLE_STRUCT(SwitchboxNode, col, row, id)

struct ChannelEdge {
  SwitchboxNode *src;
  SwitchboxNode *target;
  int maxCapacity;
  StrmSwPortType bundle;

  ChannelEdge(SwitchboxNode *src, SwitchboxNode *target);
};

// A node holds a pointer
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

SwitchboxNode::SwitchboxNode(int col, int row, int id,
                             const AMDAIEDeviceModel &targetModel)
    : Switchbox(col, row), id{id} {
  std::vector<StrmSwPortType> allBundles = {
      StrmSwPortType::CORE,  StrmSwPortType::DMA,  StrmSwPortType::FIFO,
      StrmSwPortType::SOUTH, StrmSwPortType::WEST, StrmSwPortType::NORTH,
      StrmSwPortType::EAST,  StrmSwPortType::PLIO, StrmSwPortType::NOC,
      StrmSwPortType::TRACE, StrmSwPortType::CTRL};
  for (StrmSwPortType bundle : allBundles) {
    int maxCapacity =
        targetModel.getNumSourceSwitchboxConnections(col, row, bundle);
    if (targetModel.isShimNOCorPLTile(col, row) && maxCapacity == 0) {
      // wordaround for shimMux, todo: integrate shimMux into routable grid
      maxCapacity =
          targetModel.getNumSourceShimMuxConnections(col, row, bundle);
    }

    for (int channel = 0; channel < maxCapacity; channel++) {
      inPortToId[{bundle, channel}] = inPortId;
      inPortId++;
    }

    maxCapacity = targetModel.getNumDestSwitchboxConnections(col, row, bundle);
    // wordaround for shimMux, todo: integrate shimMux into routable grid
    if (targetModel.isShimNOCorPLTile(col, row) && maxCapacity == 0) {
      maxCapacity = targetModel.getNumDestShimMuxConnections(col, row, bundle);
    }

    for (int channel = 0; channel < maxCapacity; channel++) {
      outPortToId[{bundle, channel}] = outPortId;
      outPortId++;
    }
  }

  std::vector<StrmSwPortType> shimBundles = {
      StrmSwPortType::DMA, StrmSwPortType::NOC, StrmSwPortType::PLIO};
  auto isBundleInList = [](StrmSwPortType bundle,
                           std::vector<StrmSwPortType> bundles) {
    return std::find(bundles.begin(), bundles.end(), bundle) != bundles.end();
  };
  // illegal connection
  connectionMatrix.resize(
      inPortId, std::vector<Connectivity>(outPortId, Connectivity::AVAILABLE));
  for (const auto &[inPort, inId] : inPortToId) {
    for (const auto &[outPort, outId] : outPortToId) {
      if (!targetModel.isLegalTileConnection(col, row, inPort.bundle,
                                             inPort.channel, outPort.bundle,
                                             outPort.channel)) {
        // TODO(max): can't put a continue here because these two conditionals
        // aren't mutually exclusive
        connectionMatrix[inId][outId] = Connectivity::INVALID;
      }

      if (targetModel.isShimNOCorPLTile(col, row)) {
        // wordaround for shimMux, todo: integrate shimMux into routable grid
        if (isBundleInList(inPort.bundle, shimBundles) ||
            isBundleInList(outPort.bundle, shimBundles)) {
          connectionMatrix[inId][outId] = Connectivity::AVAILABLE;
        }
      }
    }
  }
}

// given a outPort, find availble input channel
std::vector<int> SwitchboxNode::findAvailableChannelIn(StrmSwPortType inBundle,
                                                       Port outPort,
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
            for (const auto &[_outPort, _outPortId] : outPortToId) {
              if (connectionMatrix[inId][_outPortId] ==
                  Connectivity::OCCUPIED) {
                // occupied by others as circuit-switched
                available = false;
                break;
              }
            }
          } else if (inPortPktCount[inPort] >= maxPktStream) {
            // occupied by others as packet-switched but exceed max packet
            // stream capacity
            available = false;
          }
          if (available) availableChannels.push_back(inPort.channel);
        }
      }
    } else {
      for (const auto &[inPort, inId] : inPortToId) {
        if (inPort.bundle == inBundle &&
            connectionMatrix[inId][outId] == Connectivity::AVAILABLE) {
          bool available = true;
          for (const auto &[_outPort, _outPortId] : outPortToId) {
            if (connectionMatrix[inId][_outPortId] == Connectivity::OCCUPIED) {
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

bool SwitchboxNode::allocate(Port inPort, Port outPort, bool isPkt) {
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
      for (const auto &[_outPort, _outPortId] : outPortToId) {
        // occupied by others as circuit-switched, allocation fail!
        if (connectionMatrix[inId][_outPortId] == Connectivity::OCCUPIED)
          return false;
      }
      // empty channel, allocation succeed!
      inPortPktCount[inPort] = 1;
      connectionMatrix[inId][outId] = Connectivity::OCCUPIED;
      return true;
    } else if (inPortPktCount[inPort] >= maxPktStream) {
      // occupied by others as packet-switched but exceed max packet stream
      // capacity, allocation fail!
      return false;
    } else {
      // valid packet-switched, allocation succeed!
      inPortPktCount[inPort]++;
      return true;
    }
  } else if (connectionMatrix[inId][outId] == Connectivity::AVAILABLE) {
    // empty channel, allocation succeed!
    connectionMatrix[inId][outId] = Connectivity::OCCUPIED;
    return true;
  } else {
    // occupied by others, allocation fail!
    return false;
  }
}

void SwitchboxNode::clearAllocation() {
  for (int inId = 0; inId < inPortId; inId++) {
    for (int outId = 0; outId < outPortId; outId++) {
      if (connectionMatrix[inId][outId] != Connectivity::INVALID) {
        connectionMatrix[inId][outId] = Connectivity::AVAILABLE;
      }
    }
  }
  inPortPktCount.clear();
}

ChannelEdge::ChannelEdge(SwitchboxNode *src, SwitchboxNode *target)
    : src(src), target(target) {
  // get bundle from src to target coordinates
  if (src->col == target->col) {
    if (src->row > target->row) {
      bundle = StrmSwPortType::SOUTH;
    } else {
      bundle = StrmSwPortType::NORTH;
    }
  } else {
    if (src->col > target->col) {
      bundle = StrmSwPortType::WEST;
    } else {
      bundle = StrmSwPortType::EAST;
    }
  }

  // maximum number of routing resources
  maxCapacity = 0;
  for (auto &[outPort, _] : src->outPortToId) {
    if (outPort.bundle == bundle) maxCapacity++;
  }
}

struct RouterImpl {
  std::vector<FlowNode> flows;
  std::map<TileLoc, SwitchboxNode> grid;
  std::list<ChannelEdge> edges;
};

Router::Router() { impl = new RouterImpl(); }
Router::~Router() { delete impl; }

void Router::initialize(int maxCol, int maxRow,
                        const AMDAIEDeviceModel &targetModel) {
  // make grid of switchboxes
  int id = 0;
  for (int row = 0; row <= maxRow; row++) {
    for (int col = 0; col <= maxCol; col++) {
      impl->grid.insert(
          {{col, row}, SwitchboxNode{col, row, id++, targetModel}});
      SwitchboxNode &thisNode = impl->grid.at({col, row});
      if (row > 0) {  // if not in row 0 add channel to North/South
        SwitchboxNode &southernNeighbor = impl->grid.at({col, row - 1});
        // get the number of outgoing connections on the south side - outgoing
        // because these correspond to rhs of a connect op
        if (targetModel.getNumDestSwitchboxConnections(col, row,
                                                       StrmSwPortType::SOUTH)) {
          impl->edges.emplace_back(&thisNode, &southernNeighbor);
        }
        // get the number of incoming connections on the south side - incoming
        // because they correspond to connections on the southside that are then
        // routed using internal connect ops through the switchbox (i.e., lhs of
        // connect ops)
        if (targetModel.getNumSourceSwitchboxConnections(
                col, row, StrmSwPortType::SOUTH)) {
          impl->edges.emplace_back(&southernNeighbor, &thisNode);
        }
      }

      if (col > 0) {
        // if not in col 0 add channel to East/West
        SwitchboxNode &westernNeighbor = impl->grid.at({col - 1, row});
        if (targetModel.getNumDestSwitchboxConnections(col, row,
                                                       StrmSwPortType::WEST)) {
          impl->edges.emplace_back(&thisNode, &westernNeighbor);
        }
        if (targetModel.getNumSourceSwitchboxConnections(
                col, row, StrmSwPortType::WEST)) {
          impl->edges.emplace_back(&westernNeighbor, &thisNode);
        }
      }
    }
  }
}

// Add a flow from src to dst can have an arbitrary number of dst locations due
// to fanout.
void Router::addFlow(TileLoc srcCoords, Port srcPort, TileLoc dstCoords,
                     Port dstPort, bool isPacketFlow) {
  // check if a flow with this source already exists
  for (auto &[isPkt, src, dsts] : impl->flows) {
    SwitchboxNode *existingSrcPtr = src.sb;
    assert(existingSrcPtr && "nullptr flow source");
    if (Port existingPort = src.port; existingSrcPtr->col == srcCoords.col &&
                                      existingSrcPtr->row == srcCoords.row &&
                                      existingPort == srcPort) {
      // find the vertex corresponding to the destination
      SwitchboxNode *matchingDstSbPtr = &impl->grid.at(dstCoords);
      dsts.emplace_back(matchingDstSbPtr, dstPort);
      return;
    }
  }

  // If no existing flow was found with this source, create a new flow.
  SwitchboxNode *matchingSrcSbPtr = &impl->grid.at(srcCoords);
  SwitchboxNode *matchingDstSbPtr = &impl->grid.at(dstCoords);
  impl->flows.push_back(
      {isPacketFlow, PathEndPointNode{matchingSrcSbPtr, srcPort},
       std::vector<PathEndPointNode>{{matchingDstSbPtr, dstPort}}});
}

// Keep track of connections already used in the AIE; Pathfinder algorithm will
// avoid using these.
bool Router::addFixedConnection(
    int col, int row,
    const std::vector<std::tuple<StrmSwPortType, int, StrmSwPortType, int>>
        &connects) {
  SwitchboxNode &sb = impl->grid.at({col, row});
  std::set<int> invalidInId, invalidOutId;
  for (auto &[sourceBundle, sourceChannel, destBundle, destChannel] :
       connects) {
    Port srcPort = {sourceBundle, sourceChannel};
    Port destPort = {destBundle, destChannel};
    if (sb.inPortToId.count(srcPort) == 0 ||
        sb.outPortToId.count(destPort) == 0) {
      return false;
    }
    int inId = sb.inPortToId.at(srcPort);
    int outId = sb.outPortToId.at(destPort);
    if (sb.connectionMatrix[inId][outId] !=
        SwitchboxNode::Connectivity::AVAILABLE) {
      return false;
    }
    invalidInId.insert(inId);
    invalidOutId.insert(outId);
  }

  for (const auto &[inPort, inId] : sb.inPortToId) {
    for (const auto &[outPort, outId] : sb.outPortToId) {
      if (invalidInId.find(inId) != invalidInId.end() ||
          invalidOutId.find(outId) != invalidOutId.end()) {
        sb.connectionMatrix[inId][outId] = SwitchboxNode::Connectivity::INVALID;
      }
    }
  }
  return true;
}

std::map<SwitchboxNode *, SwitchboxNode *> dijkstraShortestPaths(
    SwitchboxNode *src, std::map<TileLoc, SwitchboxNode> &grid,
    std::list<ChannelEdge> &edges,
    const std::map<ChannelEdge *, double> &demand) {
  // Use std::map instead of DenseMap because DenseMap doesn't let you overwrite
  // tombstones.
  auto distance = std::map<SwitchboxNode *, double>{};
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
    for (auto &e : edges)
      if (e.src == sbPtr) channels[sbPtr].push_back(&e);

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
      bool relax = distance[src] + demand.at(e) < distance[dest];
      if (colors[dest] == WHITE) {
        if (relax) {
          distance[dest] = distance[src] + demand.at(e);
          preds[dest] = src;
          colors[dest] = GRAY;
        }
        Q.push(dest);
      } else if (colors[dest] == GRAY && relax) {
        distance[dest] = distance[src] + demand.at(e);
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
std::optional<std::map<PathEndPoint, SwitchSettings>> Router::findPaths(
    const int maxIterations) {
  int iterationCount = 0;
  std::map<PathEndPoint, SwitchSettings> routingSolution;
  std::map<ChannelEdge *, int> overCapacity;
  std::map<ChannelEdge *, int> usedCapacity;
  std::map<ChannelEdge *, double> demand;

  // initialize all Channel histories to 0
  for (auto &ch : impl->edges) {
    overCapacity[&ch] = 0;
    usedCapacity[&ch] = 0;
  }
  // assume legal until found otherwise
  bool isLegal = true;

  do {
    for (auto &ch : impl->edges) {
      double history = 1.0 + OVER_CAPACITY_COEFF * overCapacity[&ch];
      double congestion = 1.0 + USED_CAPACITY_COEFF * usedCapacity[&ch];
      demand[&ch] = history * congestion;
    }
    // if reach maxIterations, throw an error since no routing can be found
    if (++iterationCount > maxIterations) return std::nullopt;

    // "rip up" all routes
    routingSolution.clear();
    for (auto &[tileID, node] : impl->grid) node.clearAllocation();
    for (auto &ch : impl->edges) usedCapacity[&ch] = 0;
    isLegal = true;

    auto findIncomingEdge =
        [&](std::map<SwitchboxNode *, SwitchboxNode *> preds,
            SwitchboxNode *sb) -> ChannelEdge * {
      for (auto &e : impl->edges) {
        if (e.src == preds[sb] && e.target == sb) return &e;
      }
      return nullptr;
    };

    // for each flow, find the shortest path from source to destination
    // update used_capacity for the path between them
    for (const auto &[isPkt, src, dsts] : impl->flows) {
      // Use dijkstra to find path given current demand from the start
      // switchbox; find the shortest paths to each other switchbox. Output is
      // in the predecessor map, which must then be processed to get individual
      // switchbox settings
      assert(src.sb && "nonexistent flow source");
      std::set<SwitchboxNode *> processed;
      std::map<SwitchboxNode *, SwitchboxNode *> preds =
          dijkstraShortestPaths(src.sb, impl->grid, impl->edges, demand);

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
          ChannelEdge *ch = findIncomingEdge(preds, curr);
          assert(ch != nullptr && "couldn't find ch");
          int channel;
          // find all available channels in
          std::vector<int> availableChannels = curr->findAvailableChannelIn(
              getConnectingBundle(ch->bundle), lastDestPort, isPkt);
          if (!availableChannels.empty()) {
            // if possible, choose the channel that predecessor can also use
            // todo: consider all predecessors?
            int bFound = false;
            auto &pred = preds[curr];
            if (!processed.count(pred) && pred != src.sb) {
              ChannelEdge *predCh = findIncomingEdge(preds, pred);
              assert(predCh != nullptr && "couldn't find ch");
              for (int availableCh : availableChannels) {
                channel = availableCh;
                std::vector<int> availablePredChannels =
                    pred->findAvailableChannelIn(
                        getConnectingBundle(predCh->bundle),
                        {ch->bundle, channel}, isPkt);
                if (!availablePredChannels.empty()) {
                  bFound = true;
                  break;
                }
              }
            }
            if (!bFound) channel = availableChannels[0];
            if (!curr->allocate({getConnectingBundle(ch->bundle), channel},
                                lastDestPort, isPkt)) {
              llvm::errs() << "failed to make switchbox allocation";
              return std::nullopt;
            }
          } else {
            // if no channel available, use a virtual channel id and mark
            // routing as being invalid
            channel = usedCapacity[ch];
            if (isLegal) overCapacity[ch]++;
            isLegal = false;
          }
          if (!isLegal) break;

          usedCapacity[ch]++;

          // add the entrance port for this Switchbox
          Port currSourcePort = {getConnectingBundle(ch->bundle), channel};
          switchSettings[*curr].src = {currSourcePort};

          // add the current Switchbox to the map of the predecessor
          Port PredDestPort = {ch->bundle, channel};
          switchSettings[*preds[curr]].dsts.insert(PredDestPort);
          lastDestPort = PredDestPort;

          // if at capacity, bump demand to discourage using this Channel
          // this means the order matters!
          if (usedCapacity[ch] >= ch->maxCapacity) demand[ch] *= DEMAND_COEFF;

          processed.insert(curr);
          curr = preds[curr];

          // allocation may fail, as we start from the dest of flow while
          // src.port is not chosen by router
          if (curr == src.sb &&
              std::find(srcDestPorts.begin(), srcDestPorts.end(),
                        lastDestPort) == srcDestPorts.end()) {
            if (!src.sb->allocate(src.port, lastDestPort, isPkt)) {
              isLegal = false;
              overCapacity[ch]++;
            }
            if (!isLegal) break;
            srcDestPorts.push_back(lastDestPort);
          }
          if (!isLegal) break;
        }
        if (!isLegal) break;
      }
      if (!isLegal) break;
      // add this flow to the proposed solution
      routingSolution[src].swap(switchSettings);
    }

  } while (!isLegal);  // continue iterations until a legal routing is found

  return routingSolution;
}

std::vector<std::pair<Switchbox, Connect>> emitConnections(
    const std::map<PathEndPoint, SwitchSettings> &flowSolutions,
    const PathEndPoint &srcPoint, const AMDAIEDeviceModel &targetModel) {
  auto srcBundle = srcPoint.port.bundle;
  auto srcChannel = srcPoint.port.channel;
  Switchbox srcSB = srcPoint.sb;
  // the first sb isn't necessary here at all but it's just to agree with
  // ordering in mlir-aie tests (see
  // ConvertFlowsToInterconnect::matchAndRewrite).
  std::vector<std::pair<Switchbox, Connect>> connections;
  auto addConnection = [&connections](const Switchbox &currSb,
                                      StrmSwPortType inBundle, int inIndex,
                                      StrmSwPortType outBundle, int outIndex,
                                      Connect::Interconnect op, uint8_t col = 0,
                                      uint8_t row = 0) {
    connections.emplace_back(
        currSb, Connect(Port{inBundle, inIndex}, Port{outBundle, outIndex}, op,
                        col, row));
  };
  SwitchSettings settings = flowSolutions.at(srcPoint);
  for (const auto &[curr, setting] : settings) {
    int shimCh = srcChannel;
    // TODO: must reserve N3, N7, S2, S3 for DMA connections
    if (curr == srcSB && targetModel.isShimNOCTile(srcSB.col, srcSB.row)) {
      // shim DMAs at start of flows
      auto shimMuxOp = std::pair(Connect::Interconnect::shimMuxOp, srcSB.col);
      if (srcBundle == StrmSwPortType::DMA) {
        // must be either DMA0 -> N3 or DMA1 -> N7
        shimCh = srcChannel == 0 ? 3 : 7;
        addConnection(curr, srcBundle, srcChannel, StrmSwPortType::NORTH,
                      shimCh, shimMuxOp.first, shimMuxOp.second);
      } else if (srcBundle == StrmSwPortType::NOC) {
        // must be NOC0/NOC1 -> N2/N3 or NOC2/NOC3 -> N6/N7
        shimCh = srcChannel >= 2 ? srcChannel + 4 : srcChannel + 2;
        addConnection(curr, srcBundle, srcChannel, StrmSwPortType::NORTH,
                      shimCh, shimMuxOp.first, shimMuxOp.second);
      } else if (srcBundle == StrmSwPortType::PLIO) {
        // PLIO at start of flows with mux
        if (srcChannel == 2 || srcChannel == 3 || srcChannel == 6 ||
            srcChannel == 7) {
          // Only some PLIO requrie mux
          addConnection(curr, srcBundle, srcChannel, StrmSwPortType::NORTH,
                        shimCh, shimMuxOp.first, shimMuxOp.second);
        }
      }
    }

    auto swOp =
        std::make_tuple(Connect::Interconnect::swOp, curr.col, curr.row);
    for (const auto &[bundle, channel] : setting.dsts) {
      // handle special shim connectivity
      if (curr == srcSB &&
          targetModel.isShimNOCorPLTile(srcSB.col, srcSB.row)) {
        addConnection(curr, StrmSwPortType::SOUTH, shimCh, bundle, channel,
                      std::get<0>(swOp), std::get<1>(swOp), std::get<2>(swOp));
      } else if (targetModel.isShimNOCorPLTile(curr.col, curr.row) &&
                 (bundle == StrmSwPortType::DMA ||
                  bundle == StrmSwPortType::PLIO ||
                  bundle == StrmSwPortType::NOC)) {
        auto shimMuxOp =
            std::make_pair(Connect::Interconnect::shimMuxOp, curr.col);
        shimCh = channel;
        if (targetModel.isShimNOCTile(curr.col, curr.row)) {
          // shim DMAs at end of flows
          if (bundle == StrmSwPortType::DMA) {
            // must be either N2 -> DMA0 or N3 -> DMA1
            shimCh = channel == 0 ? 2 : 3;
            addConnection(curr, StrmSwPortType::NORTH, shimCh, bundle, channel,
                          shimMuxOp.first, shimMuxOp.second);
          } else if (bundle == StrmSwPortType::NOC) {
            // must be either N2/3/4/5 -> NOC0/1/2/3
            shimCh = channel + 2;
            addConnection(curr, StrmSwPortType::NORTH, shimCh, bundle, channel,
                          shimMuxOp.first, shimMuxOp.second);
          } else if (channel >= 2) {
            // must be PLIO...only PLIO >= 2 require mux
            addConnection(curr, StrmSwPortType::NORTH, shimCh, bundle, channel,
                          shimMuxOp.first, shimMuxOp.second);
          }
        }
        addConnection(curr, setting.src.bundle, setting.src.channel,
                      StrmSwPortType::SOUTH, shimCh, std::get<0>(swOp),
                      std::get<1>(swOp), std::get<2>(swOp));
      } else {
        // otherwise, regular switchbox connection
        addConnection(curr, setting.src.bundle, setting.src.channel, bundle,
                      channel, std::get<0>(swOp), std::get<1>(swOp),
                      std::get<2>(swOp));
      }
    }
  }

  return connections;
}

bool existsPathToDest(const SwitchSettings &settings, TileLoc currTile,
                      StrmSwPortType currDestBundle, int currDestChannel,
                      TileLoc finalTile, StrmSwPortType finalDestBundle,
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
        if (existsPathToDest(settings, neighbourTile, bundle, channel,
                             finalTile, finalDestBundle, finalDestChannel)) {
          return true;
        }
      }
    }
  }

  return false;
}

}  // namespace mlir::iree_compiler::AMDAIE
