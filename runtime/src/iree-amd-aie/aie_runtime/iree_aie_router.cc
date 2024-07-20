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
                        [](const llvm::detail::DenseMapPair<SwitchboxNode,
                                                            SwitchSetting> &p) {
                          return to_string(p.getFirst()) + ": " +
                                 to_string(p.getSecond());
                        }),
                    ", ") +
         ")";
}

std::string to_string(const Connectivity &value) {
  switch (value) {
    STRINGIFY_ENUM_CASE(Connectivity::INVALID)
    STRINGIFY_ENUM_CASE(Connectivity::AVAILABLE)
    STRINGIFY_ENUM_CASE(Connectivity::OCCUPIED)
  }

  llvm::report_fatal_error("Unhandled Connectivity case");
}

STRINGIFY_2TUPLE_STRUCT(Port, bundle, channel)
STRINGIFY_2TUPLE_STRUCT(Connect, src, dst)
STRINGIFY_3TUPLE_STRUCT(SwitchboxNode, col, row, id)
STRINGIFY_2TUPLE_STRUCT(PathEndPoint, sb, port)

BOTH_OSTREAM_OPS_FORALL_ROUTER_TYPES(OSTREAM_OP_DEFN, BOTH_OSTREAM_OP)

static constexpr double INF = std::numeric_limits<double>::max();
void Pathfinder::initialize(int maxCol, int maxRow,
                            const AMDAIEDeviceModel &targetModel) {
  // make grid of switchboxes
  int id = 0;
  for (int row = 0; row <= maxRow; row++) {
    for (int col = 0; col <= maxCol; col++) {
      grid.insert({{col, row}, SwitchboxNode{col, row, id++, targetModel}});
      SwitchboxNode &thisNode = grid.at({col, row});
      if (row > 0) {  // if not in row 0 add channel to North/South
        SwitchboxNode &southernNeighbor = grid.at({col, row - 1});
        // get the number of outgoing connections on the south side - outgoing
        // because these correspond to rhs of a connect op
        if (targetModel.getNumDestSwitchboxConnections(col, row,
                                                       StrmSwPortType::SOUTH)) {
          edges.emplace_back(&thisNode, &southernNeighbor);
        }
        // get the number of incoming connections on the south side - incoming
        // because they correspond to connections on the southside that are then
        // routed using internal connect ops through the switchbox (i.e., lhs of
        // connect ops)
        if (targetModel.getNumSourceSwitchboxConnections(
                col, row, StrmSwPortType::SOUTH)) {
          edges.emplace_back(&southernNeighbor, &thisNode);
        }
      }

      if (col > 0) {  // if not in col 0 add channel to East/West
        SwitchboxNode &westernNeighbor = grid.at({col - 1, row});
        if (targetModel.getNumDestSwitchboxConnections(col, row,
                                                       StrmSwPortType::WEST)) {
          edges.emplace_back(&thisNode, &westernNeighbor);
        }
        if (targetModel.getNumSourceSwitchboxConnections(
                col, row, StrmSwPortType::WEST)) {
          edges.emplace_back(&westernNeighbor, &thisNode);
        }
      }
    }
  }
}

// Add a flow from src to dst can have an arbitrary number of dst locations due
// to fanout.
void Pathfinder::addFlow(TileLoc srcCoords, Port srcPort, TileLoc dstCoords,
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
bool Pathfinder::addFixedConnection(
    int col, int row,
    const std::vector<std::tuple<StrmSwPortType, int, StrmSwPortType, int>>
        &connects) {
  SwitchboxNode &sb = grid.at({col, row});
  std::set<int> invalidInId, invalidOutId;
  for (auto &[sourceBundle, sourceChannel, destBundle, destChannel] :
       connects) {
    Port srcPort = {sourceBundle, sourceChannel};
    Port destPort = {destBundle, destChannel};
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
                       << getConnectingBundle(ch->bundle) << channel << " -> "
                       << lastDestPort.bundle << lastDestPort.channel << "\n");
          } else {
            // if no channel available, use a virtual channel id and mark
            // routing as being invalid
            channel = usedCapacity[ch];
            if (isLegal) {
              overCapacity[ch]++;
              LLVM_DEBUG(llvm::dbgs()
                         << *curr
                         << ", congestion: " << getConnectingBundle(ch->bundle)
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
            LLVM_DEBUG(llvm::dbgs() << *curr << ", bump demand: "
                                    << getConnectingBundle(ch->bundle)
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
                         << *curr << ", unable to connect: " << src.port.bundle
                         << src.port.channel << " -> " << lastDestPort.bundle
                         << lastDestPort.channel << "\n");
            }
            srcDestPorts.push_back(lastDestPort);
          }
        }
      }
      // add this flow to the proposed solution
      routingSolution[src].swap(switchSettings);
    }

  } while (!isLegal);  // continue iterations until a legal routing is found

  return routingSolution;
}

}  // namespace mlir::iree_compiler::AMDAIE
