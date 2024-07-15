// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions. See
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: # Apache-2.0 WITH LLVM-exception

#include "iree_aie_router.h"

#include <set>

#include "d_ary_heap.h"
#include "iree_aie_runtime.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DirectedGraph.h"

#define DEBUG_TYPE "iree-aie-runtime-router"

#define OVER_CAPACITY_COEFF 0.02
#define USED_CAPACITY_COEFF 0.02
#define DEMAND_COEFF 1.1
static constexpr double INF = std::numeric_limits<double>::max();

namespace mlir::iree_compiler::AMDAIE {

/// Create the initial graph of nearest neighbor relationships between
/// switchboxes, including link capacity.
void Router::initialize(int maxCol, int maxRow,
                        AMDAIEDeviceModel &deviceModel) {
  int nodeId = 0;
  for (int row = 0; row <= maxRow; row++) {
    for (int col = 0; col <= maxCol; col++) {
      auto [it, _] =
          grid.insert({{col, row}, SwitchboxNode{col, row, nodeId++}});
      (void)graph.addNode(it->second);
      SwitchboxNode &thisNode = grid.at({col, row});
      if (row > 0) {  // if not in row 0 add channel to North/South
        SwitchboxNode &southernNeighbor = grid.at({col, row - 1});
        // get the number of outgoing connections on the south side - outgoing
        // because these correspond to rhs of a connect op
        if (uint32_t maxCapacity = deviceModel.getNumDestSwitchboxConnections(
                col, row, StrmSwPortType::SOUTH)) {
          edges.emplace_back(thisNode, southernNeighbor, StrmSwPortType::SOUTH,
                             maxCapacity);
          (void)graph.connect(thisNode, southernNeighbor, edges.back());
        }
        // get the number of incoming connections on the south side - incoming
        // because they correspond to connections on the southside that are then
        // routed using internal connect ops through the switchbox (i.e., lhs of
        // connect ops)
        if (uint32_t maxCapacity = deviceModel.getNumSourceSwitchboxConnections(
                col, row, StrmSwPortType::SOUTH)) {
          edges.emplace_back(southernNeighbor, thisNode, StrmSwPortType::NORTH,
                             maxCapacity);
          (void)graph.connect(southernNeighbor, thisNode, edges.back());
        }
      }

      if (col > 0) {  // if not in col 0 add channel to East/West
        SwitchboxNode &westernNeighbor = grid.at({col - 1, row});
        if (uint32_t maxCapacity = deviceModel.getNumDestSwitchboxConnections(
                col, row, StrmSwPortType::WEST)) {
          edges.emplace_back(thisNode, westernNeighbor, StrmSwPortType::WEST,
                             maxCapacity);
          (void)graph.connect(thisNode, westernNeighbor, edges.back());
        }
        if (uint32_t maxCapacity = deviceModel.getNumSourceSwitchboxConnections(
                col, row, StrmSwPortType::WEST)) {
          edges.emplace_back(westernNeighbor, thisNode, StrmSwPortType::EAST,
                             maxCapacity);
          (void)graph.connect(westernNeighbor, thisNode, edges.back());
        }
      }
    }
  }
}

/// Add a flow from src to dst can have an arbitrary number of dst locations due
/// to fanout.
void Router::addFlow(TileLoc srcCoords, Port srcPort, TileLoc dstCoords,
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

/// Keep track of connections already used in the AIE; Pathfinder algorithm will
/// avoid using these.
bool Router::addFixedConnection(Connect connectOp) {
  Switchbox sb = connectOp.sb;
  TileLoc sbTile(sb.col, sb.row);
  StrmSwPortType sourceBundle = connectOp.src.bundle;
  StrmSwPortType destBundle = connectOp.dst.bundle;

  // find the correct SwitchBoxConnection and indicate the fixed direction
  // outgoing connection
  auto matchingCh = std::find_if(
      edges.begin(), edges.end(), [&](SwitchBoxConnectionEdge &ch) {
        return static_cast<TileLoc>(ch.src) == sbTile &&
               ch.bundle == destBundle;
      });
  if (matchingCh != edges.end()) {
    return matchingCh->fixedCapacity.insert(connectOp.dst.channel).second ||
           true;
  }

  // incoming connection
  matchingCh = std::find_if(
      edges.begin(), edges.end(), [&](SwitchBoxConnectionEdge &ch) {
        return static_cast<TileLoc>(ch.target) == sbTile &&
               ch.bundle == getConnectingBundle(sourceBundle);
      });
  if (matchingCh != edges.end()) {
    return matchingCh->fixedCapacity.insert(connectOp.src.channel).second ||
           true;
  }

  return false;
}

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

  std::map<SwitchboxNode *, std::vector<SwitchBoxConnectionEdge *>> edges;

  enum Color { WHITE, GRAY, BLACK };
  std::map<SwitchboxNode *, Color> colors;
  for (SwitchboxNode *sb : graph) {
    colors[sb] = WHITE;
    edges[sb] = {sb->getEdges().begin(), sb->getEdges().end()};
    std::sort(
        edges[sb].begin(), edges[sb].end(),
        [](const SwitchBoxConnectionEdge *c1, SwitchBoxConnectionEdge *c2) {
          return c1->getTargetNode().id < c2->getTargetNode().id;
        });
  }

  Q.push(src);
  while (!Q.empty()) {
    src = Q.top();
    Q.pop();
    for (SwitchBoxConnectionEdge *e : edges[src]) {
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

/// Perform congestion-aware routing for all flows which have been added.
/// Use Dijkstra's shortest path to find routes, and use "demand" as the
/// weights. If the routing finds too much congestion, update the demand weights
/// and repeat the process until a valid solution is found.
/// Returns a map specifying switchbox settings for all flows.
/// If no legal routing can be found after maxIterations, returns empty vector.
std::optional<std::map<PathEndPoint, SwitchSettings>> Router::findPaths(
    const int maxIterations) {
  int iterationCount = 0;
  std::map<PathEndPoint, SwitchSettings> routingSolution;

  // initialize all SwitchBoxConnection histories to 0
  for (auto &ch : edges) ch.overCapacityCount = 0;

  // Check that every channel does not exceed max capacity.
  auto isLegal = [&] {
    bool legal = true;  // assume legal until found otherwise
    for (auto &e : edges) {
      if (e.usedCapacity > e.maxCapacity) {
        e.overCapacityCount++;
        legal = false;
        break;
      }
    }

    return legal;
  };

  do {
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
    if (++iterationCount > maxIterations) return std::nullopt;

    // "rip up" all routes, i.e. set used capacity in each SwitchBoxConnection
    // to 0
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
          SmallVector<SwitchBoxConnectionEdge *, 10> channels;
          graph.findIncomingEdgesToNode(*curr, channels);
          auto *matchingCh = std::find_if(channels.begin(), channels.end(),
                                          [&](SwitchBoxConnectionEdge *ch) {
                                            return ch->src == *preds[curr];
                                          });
          assert(matchingCh != channels.end() && "couldn't find ch");
          // incoming edge
          SwitchBoxConnectionEdge *ch = *matchingCh;

          // don't use fixed channels
          while (ch->fixedCapacity.count(ch->usedCapacity)) ch->usedCapacity++;

          // add the entrance port for this Switchbox
          switchSettings[*curr].src = {getConnectingBundle(ch->bundle),
                                       ch->usedCapacity};
          // add the current Switchbox to the map of the predecessor
          switchSettings[*preds[curr]].dsts.insert(
              {ch->bundle, ch->usedCapacity});

          ch->usedCapacity++;
          // if at capacity, bump demand to discourage using this
          // SwitchBoxConnection
          if (ch->usedCapacity >= ch->maxCapacity) ch->demand *= DEMAND_COEFF;

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

Switchbox *Router::getSwitchbox(TileLoc coords) {
  auto *sb = std::find_if(graph.begin(), graph.end(), [&](SwitchboxNode *sb) {
    return sb->col == coords.col && sb->row == coords.row;
  });
  assert(sb != graph.end() && "couldn't find sb");
  return *sb;
}

}  // namespace mlir::iree_compiler::AMDAIE
