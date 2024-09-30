// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions. See
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: # Apache-2.0 WITH LLVM-exception

#include "iree_aie_router.h"

#include <tuple>

#include "amsel_generator.h"
#include "d_ary_heap.h"
#include "iree_aie_runtime.h"
#include "llvm/ADT/StringExtras.h"

#define DEBUG_TYPE "iree-aie-runtime-router"

#define OVER_CAPACITY_COEFF 0.1
#define USED_CAPACITY_COEFF 0.1
// NOTE(jornt): Set demand coeff high enough to find solutions. A coeff of 1.1
// - 1.3 failed to find solutions in some tests.
#define DEMAND_COEFF 1.5
#define DEMAND_BASE 1.0
#define MAX_CIRCUIT_STREAM_CAPACITY 1
#define MAX_PACKET_STREAM_CAPACITY 32

static constexpr double INF = std::numeric_limits<double>::max();

namespace MLIRAIELegacy {
extern uint32_t getNumDestShimMuxConnections(
    int col, int row, mlir::iree_compiler::AMDAIE::StrmSwPortType bundle,
    const mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel &deviceModel);
extern uint32_t getNumSourceShimMuxConnections(
    int col, int row, mlir::iree_compiler::AMDAIE::StrmSwPortType bundle,
    const mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel &deviceModel);
extern uint32_t getNumDestSwitchBoxConnections(
    int col, int row, mlir::iree_compiler::AMDAIE::StrmSwPortType bundle,
    const mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel &deviceModel);
extern uint32_t getNumSourceSwitchBoxConnections(
    int col, int row, mlir::iree_compiler::AMDAIE::StrmSwPortType bundle,
    const mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel &deviceModel);
extern bool isLegalTileConnection(
    int col, int row, mlir::iree_compiler::AMDAIE::StrmSwPortType srcBundle,
    int srcChan, mlir::iree_compiler::AMDAIE::StrmSwPortType dstBundle,
    int dstChan,
    const mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel &deviceModel);
int rows(const mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel &deviceModel);
int columns(const mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel &deviceModel);
}  // namespace MLIRAIELegacy

namespace mlir::iree_compiler::AMDAIE {

enum class Connectivity { INVALID = 0, AVAILABLE = 1 };

struct SwitchboxConnect {
  SwitchboxConnect() = default;
  SwitchboxConnect(TileLoc tileLoc) : srcCoords(tileLoc), dstCoords(tileLoc) {}
  SwitchboxConnect(TileLoc srcCoords, TileLoc dstCoords)
      : srcCoords(srcCoords), dstCoords(dstCoords) {}

  TileLoc srcCoords, dstCoords;
  std::vector<Port> srcPorts;
  std::vector<Port> dstPorts;
  // connectivity between ports
  std::vector<std::vector<Connectivity>> connectivity;
  // weights of Dijkstra's shortest path
  std::vector<std::vector<double>> demand;
  // history of Channel being over capacity
  std::vector<std::vector<int>> overCapacity;
  // how many circuit streams are actually using this Channel
  std::vector<std::vector<int>> usedCapacity;
  // how many packet streams are actually using this Channel
  std::vector<std::vector<int>> packetFlowCount;

  // resize the matrices to the size of srcPorts and dstPorts
  void resize() {
    connectivity.resize(
        srcPorts.size(),
        std::vector<Connectivity>(dstPorts.size(), Connectivity::INVALID));
    demand.resize(srcPorts.size(), std::vector<double>(dstPorts.size(), 0.0));
    overCapacity.resize(srcPorts.size(), std::vector<int>(dstPorts.size(), 0));
    usedCapacity.resize(srcPorts.size(), std::vector<int>(dstPorts.size(), 0));
    packetFlowCount.resize(srcPorts.size(),
                           std::vector<int>(dstPorts.size(), 0));
  }

  // update demand at the beginning of each dijkstraShortestPaths iteration
  void updateDemand() {
    for (size_t i = 0; i < srcPorts.size(); i++) {
      for (size_t j = 0; j < dstPorts.size(); j++) {
        double history = DEMAND_BASE + OVER_CAPACITY_COEFF * overCapacity[i][j];
        double congestion =
            DEMAND_BASE + USED_CAPACITY_COEFF * usedCapacity[i][j];
        demand[i][j] = history * congestion;
      }
    }
  }

  // inside each dijkstraShortestPaths interation, bump demand when exceeds
  // capacity
  void bumpDemand(size_t i, size_t j) {
    if (usedCapacity[i][j] >= MAX_CIRCUIT_STREAM_CAPACITY) {
      demand[i][j] *= DEMAND_COEFF;
    }
  }
};

struct Flow {
  bool isPacketFlow;
  PathEndPoint src;
  std::vector<PathEndPoint> dsts;
};

struct RouterImpl {
  // Flows to be routed
  std::vector<Flow> flows;
  // Represent all routable paths as a graph
  // The key is a pair of TileIDs representing the connectivity from srcTile to
  // dstTile If srcTile == dstTile, it represents connections inside the same
  // switchbox otherwise, it represents connections (South, North, West, East)
  // accross two switchboxes
  std::map<std::pair<TileLoc, TileLoc>, SwitchboxConnect> graph;
  // Channels available in the network
  // The key is a PathEndPoint representing the start of a path
  // The value is a vector of PathEndPoints representing the possible ends of
  // the path
  std::map<PathEndPoint, std::vector<PathEndPoint>> channels;
};

Router::Router() { impl = new RouterImpl(); }
Router::~Router() { delete impl; }

void Router::initialize(int maxCol, int maxRow,
                        const AMDAIEDeviceModel &deviceModel) {
  std::map<StrmSwPortType, int> maxChannels;
  auto intraconnect = [&](int col, int row) {
    TileLoc tileLoc = {col, row};
    SwitchboxConnect sb = {tileLoc};

    const std::vector<StrmSwPortType> bundles = {
        StrmSwPortType::CORE,  StrmSwPortType::DMA,   StrmSwPortType::FIFO,
        StrmSwPortType::SOUTH, StrmSwPortType::WEST,  StrmSwPortType::NORTH,
        StrmSwPortType::EAST,  StrmSwPortType::TRACE, StrmSwPortType::UCTRLR,
        StrmSwPortType::CTRL};
    for (StrmSwPortType bundle : bundles) {
      // get all ports into current switchbox
      int channels =
          deviceModel.getNumSourceSwitchBoxConnections(col, row, bundle);
      if (channels == 0 && deviceModel.isShimNOCorPLTile(col, row)) {
        // wordaround for shimMux
        channels = MLIRAIELegacy::getNumSourceShimMuxConnections(
            col, row, bundle, deviceModel);
      }
      for (int channel = 0; channel < channels; channel++) {
        sb.srcPorts.push_back(Port{bundle, channel});
      }
      // get all ports out of current switchbox
      channels = deviceModel.getNumDestSwitchBoxConnections(col, row, bundle);
      if (channels == 0 && deviceModel.isShimNOCorPLTile(col, row)) {
        // wordaround for shimMux
        channels = MLIRAIELegacy::getNumDestShimMuxConnections(col, row, bundle,
                                                               deviceModel);
      }
      for (int channel = 0; channel < channels; channel++) {
        sb.dstPorts.push_back(Port{bundle, channel});
      }
      maxChannels[bundle] = channels;
    }
    // initialize matrices
    sb.resize();
    for (size_t i = 0; i < sb.srcPorts.size(); i++) {
      for (size_t j = 0; j < sb.dstPorts.size(); j++) {
        auto &pIn = sb.srcPorts[i];
        auto &pOut = sb.dstPorts[j];
        if (deviceModel.isLegalTileConnection(col, row, pIn.bundle, pIn.channel,
                                              pOut.bundle, pOut.channel))
          sb.connectivity[i][j] = Connectivity::AVAILABLE;
        else {
          sb.connectivity[i][j] = Connectivity::INVALID;
          if (deviceModel.isShimNOCorPLTile(col, row)) {
            // wordaround for shimMux
            auto isBundleInList = [](StrmSwPortType bundle,
                                     std::vector<StrmSwPortType> bundles) {
              return std::find(bundles.begin(), bundles.end(), bundle) !=
                     bundles.end();
            };
            const std::vector<StrmSwPortType> bundles = {
                StrmSwPortType::DMA, StrmSwPortType::UCTRLR};
            if (isBundleInList(pIn.bundle, bundles) ||
                isBundleInList(pOut.bundle, bundles))
              sb.connectivity[i][j] = Connectivity::AVAILABLE;
          }
        }
      }
    }
    impl->graph[std::make_pair(tileLoc, tileLoc)] = sb;
  };

  auto interconnect = [&](int col, int row, int targetCol, int targetRow,
                          StrmSwPortType srcBundle, StrmSwPortType dstBundle) {
    SwitchboxConnect sb = {{col, row}, {targetCol, targetRow}};
    for (int channel = 0; channel < maxChannels[srcBundle]; channel++) {
      sb.srcPorts.push_back(Port{srcBundle, channel});
      sb.dstPorts.push_back(Port{dstBundle, channel});
    }
    sb.resize();
    for (size_t i = 0; i < sb.srcPorts.size(); i++) {
      sb.connectivity[i][i] = Connectivity::AVAILABLE;
    }
    impl->graph[std::make_pair(TileLoc{col, row},
                               TileLoc{targetCol, targetRow})] = sb;
  };

  for (int row = 0; row <= maxRow; row++) {
    for (int col = 0; col <= maxCol; col++) {
      maxChannels.clear();
      // connections within the same switchbox
      intraconnect(col, row);

      // connections between switchboxes
      if (row > 0) {
        // from south to north
        interconnect(col, row, col, row - 1, StrmSwPortType::SOUTH,
                     StrmSwPortType::NORTH);
      }
      if (row < maxRow) {
        // from north to south
        interconnect(col, row, col, row + 1, StrmSwPortType::NORTH,
                     StrmSwPortType::SOUTH);
      }
      if (col > 0) {
        // from east to west
        interconnect(col, row, col - 1, row, StrmSwPortType::WEST,
                     StrmSwPortType::EAST);
      }
      if (col < maxCol) {
        // from west to east
        interconnect(col, row, col + 1, row, StrmSwPortType::EAST,
                     StrmSwPortType::WEST);
      }
    }
  }
}

void Router::addFlow(TileLoc srcCoords, Port srcPort, TileLoc dstCoords,
                     Port dstPort, bool isPacketFlow) {
  // check if a flow with this source already exists
  for (auto &[isPkt, src, dsts] : impl->flows) {
    if (src.tileLoc == srcCoords && src.port == srcPort) {
      dsts.emplace_back(PathEndPoint{dstCoords, dstPort});
      return;
    }
  }

  // If no existing flow was found with this source, create a new flow.
  impl->flows.push_back(
      Flow{isPacketFlow, PathEndPoint{srcCoords, srcPort},
           std::vector<PathEndPoint>{PathEndPoint{dstCoords, dstPort}}});
}

// Keep track of connections already used in the AIE; Pathfinder algorithm will
// avoid using these.
bool Router::addFixedConnection(
    int col, int row, const std::vector<std::tuple<Port, Port>> &connects) {
  TileLoc tileLoc = {col, row};
  auto &sb = impl->graph[std::make_pair(tileLoc, tileLoc)];
  for (auto &[sourcePort, destPort] : connects) {
    bool found = false;
    for (size_t i = 0; i < sb.srcPorts.size(); i++) {
      for (size_t j = 0; j < sb.dstPorts.size(); j++) {
        if (sb.srcPorts[i] == sourcePort && sb.dstPorts[j] == destPort &&
            sb.connectivity[i][j] == Connectivity::AVAILABLE) {
          sb.connectivity[i][j] = Connectivity::INVALID;
          found = true;
        }
      }
    }
    if (!found) {
      // could not add such a fixed connection
      return false;
    }
  }
  return true;
}

std::map<PathEndPoint, PathEndPoint> Router::dijkstraShortestPaths(
    PathEndPoint src) {
  // Use std::map instead of DenseMap because DenseMap doesn't let you
  // overwrite tombstones.
  std::map<PathEndPoint, double> distance;
  std::map<PathEndPoint, PathEndPoint> preds;
  std::map<PathEndPoint, uint64_t> indexInHeap;
  enum Color { WHITE, GRAY, BLACK };
  std::map<PathEndPoint, Color> colors;
  typedef d_ary_heap_indirect<
      /*Value=*/PathEndPoint, /*Arity=*/4,
      /*IndexInHeapPropertyMap=*/std::map<PathEndPoint, uint64_t>,
      /*DistanceMap=*/std::map<PathEndPoint, double> &,
      /*Compare=*/std::less<>>
      MutableQueue;
  MutableQueue Q(distance, indexInHeap);

  distance[src] = 0.0;
  Q.push(src);
  while (!Q.empty()) {
    src = Q.top();
    Q.pop();

    // get all channels src connects to
    if (impl->channels.count(src) == 0) {
      auto &sb = impl->graph[std::make_pair(src.tileLoc, src.tileLoc)];
      for (size_t i = 0; i < sb.srcPorts.size(); i++) {
        for (size_t j = 0; j < sb.dstPorts.size(); j++) {
          if (sb.srcPorts[i] == src.port &&
              sb.connectivity[i][j] == Connectivity::AVAILABLE) {
            // connections within the same switchbox
            impl->channels[src].push_back(
                PathEndPoint{src.tileLoc, sb.dstPorts[j]});
          }
        }
      }
      // connections to neighboring switchboxes
      std::vector<std::pair<TileLoc, Port>> neighbors = {
          {{src.tileLoc.col, src.tileLoc.row - 1},
           {StrmSwPortType::NORTH, src.port.channel}},
          {{src.tileLoc.col - 1, src.tileLoc.row},
           {StrmSwPortType::EAST, src.port.channel}},
          {{src.tileLoc.col, src.tileLoc.row + 1},
           {StrmSwPortType::SOUTH, src.port.channel}},
          {{src.tileLoc.col + 1, src.tileLoc.row},
           {StrmSwPortType::WEST, src.port.channel}}};

      for (const auto &[neighborCoords, neighborPort] : neighbors) {
        if (impl->graph.count(std::make_pair(src.tileLoc, neighborCoords)) >
                0 &&
            src.port.bundle == getConnectingBundle(neighborPort.bundle)) {
          auto &sb = impl->graph[std::make_pair(src.tileLoc, neighborCoords)];
          if (std::find(sb.dstPorts.begin(), sb.dstPorts.end(), neighborPort) !=
              sb.dstPorts.end())
            impl->channels[src].push_back({neighborCoords, neighborPort});
        }
      }
      std::sort(impl->channels[src].begin(), impl->channels[src].end());
    }

    for (auto &dest : impl->channels[src]) {
      if (distance.count(dest) == 0) distance[dest] = INF;
      auto &sb = impl->graph[std::make_pair(src.tileLoc, dest.tileLoc)];
      size_t i = std::distance(
          sb.srcPorts.begin(),
          std::find(sb.srcPorts.begin(), sb.srcPorts.end(), src.port));
      size_t j = std::distance(
          sb.dstPorts.begin(),
          std::find(sb.dstPorts.begin(), sb.dstPorts.end(), dest.port));
      assert(i < sb.srcPorts.size());
      assert(j < sb.dstPorts.size());
      bool relax = distance[src] + sb.demand[i][j] < distance[dest];
      if (colors.count(dest) == 0) {
        // was WHITE
        if (relax) {
          distance[dest] = distance[src] + sb.demand[i][j];
          preds[dest] = src;
          colors[dest] = GRAY;
        }
        Q.push(dest);
      } else if (colors[dest] == GRAY && relax) {
        distance[dest] = distance[src] + sb.demand[i][j];
        preds[dest] = src;
      }
    }
    colors[src] = BLACK;
  }

  return preds;
}

// Perform congestion-aware routing for all flows which have been added.
// Use Dijkstra's shortest path to find routes, and use "demand" as the
// weights. If the routing finds too much congestion, update the demand
// weights and repeat the process until a valid solution is found. Returns a
// map specifying switchbox settings for all flows. If no legal routing can be
// found after maxIterations, returns empty vector.
std::optional<std::map<PathEndPoint, SwitchSettings>> Router::findPaths(
    const int maxIterations) {
  LLVM_DEBUG(llvm::dbgs() << "\t---Begin Pathfinder::findPaths---\n");
  std::map<PathEndPoint, SwitchSettings> routingSolution;
  // initialize all Channel histories to 0
  for (auto &[_, sb] : impl->graph) {
    for (size_t i = 0; i < sb.srcPorts.size(); i++) {
      for (size_t j = 0; j < sb.dstPorts.size(); j++) {
        sb.usedCapacity[i][j] = 0;
        sb.overCapacity[i][j] = 0;
      }
    }
  }

  int iterationCount = -1;
  int illegalEdges = 0;
  int totalPathLength = 0;
  do {
    // if reach maxIterations, throw an error since no routing can be found
    if (++iterationCount >= maxIterations) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\t\tPathfinder: maxIterations has been exceeded ("
                 << maxIterations
                 << " iterations)...unable to find routing for flows.\n");
      return std::nullopt;
    }

    LLVM_DEBUG(llvm::dbgs() << "\t\t---Begin findPaths iteration #"
                            << iterationCount << "---\n");
    // update demand at the beginning of each iteration
    for (auto &[_, sb] : impl->graph) {
      sb.updateDemand();
    }

    // "rip up" all routes
    illegalEdges = 0;
    totalPathLength = 0;
    routingSolution.clear();
    for (auto &[_, sb] : impl->graph) {
      for (size_t i = 0; i < sb.srcPorts.size(); i++) {
        for (size_t j = 0; j < sb.dstPorts.size(); j++) {
          sb.usedCapacity[i][j] = 0;
          sb.packetFlowCount[i][j] = 0;
        }
      }
    }

    // for each flow, find the shortest path from source to destination
    // update used_capacity for the path between them
    for (const auto &[isPkt, src, dsts] : impl->flows) {
      // Use dijkstra to find path given current demand from the start
      // switchbox; find the shortest paths to each other switchbox. Output is
      // in the predecessor map, which must then be processed to get
      // individual switchbox settings
      std::set<PathEndPoint> processed;
      std::map<PathEndPoint, PathEndPoint> preds = dijkstraShortestPaths(src);

      // trace the path of the flow backwards via predecessors
      // increment used_capacity for the associated channels
      SwitchSettings switchSettings;
      processed.insert(src);
      for (auto endPoint : dsts) {
        if (endPoint == src) {
          // route to self
          switchSettings[src.tileLoc].srcs.push_back(src.port);
          switchSettings[src.tileLoc].dsts.push_back(src.port);
        }
        auto curr = endPoint;
        // trace backwards until a vertex already processed is reached
        while (!processed.count(curr)) {
          auto &sb =
              impl->graph[std::make_pair(preds[curr].tileLoc, curr.tileLoc)];
          size_t i =
              std::distance(sb.srcPorts.begin(),
                            std::find(sb.srcPorts.begin(), sb.srcPorts.end(),
                                      preds[curr].port));
          size_t j = std::distance(
              sb.dstPorts.begin(),
              std::find(sb.dstPorts.begin(), sb.dstPorts.end(), curr.port));
          assert(i < sb.srcPorts.size());
          assert(j < sb.dstPorts.size());
          if (isPkt) {
            sb.packetFlowCount[i][j]++;
            // maximum packet stream per channel
            if (sb.packetFlowCount[i][j] >= MAX_PACKET_STREAM_CAPACITY) {
              sb.packetFlowCount[i][j] = 0;
              sb.usedCapacity[i][j]++;
            }
          } else {
            sb.packetFlowCount[i][j] = 0;
            sb.usedCapacity[i][j]++;
          }
          // if at capacity, bump demand to discourage using this Channel
          // this means the order matters!
          sb.bumpDemand(i, j);
          if (preds[curr].tileLoc == curr.tileLoc) {
            switchSettings[preds[curr].tileLoc].srcs.push_back(
                preds[curr].port);
            switchSettings[curr.tileLoc].dsts.push_back(curr.port);
          }
          processed.insert(curr);
          curr = preds[curr];
        }
      }
      // add this flow to the proposed solution
      routingSolution[src] = switchSettings;
    }

    for (auto &[_, sb] : impl->graph) {
      for (size_t i = 0; i < sb.srcPorts.size(); i++) {
        for (size_t j = 0; j < sb.dstPorts.size(); j++) {
          // fix used capacity for packet flows
          if (sb.packetFlowCount[i][j] > 0) {
            sb.packetFlowCount[i][j] = 0;
            sb.usedCapacity[i][j]++;
          }
          // check that every channel does not exceed max capacity
          if (sb.usedCapacity[i][j] > MAX_CIRCUIT_STREAM_CAPACITY) {
            sb.overCapacity[i][j]++;
            illegalEdges++;
            LLVM_DEBUG(llvm::dbgs()
                       << "\t\t\tToo much capacity on (" << sb.srcCoords.col
                       << "," << sb.srcCoords.row << ") " << sb.srcPorts[i]
                       << " -> (" << sb.dstCoords.col << "," << sb.dstCoords.row
                       << ") " << sb.dstPorts[j]
                       << ", used_capacity = " << sb.usedCapacity[i][j]
                       << ", demand = " << sb.demand[i][j]
                       << ", over_capacity_count = " << sb.overCapacity[i][j]
                       << "\n");
          }
          // calculate total path length (across switchboxes)
          if (sb.srcCoords != sb.dstCoords) {
            totalPathLength += sb.usedCapacity[i][j];
          }
        }
      }
    }

#ifndef NDEBUG
    for (const auto &[PathEndPoint, switchSetting] : routingSolution) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\t\t\tFlow starting at (" << PathEndPoint.tileLoc.col
                 << "," << PathEndPoint.tileLoc.row << "):\t");
    }
#endif
    LLVM_DEBUG(llvm::dbgs()
               << "\t\t---End findPaths iteration #" << iterationCount
               << " , illegal edges count = " << illegalEdges
               << ", total path length = " << totalPathLength << "---\n");
  } while (illegalEdges >
           0);  // continue iterations until a legal routing is found

  LLVM_DEBUG(llvm::dbgs() << "\t---End Pathfinder::findPaths---\n");
  return routingSolution;
}

/// Transform outputs produced by the router into representations (structs) that
/// directly map to stream switch configuration ops (soon-to-be aie-rt calls).
/// Namely pairs of (switchbox, internal connections).
std::map<TileLoc, std::vector<Connect>> emitConnections(
    const std::map<PathEndPoint, SwitchSettings> &flowSolutions,
    const PathEndPoint &srcPoint, const AMDAIEDeviceModel &deviceModel) {
  auto srcBundle = srcPoint.port.bundle;
  auto srcChannel = srcPoint.port.channel;
  TileLoc srcTileLoc = srcPoint.tileLoc;
  // the first sb isn't necessary here at all but it's just to agree with
  // ordering in mlir-aie tests (see
  // ConvertFlowsToInterconnect::matchAndRewrite).
  std::map<TileLoc, std::vector<Connect>> connections;
  auto addConnection = [&connections](const TileLoc &currSb,
                                      StrmSwPortType inBundle, int inIndex,
                                      StrmSwPortType outBundle, int outIndex,
                                      Connect::Interconnect op, uint8_t col = 0,
                                      uint8_t row = 0) {
    connections[currSb].emplace_back(Port{inBundle, inIndex},
                                     Port{outBundle, outIndex}, op, col, row);
  };
  SwitchSettings settings = flowSolutions.at(srcPoint);
  for (const auto &[curr, setting] : settings) {
    int shimCh = srcChannel;
    // TODO: must reserve N3, N7, S2, S3 for DMA connections
    if (curr == srcTileLoc &&
        deviceModel.isShimNOCTile(srcTileLoc.col, srcTileLoc.row)) {
      // shim DMAs at start of flows
      auto shimMux = std::pair(Connect::Interconnect::SHIMMUX, srcTileLoc.col);
      if (srcBundle == StrmSwPortType::DMA) {
        // must be either DMA0 -> N3 or DMA1 -> N7
        shimCh = srcChannel == 0 ? 3 : 7;
        addConnection(curr, srcBundle, srcChannel, StrmSwPortType::NORTH,
                      shimCh, shimMux.first, shimMux.second);
      } else if (srcBundle == StrmSwPortType::NOC) {
        // must be NOC0/NOC1 -> N2/N3 or NOC2/NOC3 -> N6/N7
        shimCh = srcChannel >= 2 ? srcChannel + 4 : srcChannel + 2;
        addConnection(curr, srcBundle, srcChannel, StrmSwPortType::NORTH,
                      shimCh, shimMux.first, shimMux.second);
      }
    }

    auto sw = std::make_tuple(Connect::Interconnect::SWB, curr.col, curr.row);
    assert(setting.srcs.size() == setting.dsts.size());
    for (size_t i = 0; i < setting.srcs.size(); i++) {
      Port src = setting.srcs[i];
      Port dst = setting.dsts[i];
      StrmSwPortType bundle = dst.bundle;
      int channel = dst.channel;
      // handle special shim connectivity
      if (curr == srcTileLoc &&
          deviceModel.isShimNOCorPLTile(srcTileLoc.col, srcTileLoc.row)) {
        addConnection(curr, StrmSwPortType::SOUTH, shimCh, bundle, channel,
                      std::get<0>(sw), std::get<1>(sw), std::get<2>(sw));
      } else if (deviceModel.isShimNOCorPLTile(curr.col, curr.row) &&
                 (bundle == StrmSwPortType::DMA ||
                  bundle == StrmSwPortType::NOC)) {
        auto shimMux = std::make_pair(Connect::Interconnect::SHIMMUX, curr.col);
        shimCh = channel;
        if (deviceModel.isShimNOCTile(curr.col, curr.row)) {
          // shim DMAs at end of flows
          if (bundle == StrmSwPortType::DMA) {
            // must be either N2 -> DMA0 or N3 -> DMA1
            shimCh = channel == 0 ? 2 : 3;
            addConnection(curr, StrmSwPortType::NORTH, shimCh, bundle, channel,
                          shimMux.first, shimMux.second);
          } else if (bundle == StrmSwPortType::NOC) {
            // must be either N2/3/4/5 -> NOC0/1/2/3
            shimCh = channel + 2;
            addConnection(curr, StrmSwPortType::NORTH, shimCh, bundle, channel,
                          shimMux.first, shimMux.second);
          }
        }
        addConnection(curr, src.bundle, src.channel, StrmSwPortType::SOUTH,
                      shimCh, std::get<0>(sw), std::get<1>(sw),
                      std::get<2>(sw));
      } else {
        addConnection(curr, src.bundle, src.channel, bundle, channel,
                      std::get<0>(sw), std::get<1>(sw), std::get<2>(sw));
      }
    }
  }
  // sort for deterministic order in IR
  for (auto &[_, conns] : connections) std::sort(conns.begin(), conns.end());

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
  TileLoc neighbourTile{-1, -1};
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
  assert(neighbourTile.col != -1 && neighbourTile.row != -1);

  int neighbourSourceChannel = currDestChannel;
  for (const auto &[sbNode, setting] : settings) {
    TileLoc tile = {sbNode.col, sbNode.row};
    if (tile == neighbourTile) {
      assert(setting.srcs.size() == setting.dsts.size());
      for (size_t i = 0; i < setting.srcs.size(); i++) {
        Port src = setting.srcs[i];
        Port dest = setting.dsts[i];
        if ((src.bundle == neighbourSourceBundle) &&
            (src.channel == neighbourSourceChannel)) {
          if (existsPathToDest(settings, neighbourTile, dest.bundle,
                               dest.channel, finalTile, finalDestBundle,
                               finalDestChannel)) {
            return true;
          }
        }
      }
    }
  }

  return false;
}

std::tuple<SlaveGroupsT, SlaveMasksT> emitSlaveGroupsAndMasksRoutingConfig(
    ArrayRef<PhysPortAndID> slavePorts, const PacketFlowMapT &packetFlows) {
  // Convert packet flow map into a map from src 'port and id's to destination
  // ports, so that multiple flows with different packet IDs, but the same
  // ports, can be merged.
  DenseMap<PhysPortAndID, llvm::SetVector<PhysPort>> physPortAndIDToPhysPort;
  for (auto &&[src, dsts] : packetFlows) {
    SmallVector<PhysPort> physPorts =
        llvm::map_to_vector(dsts, [](const PhysPortAndID &physPortAndID) {
          return physPortAndID.physPort;
        });
    physPortAndIDToPhysPort[src].insert(physPorts.begin(), physPorts.end());
  }
  // Compute mask values
  // Merging as many stream flows as possible
  // The flows must originate from the same source port and have different IDs
  // Two flows can be merged if they share the same destinations
  SlaveGroupsT slaveGroups;
  SmallVector<PhysPortAndID> workList(slavePorts.begin(), slavePorts.end());
  while (!workList.empty()) {
    PhysPortAndID slave1 = workList.pop_back_val();
    Port slavePort1 = slave1.physPort.port;

    bool foundgroup = false;
    for (auto &group : slaveGroups) {
      PhysPortAndID slave2 = group.front();
      if (Port slavePort2 = slave2.physPort.port; slavePort1 != slavePort2)
        continue;

      const llvm::SetVector<PhysPort> &dests1 =
          physPortAndIDToPhysPort.at(slave1);
      const llvm::SetVector<PhysPort> &dests2 =
          physPortAndIDToPhysPort.at(slave2);
      if (dests1.size() != dests2.size()) continue;
      if (std::all_of(dests1.begin(), dests1.end(),
                      [&dests2](const PhysPort &dest1) {
                        return dests2.count(dest1);
                      })) {
        group.push_back(slave1);
        foundgroup = true;
        break;
      }
    }

    if (!foundgroup) {
      slaveGroups.emplace_back(std::vector<PhysPortAndID>{slave1});
    }
  }

  SlaveMasksT slaveMasks;
  for (const auto &group : slaveGroups) {
    // Iterate over all the ID values in a group
    // If bit n-th (n <= 5) of an ID value differs from bit n-th of another ID
    // value, the bit position should be "don't care", and we will set the
    // mask bit of that position to 0
    int mask[5] = {-1, -1, -1, -1, -1};
    for (PhysPortAndID port : group) {
      for (int i = 0; i < 5; i++) {
        if (mask[i] == -1) {
          mask[i] = port.id >> i & 0x1;
        } else if (mask[i] != (port.id >> i & 0x1)) {
          // found bit difference --> mark as "don't care"
          mask[i] = 2;
        }
      }
    }

    int maskValue = 0;
    for (int i = 4; i >= 0; i--) {
      if (mask[i] == 2) {
        // don't care
        mask[i] = 0;
      } else {
        mask[i] = 1;
      }
      maskValue = (maskValue << 1) + mask[i];
    }
    for (PhysPortAndID port : group) slaveMasks[port] = maskValue;
  }

  // sort for deterministic IR output
  for (auto &item : slaveGroups) std::sort(item.begin(), item.end());
  std::sort(slaveGroups.begin(), slaveGroups.end());
  return std::make_tuple(slaveGroups, slaveMasks);
}

/// Given switchbox configuration data produced by the router, emit
/// configuration data for packet routing along those same switchboxes.
FailureOr<std::tuple<MasterSetsT, SlaveAMSelsT>> emitPacketRoutingConfiguration(
    const AMDAIEDeviceModel &deviceModel, const PacketFlowMapT &packetFlows) {
  // The generator for finding `(arbiter, msel)` pairs for packet flow
  // connections.
  AMSelGenerator amselGenerator;

  SmallVector<std::pair<PhysPortAndID, llvm::SetVector<PhysPortAndID>>>
      sortedPacketFlows(packetFlows.begin(), packetFlows.end());

  // To get determinsitic behaviour
  std::sort(
      sortedPacketFlows.begin(), sortedPacketFlows.end(),
      [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

  // A map from Tile and master selectValue to the ports targeted by that
  // master select.
  std::map<std::pair<TileLoc, std::pair<uint8_t, uint8_t>>, std::set<Port>>
      masterAMSels;
  SlaveAMSelsT slaveAMSels;
  for (const auto &[physPortAndID, packetFlowports] : sortedPacketFlows) {
    // The Source Tile of the flow
    TileLoc tileLoc = physPortAndID.physPort.tileLoc;
    // Make sure the generator for the tile is initialized with the correct
    // number of arbiters and msels.
    uint8_t numArbiters =
        1 + deviceModel.getStreamSwitchArbiterMax(tileLoc.col, tileLoc.row);
    uint8_t numMSels =
        1 + deviceModel.getStreamSwitchMSelMax(tileLoc.col, tileLoc.row);
    amselGenerator.initTileIfNotExists(tileLoc, numArbiters, numMSels);
    SmallVector<PhysPortAndID> dstPorts(packetFlowports.begin(),
                                        packetFlowports.end());
    if (failed(
            amselGenerator.addConnection(tileLoc, physPortAndID, dstPorts))) {
      return failure();
    }
  }
  if (failed(amselGenerator.solve())) return failure();

  for (const auto &[physPortAndID, packetFlowports] : sortedPacketFlows) {
    // The Source Tile of the flow
    TileLoc tileLoc = physPortAndID.physPort.tileLoc;
    std::optional<std::pair<uint8_t, uint8_t>> maybeAMSel =
        amselGenerator.getAMSel(tileLoc, physPortAndID);
    if (!maybeAMSel) return failure();
    auto [arbiter, msel] = maybeAMSel.value();

    for (PhysPortAndID dest : packetFlowports) {
      masterAMSels[{tileLoc, {arbiter, msel}}].insert(dest.physPort.port);
    }
    slaveAMSels[physPortAndID] = {arbiter, msel};
  }

  // Compute the master set IDs
  MasterSetsT mastersets;
  for (const auto &[physPort, ports] : masterAMSels) {
    for (Port port : ports) {
      mastersets[PhysPort{physPort.first, port}].push_back(physPort.second);
    }
  }

  // sort for deterministic IR output
  for (auto &[_, amsels] : mastersets) std::sort(amsels.begin(), amsels.end());
  return std::make_tuple(mastersets, slaveAMSels);
}

/// ============================= BEGIN ==================================
/// ================== stringification utils =============================
/// ======================================================================

std::string to_string(const SwitchSetting &setting) {
  return "SwitchSetting(" +
         llvm::join(
             llvm::map_range(setting.srcs,
                             [](const Port &port) { return to_string(port); }),
             ", ") +
         +" -> " + "{" +
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
                        [](const llvm::detail::DenseMapPair<TileLoc,
                                                            SwitchSetting> &p) {
                          return to_string(p.getFirst()) + ": " +
                                 to_string(p.getSecond());
                        }),
                    ", ") +
         ")";
}

STRINGIFY_2TUPLE_STRUCT(Port, bundle, channel)
STRINGIFY_2TUPLE_STRUCT(Connect, src, dst)
STRINGIFY_2TUPLE_STRUCT(PathEndPoint, tileLoc, port)
STRINGIFY_2TUPLE_STRUCT(PhysPort, tileLoc, port)
STRINGIFY_2TUPLE_STRUCT(PhysPortAndID, physPort, id)

BOTH_OSTREAM_OPS_FORALL_ROUTER_TYPES(OSTREAM_OP_DEFN, BOTH_OSTREAM_OP)

}  // namespace mlir::iree_compiler::AMDAIE
