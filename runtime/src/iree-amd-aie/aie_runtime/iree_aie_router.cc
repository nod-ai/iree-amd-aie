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
  // The number of fixed, pre-existing packet streams using each channel
  // before the pathfinder algorithm starts.
  std::vector<std::vector<int>> fixedPacketFlowCount;
  // only sharing the channel with the same packet group id
  std::vector<std::vector<int>> packetGroupId;

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
    fixedPacketFlowCount.resize(srcPorts.size(),
                                std::vector<int>(dstPorts.size(), 0));
    packetGroupId.resize(srcPorts.size(), std::vector<int>(dstPorts.size(), 0));
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
  int packetGroupId;
  PathEndPoint src;
  std::vector<PathEndPoint> dsts;
};

struct RouterImpl {
  RouterImpl(int maxCol, int maxRow) : maxCol(maxCol), maxRow(maxRow) {}
  // The dimensions (columns and rows) of the AIE device to be routed.
  // These values may be smaller than the actual device size if only a portion
  // of the device is intended to be routed.
  int maxCol, maxRow;

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

Router::Router(int maxCol, int maxRow) {
  impl = new RouterImpl(maxCol, maxRow);
}
Router::~Router() { delete impl; }

void Router::initialize(const AMDAIEDeviceModel &deviceModel) {
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

  for (int row = 0; row <= impl->maxRow; row++) {
    for (int col = 0; col <= impl->maxCol; col++) {
      maxChannels.clear();
      // connections within the same switchbox
      intraconnect(col, row);

      // connections between switchboxes
      if (row > 0) {
        // from south to north
        interconnect(col, row, col, row - 1, StrmSwPortType::SOUTH,
                     StrmSwPortType::NORTH);
      }
      if (row < impl->maxRow) {
        // from north to south
        interconnect(col, row, col, row + 1, StrmSwPortType::NORTH,
                     StrmSwPortType::SOUTH);
      }
      if (col > 0) {
        // from east to west
        interconnect(col, row, col - 1, row, StrmSwPortType::WEST,
                     StrmSwPortType::EAST);
      }
      if (col < impl->maxCol) {
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
  for (auto &[_, src, dsts] : impl->flows) {
    if (src.tileLoc == srcCoords && src.port == srcPort) {
      dsts.emplace_back(PathEndPoint{dstCoords, dstPort});
      return;
    }
  }

  // Assign a group ID for packet flows
  // any overlapping in source/destination will lead to the same group ID
  // channel sharing will happen within the same group ID
  // for circuit flows, group ID is always -1, and no channel sharing
  int packetGroupId = -1;
  if (isPacketFlow) {
    bool found = false;
    for (auto &[existingId, src, dsts] : impl->flows) {
      if (src.tileLoc == srcCoords && src.port == srcPort) {
        packetGroupId = existingId;
        found = true;
        break;
      }
      for (auto &dst : dsts) {
        if (dst.tileLoc == dstCoords && dst.port == dstPort) {
          packetGroupId = existingId;
          found = true;
          break;
        }
      }
      packetGroupId = std::max(packetGroupId, existingId);
    }
    if (!found) {
      packetGroupId++;
    }
  }

  // If no existing flow was found with this source, create a new flow.
  impl->flows.push_back(
      Flow{packetGroupId, PathEndPoint{srcCoords, srcPort},
           std::vector<PathEndPoint>{PathEndPoint{dstCoords, dstPort}}});
}

// Keep track of connections already used in the AIE; Pathfinder algorithm will
// avoid using these.
bool Router::addFixedCircuitConnection(
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

bool Router::addFixedPacketConnection(const PhysPort &srcPhyPort,
                                      const PhysPort &destPhyPort) {
  SwitchboxConnect &sb =
      impl->graph[std::make_pair(srcPhyPort.tileLoc, destPhyPort.tileLoc)];
  size_t i = std::distance(
      sb.srcPorts.begin(),
      std::find(sb.srcPorts.begin(), sb.srcPorts.end(), srcPhyPort.port));
  size_t j = std::distance(
      sb.dstPorts.begin(),
      std::find(sb.dstPorts.begin(), sb.dstPorts.end(), destPhyPort.port));
  // Could not find the source or destination port.
  if (i >= sb.srcPorts.size() || j >= sb.dstPorts.size()) return false;
  // Increment the fixed packet flow count.
  sb.fixedPacketFlowCount[i][j]++;
  // Populate to the neighboring switchboxes.
  if (srcPhyPort.tileLoc == destPhyPort.tileLoc) {
    int col = srcPhyPort.tileLoc.col;
    int row = srcPhyPort.tileLoc.row;
    int srcChannel = srcPhyPort.port.channel;
    int destChannel = destPhyPort.port.channel;
    // Populate to the neighboring SOUTH switchbox.
    if (row > 0) {
      if (srcPhyPort.port.bundle == StrmSwPortType::SOUTH &&
          !addFixedPacketConnection({{col, row - 1},
                                     {StrmSwPortType::NORTH, srcChannel},
                                     PhysPort::Direction::SRC},
                                    {{col, row},
                                     {StrmSwPortType::SOUTH, srcChannel},
                                     PhysPort::Direction::DST})) {
        return false;
      }
      if (destPhyPort.port.bundle == StrmSwPortType::SOUTH &&
          !addFixedPacketConnection({{col, row},
                                     {StrmSwPortType::SOUTH, destChannel},
                                     PhysPort::Direction::SRC},
                                    {{col, row - 1},
                                     {StrmSwPortType::NORTH, destChannel},
                                     PhysPort::Direction::DST})) {
        return false;
      }
    }
    // Populate to the neighboring NORTH switchbox.
    if (row < impl->maxRow) {
      if (srcPhyPort.port.bundle == StrmSwPortType::NORTH &&
          !addFixedPacketConnection({{col, row + 1},
                                     {StrmSwPortType::SOUTH, srcChannel},
                                     PhysPort::Direction::SRC},
                                    {{col, row},
                                     {StrmSwPortType::NORTH, srcChannel},
                                     PhysPort::Direction::DST})) {
        return false;
      }
      if (destPhyPort.port.bundle == StrmSwPortType::NORTH &&
          !addFixedPacketConnection({{col, row},
                                     {StrmSwPortType::NORTH, destChannel},
                                     PhysPort::Direction::SRC},
                                    {{col, row + 1},
                                     {StrmSwPortType::SOUTH, destChannel},
                                     PhysPort::Direction::DST})) {
        return false;
      }
    }
    // Populate to the neighboring WEST switchbox.
    if (col > 0) {
      if (srcPhyPort.port.bundle == StrmSwPortType::WEST &&
          !addFixedPacketConnection({{col - 1, row},
                                     {StrmSwPortType::EAST, srcChannel},
                                     PhysPort::Direction::SRC},
                                    {{col, row},
                                     {StrmSwPortType::WEST, srcChannel},
                                     PhysPort::Direction::DST})) {
        return false;
      }
      if (destPhyPort.port.bundle == StrmSwPortType::WEST &&
          !addFixedPacketConnection({{col, row},
                                     {StrmSwPortType::WEST, destChannel},
                                     PhysPort::Direction::SRC},
                                    {{col - 1, row},
                                     {StrmSwPortType::EAST, destChannel},
                                     PhysPort::Direction::DST})) {
        return false;
      }
    }
    // Populate to the neighboring EAST switchbox.
    if (col < impl->maxCol) {
      if (srcPhyPort.port.bundle == StrmSwPortType::EAST &&
          !addFixedPacketConnection({{col + 1, row},
                                     {StrmSwPortType::WEST, srcChannel},
                                     PhysPort::Direction::SRC},
                                    {{col, row},
                                     {StrmSwPortType::EAST, srcChannel},
                                     PhysPort::Direction::DST})) {
        return false;
      }
      if (destPhyPort.port.bundle == StrmSwPortType::EAST &&
          !addFixedPacketConnection({{col, row},
                                     {StrmSwPortType::EAST, destChannel},
                                     PhysPort::Direction::SRC},
                                    {{col + 1, row},
                                     {StrmSwPortType::WEST, destChannel},
                                     PhysPort::Direction::DST})) {
        return false;
      }
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

  // group flows based on packetGroupId
  std::map<int, std::vector<Flow>> groupedFlows;
  for (auto &f : impl->flows) {
    groupedFlows[f.packetGroupId].push_back(f);
  }

  int iterationCount = -1;
  int illegalEdges = 0;
  [[maybe_unused]] int totalPathLength = 0;
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
          // If there is no fixed, pre-routed packet flow,
          // `fixedPacketFlowCount` will be just 0.
          sb.packetFlowCount[i][j] = sb.fixedPacketFlowCount[i][j];
          sb.packetGroupId[i][j] = -1;
        }
      }
    }

    for (const auto &[_, flows] : groupedFlows) {
      for (const auto &[packetGroupId, src, dsts] : flows) {
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
            if (packetGroupId >= 0 &&
                (sb.packetGroupId[i][j] == -1 ||
                 sb.packetGroupId[i][j] == packetGroupId)) {
              for (size_t k = 0; k < sb.srcPorts.size(); k++) {
                for (size_t l = 0; l < sb.dstPorts.size(); l++) {
                  if (k == i || l == j) {
                    sb.packetGroupId[k][l] = packetGroupId;
                  }
                }
              }
              sb.packetFlowCount[i][j]++;
              // maximum packet stream sharing per channel
              if (sb.packetFlowCount[i][j] >= MAX_PACKET_STREAM_CAPACITY) {
                sb.packetFlowCount[i][j] = 0;
                sb.usedCapacity[i][j]++;
              }
            } else {
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
            sb.bumpDemand(i, j);
          }
        }
      }
    }

    for (auto &[_, sb] : impl->graph) {
      for (size_t i = 0; i < sb.srcPorts.size(); i++) {
        for (size_t j = 0; j < sb.dstPorts.size(); j++) {
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

/// Transform outputs produced by the router into representations (structs)
/// that directly map to stream switch configuration ops (soon-to-be aie-rt
/// calls). Namely pairs of (switchbox, internal connections).
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
    std::optional<StrmSwPortType> newSrcBundle = std::nullopt;
    std::optional<int> newSrcChannel = std::nullopt;
    // TODO: must reserve N3, N7, S2, S3 for DMA connections
    if (std::optional<std::pair<StrmSwPortType, uint8_t>> mappedShimMuxPort =
            deviceModel.getShimMuxPortMappingForDmaOrNoc(srcBundle, srcChannel,
                                                         DMAChannelDir::MM2S);
        mappedShimMuxPort.has_value() && curr == srcTileLoc &&
        deviceModel.isShimNOCTile(srcTileLoc.col, srcTileLoc.row)) {
      // Check for special shim connectivity at the start (based on
      // `srcTileLoc` and `srcBundle`) of the flow. Shim DMAs/NOCs require
      // special handling.
      newSrcBundle = mappedShimMuxPort->first;
      newSrcChannel = mappedShimMuxPort->second;
      // The connection now is `src.bundle/src.channel` ->
      // `newSrcBundle/newSrcChannel` -> `destBundle/destChannel`.
      // Here, we connect the first half, and the second half will be
      // dealt with later.
      addConnection(curr, srcBundle, srcChannel, newSrcBundle.value(),
                    newSrcChannel.value(), Connect::Interconnect::SHIMMUX,
                    curr.col, curr.row);
      // Flip the bundle direction as we are going from MM2S to S2MM.
      newSrcBundle = getConnectingBundle(newSrcBundle.value());
    }

    assert(setting.srcs.size() == setting.dsts.size());
    for (size_t i = 0; i < setting.srcs.size(); i++) {
      Port src = setting.srcs[i];
      Port dst = setting.dsts[i];
      StrmSwPortType destBundle = dst.bundle;
      int destChannel = dst.channel;
      // Check for special shim connectivity at the start (based on
      // `srcTileLoc` `newSrcBundle`) or at the end (based on `curr` and
      // `destBundle`) of the flow. Shim DMAs/NOCs require special handling.
      if (curr == srcTileLoc &&
          deviceModel.isShimNOCorPLTile(srcTileLoc.col, srcTileLoc.row) &&
          newSrcBundle.has_value() && newSrcChannel.has_value()) {
        // Complete the second half of `src.bundle/src.channel` ->
        // `newSrcBundle/newSrcChannel` -> `destBundle/destChannel`.
        addConnection(curr, newSrcBundle.value(), newSrcChannel.value(),
                      destBundle, destChannel, Connect::Interconnect::SWB,
                      curr.col, curr.row);
      } else if (std::optional<std::pair<StrmSwPortType, uint8_t>>
                     mappedShimMuxPort =
                         deviceModel.getShimMuxPortMappingForDmaOrNoc(
                             destBundle, destChannel, DMAChannelDir::S2MM);
                 mappedShimMuxPort &&
                 deviceModel.isShimNOCorPLTile(curr.col, curr.row)) {
        StrmSwPortType newDestBundle = mappedShimMuxPort->first;
        int newDestChannel = mappedShimMuxPort->second;
        // The connection now is `src.bundle/src.channel` ->
        // `newDestBundle/newDestChannel` -> `destBundle/destChannel`.
        addConnection(curr, newDestBundle, newDestChannel, destBundle,
                      destChannel, Connect::Interconnect::SHIMMUX, curr.col,
                      curr.row);
        addConnection(curr, src.bundle, src.channel,
                      getConnectingBundle(newDestBundle), newDestChannel,
                      Connect::Interconnect::SWB, curr.col, curr.row);
      } else {
        // Otherwise, add the regular switchbox connection.
        addConnection(curr, src.bundle, src.channel, destBundle, destChannel,
                      Connect::Interconnect::SWB, curr.col, curr.row);
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

/// Update the mask value for all the IDs in the group.
void updateGroupMask(const PhysPort &slavePort, std::set<int> &group,
                     std::map<PhysPortAndID, int> &slaveMasks) {
  // Iterate over all the ID values in a group.If bit n-th (n <= 5) of an ID
  // value differs from bit n-th of another ID value, the bit position should be
  // "don't care", and we will set the mask bit of that position to 0.
  int mask[5] = {-1, -1, -1, -1, -1};
  for (int id : group) {
    for (int i = 0; i < 5; i++) {
      if (mask[i] == -1) {
        mask[i] = id >> i & 0x1;
      } else if (mask[i] != (id >> i & 0x1)) {
        // Found bit difference --> mark as "don't care".
        mask[i] = 2;
      }
    }
  }

  int maskValue = 0;
  for (int i = 4; i >= 0; i--) {
    if (mask[i] == 2) {
      // Don't care.
      mask[i] = 0;
    } else {
      mask[i] = 1;
    }
    maskValue = (maskValue << 1) + mask[i];
  }
  for (int id : group) slaveMasks[PhysPortAndID{slavePort, id}] = maskValue;
}

/// Sort groups primarily by the mask value. If the mask values are the same,
/// sort by the group itself.
///
/// Example:
/// Consider two slave groups, A and B, that share the same `physPort`
/// (i.e., the same `tileLoc`, `bundle`, and `channel`). These groups
/// will later be merged into a single `packet_rules` operation,
/// where each group contributes a `packet_rule` entry. The order
/// of these entries is critical because the first matching `packet_rule`
/// takes precedence.
///
/// - `Group A` contains IDs `{0x3, 0x4, 0x5}` with `mask = 0x18`,
///   and defines a `packet_rule`: `(ID & 0x18) == 0x00`.
/// - `Group B` contains ID `{0x2}` with `mask = 0x1F`,
///   and defines a `packet_rule`: `(ID & 0x1F) == 0x02`.
///
/// In this case, `Group B`'s `packet_rule` must precede `Group A`'s
/// within the `packet_rules` operation. Otherwise, ID `0x02`
/// would incorrectly match `(ID & 0x18) == 0x00`, leading to incorrect
/// behavior.
///
/// Sorting Rationale:
/// A larger mask can represent a stricter rule, as it constrains more bits.
/// Therefore, rules with larger masks should be placed first to prevent
/// broader (less strict) rules from matching unintended IDs.
void sortGroupsByMask(PhysPort slavePort, std::vector<std::set<int>> &groups,
                      const std::map<PhysPortAndID, int> &slaveMasks) {
  auto sortByMask = [&slaveMasks, &slavePort](auto &lhs, auto &rhs) {
    PhysPortAndID lhsSlavePort{slavePort, *lhs.begin()};
    PhysPortAndID rhsSlavePort{slavePort, *rhs.begin()};
    // Sort by mask in descending order.
    if (slaveMasks.at(lhsSlavePort) != slaveMasks.at(rhsSlavePort))
      return slaveMasks.at(lhsSlavePort) > slaveMasks.at(rhsSlavePort);
    // Sort by the group itself in ascending order.
    return lhs < rhs;
  };
  std::sort(groups.begin(), groups.end(), sortByMask);
}

/// Verifies the correctness of ID groupings before putting them into a single
/// `packet_rules` set. Each group contributes a `packet_rule` entry, and this
/// function checks if any ID in a later group incorrectly matches a preceding
/// group's masked ID.
///
/// Example:
/// Consider three groups: A, B, and C.
/// - `Group A`: Contains IDs `{0x0}` with `mask = 0x1F`.
///   - Packet rule: `(ID & 0x1F) ?= (0x0 & 0x1F)`.
/// - `Group B`: Contains IDs `{0x1, 0x2, 0x3, 0x4, 0x5}` with `mask = 0x18`.
///   - Packet rule: `(ID & 0x18) ?= (0x1 & 0x18)`.
/// - `Group C`: Contains IDs `{0x6, 0x7}` with `mask = 0x18`.
///   - Packet rule: `(ID & 0x18) ?= (0x6 & 0x18)`.
///
/// ID `0x6` belongs to `Group C`, however, due to the limitation of masking, it
/// matches both Group B and Group C's rules. Since Group B precedes Group C,
/// and `packet_rule` entries are evaluated in order, the function returns
/// `false` to indicate an invalid grouping.
bool verifyGroupsByMask(PhysPort slavePort,
                        const std::vector<std::set<int>> &groups,
                        const std::map<PhysPortAndID, int> &slaveMasks) {
  for (size_t i = 0; i < groups.size(); ++i) {
    int iPktId = *groups[i].begin();
    int iMask = slaveMasks.at({slavePort, iPktId});
    int iMaskedId = iPktId & iMask;
    for (size_t j = i + 1; j < groups.size(); ++j) {
      for (int jPktId : groups[j]) {
        if ((jPktId & iMask) == iMaskedId) return false;
      }
    }
  }
  return true;
}

std::tuple<SlaveGroupsT, SlaveMasksT> emitSlaveGroupsAndMasksRoutingConfig(
    ArrayRef<PhysPortAndID> slavePorts, const PacketFlowMapT &packetFlows,
    ArrayRef<PhysPortAndID> existingSlavePorts,
    const PacketFlowMapT &existingPacketFlows) {
  // Convert packet flow map into a map from src 'port and id's to destination
  // ports, so that multiple flows with different packet IDs, but the same
  // ports, can be merged.
  DenseMap<PhysPortAndID, llvm::SetVector<PhysPort>> physPortAndIDToPhysPort;
  // Put existing packet flows with the new ones together, with existing ones
  // first to respect their relative order.
  PacketFlowMapT allPacketFlows = existingPacketFlows;
  allPacketFlows.insert(packetFlows.begin(), packetFlows.end());
  for (auto &&[src, dsts] : allPacketFlows) {
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
  SlaveMasksT slaveMasks;
  // Put existing slave ports with the new ones together, with existing ones
  // first to respect their relative order.
  SmallVector<PhysPortAndID> workList(existingSlavePorts.begin(),
                                      existingSlavePorts.end());
  workList.append(slavePorts.begin(), slavePorts.end());
  while (!workList.empty()) {
    PhysPortAndID slave1 = workList.pop_back_val();
    std::optional<int> matchedGroupIdx;
    // Try to find a matching group that can be merged with.
    std::vector<std::set<int>> &groups = slaveGroups[slave1.physPort];
    for (size_t i = 0; i < groups.size(); ++i) {
      PhysPortAndID slave2 = {slave1.physPort, *groups[i].begin()};
      const llvm::SetVector<PhysPort> &dests1 =
          physPortAndIDToPhysPort.at(slave1);
      const llvm::SetVector<PhysPort> &dests2 =
          physPortAndIDToPhysPort.at(slave2);
      if (dests1.size() != dests2.size()) continue;
      if (std::all_of(dests1.begin(), dests1.end(),
                      [&dests2](const PhysPort &dest1) {
                        return dests2.count(dest1);
                      })) {
        // Found a matching group.
        matchedGroupIdx = i;
        break;
      }
    }
    // Attempt to merge, and verify that the merged group is still valid.
    if (matchedGroupIdx.has_value()) {
      // Make a copy of the groups in case the merge is invalid.
      std::vector<std::set<int>> groupsCopy = groups;
      std::set<int> &group = groups[matchedGroupIdx.value()];
      // Merge `slave1.id` into the group.
      group.insert(slave1.id);
      updateGroupMask(slave1.physPort, group, slaveMasks);
      sortGroupsByMask(slave1.physPort, groups, slaveMasks);
      // If the merge is valid, simply continue the while loop on `workList`.
      if (verifyGroupsByMask(slave1.physPort, groups, slaveMasks)) continue;
      // Not a valid merge, so revert the changes on `groups` and `slaveMasks`.
      slaveGroups[slave1.physPort] = groupsCopy;
      updateGroupMask(slave1.physPort, group, slaveMasks);
    }
    // No matching group was found, create a new group.
    std::set<int> group = {slave1.id};
    groups.emplace_back(group);
    updateGroupMask(slave1.physPort, group, slaveMasks);
    sortGroupsByMask(slave1.physPort, groups, slaveMasks);
  }
  return std::make_tuple(slaveGroups, slaveMasks);
}

/// Given switchbox configuration data produced by the router, emit
/// configuration data for packet routing along those same switchboxes.
FailureOr<std::tuple<MasterSetsT, SlaveAMSelsT>> emitPacketRoutingConfiguration(
    const AMDAIEDeviceModel &deviceModel, const PacketFlowMapT &packetFlows,
    const PacketFlowMapT &existingPacketFlows) {
  // The generator for finding `(arbiter, msel)` pairs for packet flow
  // connections.
  AMSelGenerator amselGenerator;

  // Put existing packet flows with the new ones together, with existing ones
  // first to respect their relative order.
  SmallVector<std::pair<PhysPortAndID, llvm::SetVector<PhysPortAndID>>>
      sortedPacketFlows(existingPacketFlows.begin(), existingPacketFlows.end());
  size_t existingPacketFlowsSize = existingPacketFlows.size();
  sortedPacketFlows.append(packetFlows.begin(), packetFlows.end());

  // Sort for deterministic AMSel allocation.
  // Note that `first.id` is intentionally excluded from the sorting key.
  // This is because `existingPacketFlows` may contain a superset of actual
  // packet IDs, as they are inferred from packet rule masks and values.
  // Including `first.id` in the comparison could lead to inconsistent
  // ordering between multiple runs of the router.
  auto sortByLocPortPktid = [](const auto &lhs, const auto &rhs) {
    return std::tuple(lhs.first.physPort.tileLoc.col,
                      lhs.first.physPort.tileLoc.row, lhs.first.physPort.port,
                      std::vector(lhs.second.begin(), lhs.second.end())) <
           std::tuple(rhs.first.physPort.tileLoc.col,
                      rhs.first.physPort.tileLoc.row, rhs.first.physPort.port,
                      std::vector(rhs.second.begin(), rhs.second.end()));
  };
  // Existing flows and new flows are sorted separately to maintain their
  // relative order within their respective groups.
  std::sort(sortedPacketFlows.begin(),
            sortedPacketFlows.begin() + existingPacketFlowsSize,
            sortByLocPortPktid);
  std::sort(sortedPacketFlows.begin() + existingPacketFlowsSize,
            sortedPacketFlows.end(), sortByLocPortPktid);

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
      mastersets[PhysPort{physPort.first, port, PhysPort::Direction::DST}]
          .push_back(physPort.second);
    }
  }

  // sort for deterministic IR output
  for (auto &[_, amsels] : mastersets) std::sort(amsels.begin(), amsels.end());
  return std::make_tuple(mastersets, slaveAMSels);
}

/// ============================= BEGIN ==================================
/// ================== stringification utils =============================
/// ======================================================================

std::string to_string(const PhysPort::Direction &direction) {
  switch (direction) {
    STRINGIFY_ENUM_CASE(PhysPort::Direction::SRC)
    STRINGIFY_ENUM_CASE(PhysPort::Direction::DST)
  }
  llvm::report_fatal_error("Unhandled PhysPortDirection case");
}

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
STRINGIFY_3TUPLE_STRUCT(PhysPort, tileLoc, port, direction)
STRINGIFY_2TUPLE_STRUCT(PhysPortAndID, physPort, id)

BOTH_OSTREAM_OPS_FORALL_ROUTER_TYPES(OSTREAM_OP_DEFN, BOTH_OSTREAM_OP)

}  // namespace mlir::iree_compiler::AMDAIE
