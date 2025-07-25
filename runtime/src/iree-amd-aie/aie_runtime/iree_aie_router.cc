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
  // The number of fixed, previously-routed packet streams using each channel
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
  PhysPort src;
  SmallVector<PhysPort> dsts;
};

struct RouterImpl {
  RouterImpl(int maxCol, int maxRow) : maxCol(maxCol), maxRow(maxRow) {}
  // Flows to be routed
  std::vector<Flow> flows;
  // Represent all routable paths as a graph
  // The key is a pair of TileIDs representing the connectivity from srcTile to
  // dstTile If srcTile == dstTile, it represents connections inside the same
  // switchbox otherwise, it represents connections (South, North, West, East)
  // accross two switchboxes
  std::map<std::pair<TileLoc, TileLoc>, SwitchboxConnect> graph;
  // Channels available in the network.
  // The key is a `PhysPort` representing the start of a path.
  // The value is a vector of `PhysPorts` representing the possible ends of
  // the path.
  std::map<PhysPort, SmallVector<PhysPort>> channels;
  /// The dimensions (columns and rows) of the AIE device to be routed.
  /// These values may be smaller than the actual device size if only a portion
  /// of the device is intended to be routed.
  int maxCol, maxRow;
  /// Map to get the neighbor tile location given a direction.
  static const llvm::SmallDenseMap<std::pair<StrmSwPortType, StrmSwPortType>,
                                   std::pair<int, int>>
      directionToDeltaCoordsMap;
};

const llvm::SmallDenseMap<std::pair<StrmSwPortType, StrmSwPortType>,
                          std::pair<int, int>>
    RouterImpl::directionToDeltaCoordsMap = {
        {{StrmSwPortType::SOUTH, StrmSwPortType::NORTH},
         {0, -1}},  // South to North, going down, row - 1
        {{StrmSwPortType::NORTH, StrmSwPortType::SOUTH},
         {0, 1}},  // North to South, going up, row + 1
        {{StrmSwPortType::WEST, StrmSwPortType::EAST},
         {-1, 0}},  // West to East, going left, col - 1
        {{StrmSwPortType::EAST, StrmSwPortType::WEST},
         {1, 0}}};  // East to West, going right, col + 1

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
      for (auto &[direction, deltaCoords] : impl->directionToDeltaCoordsMap) {
        int neighborCol = col + deltaCoords.first;
        int neighborRow = row + deltaCoords.second;
        // Check if the neighboring switchbox exists.
        if (neighborCol < 0 || neighborCol > impl->maxCol || neighborRow < 0 ||
            neighborRow > impl->maxRow) {
          continue;
        }
        interconnect(col, row, neighborCol, neighborRow, direction.first,
                     direction.second);
      }
    }
  }
}

void Router::addFlow(TileLoc srcCoords, Port srcPort, TileLoc dstCoords,
                     Port dstPort, bool isPacketFlow) {
  // check if a flow with this source already exists
  for (auto &[_, src, dsts] : impl->flows) {
    if (src.tileLoc == srcCoords && src.port == srcPort) {
      dsts.emplace_back(PhysPort{dstCoords, dstPort, PhysPort::Direction::DST});
      return;
    }
  }

  // Assign a group ID to packet flows:
  // - If the source or any destination overlaps with an existing flow, reuse
  // its group ID.
  // - Flows sharing the same group ID can share channels.
  // - For circuit flows, always assign group ID -1 (no channel sharing).
  int32_t packetGroupId = [&]() -> int32_t {
    if (!isPacketFlow) return -1;
    for (auto &[existingId, src, dsts] : impl->flows) {
      if (src.tileLoc == srcCoords && src.port == srcPort) return existingId;
      if (llvm::any_of(dsts, [&](const auto &dst) {
            return dst.tileLoc == dstCoords && dst.port == dstPort;
          })) {
        return existingId;
      }
    }
    return nextAvailablePacketGroupId++;
  }();

  // If no existing flow was found with this source, create a new flow.
  PhysPort srcPhysPort{srcCoords, srcPort, PhysPort::Direction::SRC};
  PhysPort dstPhysPort{dstCoords, dstPort, PhysPort::Direction::DST};
  impl->flows.push_back(
      Flow{packetGroupId, srcPhysPort, SmallVector<PhysPort>{dstPhysPort}});
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

/// Tracks previously-routed packet flows by setting
/// `fixedPacketFlowCount`. This ensures that the pathfinder algorithm does not
/// reuse the same channel for a new circuit flow. Reuse for new packet flows is
/// allowed, as packet IDs will differentiate them.
bool Router::addFixedPacketConnection(const PhysPort &srcPhyPort,
                                      const PhysPort &destPhyPort) {
  SwitchboxConnect &sb =
      impl->graph[std::make_pair(srcPhyPort.tileLoc, destPhyPort.tileLoc)];
  size_t srcPortIdx = std::distance(
      sb.srcPorts.begin(),
      std::find(sb.srcPorts.begin(), sb.srcPorts.end(), srcPhyPort.port));
  size_t destPortIdx = std::distance(
      sb.dstPorts.begin(),
      std::find(sb.dstPorts.begin(), sb.dstPorts.end(), destPhyPort.port));
  // Could not find the source or destination port, something is wrong.
  if (srcPortIdx >= sb.srcPorts.size() || destPortIdx >= sb.dstPorts.size())
    return false;
  // Increment the fixed packet flow count, to indicate there is a
  // previously-routed packet stream using this channel.
  sb.fixedPacketFlowCount[srcPortIdx][destPortIdx]++;
  // There are two types of `SwitchboxConnect` in `impl->graph`:
  // 1. intra-switchbox connections (`srcPhyPort.tileLoc ==
  // destPhyPort.tileLoc`), representing configurable routes within a single
  // switchbox.
  // 2. inter-switchbox connections (`srcPhyPort.tileLoc !=
  // destPhyPort.tileLoc`): representing unconfigurable connections between
  // neighboring switchboxes. For example, tile (0,0) North is always tied
  // to tile (0,1) South.
  // When handling intra-switchbox connections connections,
  // we must also propagate to inter-switchbox connections.
  if (srcPhyPort.tileLoc == destPhyPort.tileLoc) {
    int col = srcPhyPort.tileLoc.col;
    int row = srcPhyPort.tileLoc.row;
    int srcChannel = srcPhyPort.port.channel;
    int destChannel = destPhyPort.port.channel;
    for (auto &[direction, deltaCoords] : impl->directionToDeltaCoordsMap) {
      int neighborCol = col + deltaCoords.first;
      int neighborRow = row + deltaCoords.second;
      // Check if the neighboring switchbox exists.
      if (neighborCol < 0 || neighborCol > impl->maxCol || neighborRow < 0 ||
          neighborRow > impl->maxRow) {
        continue;
      }
      // Propagate to the neighboring switchbox from `srcPhyPort`.
      if (srcPhyPort.port.bundle == direction.first &&
          !addFixedPacketConnection({{neighborCol, neighborRow},
                                     {direction.second, srcChannel},
                                     PhysPort::Direction::SRC},
                                    {{col, row},
                                     {direction.first, srcChannel},
                                     PhysPort::Direction::DST})) {
        return false;
      }
      // Propagate to the neighboring switchbox from `destPhyPort`.
      if (destPhyPort.port.bundle == direction.first &&
          !addFixedPacketConnection({{col, row},
                                     {direction.first, destChannel},
                                     PhysPort::Direction::SRC},
                                    {{neighborCol, neighborRow},
                                     {direction.second, destChannel},
                                     PhysPort::Direction::DST})) {
        return false;
      }
    }
  }
  return true;
}

std::map<PhysPort, PhysPort> Router::dijkstraShortestPaths(PhysPort src) {
  // Use std::map instead of DenseMap because DenseMap doesn't let you
  // overwrite tombstones.
  std::map<PhysPort, double> distance;
  std::map<PhysPort, PhysPort> preds;
  std::map<PhysPort, uint64_t> indexInHeap;
  enum Color { WHITE, GRAY, BLACK };
  std::map<PhysPort, Color> colors;
  typedef d_ary_heap_indirect<
      /*Value=*/PhysPort, /*Arity=*/4,
      /*IndexInHeapPropertyMap=*/std::map<PhysPort, uint64_t>,
      /*DistanceMap=*/std::map<PhysPort, double> &,
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
            impl->channels[src].push_back(PhysPort{src.tileLoc, sb.dstPorts[j],
                                                   PhysPort::Direction::DST});
          }
        }
      }
      // connections to neighboring switchboxes
      for (const auto &[direction, deltaCoords] :
           impl->directionToDeltaCoordsMap) {
        TileLoc neighborCoords = {src.tileLoc.col + deltaCoords.first,
                                  src.tileLoc.row + deltaCoords.second};
        Port neighborPort = {direction.second, src.port.channel};
        if (impl->graph.count(std::make_pair(src.tileLoc, neighborCoords)) >
                0 &&
            src.port.bundle == getConnectingBundle(neighborPort.bundle)) {
          auto &sb = impl->graph[std::make_pair(src.tileLoc, neighborCoords)];
          if (std::find(sb.dstPorts.begin(), sb.dstPorts.end(), neighborPort) !=
              sb.dstPorts.end())
            impl->channels[src].push_back(
                {neighborCoords, neighborPort, PhysPort::Direction::DST});
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
std::optional<std::map<PhysPort, SwitchSettings>> Router::findPaths(
    const int maxIterations) {
  LLVM_DEBUG(llvm::dbgs() << "\t---Begin Pathfinder::findPaths---\n");
  std::map<PhysPort, SwitchSettings> routingSolution;
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
          // If there is no previously-routed packet flow,
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
        DenseSet<PhysPort> processed;
        std::map<PhysPort, PhysPort> preds = dijkstraShortestPaths(src);

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
    for (const auto &[PhysPort, switchSetting] : routingSolution) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\t\t\tFlow starting at (" << PhysPort.tileLoc.col << ","
                 << PhysPort.tileLoc.row << "):\t");
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
    const std::map<PhysPort, SwitchSettings> &flowSolutions,
    const PhysPort &srcPoint, const AMDAIEDeviceModel &deviceModel) {
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
    // Check the source (based on `srcTileLoc` and `srcBundle`) of the flow.
    // Shim DMAs/NOCs require special handling.
    std::optional<StrmSwPortType> newSrcBundle = std::nullopt;
    std::optional<int> newSrcChannel = std::nullopt;
    // TODO: must reserve N3, N7, S2, S3 for DMA connections
    if (std::optional<std::pair<StrmSwPortType, uint8_t>> mappedShimMuxPort =
            deviceModel.getShimMuxPortMappingForDmaOrNoc(srcBundle, srcChannel,
                                                         DMAChannelDir::MM2S);
        mappedShimMuxPort.has_value() && curr == srcTileLoc &&
        deviceModel.isShimNOCTile(srcTileLoc.col, srcTileLoc.row)) {
      newSrcBundle = mappedShimMuxPort->first;
      newSrcChannel = mappedShimMuxPort->second;
      // The connection is updated as: `srcBundle/srcChannel` ->
      // `newSrcBundle/newSrcChannel` -> `destBundle/destChannel`. The following
      // line establishes the first half of the connection; the second half will
      // be handled later.
      addConnection(curr, srcBundle, srcChannel, newSrcBundle.value(),
                    newSrcChannel.value(), Connect::Interconnect::SHIMMUX,
                    curr.col, curr.row);
    }

    assert(setting.srcs.size() == setting.dsts.size());
    for (size_t i = 0; i < setting.srcs.size(); i++) {
      Port src = setting.srcs[i];
      Port dst = setting.dsts[i];
      StrmSwPortType destBundle = dst.bundle;
      int destChannel = dst.channel;
      if (newSrcBundle.has_value() && newSrcChannel.has_value()) {
        // Complete the second half of `src.bundle/src.channel` ->
        // `newSrcBundle/newSrcChannel` -> `destBundle/destChannel`.
        // `getConnectingBundle` is used to update bundle direction from shim
        // mux to shim switchbox.
        addConnection(curr, getConnectingBundle(newSrcBundle.value()),
                      newSrcChannel.value(), destBundle, destChannel,
                      Connect::Interconnect::SWB, curr.col, curr.row);
      } else if (std::optional<std::pair<StrmSwPortType, uint8_t>>
                     mappedShimMuxPort =
                         deviceModel.getShimMuxPortMappingForDmaOrNoc(
                             destBundle, destChannel, DMAChannelDir::S2MM);
                 mappedShimMuxPort &&
                 deviceModel.isShimNOCTile(curr.col, curr.row)) {
        // Check for special shim connectivity at the destination (based on
        // `curr` and `destBundle`) of the flow. Shim DMAs/NOCs require special
        // handling.
        StrmSwPortType newDestBundle = mappedShimMuxPort->first;
        int newDestChannel = mappedShimMuxPort->second;
        // The connection is updated as: `src.bundle/src.channel` ->
        // `newDestBundle/newDestChannel` -> `destBundle/destChannel`.
        // `getConnectingBundle` is used to update bundle direction from shim
        // mux to shim switchbox.
        addConnection(curr, src.bundle, src.channel,
                      getConnectingBundle(newDestBundle), newDestChannel,
                      Connect::Interconnect::SWB, curr.col, curr.row);
        addConnection(curr, newDestBundle, newDestChannel, destBundle,
                      destChannel, Connect::Interconnect::SHIMMUX, curr.col,
                      curr.row);
      } else {
        // Otherwise, add the regular switchbox connection.
        addConnection(curr, src.bundle, src.channel, destBundle, destChannel,
                      Connect::Interconnect::SWB, curr.col, curr.row);
      }
    }
  }
  // Sort for deterministic order in IR.
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

/// Generate the mask value for all the IDs in the group.
/// Iterate over all the ID values in a group. If the i-th bit (i <=
/// `numMaskBits`) of an ID value differs from the i-th bit of another ID value,
/// the bit position should be "don't care", and the mask value should be 0 at
/// that bit position. Otherwise, the mask value should be 1 at that bit
/// position.
///
/// Example:
/// Consider a group of IDs: {0x1, 0x2, 0x3} and `numMaskBits`=5. Counting from
/// the LSB,
/// - 1st bit of 0x1 is 1, 1st bit of 0x2 is 0, and 1st bit of 0x3 is 1;
/// - 2nd bit of 0x1 is 0, 2nd bit of 0x2 is 1, and 2nd bit of 0x3 is 1;
/// - 3rd bit of 0x1 is 0, 3rd bit of 0x2 is 0, and 3rd bit of 0x3 is 0;
/// - 4th bit of 0x1 is 0, 4th bit of 0x2 is 0, and 4th bit of 0x3 is 0;
/// - 5th bit of 0x1 is 0, 5th bit of 0x2 is 0, and 5th bit of 0x3 is 0.
/// Therefore, the 1st and 2nd bits of the mask value should be "don't care"
/// (marked as 0), and the 3rd, 4th and 5th bits of the mask value should be 1,
/// resulting in a final mask value of 0b11100(0x1C).
void updateGroupMask(const PhysPort &slavePort, std::set<uint32_t> &group,
                     std::map<PhysPortAndID, uint32_t> &slaveMasks,
                     uint32_t numMaskBits) {
  if (group.empty()) return;
  assert(numMaskBits <= 32 && "Invalid number of mask bits");
  // Initialize the mask value to all 1s.
  uint32_t mask = (numMaskBits == 32) ? ~0u : ((uint32_t)1 << numMaskBits) - 1;
  // Iterate through `group`, use XOR to find differing bits from `firstId`, and
  // set them as 0 in `mask`.
  uint32_t firstId = *group.begin();
  for (uint32_t id : group) mask = mask & ~(id ^ firstId);
  // Update the final mask value for all the IDs in the group.
  for (uint32_t id : group) slaveMasks[PhysPortAndID(slavePort, id)] = mask;
}

/// Sort groups by their size in ascending order. A smaller group size can
/// represent a stricter `packet_rule`, which should be placed first to prevent
/// other broader (less strict) rules from matching unintended IDs.
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
void sortGroupsBySize(SmallVector<std::set<uint32_t>> &groups) {
  auto sortBySize = [](auto &lhs, auto &rhs) {
    if (lhs.size() != rhs.size()) return lhs.size() < rhs.size();
    return lhs < rhs;
  };
  std::sort(groups.begin(), groups.end(), sortBySize);
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
/// - `Group B`: Contains IDs `{0x6, 0x7}` with `mask = 0x18`.
///   - Packet rule: `(ID & 0x18) ?= (0x6 & 0x18)`.
/// - `Group C`: Contains IDs `{0x1, 0x2, 0x3, 0x4, 0x5}` with `mask = 0x18`.
///   - Packet rule: `(ID & 0x18) ?= (0x1 & 0x18)`.
///
/// ID `0x1` belongs to `Group C`, however, due to the limitation of masking, it
/// matches both `Group B` and `C`'s rules. Since `Group B` precedes `Group C`,
/// and `packet_rule` entries are evaluated in order, the function returns
/// `false` to indicate an invalid grouping.
bool verifyGroupsByMask(PhysPort slavePort,
                        const SmallVector<std::set<uint32_t>> &groups,
                        const std::map<PhysPortAndID, uint32_t> &slaveMasks) {
  for (size_t i = 0; i < groups.size(); ++i) {
    uint32_t iPktId = *groups[i].begin();
    uint32_t iMask = slaveMasks.at(PhysPortAndID(slavePort, iPktId));
    uint32_t iMaskedId = iPktId & iMask;
    for (size_t j = i + 1; j < groups.size(); ++j) {
      for (uint32_t jPktId : groups[j]) {
        if ((jPktId & iMask) == iMaskedId) return false;
      }
    }
  }
  return true;
}

std::tuple<SlaveGroupsT, SlaveMasksT> emitSlaveGroupsAndMasksRoutingConfig(
    ArrayRef<PhysPortAndID> slavePorts, const PacketFlowMapT &packetFlows,
    ArrayRef<PhysPortAndID> priorSlavePorts,
    const PacketFlowMapT &priorPacketFlows, uint32_t numMaskBits) {
  // Convert packet flow map into a map from src 'port and id's to destination
  // ports, so that multiple flows with different packet IDs, but the same
  // ports, can be merged.
  DenseMap<PhysPortAndID, llvm::SetVector<PhysPort>> physPortAndIDToPhysPort;
  // Put previously-routed packet flows with the unrouted ones together,
  // with previously-routed ones first to respect their relative order.
  PacketFlowMapT allPacketFlows = priorPacketFlows;
  allPacketFlows.insert(packetFlows.begin(), packetFlows.end());
  for (auto &&[src, dsts] : allPacketFlows) {
    SmallVector<PhysPort> physPorts =
        llvm::map_to_vector(dsts, [](const PhysPortAndID &physPortAndID) {
          return physPortAndID.physPort;
        });
    physPortAndIDToPhysPort[src].insert(physPorts.begin(), physPorts.end());
  }
  // `slaveGroups` maps a slave port to groups of packet IDs. The groups will be
  // later used for generating `packet_rules`.
  SlaveGroupsT slaveGroups;
  // `slaveMasks` maps a slave port and packet ID to a mask value, used for
  // `packet_rule` entries.
  SlaveMasksT slaveMasks;
  // Put previously-routed slave ports with the unrouted ones together, with
  // previously-routed ones first to respect their relative order.
  SmallVector<PhysPortAndID> workList(priorSlavePorts.begin(),
                                      priorSlavePorts.end());
  workList.append(slavePorts.begin(), slavePorts.end());
  for (PhysPortAndID &slave1 : workList) {
    // Try to find a matching group that can be merged with.
    std::optional<uint32_t> matchedGroupIdx;
    SmallVector<std::set<uint32_t>> &groups = slaveGroups[slave1.physPort];
    for (size_t i = 0; i < groups.size(); ++i) {
      PhysPortAndID slave2(slave1.physPort, *groups[i].begin());
      // Can be merged if `slave1` and `slave2` share the same destinations.
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
      SmallVector<std::set<uint32_t>> groupsCopy = groups;
      std::set<uint32_t> &group = groups[matchedGroupIdx.value()];
      // Merge `slave1.id` into the group.
      group.insert(slave1.id);
      updateGroupMask(slave1.physPort, group, slaveMasks, numMaskBits);
      sortGroupsBySize(groups);
      // If the merge is valid, simply continue the while loop on `workList`.
      if (verifyGroupsByMask(slave1.physPort, groups, slaveMasks)) continue;
      // Not a valid merge, so revert the changes on `groups` and `slaveMasks`.
      slaveGroups[slave1.physPort] = groupsCopy;
      updateGroupMask(slave1.physPort, group, slaveMasks, numMaskBits);
    }
    // No mergeable group, create a new group instead.
    std::set<uint32_t> group = {static_cast<uint32_t>(slave1.id)};
    groups.emplace_back(group);
    updateGroupMask(slave1.physPort, group, slaveMasks, numMaskBits);
    sortGroupsBySize(groups);
  }
  return std::make_tuple(slaveGroups, slaveMasks);
}

/// Given switchbox configuration data produced by the router, emit
/// configuration data for packet routing along those same switchboxes.
FailureOr<std::tuple<MasterSetsT, SlaveAMSelsT>> emitPacketRoutingConfiguration(
    const AMDAIEDeviceModel &deviceModel, const PacketFlowMapT &packetFlows,
    const PacketFlowMapT &priorPacketFlows) {
  // The generator for finding `(arbiter, msel)` pairs for packet flow
  // connections.
  AMSelGenerator amselGenerator;

  // Put previously-routed packet flows with the unrouted ones together, with
  // previously-routed ones first to respect their relative order.
  SmallVector<std::pair<PhysPortAndID, llvm::SetVector<PhysPortAndID>>>
      sortedPacketFlows(priorPacketFlows.begin(), priorPacketFlows.end());
  size_t priorPacketFlowsSize = priorPacketFlows.size();
  sortedPacketFlows.append(packetFlows.begin(), packetFlows.end());

  // Sort for deterministic AMSel allocation, using the source PhysPortAndID as
  // the key.
  auto sortBySrc = [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  };
  // Previously-routed flows and unrouted flows are sorted separately to
  // maintain their relative order.
  std::sort(sortedPacketFlows.begin(),
            sortedPacketFlows.begin() + priorPacketFlowsSize, sortBySrc);
  std::sort(sortedPacketFlows.begin() + priorPacketFlowsSize,
            sortedPacketFlows.end(), sortBySrc);

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
STRINGIFY_3TUPLE_STRUCT(PhysPort, tileLoc, port, direction)
STRINGIFY_2TUPLE_STRUCT(PhysPortAndID, physPort, id)

BOTH_OSTREAM_OPS_FORALL_ROUTER_TYPES(OSTREAM_OP_DEFN, BOTH_OSTREAM_OP)

}  // namespace mlir::iree_compiler::AMDAIE
