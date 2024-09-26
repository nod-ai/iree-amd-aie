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

#define OVER_CAPACITY_COEFF 0.02
#define USED_CAPACITY_COEFF 0.02
#define DEMAND_COEFF 1.1
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
struct SwitchBoxNode : SwitchBox {
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

  SwitchBoxNode(int col, int row, int id, const AMDAIEDeviceModel &deviceModel);
  std::vector<int> findAvailableChannelIn(StrmSwPortType inBundle, Port outPort,
                                          bool isPkt);
  bool allocate(Port inPort, Port outPort, bool isPkt);
  void clearAllocation();

  bool operator==(const SwitchBoxNode &rhs) const {
    // TODO(max): do i really need to write this all out by hand?
    return std::tie(col, row, id, inPortId, outPortId, inPortToId, outPortToId,
                    connectionMatrix, inPortPktCount, maxPktStream) ==
           std::tie(rhs.col, rhs.row, rhs.id, rhs.inPortId, rhs.outPortId,
                    rhs.inPortToId, rhs.outPortToId, rhs.connectionMatrix,
                    rhs.inPortPktCount, rhs.maxPktStream);
  }
};

struct SwitchBoxConnectionEdge {
  SwitchBoxNode &src;
  SwitchBoxNode &target;
  int maxCapacity;
  StrmSwPortType bundle;

  SwitchBoxConnectionEdge(SwitchBoxNode &src, SwitchBoxNode &target);
};

// A node holds a pointer
struct PathEndPointNode : PathEndPoint {
  PathEndPointNode(SwitchBoxNode &sb, Port port)
      : PathEndPoint{static_cast<SwitchBox>(sb), port}, sb(sb) {}
  SwitchBoxNode &sb;
};

struct FlowNode {
  bool isPacketFlow;
  PathEndPointNode src;
  std::vector<PathEndPointNode> dsts;
};

SwitchBoxNode::SwitchBoxNode(int col, int row, int id,
                             const AMDAIEDeviceModel &deviceModel)
    : SwitchBox(col, row), id{id} {
  std::vector<StrmSwPortType> allBundles = {
      StrmSwPortType::CORE,  StrmSwPortType::DMA,  StrmSwPortType::FIFO,
      StrmSwPortType::SOUTH, StrmSwPortType::WEST, StrmSwPortType::NORTH,
      StrmSwPortType::EAST,  StrmSwPortType::NOC,  StrmSwPortType::TRACE,
      StrmSwPortType::CTRL};
  for (StrmSwPortType bundle : allBundles) {
    uint32_t maxCapacity =
        deviceModel.getNumSourceSwitchBoxConnections(col, row, bundle);
    if (deviceModel.isShimNOCorPLTile(col, row) && maxCapacity == 0) {
      // TODO(max): investigate copy-pasted todo; wordaround for shimMux, todo:
      // integrate shimMux into routable grid
      maxCapacity = MLIRAIELegacy::getNumSourceShimMuxConnections(
          col, row, bundle, deviceModel);
    }

    for (int channel = 0; channel < maxCapacity; channel++) {
      inPortToId[{bundle, channel}] = inPortId;
      inPortId++;
    }

    maxCapacity = deviceModel.getNumDestSwitchBoxConnections(col, row, bundle);
    // TODO(max): investigate copy-pasted todo; wordaround for shimMux, todo:
    // integrate shimMux into routable grid
    if (deviceModel.isShimNOCorPLTile(col, row) && maxCapacity == 0) {
      maxCapacity = MLIRAIELegacy::getNumDestShimMuxConnections(
          col, row, bundle, deviceModel);
    }

    for (int channel = 0; channel < maxCapacity; channel++) {
      outPortToId[{bundle, channel}] = outPortId;
      outPortId++;
    }
  }

  std::set<StrmSwPortType> shimBundles = {StrmSwPortType::DMA,
                                          StrmSwPortType::NOC};
  connectionMatrix.resize(
      inPortId, std::vector<Connectivity>(outPortId, Connectivity::AVAILABLE));
  for (const auto &[inPort, inId] : inPortToId) {
    for (const auto &[outPort, outId] : outPortToId) {
      if (!deviceModel.isLegalTileConnection(col, row, inPort.bundle,
                                             inPort.channel, outPort.bundle,
                                             outPort.channel)) {
        // TODO(max): can't put a continue here because these two conditionals
        // aren't mutually exclusive
        connectionMatrix[inId][outId] = Connectivity::INVALID;
      }

      if (deviceModel.isShimNOCorPLTile(col, row)) {
        // TODO(max): investigate copy-pasted todo; wordaround for shimMux,
        // todo: integrate shimMux into routable grid
        if (shimBundles.count(inPort.bundle) ||
            shimBundles.count(outPort.bundle)) {
          connectionMatrix[inId][outId] = Connectivity::AVAILABLE;
        }
      }
    }
  }
}

std::vector<int> SwitchBoxNode::findAvailableChannelIn(StrmSwPortType inBundle,
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

bool SwitchBoxNode::allocate(Port inPort, Port outPort, bool isPkt) {
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
        if (connectionMatrix[inId][_outPortId] == Connectivity::OCCUPIED) {
          return false;
        }
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

void SwitchBoxNode::clearAllocation() {
  for (int inId = 0; inId < inPortId; inId++) {
    for (int outId = 0; outId < outPortId; outId++) {
      if (connectionMatrix[inId][outId] != Connectivity::INVALID) {
        connectionMatrix[inId][outId] = Connectivity::AVAILABLE;
      }
    }
  }
  inPortPktCount.clear();
}

SwitchBoxConnectionEdge::SwitchBoxConnectionEdge(SwitchBoxNode &src,
                                                 SwitchBoxNode &target)
    : src(src), target(target) {
  // get bundle from src to target coordinates
  if (src.col == target.col) {
    if (src.row > target.row) {
      bundle = StrmSwPortType::SOUTH;
    } else {
      bundle = StrmSwPortType::NORTH;
    }
  } else {
    if (src.col > target.col) {
      bundle = StrmSwPortType::WEST;
    } else {
      bundle = StrmSwPortType::EAST;
    }
  }

  // maximum number of routing resources
  maxCapacity = 0;
  for (auto &[outPort, _] : src.outPortToId) {
    if (outPort.bundle == bundle) maxCapacity++;
  }
}

struct RouterImpl {
  std::vector<FlowNode> flows;
  std::map<TileLoc, SwitchBoxNode> grid;
  std::list<SwitchBoxConnectionEdge> edges;
};

Router::Router() { impl = new RouterImpl(); }
Router::~Router() { delete impl; }

void Router::initialize(int maxCol, int maxRow,
                        const AMDAIEDeviceModel &deviceModel) {
  // make grid of switchboxes
  int nodeId = 0;
  for (int row = 0; row <= maxRow; row++) {
    for (int col = 0; col <= maxCol; col++) {
      impl->grid.insert(
          {TileLoc{col, row}, SwitchBoxNode{col, row, nodeId++, deviceModel}});
      SwitchBoxNode &thisNode = impl->grid.at({col, row});
      if (row > 0) {  // if not in row 0 add channel to North/South
        SwitchBoxNode &southernNeighbor = impl->grid.at({col, row - 1});
        // get the number of outgoing connections on the south side - outgoing
        // because these correspond to rhs of a connect op
        if (deviceModel.getNumDestSwitchBoxConnections(col, row,
                                                       StrmSwPortType::SOUTH)) {
          impl->edges.emplace_back(thisNode, southernNeighbor);
        }
        // get the number of incoming connections on the south side - incoming
        // because they correspond to connections on the southside that are then
        // routed using internal connect ops through the switchbox (i.e., lhs of
        // connect ops)
        if (deviceModel.getNumSourceSwitchBoxConnections(
                col, row, StrmSwPortType::SOUTH)) {
          impl->edges.emplace_back(southernNeighbor, thisNode);
        }
      }

      if (col > 0) {
        // if not in col 0 add channel to East/West
        SwitchBoxNode &westernNeighbor = impl->grid.at({col - 1, row});
        if (deviceModel.getNumDestSwitchBoxConnections(col, row,
                                                       StrmSwPortType::WEST)) {
          impl->edges.emplace_back(thisNode, westernNeighbor);
        }
        if (deviceModel.getNumSourceSwitchBoxConnections(
                col, row, StrmSwPortType::WEST)) {
          impl->edges.emplace_back(westernNeighbor, thisNode);
        }
      }
    }
  }
}

void Router::addFlow(TileLoc srcCoords, Port srcPort, TileLoc dstCoords,
                     Port dstPort, bool isPacketFlow) {
  // check if a flow with this source already exists
  for (auto &[isPkt, src, dsts] : impl->flows) {
    if (Port existingPort = src.port; src.sb.col == srcCoords.col &&
                                      src.sb.row == srcCoords.row &&
                                      existingPort == srcPort) {
      // find the vertex corresponding to the destination
      SwitchBoxNode &matchingDstSbPtr = impl->grid.at(dstCoords);
      dsts.emplace_back(matchingDstSbPtr, dstPort);
      return;
    }
  }

  // If no existing flow was found with this source, create a new flow.
  SwitchBoxNode &matchingSrcSbPtr = impl->grid.at(srcCoords);
  SwitchBoxNode &matchingDstSbPtr = impl->grid.at(dstCoords);
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
  SwitchBoxNode &sb = impl->grid.at({col, row});
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
        SwitchBoxNode::Connectivity::AVAILABLE) {
      return false;
    }
    invalidInId.insert(inId);
    invalidOutId.insert(outId);
  }

  for (const auto &[inPort, inId] : sb.inPortToId) {
    for (const auto &[outPort, outId] : sb.outPortToId) {
      if (invalidInId.count(inId) || invalidOutId.count(outId)) {
        sb.connectionMatrix[inId][outId] = SwitchBoxNode::Connectivity::INVALID;
      }
    }
  }
  return true;
}

std::map<SwitchBoxNode *, SwitchBoxNode *> dijkstraShortestPaths(
    SwitchBoxNode *src, std::map<TileLoc, SwitchBoxNode> &grid,
    std::list<SwitchBoxConnectionEdge> &edges,
    const std::map<SwitchBoxConnectionEdge *, double> &demand) {
  // Use std::map instead of DenseMap because DenseMap doesn't let you overwrite
  // tombstones.
  auto distance = std::map<SwitchBoxNode *, double>{};
  auto preds = std::map<SwitchBoxNode *, SwitchBoxNode *>();
  std::map<SwitchBoxNode *, uint64_t> indexInHeap;
  typedef d_ary_heap_indirect<
      /*Value=*/SwitchBoxNode *, /*Arity=*/4,
      /*IndexInHeapPropertyMap=*/std::map<SwitchBoxNode *, uint64_t>,
      /*DistanceMap=*/std::map<SwitchBoxNode *, double> &,
      /*Compare=*/std::less<>>
      MutableQueue;
  MutableQueue Q(distance, indexInHeap);

  for (auto &[_, sb] : grid) distance.emplace(&sb, INF);
  distance[src] = 0.0;

  std::map<SwitchBoxNode *, std::vector<SwitchBoxConnectionEdge *>> edgesOut;
  enum Color { WHITE, GRAY, BLACK };
  std::map<SwitchBoxNode *, Color> colors;
  for (auto &[_, sb] : grid) {
    colors[&sb] = WHITE;
    for (auto &e : edges)
      if (e.src == sb) edgesOut[&sb].push_back(&e);

    std::sort(
        edgesOut[&sb].begin(), edgesOut[&sb].end(),
        [](const SwitchBoxConnectionEdge *c1, SwitchBoxConnectionEdge *c2) {
          return c1->target.id < c2->target.id;
        });
  }

  Q.push(src);
  while (!Q.empty()) {
    src = Q.top();
    Q.pop();
    for (SwitchBoxConnectionEdge *e : edgesOut[src]) {
      SwitchBoxNode &dest = e->target;
      bool relax = distance[src] + demand.at(e) < distance[&dest];
      if (colors[&dest] == WHITE) {
        if (relax) {
          distance[&dest] = distance[src] + demand.at(e);
          preds[&dest] = src;
          colors[&dest] = GRAY;
        }
        Q.push(&dest);
      } else if (colors[&dest] == GRAY && relax) {
        distance[&dest] = distance[src] + demand.at(e);
        preds[&dest] = src;
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
  std::map<SwitchBoxConnectionEdge *, int> overCapacity;
  std::map<SwitchBoxConnectionEdge *, int> usedCapacity;
  std::map<SwitchBoxConnectionEdge *, double> demand;

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
        [&](std::map<SwitchBoxNode *, SwitchBoxNode *> preds,
            SwitchBoxNode *sb) -> SwitchBoxConnectionEdge * {
      for (auto &e : impl->edges) {
        if (e.src == *preds[sb] && e.target == *sb) return &e;
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
      std::set<SwitchBoxNode *> processed;
      std::map<SwitchBoxNode *, SwitchBoxNode *> preds =
          dijkstraShortestPaths(&src.sb, impl->grid, impl->edges, demand);

      // trace the path of the flow backwards via predecessors
      // increment used_capacity for the associated channels
      SwitchSettings switchSettings;
      // set the input bundle for the source endpoint
      switchSettings.emplace(src.sb, SwitchSetting{src.port, {}});
      processed.insert(&src.sb);
      // track destination ports used by src.sb
      std::set<Port> srcDestPorts;
      for (const PathEndPointNode &endPoint : dsts) {
        SwitchBoxNode *curr = &endPoint.sb;
        assert(curr && "endpoint has no source switchbox");
        if (switchSettings.count(*curr)) {
          switchSettings.at(*curr).dsts.insert(endPoint.port);
        } else {
          switchSettings.emplace(
              *curr, SwitchSetting{Port{StrmSwPortType::SS_PORT_TYPE_MAX, -1},
                                   {endPoint.port}});
        }
        Port lastDestPort = endPoint.port;
        // trace backwards until a vertex already processed is reached
        while (!processed.count(curr)) {
          // find the incoming edge from the pred to curr
          SwitchBoxConnectionEdge *ch = findIncomingEdge(preds, curr);
          assert(ch != nullptr && "couldn't find ch");
          int channel;
          // find all available channels in
          std::vector<int> availableChannels = curr->findAvailableChannelIn(
              getConnectingBundle(ch->bundle), lastDestPort, isPkt);
          if (!availableChannels.empty()) {
            // if possible, choose the channel that predecessor can also use
            // TODO(max): investigate copy-pasted todo; todo: consider all
            // predecessors?
            int bFound = false;
            SwitchBoxNode *pred = preds[curr];
            if (!processed.count(pred) && *pred != src.sb) {
              SwitchBoxConnectionEdge *predCh = findIncomingEdge(preds, pred);
              assert(predCh != nullptr && "couldn't find ch");
              for (int availableCh : availableChannels) {
                channel = availableCh;
                std::vector<int> availablePredChannels =
                    pred->findAvailableChannelIn(
                        getConnectingBundle(predCh->bundle),
                        Port{ch->bundle, channel}, isPkt);
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

          // add the entrance port for this SwitchBox
          Port currSourcePort = {getConnectingBundle(ch->bundle), channel};
          assert(switchSettings.count(*curr) &&
                 "expected current node already in switchSettings");
          assert((switchSettings.at(*curr).src ==
                  Port{StrmSwPortType::SS_PORT_TYPE_MAX, -1}) &&
                 "expected src to not have been set yet");
          switchSettings.at(*curr).src = currSourcePort;
          // add the current SwitchBox to the map of the predecessor
          Port predDestPort = {ch->bundle, channel};
          if (switchSettings.count(*preds[curr])) {
            switchSettings.at(*preds[curr]).dsts.insert(predDestPort);
          } else {
            switchSettings.emplace(
                *preds[curr],
                SwitchSetting{Port{StrmSwPortType::SS_PORT_TYPE_MAX, -1},
                              {predDestPort}});
          }
          lastDestPort = predDestPort;

          // if at capacity, bump demand to discourage using this Channel
          // this means the order matters!
          if (usedCapacity[ch] >= ch->maxCapacity) demand[ch] *= DEMAND_COEFF;

          processed.insert(curr);
          curr = preds[curr];

          // allocation may fail, as we start from the dest of flow while
          // src.port is not chosen by router
          if (*curr == src.sb && !srcDestPorts.count(lastDestPort)) {
            if (!src.sb.allocate(src.port, lastDestPort, isPkt)) {
              isLegal = false;
              overCapacity[ch]++;
            }
            if (!isLegal) break;
            srcDestPorts.insert(lastDestPort);
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

/// Transform outputs produced by the router into representations (structs) that
/// directly map to stream switch configuration ops (soon-to-be aie-rt calls).
/// Namely pairs of (switchbox, internal connections).
std::map<SwitchBox, std::vector<Connect>> emitConnections(
    const std::map<PathEndPoint, SwitchSettings> &flowSolutions,
    const PathEndPoint &srcPoint, const AMDAIEDeviceModel &deviceModel) {
  auto srcBundle = srcPoint.port.bundle;
  auto srcChannel = srcPoint.port.channel;
  SwitchBox srcSB = srcPoint.sb;
  // the first sb isn't necessary here at all but it's just to agree with
  // ordering in mlir-aie tests (see
  // ConvertFlowsToInterconnect::matchAndRewrite).
  std::map<SwitchBox, std::vector<Connect>> connections;
  auto addConnection = [&connections](const SwitchBox &currSb,
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
    if (curr == srcSB && deviceModel.isShimNOCTile(srcSB.col, srcSB.row)) {
      // shim DMAs at start of flows
      auto shimMux = std::pair(Connect::Interconnect::SHIMMUX, srcSB.col);
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
    for (const auto &[bundle, channel] : setting.dsts) {
      // handle special shim connectivity
      if (curr == srcSB &&
          deviceModel.isShimNOCorPLTile(srcSB.col, srcSB.row)) {
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
        addConnection(curr, setting.src.bundle, setting.src.channel,
                      StrmSwPortType::SOUTH, shimCh, std::get<0>(sw),
                      std::get<1>(sw), std::get<2>(sw));
      } else {
        // otherwise, regular switchbox connection
        addConnection(curr, setting.src.bundle, setting.src.channel, bundle,
                      channel, std::get<0>(sw), std::get<1>(sw),
                      std::get<2>(sw));
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
    int numMsels, int numArbiters, const PacketFlowMapT &packetFlows) {
  SmallVector<std::pair<PhysPortAndID, llvm::SetVector<PhysPortAndID>>>
      sortedPacketFlows(packetFlows.begin(), packetFlows.end());

  // To get determinsitic behaviour
  std::sort(
      sortedPacketFlows.begin(), sortedPacketFlows.end(),
      [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

  AMSelGenerator amselGenerator(numArbiters, numMsels);

  // A map from Tile and master selectValue to the ports targeted by that
  // master select.
  std::map<std::pair<TileLoc, std::pair<uint8_t, uint8_t>>, std::set<Port>>
      masterAMSels;
  SlaveAMSelsT slaveAMSels;
  for (const auto &[physPortAndID, packetFlowports] : sortedPacketFlows) {
    // The Source Tile of the flow
    TileLoc tileLoc = physPortAndID.physPort.tileLoc;
    SmallVector<PhysPortAndID> dstPorts(packetFlowports.begin(),
                                        packetFlowports.end());
    amselGenerator.addConnection(tileLoc, physPortAndID, dstPorts);
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
                        [](const llvm::detail::DenseMapPair<SwitchBox,
                                                            SwitchSetting> &p) {
                          return to_string(p.getFirst()) + ": " +
                                 to_string(p.getSecond());
                        }),
                    ", ") +
         ")";
}

STRINGIFY_2TUPLE_STRUCT(Port, bundle, channel)
STRINGIFY_2TUPLE_STRUCT(Connect, src, dst)
STRINGIFY_2TUPLE_STRUCT(SwitchBox, col, row)
STRINGIFY_2TUPLE_STRUCT(PathEndPoint, sb, port)
STRINGIFY_2TUPLE_STRUCT(PhysPort, tileLoc, port)
STRINGIFY_2TUPLE_STRUCT(PhysPortAndID, physPort, id)

BOTH_OSTREAM_OPS_FORALL_ROUTER_TYPES(OSTREAM_OP_DEFN, BOTH_OSTREAM_OP)

}  // namespace mlir::iree_compiler::AMDAIE
