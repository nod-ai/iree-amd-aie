// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions. See
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: # Apache-2.0 WITH LLVM-exception

#ifndef IREE_AIE_ROUTER_H
#define IREE_AIE_ROUTER_H

#include <list>
#include <map>

#include "iree_aie_runtime.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DirectedGraph.h"
#include "llvm/ADT/GraphTraits.h"

namespace mlir::iree_compiler::AMDAIE {
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

struct ChannelEdge : ChannelEdgeBase, Channel {
  using Channel::Channel;
  SwitchboxNode &src;

  explicit ChannelEdge(SwitchboxNode &target) = delete;
  ChannelEdge(SwitchboxNode &src, SwitchboxNode &target, StrmSwPortType bundle,
              int maxCapacity)
      : ChannelEdgeBase(target),
        Channel(src, target, bundle, maxCapacity),
        src(src) {}

  // This class isn't designed to copied or moved.
  ChannelEdge(const ChannelEdge &E) = delete;
  ChannelEdge &operator=(ChannelEdge &&E) = delete;
};

class SwitchboxGraph : public SwitchboxGraphBase {
 public:
  SwitchboxGraph() = default;
  ~SwitchboxGraph() = default;
};

/// A Flow defines source and destination vertices
/// Only one source, but any number of destinations (fanout)
struct PathEndPointNode : PathEndPoint {
  SwitchboxNode *sb;
  PathEndPointNode(SwitchboxNode *sb, Port port)
      : PathEndPoint{*sb, port}, sb(sb) {}
};

struct FlowNode {
  PathEndPointNode src;
  std::vector<PathEndPointNode> dsts;
};

}  // namespace mlir::iree_compiler::AMDAIE

namespace llvm {

template <>
struct GraphTraits<mlir::iree_compiler::AMDAIE::SwitchboxNode *> {
  using NodeRef = mlir::iree_compiler::AMDAIE::SwitchboxNode *;

  static mlir::iree_compiler::AMDAIE::SwitchboxNode *SwitchboxGraphGetSwitchbox(
      DGEdge<mlir::iree_compiler::AMDAIE::SwitchboxNode,
             mlir::iree_compiler::AMDAIE::ChannelEdge> *P) {
    return &P->getTargetNode();
  }

  // Provide a mapped iterator so that the GraphTrait-based implementations
  // can find the target nodes without having to explicitly go through the
  // edges.
  using ChildIteratorType =
      mapped_iterator<mlir::iree_compiler::AMDAIE::SwitchboxNode::iterator,
                      decltype(&SwitchboxGraphGetSwitchbox)>;
  using ChildEdgeIteratorType =
      mlir::iree_compiler::AMDAIE::SwitchboxNode::iterator;

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
struct GraphTraits<mlir::iree_compiler::AMDAIE::SwitchboxGraph *>
    : GraphTraits<mlir::iree_compiler::AMDAIE::SwitchboxNode *> {
  using nodes_iterator = mlir::iree_compiler::AMDAIE::SwitchboxGraph::iterator;
  static NodeRef getEntryNode(mlir::iree_compiler::AMDAIE::SwitchboxGraph *DG) {
    return *DG->begin();
  }

  static nodes_iterator nodes_begin(
      mlir::iree_compiler::AMDAIE::SwitchboxGraph *DG) {
    return DG->begin();
  }

  static nodes_iterator nodes_end(
      mlir::iree_compiler::AMDAIE::SwitchboxGraph *DG) {
    return DG->end();
  }
};
}  // namespace llvm

namespace mlir::iree_compiler::AMDAIE {

/// The center of show: the router that builds a representation of the array,
/// executes the routing algorithm, and holds the result routes.
struct Router {
  SwitchboxGraph graph;
  std::vector<FlowNode> flows;
  std::map<TileLoc, SwitchboxNode> grid;
  // Use a list instead of a vector because nodes have an edge list of raw
  // pointers to edges (so growing a vector would invalidate the pointers).
  std::list<ChannelEdge> edges;

  Router() = default;
  void initialize(int maxCol, int maxRow, AMDAIEDeviceModel &deviceModel);
  void addFlow(TileLoc srcCoords, Port srcPort, TileLoc dstCoords,
               Port dstPort);
  bool addFixedConnection(Connect connectOp);
  std::optional<std::map<PathEndPoint, SwitchSettings>> findPaths(
      int maxIterations);
  Switchbox *getSwitchbox(TileLoc coords);
};
}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AIE_ROUTER_H
