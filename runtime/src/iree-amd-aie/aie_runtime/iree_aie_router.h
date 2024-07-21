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
static_assert(std::is_standard_layout_v<Port>,
              "Port is meant to be a standard layout type");

using PhysPort = std::pair<TileLoc, Port>;

struct Connect {
  enum class Interconnect { shimMuxOp, swOp, unk };
  Port src;
  Port dst;
  Interconnect interconnect;
  uint8_t col, row;

  Connect(const Port &src, const Port &dst,
          Interconnect interconnect = Interconnect::unk, uint8_t col = 0,
          uint8_t row = 0)
      : src(src), dst(dst), interconnect(interconnect), col(col), row(row) {}

  bool operator==(const Connect &rhs) const {
    return std::tie(src, dst) == std::tie(rhs.src, rhs.dst);
  }
};
static_assert(std::is_standard_layout_v<Connect>,
              "Connect is meant to be a standard layout type");

struct Switchbox : TileLoc {
  // Necessary for initializer construction?
  Switchbox(TileLoc t) : TileLoc(t) {}
  Switchbox(int col, int row) : TileLoc(col, row) {}
  Switchbox(std::tuple<int, int> t) : TileLoc(t) {}

  bool operator==(const Switchbox &rhs) const {
    return static_cast<TileLoc>(*this) == rhs;
  }
};
static_assert(std::is_standard_layout_v<Switchbox>,
              "Switchbox is meant to be a standard layout type");

// A SwitchSetting defines the required settings for a SwitchboxNode for a flow
// SwitchSetting.src is the incoming signal
// SwitchSetting.dsts is the fanout
struct SwitchSetting {
  Port src;
  std::set<Port> dsts;

  SwitchSetting() = default;
  SwitchSetting(Port src) : src(src) {}
  SwitchSetting(Port src, std::set<Port> dsts)
      : src(src), dsts(std::move(dsts)) {}
  bool operator<(const SwitchSetting &rhs) const { return src < rhs.src; }
};

using SwitchSettings = std::map<Switchbox, SwitchSetting>;

// A Flow defines source and destination vertices
// Only one source, but any number of destinations (fanout)
struct PathEndPoint {
  Switchbox sb;
  Port port;
  PathEndPoint(Switchbox sb, Port port) : sb(sb), port(port) {}
  PathEndPoint(int col, int row, Port port) : PathEndPoint({col, row}, port) {}
  // Needed for the std::maps that store PathEndPoint.
  bool operator<(const PathEndPoint &rhs) const {
    return std::tie(sb, port) < std::tie(rhs.sb, rhs.port);
  }

  bool operator==(const PathEndPoint &rhs) const {
    return std::tie(sb, port) == std::tie(rhs.sb, rhs.port);
  }
};
static_assert(std::is_standard_layout_v<PathEndPoint>,
              "PathEndPoint is meant to be a standard layout type");

struct RouterImpl;
struct Router {
  RouterImpl *impl;
  Router();
  ~Router();
  void initialize(int maxCol, int maxRow, const AMDAIEDeviceModel &targetModel);
  void addFlow(TileLoc srcCoords, Port srcPort, TileLoc dstCoords, Port dstPort,
               bool isPacketFlow);
  bool addFixedConnection(
      int col, int row,
      const std::vector<std::tuple<StrmSwPortType, int, StrmSwPortType, int>>
          &connects);
  std::optional<std::map<PathEndPoint, SwitchSettings>> findPaths(
      int maxIterations = 1000);
};

std::vector<std::pair<Switchbox, Connect>> emitConnections(
    const std::map<PathEndPoint, SwitchSettings> &flowSolutions,
    const PathEndPoint &srcPoint, const AMDAIEDeviceModel &targetModel);

bool existsPathToDest(const SwitchSettings &settings, TileLoc currTile,
                      StrmSwPortType currDestBundle, int currDestChannel,
                      TileLoc finalTile, StrmSwPortType finalDestBundle,
                      int finalDestChannel);

std::tuple<DenseMap<PhysPort, SmallVector<int, 4>>,
           SmallVector<SmallVector<std::pair<PhysPort, int>, 4>, 4>,
           DenseMap<std::pair<PhysPort, int>, int>,
           DenseMap<std::pair<PhysPort, int>, int>>
configurePacketFlows(
    int numMsels, int numArbiters,
    const DenseMap<TileLoc, SmallVector<std::pair<Connect, int>, 8>>
        &switchboxes,
    const SmallVector<TileLoc> &tiles);

#define TO_STRINGS(_) \
  _(Connect)          \
  _(PathEndPoint)     \
  _(Port)             \
  _(SwitchSetting)

TO_STRINGS(TO_STRING_DECL)
#undef TO_STRINGS

#define BOTH_OSTREAM_OPS_FORALL_ROUTER_TYPES(OSTREAM_OP_, _) \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::Connect)       \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::PathEndPoint)  \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::Port)          \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::SwitchSetting)

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
struct std::hash<mlir::iree_compiler::AMDAIE::PathEndPoint> {
  std::size_t operator()(
      const mlir::iree_compiler::AMDAIE::PathEndPoint &pe) const noexcept {
    std::size_t h1 = std::hash<mlir::iree_compiler::AMDAIE::Port>{}(pe.port);
    std::size_t h2 = std::hash<mlir::iree_compiler::AMDAIE::TileLoc>{}(pe.sb);
    return h1 ^ (h2 << 1);
  }
};

#endif  // IREE_AIE_ROUTER_H
