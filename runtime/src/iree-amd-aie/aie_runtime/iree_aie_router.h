// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions. See
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: # Apache-2.0 WITH LLVM-exception

#ifndef IREE_AIE_ROUTER_H
#define IREE_AIE_ROUTER_H

#include <list>
#include <map>
#include <numeric>
#include <set>

#include "iree_aie_runtime.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::iree_compiler::AMDAIE {
struct Port {
  StrmSwPortType bundle;
  int channel;

  // mlir-air legacy
  Port() : bundle(), channel() {}
  Port(StrmSwPortType b, int c) : bundle(b), channel(c) {}
  typedef std::tuple<StrmSwPortType, int> TupleType;
  Port(TupleType t) : Port(std::get<0>(t), std::get<1>(t)) {}
  operator TupleType() const { return {bundle, channel}; }
  friend llvm::hash_code hash_value(const Port &p) {
    std::size_t h1 =
        std::hash<mlir::iree_compiler::AMDAIE::StrmSwPortType>{}(p.bundle);
    std::size_t h2 = std::hash<int>{}(p.channel);
    return llvm::hash_combine(h1, h2);
  }
  TUPLE_LIKE_STRUCT_RELATIONAL_OPS(Port)
};
ASSERT_STANDARD_LAYOUT(Port);

struct Connect {
  enum class Interconnect { SHIMMUX, SWB, NOCARE };
  Port src;
  Port dst;
  Interconnect interconnect;
  uint8_t col, row;

  Connect(const Port &src, const Port &dst, Interconnect interconnect,
          uint8_t col, uint8_t row)
      : src(src), dst(dst), interconnect(interconnect), col(col), row(row) {}
  using TupleType = std::tuple<Port, Port, Interconnect, uint8_t, uint8_t>;
  Connect(TupleType t)
      : Connect(std::get<0>(t), std::get<1>(t), std::get<2>(t), std::get<3>(t),
                std::get<4>(t)) {}
  operator TupleType() const { return {src, dst, interconnect, col, row}; }
  TUPLE_LIKE_STRUCT_RELATIONAL_OPS(Connect)
};
ASSERT_STANDARD_LAYOUT(Connect);
/// A SwitchSetting defines the required configurations for an actual
/// physical/device SwitchBox.
/// SwitchSetting.srcs are the incoming signals
/// SwitchSetting.dsts is the fanout
struct SwitchSetting {
  std::vector<Port> srcs;
  std::vector<Port> dsts;
  SwitchSetting() = default;
  SwitchSetting(std::vector<Port> srcs) : srcs(std::move(srcs)) {}
  SwitchSetting(std::vector<Port> srcs, std::vector<Port> dsts)
      : srcs(std::move(srcs)), dsts(std::move(dsts)) {}
  bool operator<(const SwitchSetting &rhs) const { return srcs < rhs.srcs; }
};

using SwitchSettings = std::map<TileLoc, SwitchSetting>;

struct PathEndPoint {
  TileLoc tileLoc;
  Port port;
  PathEndPoint() = default;
  PathEndPoint(int col, int row, Port port) : PathEndPoint({col, row}, port) {}
  PathEndPoint(TileLoc tileLoc, Port port) : tileLoc(tileLoc), port(port) {}
  using TupleType = std::tuple<TileLoc, Port>;
  PathEndPoint(TupleType t) : PathEndPoint(std::get<0>(t), std::get<1>(t)) {}
  operator TupleType() const { return {tileLoc, port}; }
  TUPLE_LIKE_STRUCT_RELATIONAL_OPS(PathEndPoint)
};
ASSERT_STANDARD_LAYOUT(PathEndPoint);

struct RouterImpl;
struct Router {
  RouterImpl *impl;
  Router();
  ~Router();
  void initialize(int maxCol, int maxRow, const AMDAIEDeviceModel &targetModel);
  void addFlow(TileLoc srcCoords, Port srcPort, TileLoc dstCoords, Port dstPort,
               bool isPacketFlow);
  bool addFixedConnection(int col, int row,
                          const std::vector<std::tuple<Port, Port>> &connects);
  std::map<PathEndPoint, PathEndPoint> dijkstraShortestPaths(PathEndPoint src);
  std::optional<std::map<PathEndPoint, SwitchSettings>> findPaths(
      int maxIterations = 1000);
};

std::map<TileLoc, std::vector<Connect>> emitConnections(
    const std::map<PathEndPoint, SwitchSettings> &flowSolutions,
    const PathEndPoint &srcPoint, const AMDAIEDeviceModel &targetModel);

bool existsPathToDest(const SwitchSettings &settings, TileLoc currTile,
                      StrmSwPortType currDestBundle, int currDestChannel,
                      TileLoc finalTile, StrmSwPortType finalDestBundle,
                      int finalDestChannel);

struct PhysPort {
  TileLoc tileLoc;
  Port port;
  PhysPort(TileLoc t, Port p) : tileLoc(t), port(p) {}
  using TupleType = std::tuple<TileLoc, Port>;
  PhysPort(TupleType t) : PhysPort(std::get<0>(t), std::get<1>(t)) {}
  operator TupleType() const { return {tileLoc, port}; }
  TUPLE_LIKE_STRUCT_RELATIONAL_OPS(PhysPort)
};

struct PhysPortAndID {
  PhysPort physPort;
  int id;
  PhysPortAndID(PhysPort p, int i) : physPort(p), id(i) {}
  using TupleType = std::tuple<PhysPort, int>;
  PhysPortAndID(TupleType t) : PhysPortAndID(std::get<0>(t), std::get<1>(t)) {}
  operator TupleType() const { return {physPort, id}; }
  TUPLE_LIKE_STRUCT_RELATIONAL_OPS(PhysPortAndID)
};

// A map from a switchbox output (physical) port to the number of that port.
using MasterSetsT =
    std::map<PhysPort, std::vector<std::pair<uint8_t, uint8_t>>>;
using SlaveGroupsT = std::vector<std::vector<PhysPortAndID>>;
using SlaveMasksT = std::map<PhysPortAndID, int>;
using SlaveAMSelsT = std::map<PhysPortAndID, std::pair<uint8_t, uint8_t>>;
using ConnectionAndFlowIDT = std::pair<Connect, int>;
using TileLocToConnectionFlowIDT =
    std::map<TileLoc, DenseSet<ConnectionAndFlowIDT>>;
using PacketFlowMapT = DenseMap<PhysPortAndID, llvm::SetVector<PhysPortAndID>>;

std::tuple<SlaveGroupsT, SlaveMasksT> emitSlaveGroupsAndMasksRoutingConfig(
    ArrayRef<PhysPortAndID> slavePorts, const PacketFlowMapT &packetFlows);

FailureOr<std::tuple<MasterSetsT, SlaveAMSelsT>> emitPacketRoutingConfiguration(
    const AMDAIEDeviceModel &deviceModel, const PacketFlowMapT &packetFlows);

/// ============================= BEGIN ==================================
/// ================== stringification utils =============================
/// ======================================================================

#define TO_STRINGS(_) \
  _(Connect)          \
  _(PathEndPoint)     \
  _(Port)             \
  _(SwitchSetting)    \
  _(PhysPort)         \
  _(PhysPortAndID)

TO_STRINGS(TO_STRING_DECL)
#undef TO_STRINGS

#define BOTH_OSTREAM_OPS_FORALL_ROUTER_TYPES(OSTREAM_OP_, _) \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::Connect)       \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::PathEndPoint)  \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::Port)          \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::SwitchSetting) \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::PhysPort)      \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::PhysPortAndID)

BOTH_OSTREAM_OPS_FORALL_ROUTER_TYPES(OSTREAM_OP_DECL, BOTH_OSTREAM_OP)

}  // namespace mlir::iree_compiler::AMDAIE

template <>
struct std::hash<mlir::iree_compiler::AMDAIE::Port> {
  std::size_t operator()(
      const mlir::iree_compiler::AMDAIE::Port &p) const noexcept {
    return static_cast<std::size_t>(hash_value(p));
  }
};

namespace llvm {
template <>
struct DenseMapInfo<mlir::iree_compiler::AMDAIE::Port>
    : TupleStructDenseMapInfo<mlir::iree_compiler::AMDAIE::Port::TupleType> {};

template <>
struct DenseMapInfo<mlir::iree_compiler::AMDAIE::Connect>
    : TupleStructDenseMapInfo<mlir::iree_compiler::AMDAIE::Connect::TupleType> {
};

template <>
struct DenseMapInfo<mlir::iree_compiler::AMDAIE::PhysPort>
    : TupleStructDenseMapInfo<
          mlir::iree_compiler::AMDAIE::PhysPort::TupleType> {};

template <>
struct DenseMapInfo<mlir::iree_compiler::AMDAIE::PhysPortAndID>
    : TupleStructDenseMapInfo<
          mlir::iree_compiler::AMDAIE::PhysPortAndID::TupleType> {};

}  // namespace llvm

template <>
struct std::hash<mlir::iree_compiler::AMDAIE::PathEndPoint> {
  std::size_t operator()(
      const mlir::iree_compiler::AMDAIE::PathEndPoint &pe) const noexcept {
    std::size_t h1 = std::hash<mlir::iree_compiler::AMDAIE::Port>{}(pe.port);
    std::size_t h2 =
        std::hash<mlir::iree_compiler::AMDAIE::TileLoc>{}(pe.tileLoc);
    return h1 ^ (h2 << 1);
  }
};

#endif  // IREE_AIE_ROUTER_H
