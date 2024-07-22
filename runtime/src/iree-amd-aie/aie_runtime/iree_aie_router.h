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

namespace mlir::iree_compiler::AMDAIE {
struct Port {
  StrmSwPortType bundle;
  int channel;

  Port() = delete;

  bool operator==(const Port &rhs) const {
    return std::tie(bundle, channel) == std::tie(rhs.bundle, rhs.channel);
  }

  bool operator!=(const Port &rhs) const { return !(*this == rhs); }

  bool operator<(const Port &rhs) const {
    return std::tie(bundle, channel) < std::tie(rhs.bundle, rhs.channel);
  }
};
ASSERT_STANDARD_LAYOUT(Port);

struct Connect {
  enum class Interconnect { shimMuxOp, swOp, nocare };
  Port src;
  Port dst;
  Interconnect interconnect;
  uint8_t col, row;

  Connect(const Port &src, const Port &dst,
          Interconnect interconnect = Interconnect::nocare, uint8_t col = 0,
          uint8_t row = 0)
      : src(src), dst(dst), interconnect(interconnect), col(col), row(row) {}

  bool operator==(const Connect &rhs) const {
    return std::tie(src, dst, interconnect, col, row) ==
           std::tie(rhs.src, rhs.dst, interconnect, col, row);
  }
};
ASSERT_STANDARD_LAYOUT(Connect);

struct SwitchBox : TileLoc {
  SwitchBox(TileLoc t) : TileLoc(t) {}
  SwitchBox(int col, int row) : TileLoc(col, row) {}
  SwitchBox(std::tuple<int, int> t) : TileLoc(t) {}

  bool operator==(const SwitchBox &rhs) const {
    return static_cast<TileLoc>(*this) == rhs;
  }
};
ASSERT_STANDARD_LAYOUT(SwitchBox);

/// A SwitchSetting defines the required conifgurations for an actual
/// physical/device SwitchBox.
/// SwitchSetting.src is the incoming signal
/// SwitchSetting.dsts is the fanout
struct SwitchSetting {
  Port src;
  std::set<Port> dsts;

  // deleted anyway because Port's is deleted
  SwitchSetting() = delete;
  SwitchSetting(Port src) : src(src) {}
  SwitchSetting(Port src, std::set<Port> dsts)
      : src(src), dsts(std::move(dsts)) {}
  bool operator<(const SwitchSetting &rhs) const { return src < rhs.src; }
};

using SwitchSettings = std::map<SwitchBox, SwitchSetting>;

struct PathEndPoint {
  SwitchBox sb;
  Port port;
  PathEndPoint(SwitchBox sb, Port port) : sb(sb), port(port) {}
  PathEndPoint(int col, int row, Port port) : PathEndPoint({col, row}, port) {}
  bool operator<(const PathEndPoint &rhs) const {
    return std::tie(sb, port) < std::tie(rhs.sb, rhs.port);
  }

  bool operator==(const PathEndPoint &rhs) const {
    return std::tie(sb, port) == std::tie(rhs.sb, rhs.port);
  }
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
  bool addFixedConnection(
      int col, int row,
      const std::vector<std::tuple<StrmSwPortType, int, StrmSwPortType, int>>
          &connects);
  std::optional<std::map<PathEndPoint, SwitchSettings>> findPaths(
      int maxIterations = 1000);
};

std::vector<std::pair<SwitchBox, Connect>> emitConnections(
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
  inline bool operator<(const PhysPort &rhs) const {
    return TupleType(*this) < TupleType(rhs);
  }
  bool operator==(const PhysPort &rhs) const {
    return TupleType(*this) == TupleType(rhs);
  }
};

struct PhysPortAndID {
  PhysPort physPort;
  int id;
  PhysPortAndID(PhysPort p, int i) : physPort(p), id(i) {}
  using TupleType = std::tuple<PhysPort, int>;
  PhysPortAndID(TupleType t) : PhysPortAndID(std::get<0>(t), std::get<1>(t)) {}
  operator TupleType() const { return {physPort, id}; }
  inline bool operator<(const PhysPortAndID &rhs) const {
    return TupleType(*this) < TupleType(rhs);
  }
  bool operator==(const PhysPortAndID &rhs) const {
    return std::tie(physPort, id) == std::tie(rhs.physPort, rhs.id);
  }
};

// A map from a switchbox output (physical) port to the number of that port.
using MasterSetsT = DenseMap<PhysPort, SmallVector<int>>;
using SlaveGroupsT = SmallVector<SmallVector<PhysPortAndID>>;
using SlaveMasksT = DenseMap<PhysPortAndID, int>;
using SlaveAMSelsT = DenseMap<PhysPortAndID, int>;
using ConnectionAndFlowIDT = std::pair<Connect, int>;
using SwitchBoxToConnectionFlowIDT =
    DenseMap<TileLoc, DenseSet<ConnectionAndFlowIDT>>;

std::tuple<MasterSetsT, SlaveGroupsT, SlaveMasksT, SlaveAMSelsT>
configurePacketFlows(int numMsels, int numArbiters,
                     const SwitchBoxToConnectionFlowIDT &switchboxes,
                     const SmallVector<TileLoc> &tiles);

/// ============================= BEGIN ==================================
/// ================== stringification utils =============================
/// ======================================================================

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

template <>
struct DenseMapInfo<mlir::iree_compiler::AMDAIE::Connect> {
  using FirstInfo = DenseMapInfo<mlir::iree_compiler::AMDAIE::Port>;
  using SecondInfo = DenseMapInfo<mlir::iree_compiler::AMDAIE::Port>;
  using ThirdInfo =
      DenseMapInfo<mlir::iree_compiler::AMDAIE::Connect::Interconnect>;
  using FourthInfo = DenseMapInfo<uint8_t>;
  using FifthInfo = DenseMapInfo<uint8_t>;

  static mlir::iree_compiler::AMDAIE::Connect getEmptyKey() {
    return {FirstInfo::getEmptyKey(), SecondInfo::getEmptyKey(),
            ThirdInfo::getEmptyKey(), FourthInfo::getEmptyKey(),
            FifthInfo::getEmptyKey()};
  }

  static mlir::iree_compiler::AMDAIE::Connect getTombstoneKey() {
    return {FirstInfo::getTombstoneKey(), SecondInfo::getTombstoneKey(),
            ThirdInfo::getTombstoneKey(), FourthInfo::getTombstoneKey(),
            FifthInfo::getTombstoneKey()};
  }

  static unsigned getHashValue(const mlir::iree_compiler::AMDAIE::Connect &d) {
    std::vector<unsigned> hashes{
        FirstInfo::getHashValue(d.src), SecondInfo::getHashValue(d.dst),
        ThirdInfo::getHashValue(d.interconnect),
        FourthInfo::getHashValue(d.col), FifthInfo::getHashValue(d.row)};

    return std::accumulate(hashes.begin(), hashes.end(), 0,
                           detail::combineHashValue);
  }

  static bool isEqual(const mlir::iree_compiler::AMDAIE::Connect &lhs,
                      const mlir::iree_compiler::AMDAIE::Connect &rhs) {
    return lhs == rhs;
  }
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
    std::size_t h2 = std::hash<mlir::iree_compiler::AMDAIE::TileLoc>{}(pe.sb);
    return h1 ^ (h2 << 1);
  }
};

#endif  // IREE_AIE_ROUTER_H
