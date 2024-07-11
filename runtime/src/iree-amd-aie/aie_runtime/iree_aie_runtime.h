// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions. See
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: # Apache-2.0 WITH LLVM-exception

#ifndef IREE_AIE_RUNTIME_H
#define IREE_AIE_RUNTIME_H

#include <optional>
#include <ostream>
#include <sstream>
#include <tuple>

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"
// clang-format off
#include "iree-amd-aie/aie_runtime/AMDAIEEnums.h"
// clang-format on

extern "C" {
#include "xaie_hwcfg.h"
#include "xaiengine.h"
#include "xaiengine/xaie_device_aieml.h"
#undef s8
#undef u8
#undef u16
#undef s32
#undef u32
#undef u64

enum byte_ordering { Little_Endian, Big_Endian };
void startCDOFileStream(const char* cdoFileName);
void endCurrentCDOFileStream();
void FileHeader();
void EnAXIdebug();
void setEndianness(bool endianness);
void configureHeader();
void insertNoOpCommand(unsigned int numPadBytes);
}

namespace mlir::iree_compiler::AMDAIE {
struct TileLoc {
  int col, row;

  TileLoc(int col, int row) : col(col), row(row) {}
  TileLoc() = delete;
  TileLoc(XAie_LocType loc) : col(loc.Col), row(loc.Row) {}
  operator XAie_LocType() const { return XAie_TileLoc(col, row); }

  inline bool operator<(const TileLoc& rhs) const {
    return std::tie(col, row) < std::tie(rhs.col, rhs.row);
  }

  bool operator==(const TileLoc& rhs) const {
    return std::tie(col, row) == std::tie(rhs.col, rhs.row);
  }

  bool operator!=(const TileLoc& rhs) const { return !(*this == rhs); }
};

struct Port {
  StrmSwPortType bundle;
  int channel;

  bool operator==(const Port& rhs) const {
    return std::tie(bundle, channel) == std::tie(rhs.bundle, rhs.channel);
  }

  bool operator!=(const Port& rhs) const { return !(*this == rhs); }

  bool operator<(const Port& rhs) const {
    return std::tie(bundle, channel) < std::tie(rhs.bundle, rhs.channel);
  }
};

struct Connect {
  Port src;
  Port dst;

  bool operator==(const Connect& rhs) const {
    return std::tie(src, dst) == std::tie(rhs.src, rhs.dst);
  }
};

struct DMAChannel {
  DMAChannelDir direction;
  int channel;

  bool operator==(const DMAChannel& rhs) const {
    return std::tie(direction, channel) == std::tie(rhs.direction, rhs.channel);
  }
};

struct Switchbox : TileLoc {
  // Necessary for initializer construction?
  Switchbox(TileLoc t) : TileLoc(t) {}
  Switchbox(int col, int row) : TileLoc{col, row} {}
  bool operator==(const Switchbox& rhs) const {
    return static_cast<TileLoc>(*this) == rhs;
  }
};

struct Channel {
  Switchbox& src;
  Switchbox& target;
  StrmSwPortType bundle;
  int maxCapacity = 0;   // maximum number of routing resources
  double demand = 0.0;   // indicates how many flows want to use this Channel
  int usedCapacity = 0;  // how many flows are actually using this Channel
  DenseSet<int> fixedCapacity;  // channels not available to the algorithm
  int overCapacityCount = 0;    // history of Channel being over capacity
  Channel(Switchbox& src, Switchbox& target, StrmSwPortType bundle,
          int maxCapacity)
      : src(src), target(target), bundle(bundle), maxCapacity(maxCapacity) {}
};

// A SwitchSetting defines the required settings for a Switchbox for a flow
// SwitchSetting.src is the incoming signal
// SwitchSetting.dsts is the fanout
struct SwitchSetting {
  Port src;
  DenseSet<Port> dsts;
  SwitchSetting() = default;
  SwitchSetting(Port src) : src(src) {}
  SwitchSetting(Port src, DenseSet<Port> dsts)
      : src(src), dsts(std::move(dsts)) {}

  bool operator<(const SwitchSetting& rhs) const { return src < rhs.src; }
};

using SwitchSettings = DenseMap<Switchbox, SwitchSetting>;

// A Flow defines source and destination vertices
// Only one source, but any number of destinations (fanout)
struct PathEndPoint {
  Switchbox sb;
  Port port;

  bool operator<(const PathEndPoint& rhs) const {
    return std::tie(sb, port) < std::tie(rhs.sb, rhs.port);
  }

  bool operator==(const PathEndPoint& rhs) const {
    return std::tie(sb, port) == std::tie(rhs.sb, rhs.port);
  }
};

StrmSwPortType getConnectingBundle(StrmSwPortType dir);

enum class AMDAIETileType : uint8_t {
  AIETILE = 0,
  SHIMNOC = 1,
  SHIMPL = 2,
  MEMTILE = 3,
  MAX = 4
};

/// Enum of DMA properties. Uses the offset within the `XAie_DmaMod` struct as
/// underlying value to easily retrieve the specified property with a single
/// getter method, while being versatile towards `XAie_DmaMod` struct changes.
enum class AMDAIEDmaProp : uint8_t {
  NumBds = offsetof(struct XAie_DmaMod, NumBds),
  NumLocks = offsetof(struct XAie_DmaMod, NumLocks),
  ChIdxOffset = offsetof(struct XAie_DmaMod, ChIdxOffset),
  NumAddrDim = offsetof(struct XAie_DmaMod, NumAddrDim),
  DoubleBuffering = offsetof(struct XAie_DmaMod, DoubleBuffering),
  Compression = offsetof(struct XAie_DmaMod, Compression),
  Padding = offsetof(struct XAie_DmaMod, Padding),
  OutofOrderBdId = offsetof(struct XAie_DmaMod, OutofOrderBdId),
  InterleaveMode = offsetof(struct XAie_DmaMod, InterleaveMode),
  FifoMode = offsetof(struct XAie_DmaMod, FifoMode),
  EnTokenIssue = offsetof(struct XAie_DmaMod, EnTokenIssue),
  RepeatCount = offsetof(struct XAie_DmaMod, RepeatCount),
  TlastSuppress = offsetof(struct XAie_DmaMod, TlastSuppress),
  StartQueueBase = offsetof(struct XAie_DmaMod, StartQueueBase),
  BaseAddr = offsetof(struct XAie_DmaMod, BaseAddr),
  IdxOffset = offsetof(struct XAie_DmaMod, IdxOffset),
  ChCtrlBase = offsetof(struct XAie_DmaMod, ChCtrlBase),
  NumChannels = offsetof(struct XAie_DmaMod, NumChannels),
  ChStatusBase = offsetof(struct XAie_DmaMod, ChStatusBase),
  ChStatusOffset = offsetof(struct XAie_DmaMod, ChStatusOffset),
  PadValueBase = offsetof(struct XAie_DmaMod, PadValueBase),
  MAX = sizeof(struct XAie_DmaMod)
};

/*
 * This struct is meant to be a thin wrapper around aie-rt, which provides
 * the canonical representation/metadata for AIE devices; attributes like number
 * of locks, bds per tile, whether certain switch connections are legal or not,
 * etc.
 *
 * This representation is parameterized by platform specific constants
 * (BASE_ADDR, COL/ROW shift, NUM_MEM_TILE_ROWS, etc.) which are available in
 * the adjacent xaie_hwcfg.h for common platforms (AIE1, AIE2, AIE2IPU).
 *
 * This struct is used in places where device specific features/attributes need
 * to be considered in order to emit efficient/legal code; utilities such as
 * stream switch configuration/routing.
 *
 * TODO(max): Refactor AMDAIETargetCDODirect and move majority of emission code
 * here, in order to make reusable for real/actual runtime facilities.
 */
struct AMDAIEDeviceModel {
  XAie_Config configPtr;
  XAie_DevInst devInst;

  explicit AMDAIEDeviceModel(uint8_t aieGen, uint64_t baseAddr,
                             uint8_t colShift, uint8_t rowShift,
                             uint8_t devNColumns, uint8_t devNRows,
                             uint8_t memTileRowStart, uint8_t nMemTileRows,
                             uint8_t nShimTileRows, int partitionNumCols,
                             int partitionStartCol, bool aieSim,
                             bool xaieDebug);

  int rows() const;
  int columns() const;

  AMDAIETileType getTileType(uint8_t col, uint8_t row) const;
  bool isCoreTile(uint8_t col, uint8_t row) const;
  bool isMemTile(uint8_t col, uint8_t row) const;
  bool isShimNOCTile(uint8_t col, uint8_t row) const;
  bool isShimPLTile(uint8_t col, uint8_t row) const;

  /// Retrieve a DMA properpty for the specified tile type.
  template <typename T>
  T getDmaProp(AMDAIETileType tileType, AMDAIEDmaProp dmaProp) const {
    const uint8_t* dmaMod = reinterpret_cast<const uint8_t*>(
        devInst.DevProp.DevMod[static_cast<uint8_t>(tileType)].DmaMod);
    return *((const T*)(dmaMod + static_cast<uint8_t>(dmaProp)));
  }

  uint32_t getNumLocks(uint8_t col, uint8_t row) const;

  std::optional<TileLoc> getMemWest(TileLoc src) const;
  std::optional<TileLoc> getMemEast(TileLoc src) const;
  std::optional<TileLoc> getMemNorth(TileLoc src) const;
  std::optional<TileLoc> getMemSouth(TileLoc src) const;

  bool hasMemWest(uint8_t srcCol, uint8_t srcRow, uint8_t dstCol,
                  uint8_t dstRow) const;
  bool hasMemEast(uint8_t srcCol, uint8_t srcRow, uint8_t dstCol,
                  uint8_t dstRow) const;
  bool hasMemNorth(uint8_t srcCol, uint8_t srcRow, uint8_t dstCol,
                   uint8_t dstRow) const;
  bool hasMemSouth(uint8_t srcCol, uint8_t srcRow, uint8_t dstCol,
                   uint8_t dstRow) const;
  /// Return true if core can access the memory in mem
  bool hasLegalMemAffinity(uint8_t coreCol, uint8_t coreRow, uint8_t memCol,
                           uint8_t memRow) const;

  uint32_t getMemInternalBaseAddress() const;
  uint32_t getMemSouthBaseAddress() const;
  uint32_t getMemWestBaseAddress() const;
  uint32_t getMemNorthBaseAddress() const;
  uint32_t getMemEastBaseAddress() const;
  uint32_t getLocalMemorySize(uint8_t col, uint8_t row) const;
  uint32_t getMemTileSize(uint8_t col, uint8_t row) const;
  uint32_t getCoreTileLocalMemorySize() const;

  uint32_t getNumBDs(uint8_t col, uint8_t row) const;

  uint32_t getNumSourceSwitchboxConnections(uint8_t col, uint8_t row,
                                            StrmSwPortType bundle) const;
  uint32_t getNumDestSwitchboxConnections(uint8_t col, uint8_t row,
                                          StrmSwPortType bundle) const;
  bool isLegalMemtileConnection(uint8_t col, uint8_t row,
                                StrmSwPortType srcBundle, uint8_t srcChan,
                                StrmSwPortType dstBundle,
                                uint8_t dstChan) const;

  uint32_t getColumnShift() const;
  uint32_t getRowShift() const;
};

struct AMDAIEDeviceModel getDeviceModel(AMDAIEDevice device);

#define OSTREAM_OP(O_TYPE, TYPE) O_TYPE& operator<<(O_TYPE& os, const TYPE& s);
#define TO_STRING(TYPE) std::string to_string(const TYPE& t);

#define TO_STRINGS(_) \
  _(AMDAIETileType)   \
  _(AMDAIEDmaProp)    \
  _(AieRC)            \
  _(Channel)          \
  _(Connect)          \
  _(DMAChannel)       \
  _(DMAChannelDir)    \
  _(Port)             \
  _(SwitchSetting)    \
  _(SwitchSettings)   \
  _(Switchbox)        \
  _(StrmSwPortType)   \
  _(TileLoc)          \
  _(XAie_LocType)     \
  _(XAie_Lock)        \
  _(XAie_Packet)

TO_STRINGS(TO_STRING)
#undef TO_STRING
#undef TO_STRINGS

#define BOTH_OSTREAM_OP(OSTREAM_OP_, TYPE) \
  OSTREAM_OP_(std::ostream, TYPE)          \
  OSTREAM_OP_(llvm::raw_ostream, TYPE)

#define BOTH_OSTREAM_OPS_FORALL_TYPES(OSTREAM_OP_, _)         \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::AMDAIETileType) \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::Channel)        \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::Connect)        \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::DMAChannel)     \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::DMAChannelDir)  \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::Port)           \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::SwitchSetting)  \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::SwitchSettings) \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::Switchbox)      \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::TileLoc)        \
  _(OSTREAM_OP_, AieRC)                                       \
  _(OSTREAM_OP_, StrmSwPortType)                              \
  _(OSTREAM_OP_, XAie_LocType)                                \
  _(OSTREAM_OP_, XAie_Lock)                                   \
  _(OSTREAM_OP_, XAie_Packet)

BOTH_OSTREAM_OPS_FORALL_TYPES(OSTREAM_OP, BOTH_OSTREAM_OP)
#undef OSTREAM_OP

// https://stackoverflow.com/a/32230306
template <typename H1>
llvm::raw_ostream& showArgs(llvm::raw_ostream& out, const char* label,
                            H1&& value) {
  if constexpr (std::is_pointer<H1>::value)
    return out << label << "="
               << "ptr";
  else
    return out << label << "=" << std::forward<H1>(value);
}

template <typename H1, typename... T>
llvm::raw_ostream& showArgs(llvm::raw_ostream& out, const char* label,
                            H1&& value, T&&... rest) {
  const char* pcomma = strchr(label, ',');
  if constexpr (std::is_pointer<H1>::value)
    return showArgs(out.write(label, pcomma - label) << "=ptr,", pcomma + 1,
                    std::forward<T>(rest)...);
  else
    return showArgs(out.write(label, pcomma - label)
                        << "=" << std::forward<H1>(value) << ',',
                    pcomma + 1, std::forward<T>(rest)...);
}

#define SHOW_ARGS(os, ...) showArgs(os, #__VA_ARGS__, __VA_ARGS__)

// So that we can use the pattern if(auto r = TRY_XAIE_API...) { // r is nonzero
// }
static_assert(XAIE_OK == 0);

#define TRY_XAIE_API_FATAL_ERROR(API, ...)                              \
  do {                                                                  \
    LLVM_DEBUG(llvm::dbgs() << "XAIE API: " << #API << " with args: "); \
    LLVM_DEBUG(SHOW_ARGS(llvm::dbgs(), __VA_ARGS__));                   \
    LLVM_DEBUG(llvm::dbgs() << "\n");                                   \
    if (auto r = API(__VA_ARGS__))                                      \
      llvm::report_fatal_error(llvm::Twine(#API " failed with ") +      \
                               to_string(r));                           \
  } while (0)

#define TRY_XAIE_API_EMIT_ERROR(OP, API, ...)                           \
  do {                                                                  \
    LLVM_DEBUG(llvm::dbgs() << "XAIE API: " << #API << " with args: "); \
    LLVM_DEBUG(SHOW_ARGS(llvm::dbgs(), __VA_ARGS__));                   \
    LLVM_DEBUG(llvm::dbgs() << "\n");                                   \
    if (auto r = API(__VA_ARGS__))                                      \
      return OP.emitOpError() << #API " failed with " << r;             \
  } while (0)

#define TRY_XAIE_API_LOGICAL_RESULT(API, ...)                           \
  do {                                                                  \
    LLVM_DEBUG(llvm::dbgs() << "XAIE API: " << #API << " with args: "); \
    LLVM_DEBUG(SHOW_ARGS(llvm::dbgs(), __VA_ARGS__));                   \
    LLVM_DEBUG(llvm::dbgs() << "\n");                                   \
    if (auto r = API(__VA_ARGS__)) {                                    \
      llvm::errs() << #API " failed with " << r;                        \
      return failure();                                                 \
    }                                                                   \
  } while (0)

StrmSwPortType getConnectingBundle(StrmSwPortType dir);

}  // namespace mlir::iree_compiler::AMDAIE

namespace llvm {
template <>
struct DenseMapInfo<mlir::iree_compiler::AMDAIE::TileLoc> {
  using FirstInfo = DenseMapInfo<int>;
  using SecondInfo = DenseMapInfo<int>;

  static mlir::iree_compiler::AMDAIE::TileLoc getEmptyKey() {
    return {FirstInfo::getEmptyKey(), SecondInfo::getEmptyKey()};
  }

  static mlir::iree_compiler::AMDAIE::TileLoc getTombstoneKey() {
    return {FirstInfo::getTombstoneKey(), SecondInfo::getTombstoneKey()};
  }

  static unsigned getHashValue(const mlir::iree_compiler::AMDAIE::TileLoc& t) {
    return llvm::detail::combineHashValue(FirstInfo::getHashValue(t.col),
                                          SecondInfo::getHashValue(t.row));
  }

  static bool isEqual(const mlir::iree_compiler::AMDAIE::TileLoc& lhs,
                      const mlir::iree_compiler::AMDAIE::TileLoc& rhs) {
    return lhs == rhs;
  }
};

template <>
struct DenseMapInfo<mlir::iree_compiler::AMDAIE::DMAChannel> {
  using FirstInfo = DenseMapInfo<mlir::iree_compiler::AMDAIE::DMAChannelDir>;
  using SecondInfo = DenseMapInfo<int>;

  static mlir::iree_compiler::AMDAIE::DMAChannel getEmptyKey() {
    return {FirstInfo::getEmptyKey(), SecondInfo::getEmptyKey()};
  }

  static mlir::iree_compiler::AMDAIE::DMAChannel getTombstoneKey() {
    return {FirstInfo::getTombstoneKey(), SecondInfo::getTombstoneKey()};
  }

  static unsigned getHashValue(
      const mlir::iree_compiler::AMDAIE::DMAChannel& d) {
    return detail::combineHashValue(FirstInfo::getHashValue(d.direction),
                                    SecondInfo::getHashValue(d.channel));
  }

  static bool isEqual(const mlir::iree_compiler::AMDAIE::DMAChannel& lhs,
                      const mlir::iree_compiler::AMDAIE::DMAChannel& rhs) {
    return lhs == rhs;
  }
};

template <>
struct DenseMapInfo<mlir::iree_compiler::AMDAIE::Port> {
  using FirstInfo = DenseMapInfo<StrmSwPortType>;
  using SecondInfo = DenseMapInfo<int>;

  static mlir::iree_compiler::AMDAIE::Port getEmptyKey() {
    return {FirstInfo::getEmptyKey(), SecondInfo::getEmptyKey()};
  }

  static mlir::iree_compiler::AMDAIE::Port getTombstoneKey() {
    return {FirstInfo::getTombstoneKey(), SecondInfo::getTombstoneKey()};
  }

  static unsigned getHashValue(const mlir::iree_compiler::AMDAIE::Port& d) {
    return detail::combineHashValue(FirstInfo::getHashValue(d.bundle),
                                    SecondInfo::getHashValue(d.channel));
  }

  static bool isEqual(const mlir::iree_compiler::AMDAIE::Port& lhs,
                      const mlir::iree_compiler::AMDAIE::Port& rhs) {
    return lhs == rhs;
  }
};

template <>
struct DenseMapInfo<mlir::iree_compiler::AMDAIE::Switchbox>
    : DenseMapInfo<mlir::iree_compiler::AMDAIE::TileLoc> {};

}  // namespace llvm

#endif  // IREE_AIE_RUNTIME_H
