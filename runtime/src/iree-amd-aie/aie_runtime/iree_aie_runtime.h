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
#include <type_traits>

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"
#include "macros.h"
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

void startCDOFileStream(const char *cdoFileName);
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
  // for std::transform
  TileLoc &operator=(const TileLoc &t) = default;

  TileLoc(XAie_LocType loc) : col(loc.Col), row(loc.Row) {}

  operator XAie_LocType() const { return XAie_TileLoc(col, row); }

  // for getting free DenseMapInfo (see below)
  using TupleType = std::tuple<int, int>;

  TileLoc(TupleType t) : TileLoc(std::get<0>(t), std::get<1>(t)) {}

  operator TupleType() const { return {col, row}; }
  TUPLE_LIKE_STRUCT_RELATIONAL_OPS(TileLoc)
};

ASSERT_STANDARD_LAYOUT(TileLoc);

static_assert(static_cast<uint8_t>(DMAChannelDir::MM2S) ==
                      static_cast<uint8_t>(XAie_DmaDirection::DMA_MM2S) &&
                  static_cast<uint8_t>(DMAChannelDir::S2MM) ==
                      static_cast<uint8_t>(XAie_DmaDirection::DMA_S2MM),
              "DMAChannelDir and XAie_DmaDirection don't line up");

struct SwitchDMAConnection {
  DMAChannelDir direction;
  uint8_t channel;

  SwitchDMAConnection(DMAChannelDir direction, uint8_t channel)
      : direction(direction), channel(channel) {}

  bool operator==(const SwitchDMAConnection &rhs) const {
    return std::tie(direction, channel) == std::tie(rhs.direction, rhs.channel);
  }
};

ASSERT_STANDARD_LAYOUT(SwitchDMAConnection);

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

/// Enum of DMA BD properties. Uses the offset within the `XAie_DmaBdProp`
/// struct as underlying value to easily retrieve the specified property with a
/// single getter method, while being versatile towards `XAie_DmaBdProp` struct
/// changes.
enum class AMDAIEDmaBdProp : uint8_t {
  AddrMax = offsetof(XAie_DmaBdProp, AddrMax),
  AddrAlignMask = offsetof(XAie_DmaBdProp, AddrAlignMask),
  AddrAlignShift = offsetof(XAie_DmaBdProp, AddrAlignShift),
  LenActualOffset = offsetof(XAie_DmaBdProp, LenActualOffset),
  StepSizeMax = offsetof(XAie_DmaBdProp, StepSizeMax),
  WrapMax = offsetof(XAie_DmaBdProp, WrapMax),
  IterStepSizeMax = offsetof(XAie_DmaBdProp, IterStepSizeMax),
  IterWrapMax = offsetof(XAie_DmaBdProp, IterWrapMax),
  IterCurrMax = offsetof(XAie_DmaBdProp, IterCurrMax),
  MAX = sizeof(XAie_DmaBdProp)
};

static_assert(static_cast<uint8_t>(StrmSwPortType::CORE) ==
                  ::StrmSwPortType::CORE,
              "mlir::iree_compiler::AMDAIE::StrmSwPortType is out of sync with "
              "aie-rt's StrmSwPortType");
static_assert(static_cast<uint8_t>(StrmSwPortType::CORE) == 0,
              "mlir::iree_compiler::AMDAIE::StrmSwPortType is out of sync with "
              "aie-rt's StrmSwPortType");
static_assert(static_cast<uint8_t>(StrmSwPortType::SS_PORT_TYPE_MAX) ==
                  ::StrmSwPortType::SS_PORT_TYPE_MAX,
              "mlir::iree_compiler::AMDAIE::StrmSwPortType is out of sync with "
              "aie-rt's StrmSwPortType");

inline ::StrmSwPortType strmTtoStrmT(StrmSwPortType t) {
  return static_cast<::StrmSwPortType>(t);
}

enum class XAie_TxnOpcode : uint8_t {
  XAIE_IO_WRITE = ::XAie_TxnOpcode::XAIE_IO_WRITE,
  XAIE_IO_BLOCKWRITE,
  XAIE_IO_BLOCKSET,
  XAIE_IO_MASKWRITE,
  XAIE_IO_MASKPOLL,
  XAIE_CONFIG_SHIMDMA_BD,
  XAIE_CONFIG_SHIMDMA_DMABUF_BD,
  XAIE_IO_CUSTOM_OP_BEGIN = ::XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_BEGIN,
  XAIE_IO_CUSTOM_OP_TCT = ::XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_BEGIN,
  XAIE_IO_CUSTOM_OP_DDR_PATCH,
  XAIE_IO_CUSTOM_OP_READ_REGS,
  XAIE_IO_CUSTOM_OP_RECORD_TIMER,
  XAIE_IO_CUSTOM_OP_MERGE_SYNC,
  XAIE_IO_CUSTOM_OP_NEXT,
  XAIE_IO_CUSTOM_OP_MAX = ::XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_MAX,
};

static_assert(static_cast<uint8_t>(XAie_TxnOpcode::XAIE_IO_WRITE) == 0,
              "mlir::iree_compiler::AMDAIE::XAie_TxnOpcode is out of sync with "
              "aie-rt's XAie_TxnOpcode");
static_assert(
    static_cast<uint8_t>(XAie_TxnOpcode::XAIE_CONFIG_SHIMDMA_DMABUF_BD) ==
        ::XAie_TxnOpcode::XAIE_CONFIG_SHIMDMA_DMABUF_BD,
    "mlir::iree_compiler::AMDAIE::XAie_TxnOpcode is out of sync with "
    "aie-rt's XAie_TxnOpcode");
static_assert(
    static_cast<uint8_t>(XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_DDR_PATCH) ==
        ::XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_DDR_PATCH,
    "mlir::iree_compiler::AMDAIE::XAie_TxnOpcode is out of sync with "
    "aie-rt's XAie_TxnOpcode");
static_assert(static_cast<uint8_t>(XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_NEXT) ==
                  ::XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_NEXT,
              "mlir::iree_compiler::AMDAIE::XAie_TxnOpcode is out of sync with "
              "aie-rt's XAie_TxnOpcode");

inline ::XAie_TxnOpcode txnToTxn(XAie_TxnOpcode t) {
  return static_cast<::XAie_TxnOpcode>(t);
}

// mlir-air legacy
enum class AIEArch : uint8_t { AIE1 = 1, AIE2 = 2 };

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
  /// Contains additional device config parameters that can't be retrieved from
  /// aie-rt for whatever reason. Make sure the parameters can't be retrieved in
  /// another way before adding new fields to this struct.
  struct AMDAIEDeviceConfig {
    /// Currently, the max arbiter/msel is hidden inside aie-rt.
    uint8_t streamSwitchCoreArbiterMax{0};
    uint8_t streamSwitchCoreMSelMax{0};
    uint8_t streamSwitchMemTileArbiterMax{0};
    uint8_t streamSwitchMemTileMSelMax{0};
    uint8_t streamSwitchShimArbiterMax{0};
    uint8_t streamSwitchShimMSelMax{0};
    AMDAIEDeviceConfig() = default;
  };
  XAie_Config configPtr;
  XAie_DevInst devInst;
  AMDAIEDeviceConfig deviceConfig;

  explicit AMDAIEDeviceModel(uint8_t aieGen, uint64_t baseAddr,
                             uint8_t colShift, uint8_t rowShift,
                             uint8_t devNColumns, uint8_t devNRows,
                             uint8_t memTileRowStart, uint8_t nMemTileRows,
                             uint8_t nShimTileRows, int partitionNumCols,
                             int partitionStartCol, uint64_t partBaseAddr,
                             uint64_t npiAddr, bool aieSim, bool xaieDebug,
                             AMDAIEDevice device,
                             AMDAIEDeviceConfig deviceConfig);

  int rows() const;
  int columns() const;

  AMDAIETileType getTileType(uint8_t col, uint8_t row) const;
  bool isCoreTile(uint8_t col, uint8_t row) const;
  bool isMemTile(uint8_t col, uint8_t row) const;
  bool isShimNOCTile(uint8_t col, uint8_t row) const;
  bool isShimPLTile(uint8_t col, uint8_t row) const;
  bool isShimNOCorPLTile(uint8_t col, uint8_t row) const;
  bool isShimTile(uint8_t col, uint8_t row) const;

  /// Retrieve a DMA property for the specified tile type.
  template <typename T>
  T getDmaProp(AMDAIETileType tileType, AMDAIEDmaProp dmaProp) const {
    const uint8_t *dmaMod = reinterpret_cast<const uint8_t *>(
        devInst.DevProp.DevMod[static_cast<uint8_t>(tileType)].DmaMod);
    return *((const T *)(dmaMod + static_cast<uint8_t>(dmaProp)));
  }

  /// Retrieve a DMA BD property for the specified tile type and BD id.
  template <typename T>
  T getDmaBdProp(AMDAIETileType tileType, uint8_t bd_id,
                 AMDAIEDmaBdProp dmaBdProp) const {
    const XAie_DmaMod *dmaMod =
        devInst.DevProp.DevMod[static_cast<uint8_t>(tileType)].DmaMod;
    assert(bd_id < dmaMod->NumBds && "BD id should be smaller than max");
    const uint8_t *dmaBdMod =
        reinterpret_cast<const uint8_t *>(&dmaMod->BdProp[bd_id]);
    return *((const T *)(dmaBdMod + static_cast<uint8_t>(dmaBdProp)));
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

  uint32_t getNumSourceSwitchBoxConnections(uint8_t col, uint8_t row,
                                            StrmSwPortType bundle) const;
  uint32_t getNumDestSwitchBoxConnections(uint8_t col, uint8_t row,
                                          StrmSwPortType bundle) const;
  bool isLegalTileConnection(uint8_t col, uint8_t row, StrmSwPortType srcBundle,
                             uint8_t srcChan, StrmSwPortType dstBundle,
                             uint8_t dstChan) const;

  uint32_t getColumnShift() const;
  uint32_t getRowShift() const;

  uint8_t getStreamSwitchArbiterMax(uint8_t col, uint8_t row) const;
  uint8_t getStreamSwitchMSelMax(uint8_t col, uint8_t row) const;

  /// Return a map from channels to valid BD ids for the requested tile type.
  /// TODO(jornt): find these ranges in the device model.
  DenseMap<uint32_t, SmallVector<uint32_t>> getChannelToValidBdIds(
      AMDAIETileType tileType) const;

  AMDAIEDevice device;

  // mlir-air legacy
  uint32_t getNumDestSwitchboxConnections(int col, int row,
                                          StrmSwPortType bundle) const;
  uint32_t getNumMemTileRows() const { return 1; }
  AIEArch getTargetArch() const { return AIEArch::AIE2; }
};

struct AMDAIEDeviceModel getDeviceModel(AMDAIEDevice device);
StrmSwPortType getConnectingBundle(StrmSwPortType dir);
bool isNPUDevice(mlir::iree_compiler::AMDAIE::AMDAIEDevice d);

/// ============================= BEGIN ==================================
/// ================== stringification utils =============================
/// ======================================================================

#define TO_STRINGS(_)     \
  _(int)                  \
  _(uint32_t)             \
  _(uint64_t)             \
  _(AMDAIEDmaProp)        \
  _(AMDAIETileType)       \
  _(AieRC)                \
  _(DMAChannelDir)        \
  _(StrmSwPortType)       \
  _(SwitchDMAConnection)  \
  _(::StrmSwPortType)     \
  _(TileLoc)              \
  _(XAie_LocType)         \
  _(XAie_Lock)            \
  _(XAie_OpHdr)           \
  _(XAie_Write32Hdr)      \
  _(XAie_BlockWrite32Hdr) \
  _(XAie_MaskWrite32Hdr)  \
  _(XAie_MaskPoll32Hdr)   \
  _(XAie_CustomOpHdr)     \
  _(XAie_TxnOpcode)       \
  _(XAie_TxnCmd)          \
  _(XAie_Packet)

TO_STRINGS(TO_STRING_DECL)
#undef TO_STRINGS

#define BOTH_OSTREAM_OPS_FORALL_TYPES(OSTREAM_OP_, _)              \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::AMDAIETileType)      \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::TileLoc)             \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::SwitchDMAConnection) \
  _(OSTREAM_OP_, mlir::iree_compiler::AMDAIE::DMAChannelDir)       \
  _(OSTREAM_OP_, AieRC)                                            \
  _(OSTREAM_OP_, StrmSwPortType)                                   \
  _(OSTREAM_OP_, ::StrmSwPortType)                                 \
  _(OSTREAM_OP_, XAie_LocType)                                     \
  _(OSTREAM_OP_, XAie_Lock)                                        \
  _(OSTREAM_OP_, XAie_OpHdr)                                       \
  _(OSTREAM_OP_, XAie_Write32Hdr)                                  \
  _(OSTREAM_OP_, XAie_BlockWrite32Hdr)                             \
  _(OSTREAM_OP_, XAie_MaskWrite32Hdr)                              \
  _(OSTREAM_OP_, XAie_MaskPoll32Hdr)                               \
  _(OSTREAM_OP_, XAie_CustomOpHdr)                                 \
  _(OSTREAM_OP_, XAie_TxnOpcode)                                   \
  _(OSTREAM_OP_, XAie_TxnCmd)                                      \
  _(OSTREAM_OP_, XAie_Packet)

BOTH_OSTREAM_OPS_FORALL_TYPES(OSTREAM_OP_DECL, BOTH_OSTREAM_OP)

// https://stackoverflow.com/a/32230306
template <typename H1>
llvm::raw_ostream &showArgs(llvm::raw_ostream &out, const char *label,
                            H1 &&value) {
  if constexpr (std::is_pointer_v<H1> ||
                std::is_pointer_v<std::remove_reference_t<H1>>) {
    return out << label << "=ptr";
  } else {
    return out << label << "=" << to_string(std::forward<H1>(value));
  }
}

template <typename H1, typename... T>
llvm::raw_ostream &showArgs(llvm::raw_ostream &out, const char *label,
                            H1 &&value, T &&...rest) {
  const char *pcomma = strchr(label, ',');
  if constexpr (std::is_pointer_v<H1> ||
                std::is_pointer_v<std::remove_reference_t<H1>>) {
    return showArgs(out.write(label, pcomma - label) << "=ptr,", pcomma + 1,
                    std::forward<T>(rest)...);
  } else {
    return showArgs(out.write(label, pcomma - label)
                        << "=" << to_string(std::forward<H1>(value)) << ',',
                    pcomma + 1, std::forward<T>(rest)...);
  }
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
    LLVM_DEBUG(llvm::dbgs().flush());                                   \
    if (auto r = API(__VA_ARGS__))                                      \
      llvm::report_fatal_error(llvm::Twine(#API " failed with ") +      \
                               to_string(r) + "\n");                    \
  } while (0)

#define TRY_XAIE_API_LOGICAL_RESULT(API, ...)                           \
  do {                                                                  \
    LLVM_DEBUG(llvm::dbgs() << "XAIE API: " << #API << " with args: "); \
    LLVM_DEBUG(SHOW_ARGS(llvm::dbgs(), __VA_ARGS__));                   \
    LLVM_DEBUG(llvm::dbgs() << "\n");                                   \
    LLVM_DEBUG(llvm::dbgs().flush());                                   \
    if (auto r = API(__VA_ARGS__)) {                                    \
      llvm::errs() << #API " failed with " << r << "\n";                \
      return failure();                                                 \
    }                                                                   \
  } while (0)
}  // namespace mlir::iree_compiler::AMDAIE

namespace llvm {
template <typename TupleT>
struct TupleStructDenseMapInfo : DenseMapInfo<TupleT> {};

template <>
struct DenseMapInfo<mlir::iree_compiler::AMDAIE::TileLoc>
    : TupleStructDenseMapInfo<mlir::iree_compiler::AMDAIE::TileLoc::TupleType> {
};
}  // namespace llvm

template <>
struct std::hash<mlir::iree_compiler::AMDAIE::TileLoc> {
  std::size_t operator()(
      const mlir::iree_compiler::AMDAIE::TileLoc &s) const noexcept {
    std::size_t h1 = std::hash<int>{}(s.col);
    std::size_t h2 = std::hash<int>{}(s.row);
    return h1 ^ (h2 << 1);
  }
};

#endif  // IREE_AIE_RUNTIME_H
