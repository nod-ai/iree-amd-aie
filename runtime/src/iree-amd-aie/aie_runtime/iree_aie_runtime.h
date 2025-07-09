// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions. See
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: # Apache-2.0 WITH LLVM-exception

#ifndef IREE_AIE_RUNTIME_H
#define IREE_AIE_RUNTIME_H

#include <optional>
#include <tuple>
#include <type_traits>

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"
#include "macros.h"
#include "mlir/IR/BuiltinTypes.h"

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

  TileLoc() = default;
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
  XAIE_IO_NOOP,
  XAIE_IO_PREEMPT,
  XAIE_IO_MASKPOLL_BUSY,
  XAIE_IO_LOADPDI,
  XAIE_IO_LOAD_PM_START,
  XAIE_IO_CREATE_SCRATCHPAD,
  XAIE_IO_UPDATE_STATE_TABLE,
  XAIE_IO_UPDATE_REG,
  XAIE_IO_UPDATE_SCRATCH,
  XAIE_CONFIG_SHIMDMA_BD,
  XAIE_CONFIG_SHIMDMA_DMABUF_BD,
  XAIE_IO_CUSTOM_OP_BEGIN = ::XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_BEGIN,
  XAIE_IO_CUSTOM_OP_TCT = ::XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_BEGIN,
  XAIE_IO_CUSTOM_OP_DDR_PATCH,
  XAIE_IO_CUSTOM_OP_READ_REGS,
  XAIE_IO_CUSTOM_OP_RECORD_TIMER,
  XAIE_IO_CUSTOM_OP_MERGE_SYNC,
  XAIE_IO_CUSTOM_OP_NEXT,
  XAIE_IO_LOAD_PM_END_INTERNAL = ::XAie_TxnOpcode::XAIE_IO_LOAD_PM_END_INTERNAL,
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
enum class AIEArch : uint8_t { AIE1 = 1, AIE2 = 2, AIE2p = 3 };

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
/// ============================== END ===================================
/// ================== stringification utils =============================
/// ======================================================================

/*
 * This struct is meant to be a thin wrapper around aie-rt, which provides
 * the canonical representation/metadata for AIE devices; attributes like number
 * of locks, bds per tile, whether certain switch connections are legal or not,
 * etc. In addition this struct is meant to contain generational specific AIE
 * VLIW processor constants, such as sizes of vectors supported for
 * load/store/matmul etc.
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
  /// aie-rt or elsewhere for whatever reason. Make sure the parameters can't be
  /// retrieved in another way before adding new fields to this struct.

  struct AMDAIEDeviceConfig {
    ///////////////////////////////////////
    // AIE Array configuration constants //
    ///////////////////////////////////////
    /// Constant specifying the number of inter-iteration dimension for DMA
    /// operations.
    ///
    /// NOTE(jornt): this number is implicitly assumed in the device model and
    /// can't be retrieved from it afaik.
    ///
    /// Some background:
    ///
    /// DMAs support multi-dimensional addressing through buffer descriptors in
    /// two ways:
    /// 1. Intra-iteration access pattern. Specified via 'strides' ('steps' in
    /// buffer descriptor lingo), 'sizes' ('wraps' in buffer descriptro lingo)
    /// and 'padding'. When a DMA executes a buffer descriptor, it will access
    /// the data (read/write) as specified by the intra-iteration access
    /// pattern.
    /// 2. Inter-iteration access pattern. Specified via an iteration 'stride',
    /// 'size' and 'current_iteration' ('stride' is the same as 'stepsize' and
    /// 'size' is the same as 'wrap' in buffer descriptor lingo). Here,
    /// 'current_iteration' keeps track of the current execution iteration of
    /// the buffer descriptor and is incremented after buffer descriptor
    /// execution. the 'stride' is the offset to be used for each execution of
    /// the buffer descriptor, relative to the previous one. When
    /// 'iteration_current' is equal to 'size', the 'iteration_current' is reset
    /// to zero.
    ///
    /// Although DMAs can have a different number of intra-iteration dimensions,
    /// all DMAs have a single inter-iteration dimension (at least in AIE2 and
    /// AIE2p).
    uint8_t dmaNbInterDims = 1;
    /// The number of shim tile rows. Not found in aie-rt data structures, but
    /// provided as `XAIE_SHIM_NUM_ROWS`.
    uint8_t shimTileNumRows{1};
    /// Set default minimum stride bitwidth/addressing granularity to 32 bits as
    /// this is the value for all current architecture versions.
    uint8_t minStrideBitWidth{32};
    /// The max packet id.
    uint8_t packetIdMaxIdx{0};
    /// The max packet type.
    uint8_t packetTypeMax{0};
    /// Suppress Tlast error in control packets.
    bool ctrlPktTlastErrorDisabled{false};

    /// The bitwidth of the packet ID mask. This is currently buried in
    /// aie-rt and not exposed for configuration.
    uint8_t packetIdMaskWidth{5};
    /// Currently, the max arbiter/msel is hidden inside aie-rt.
    uint8_t streamSwitchCoreArbiterMax{0};
    uint8_t streamSwitchCoreMSelMax{0};
    uint8_t streamSwitchMemTileArbiterMax{0};
    uint8_t streamSwitchMemTileMSelMax{0};
    uint8_t streamSwitchShimArbiterMax{0};
    uint8_t streamSwitchShimMSelMax{0};

    //////////////////////////////
    // VLIW processor constants //
    //////////////////////////////
    /// The number of bits that L1 memory must be aligned by in order
    /// to be loaded/stored into a register with a vector instruction. See for
    /// example:
    /// https://www.xilinx.com/htmldocs/xilinx2024_1/aiengine_ml_intrinsics/intrinsics/group__intr__loadstore.html
    uint32_t vectorLoadStoreAlignmentBits{256};
    /// The largest vector size supported. See for example:
    /// https://www.xilinx.com/htmldocs/xilinx2024_1/aiengine_ml_intrinsics/intrinsics/group__group__datatype__vector.html
    uint32_t maxVectorSizeBits{1024};
    /// The number of bits that each of the two vector operands of the shift
    /// intrinsic must have. See for example
    /// https://www.xilinx.com/htmldocs/xilinx2024_1/aiengine_ml_intrinsics/intrinsics/group__intr__gpvectorop__shift.html
    uint32_t shiftOperandBits{512};

    /// On aie2 (and very similar on aie2P), I have observed the
    /// following lowerings at different load sized through peano:
    ///
    /// Number of bytes | vectorized | asm lines (aie2P) |
    /// ----------------+------------+-------------------+
    /// 16              | no         | 14 (vpush.hi.8)   |
    /// 32              | yes        | 6 (1 vst)         |
    /// 64              | yes        | 7 (2 vst)         |
    /// 128             | yes        | 8 (4 vst)         |
    /// 256 LLVM ERROR: unable to legalize instruction.. |
    /// 512             | no         | ~3000 (st.s8)     |
    /// 1024            | no         | ~6000 (st.s8)     |
    /// ----------------+------------+-------------------+
    ///
    /// We choose the maximum number of bytes for which the store is vectorized:
    ///
    /// I suspect this is just 4x the `vectorLoadStoreAlignmentBits` value,
    /// and peano hasn't considered the 8x case (??).
    uint8_t preferredLoadBytes{128};

    AMDAIEDeviceConfig() = default;
  };

  /// Struct representing the format of the packet header, which includes the
  /// following fields:
  /// - [4:0] Stream ID,
  /// - [11:5] Reserved,
  /// - [14:12] Packet type,
  /// - [15] Reserved,
  /// - [20:16] Source row,
  /// - [27:21] Source column,
  /// - [30:28] Reserved,
  /// - [31] Odd parity bit.
  struct AMDAIEPacketHeaderFormat {
    uint8_t streamIdShift{0};
    uint8_t reservedShift0{5};
    uint8_t packetTypeShift{12};
    uint8_t reservedShift1{15};
    uint8_t srcRowShift{16};
    uint8_t srcColShift{21};
    uint8_t reservedShift2{28};
    uint8_t parityShift{31};
  };

  /// Struct representing the format of the control header, which
  /// includes the following fields:
  /// - [19:0] Address,
  /// - [21:20] Beat, the number of 32-bit words data in the packet,
  /// - [23:22] Operation,
  /// - [28:24] Stream ID, for return packet,
  /// - [30:29] Reserved,
  /// - [31] Odd parity bit.
  struct AMDAIEControlHeaderFormat {
    uint8_t addressShift{0};
    uint8_t beatShift{20};
    uint8_t operationShift{22};
    uint8_t streamIdShift{24};
    uint8_t reservedShift{29};
    uint8_t parityShift{31};
  };

  XAie_Config configPtr;
  XAie_DevInst devInst;
  AMDAIEDeviceConfig deviceConfig;
  AMDAIEPacketHeaderFormat packetHeaderFormat;
  AMDAIEControlHeaderFormat controlHeaderFormat;

  explicit AMDAIEDeviceModel(uint8_t aieGen, uint64_t baseAddr,
                             uint8_t colShift, uint8_t rowShift,
                             uint8_t devNColumns, uint8_t devNRows,
                             uint8_t memTileRowStart, uint8_t nMemTileRows,
                             uint8_t nShimTileRows, int partitionNumCols,
                             int partitionStartCol, uint64_t partBaseAddr,
                             uint64_t npiAddr, bool aieSim, bool xaieDebug,
                             AMDAIEDevice device,
                             AMDAIEDeviceConfig deviceConfig);

  uint8_t getMinStrideBitWidth() const;
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
  FailureOr<T> getDmaProp(AMDAIETileType tileType,
                          AMDAIEDmaProp dmaProp) const {
    if (tileType == AMDAIETileType::SHIMPL || tileType == AMDAIETileType::MAX) {
      llvm::errs() << "tileType: " << to_string(tileType)
                   << " does not have DMA properties\n";
      return failure();
    }
    const uint8_t *dmaMod = reinterpret_cast<const uint8_t *>(
        devInst.DevProp.DevMod[static_cast<uint8_t>(tileType)].DmaMod);
    return *((const T *)(dmaMod + static_cast<uint8_t>(dmaProp)));
  }

  /// Retrieve a DMA BD property for the specified tile type and BD id.
  template <typename T>
  FailureOr<T> getDmaBdProp(AMDAIETileType tileType, uint8_t bd_id,
                            AMDAIEDmaBdProp dmaBdProp) const {
    if (tileType == AMDAIETileType::SHIMPL || tileType == AMDAIETileType::MAX) {
      llvm::errs() << "tileType: " << to_string(tileType)
                   << " does not have DMA properties\n";
      return failure();
    }
    const XAie_DmaMod *dmaMod =
        devInst.DevProp.DevMod[static_cast<uint8_t>(tileType)].DmaMod;
    assert(bd_id < dmaMod->NumBds && "BD id should be smaller than max");
    const uint8_t *dmaBdMod =
        reinterpret_cast<const uint8_t *>(&dmaMod->BdProp[bd_id]);
    return *((const T *)(dmaBdMod + static_cast<uint8_t>(dmaBdProp)));
  }

  uint8_t getDmaMaxQueueSize(uint8_t col, uint8_t row) const;

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

  /// Construct a packet header from the specified fields.
  FailureOr<uint32_t> getPacketHeader(uint32_t packetId, uint32_t packetType,
                                      uint32_t srcRow, uint32_t srcCol) const;

  /// Construct a control header from the specified fields.
  FailureOr<uint32_t> getControlHeader(uint32_t address, uint32_t beat,
                                       uint32_t opcode,
                                       uint32_t streamId) const;

  /// Get the maximum for the `address` field in the control packet header.
  uint32_t getCtrlPktMaxAddress() const;
  /// Gets the maximum data length (in beats) of a control packet.
  /// The data length `i` is encoded as `i - 1` in the control packet header.
  /// For example, if 3 bits are allocated for the `length` field, the maximum
  /// length is `2^3 = 8` beats, not 7.
  uint32_t getCtrlPktMaxLength() const;
  /// Get the maximum for the `opcode` field in the control packet header.
  uint32_t getCtrlPktMaxOpcode() const;

  uint32_t getMemInternalBaseAddress() const;
  uint32_t getMemSouthBaseAddress() const;
  uint32_t getMemWestBaseAddress() const;
  uint32_t getMemNorthBaseAddress() const;
  uint32_t getMemEastBaseAddress() const;
  uint32_t getLocalMemorySize(uint8_t col, uint8_t row) const;
  uint32_t getMemTileSizeInBytes() const;
  uint32_t getMemTileSize(uint8_t col, uint8_t row) const;
  uint32_t getCoreTileLocalMemorySize() const;
  uint32_t getTileMemorySizeInBytes(uint8_t col, uint8_t row) const;

  SmallVector<uint32_t> getMemSpaceRows(uint8_t memSpace) const;

  uint32_t getNumSourceSwitchBoxConnections(uint8_t col, uint8_t row,
                                            StrmSwPortType bundle) const;
  uint32_t getNumDestSwitchBoxConnections(uint8_t col, uint8_t row,
                                          StrmSwPortType bundle) const;
  bool isLegalTileConnection(uint8_t col, uint8_t row, StrmSwPortType srcBundle,
                             uint8_t srcChan, StrmSwPortType dstBundle,
                             uint8_t dstChan) const;

  /// Maps an MM2S (shim DMA or NOC) port to its corresponding special shim mux
  /// port.
  static const llvm::SmallDenseMap<std::pair<StrmSwPortType, uint8_t>,
                                   std::pair<StrmSwPortType, uint8_t>>
      mm2sDmaNocToSpecialShimPortMap;

  /// Maps an S2MM (shim DMA or NOC) port to its corresponding special shim mux
  /// port.
  static const llvm::SmallDenseMap<std::pair<StrmSwPortType, uint8_t>,
                                   std::pair<StrmSwPortType, uint8_t>>
      s2mmDmaNocToSpecialShimPortMap;

  /// Retrieves the speicial shim mux port that connects a given MM2S or S2MM
  /// DMA/NOC port. The shim DMA and NOC ports must go through
  /// this special shim mux connection before being further routed to the rest
  /// of the device.
  std::optional<std::pair<StrmSwPortType, uint8_t>>
  getShimMuxPortMappingForDmaOrNoc(StrmSwPortType port, uint8_t channel,
                                   DMAChannelDir direction) const;

  /// Retrieves the original DMA port that corresponds to a given
  /// shim mux port. This performs the reverse lookup for
  /// `getShimMuxPortMappingForDmaOrNoc()`.
  std::optional<std::pair<StrmSwPortType, uint8_t>>
  getDmaFromShimMuxPortMapping(StrmSwPortType port, uint8_t channel,
                               DMAChannelDir direction) const;

  /// The returned string is used by `chess` to identify the device.
  std::optional<std::string> getNPUVersionString() const;
  /// The returned string is used by `peano` to identify the device.
  std::optional<std::string> getTargetArchString() const;

  uint32_t getColumnShift() const;
  uint32_t getRowShift() const;

  /// Returns the starting row index of the core tiles (i.e., AIE tiles).
  uint32_t getCoreTileRowStart() const;

  // Return the magic location in the ELF files containing the size of the
  // program in bytes. The location is returned as a byte offset and number of
  // bytes being used to store the number. NOTE: this could potentially change
  // at any moment in the future.
  std::pair<uint32_t, uint32_t> getElfPmSizeLocationAndNumBytes() const {
    return {72, 4};
  }

  /// Extract the column from a register address.
  uint32_t getColumnFromAddress(uint32_t address) const;
  /// Extract the row from a register address.
  uint32_t getRowFromAddress(uint32_t address) const;
  /// Extract the offset from a register address.
  uint32_t getOffsetFromAddress(uint32_t address) const;

  /// Get the maximum for the `packetId` field in the packet header.
  uint8_t getPacketIdMaxIdx() const;
  /// Get the maximum for the `packetType` field in the packet header.
  uint8_t getpacketTypeMax() const;
  /// Get the bitwidth of the packet id mask.
  uint8_t getPacketIdMaskWidth() const;
  /// Get the maximum number of packet rule slots available for each slave port.
  uint8_t getNumPacketRuleSlots(uint8_t col, uint8_t row) const;
  /// Get the boolean flag indicating whether the device has the control packet
  /// TLAST missing error disabled.
  bool getCtrlPktTlastErrorDisabled() const;

  uint8_t getStreamSwitchArbiterMax(uint8_t col, uint8_t row) const;
  uint8_t getStreamSwitchMSelMax(uint8_t col, uint8_t row) const;

  uint32_t getVectorLoadStoreAlignmentBits() const {
    return deviceConfig.vectorLoadStoreAlignmentBits;
  }

  uint32_t getMaxVectorSizeBits() const {
    return deviceConfig.maxVectorSizeBits;
  }

  uint32_t getNumCoreRows() const { return configPtr.AieTileNumRows; }
  uint32_t getNumCoreCols() const { return columns(); }

  uint32_t getShiftOperandBits() const { return deviceConfig.shiftOperandBits; }

  uint32_t getPreferredLoadBytes() const {
    return deviceConfig.preferredLoadBytes;
  }

  uint32_t getMaxRepeatCount(AMDAIETileType tileType) const {
    return devInst.DevProp.DevMod[static_cast<uint8_t>(tileType)]
        .DmaMod->ChProp->MaxRepeatCount;
  }
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

  FailureOr<std::array<uint32_t, 3>> getAIEMatmulInstructionSize(
      Type elTypeLhs, Type elTypeRhs, Type elTypeAcc) const;

  uint32_t getNumBanks(int col, int row) const {
    return isMemTile(col, row) ? 8 : 4;
  }
};

struct AMDAIEDeviceModel getDeviceModel(AMDAIEDevice device);
StrmSwPortType getConnectingBundle(StrmSwPortType dir);
bool isNPUDevice(mlir::iree_compiler::AMDAIE::AMDAIEDevice d);

/// Given a 32-bit word, set its most significant bit for odd parity.
void setOddParityBit(uint32_t &word);

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
