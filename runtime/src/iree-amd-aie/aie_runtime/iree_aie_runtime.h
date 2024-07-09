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

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"

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

enum class AMDAIEDevice : uint32_t {
  xcvc1902 = 1,
  xcve2302 = 2,
  xcve2802 = 3,
  npu1 = 4,
  npu1_1col = 5,
  npu1_2col = 6,
  npu1_3col = 7,
  npu1_4col = 8,
};

enum class AMDAIETileType : uint8_t {
  AIETILE = 0,
  SHIMNOC = 1,
  SHIMPL = 2,
  MEMTILE = 3,
  MAX = 4
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
  _(AieRC)            \
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
    return out << label << "=" << "ptr";
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
}  // namespace llvm

#endif  // IREE_AIE_RUNTIME_H
