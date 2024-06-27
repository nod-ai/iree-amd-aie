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

#ifdef _WIN32
#ifndef IREE_AIE_RUNTIME_EXPORT
#ifdef iree_aie_runtime_EXPORTS
// We are building this library
#define IREE_AIE_RUNTIME_EXPORT __declspec(dllexport)
#else
// We are using this library
#define IREE_AIE_RUNTIME_EXPORT __declspec(dllimport)
#endif  // iree_aie_runtime_EXPORTS
#endif  // IREE_AIE_RUNTIME_EXPORT
#else
// Non-windows: use visibility attributes.
#define IREE_AIE_RUNTIME_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32

extern "C" {
#include "xaiengine.h"
#include "xaiengine/xaie_device_aieml.h"

enum byte_ordering { Little_Endian, Big_Endian };
void startCDOFileStream(const char *cdoFileName);
void endCurrentCDOFileStream();
void FileHeader();
void EnAXIdebug();
void setEndianness(bool endianness);
void configureHeader();
void insertNoOpCommand(unsigned int numPadBytes);
}

#define s8
#define u8
#define u16
#define s32
#define u32
#define u64

#define XAIE2_BASE_ADDR 0x40000000
#define XAIE1_BASE_ADDR 0x20000000000
#define XAIE1_COL_SHIFT 23
#define XAIE1_ROW_SHIFT 18
#define XAIE2_BASE_ADDR 0x40000000
#define XAIE2_COL_SHIFT 25
#define XAIE2_ROW_SHIFT 20
#define XAIE_MEM_TILE_ROW_START 1
#define XAIE_NUM_MEM_TILE_ROWS 1
#define XAIE_NUM_SHIM_TILE_ROWS 1
#define XAIE_PARTITION_BASE_ADDR 0x0
#define XAIE_SHIM_ROW 0

#define NPI_ADDR 0x0
#define NUM_LOCKS 16
#define MEM_TILE_LOCK_ID_INCR 64
#define BASE_ADDR_A_INCR 0x80000

struct TileLoc {
  inline bool operator<(const TileLoc &rhs) const {
    return std::tie(col, row) < std::tie(rhs.col, rhs.row);
  }

  bool operator==(const TileLoc &rhs) const {
    return std::tie(col, row) == std::tie(rhs.col, rhs.row);
  }

  bool operator!=(const TileLoc &rhs) const { return !(*this == rhs); }

  operator XAie_LocType() const { return XAie_TileLoc(col, row); }
  TileLoc(XAie_LocType loc) : col(loc.Col), row(loc.Row) {}
  TileLoc(int col, int row) : col(col), row(row) {}

  int col, row;
};

enum class AMDAIEDevice : uint32_t {
  xcvc1902 = 1,
  xcve2302 = 2,
  xcve2802 = 3,
  npu = 4,
  npu1 = 4,
  npu1_1col = 5,
  npu1_2col = 6,
  npu1_3col = 7,
  npu1_4col = 8,
};

enum class AMDAIETileType : uint8_t {
  AIETILE = 0U,
  SHIMNOC = 1U,
  SHIMPL = 2U,
  MEMTILE = 3U,
  MAX = 4U
};

struct AMDAIENPUDeviceModel {
  XAie_Config configPtr;
  XAie_DevInst devInst;

  explicit AMDAIENPUDeviceModel(uint8_t aieGen, uint64_t baseAddr,
                                uint8_t colShift, uint8_t rowShift,
                                uint8_t nColumns, uint8_t,
                                uint8_t memTileRowStart, uint8_t nMemTileRows,
                                uint8_t nShimTileRows, uint8_t partitionNumCols,
                                uint8_t partitionStartCol = 1,
                                bool aieSim = false, bool xaieDebug = false);

  int rows() const;
  int columns() const;

  AMDAIETileType getTileType(uint8_t col, uint8_t row);
  bool isCoreTile(uint8_t col, uint8_t row);
  bool isMemTile(uint8_t col, uint8_t row);
  bool isShimNOCTile(uint8_t col, uint8_t row);
  bool isShimPLTile(uint8_t col, uint8_t row);

  uint32_t getNumLocks(uint8_t col, uint8_t row);

  std::optional<TileLoc> getMemWest(TileLoc src);
  std::optional<TileLoc> getMemEast(TileLoc src);
  std::optional<TileLoc> getMemNorth(TileLoc src);
  std::optional<TileLoc> getMemSouth(TileLoc src);

  bool hasMemWest(uint8_t srcCol, uint8_t srcRow, uint8_t dstCol,
                  uint8_t dstRow);
  bool hasMemEast(uint8_t srcCol, uint8_t srcRow, uint8_t dstCol,
                  uint8_t dstRow);
  bool hasMemNorth(uint8_t srcCol, uint8_t srcRow, uint8_t dstCol,
                   uint8_t dstRow);
  bool hasMemSouth(uint8_t srcCol, uint8_t srcRow, uint8_t dstCol,
                   uint8_t dstRow);
  /// Return true if core can access the memory in mem
  bool hasLegalMemAffinity(uint8_t coreCol, uint8_t coreRow, uint8_t memCol,
                           uint8_t memRow);

  uint32_t getMemInternalBaseAddress();
  uint32_t getMemSouthBaseAddress();
  uint32_t getMemWestBaseAddress();
  uint32_t getMemNorthBaseAddress();
  uint32_t getMemEastBaseAddress();
  uint32_t getLocalMemorySize(uint8_t col, uint8_t row);
  uint32_t getMemTileSize(uint8_t col, uint8_t row);

  uint32_t getNumBDs(uint8_t col, uint8_t row);

  uint32_t getNumSourceSwitchboxConnections(uint8_t col, uint8_t row,
                                            StrmSwPortType bundle);
  uint32_t getNumDestSwitchboxConnections(uint8_t col, uint8_t row,
                                          StrmSwPortType bundle);
  bool isLegalMemtileConnection(uint8_t col, uint8_t row,
                                StrmSwPortType srcBundle, uint8_t srcChan,
                                StrmSwPortType dstBundle, uint8_t dstChan);
};

namespace mlir::iree_compiler::AMDAIE {

struct AMDAIENPUDeviceModel getDeviceModel(AMDAIEDevice device);

}  // namespace mlir::iree_compiler::AMDAIE

StrmSwPortType getConnectingStrmSwPortType(StrmSwPortType dir);

#define OSTREAM_OP(O_TYPE, TYPE) O_TYPE &operator<<(O_TYPE &os, const TYPE &s);

namespace mlir::iree_compiler::AMDAIE {
#define TO_STRING(TYPE) std::string to_string(const TYPE &t);

#define TO_STRINGS(_) \
  _(AieRC)            \
  _(XAie_LocType)     \
  _(XAie_Lock)        \
  _(XAie_Packet)

    TO_STRINGS(TO_STRING)
#undef TO_STRING
#undef TO_STRINGS
}  // namespace mlir::iree_compiler::AMDAIE

#define BOTH_OSTREAM_OP(OSTREAM_OP_, TYPE) \
  OSTREAM_OP_(std::ostream, TYPE)          \
  OSTREAM_OP_(llvm::raw_ostream, TYPE)

#define BOTH_OSTREAM_OPS_FORALL_TYPES(OSTREAM_OP_, _) \
  _(OSTREAM_OP_, AieRC)                               \
  _(OSTREAM_OP_, StrmSwPortType)                      \
  _(OSTREAM_OP_, XAie_LocType)                        \
  _(OSTREAM_OP_, XAie_Lock)                           \
  _(OSTREAM_OP_, XAie_Packet)

BOTH_OSTREAM_OPS_FORALL_TYPES(OSTREAM_OP, BOTH_OSTREAM_OP)
#undef OSTREAM_OP

// https://stackoverflow.com/a/32230306
template <typename H1>
llvm::raw_ostream &showArgs(llvm::raw_ostream &out, const char *label,
                            H1 &&value) {
    if constexpr (std::is_pointer<H1>::value)
        return out << label << "=" << "ptr";
    else
        return out << label << "=" << std::forward<H1>(value);
}

template <typename H1, typename... T>
llvm::raw_ostream &showArgs(llvm::raw_ostream &out, const char *label,
                            H1 &&value, T &&...rest) {
    const char *pcomma = strchr(label, ',');
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

#define TRY_XAIE_API_FATAL_ERROR(API, ...)                                 \
  do {                                                                     \
    LLVM_DEBUG(llvm::dbgs() << "XAIE API: " << #API << " with args: ");    \
    LLVM_DEBUG(SHOW_ARGS(llvm::dbgs(), __VA_ARGS__));                      \
    LLVM_DEBUG(llvm::dbgs() << "\n");                                      \
    if (auto r = API(__VA_ARGS__))                                         \
      llvm::report_fatal_error(llvm::Twine(#API " failed with ") +         \
                               mlir::iree_compiler::AMDAIE::to_string(r)); \
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

#endif  // IREE_AIE_RUNTIME_H
