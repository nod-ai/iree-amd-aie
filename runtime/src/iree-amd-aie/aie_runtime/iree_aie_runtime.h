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

#define s8
#define u8
#define u16
#define s32
#define u32
#define u64

enum byte_ordering { Little_Endian, Big_Endian };
void startCDOFileStream(const char *cdoFileName);
void endCurrentCDOFileStream();
void FileHeader();
void EnAXIdebug();
void setEndianness(bool endianness);
void configureHeader();
void insertNoOpCommand(unsigned int numPadBytes);
}

#define XAIE2_BASE_ADDR 0x40000000
#define XAIE1_BASE_ADDR 0x20000000000
#define XAIE2_COL_SHIFT 25
#define XAIE2_ROW_SHIFT 20
#define XAIE1_COL_SHIFT 23
#define XAIE1_ROW_SHIFT 18
#define XAIE_SHIM_ROW 0
#define XAIE_MEM_TILE_ROW_START 1
#define XAIE_PARTITION_BASE_ADDR 0x0

#define NPI_ADDR 0x0
#define NUM_LOCKS 16
#define MEM_TILE_LOCK_ID_INCR 64
#define BASE_ADDR_A_INCR 0x80000

std::string AIERCTOSTR(AieRC rc);

// https://stackoverflow.com/a/32230306
template <typename H1>
llvm::raw_ostream &showArgs(llvm::raw_ostream &out, const char *label,
                            H1 &&value) {
  return out << label << "=" << std::forward<H1>(value);
}

template <typename H1, typename... T>
llvm::raw_ostream &showArgs(llvm::raw_ostream &out, const char *label,
                            H1 &&value, T &&...rest) {
  const char *pcomma = strchr(label, ',');
  return showArgs(out.write(label, pcomma - label)
                      << "=" << std::forward<H1>(value) << ',',
                  pcomma + 1, std::forward<T>(rest)...);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const XAie_LocType &loc);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const XAie_Lock &lock);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const XAie_Packet &packet);

#define SHOW_ARGS(os, ...) showArgs(os, #__VA_ARGS__, __VA_ARGS__)
#define TRY_XAIE_API_FATAL_ERROR(API, ...)                              \
  do {                                                                  \
    LLVM_DEBUG(llvm::dbgs() << "XAIE API: " << #API << " with args: "); \
    LLVM_DEBUG(SHOW_ARGS(llvm::dbgs(), __VA_ARGS__));                   \
    LLVM_DEBUG(llvm::dbgs() << "\n");                                   \
    if (auto r = API(__VA_ARGS__))                                      \
      llvm::report_fatal_error(llvm::Twine(#API " failed with ") +      \
                               AIERCTOSTR(r));                          \
  } while (0)

struct TileLoc {
  // friend definition (will define the function as a non-member function in the
  // namespace surrounding the class).
  friend std::ostream &operator<<(std::ostream &os, const TileLoc &s) {
    os << "TileLoc(" << s.col << ", " << s.row << ")";
    return os;
  }

  friend std::string to_string(const TileLoc &s) {
    std::ostringstream ss;
    ss << s;
    return ss.str();
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const TileLoc &s) {
    os << to_string(s);
    return os;
  }

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

struct AMDAIENPUDeviceModel {
  XAie_Config configPtr;
  XAie_DevInst devInst;

  explicit AMDAIENPUDeviceModel(
      uint8_t aieGen, uint64_t baseAddr, uint8_t colShift, uint8_t rowShift,
      uint8_t nColumns, uint8_t, uint8_t memTileRowStart, uint8_t nMemTileRows,
      uint8_t nShimTileRows, uint8_t partitionStartCol,
      uint8_t partitionNumCols, bool aieSim = false, bool xaieDebug = false);

  int rows() const;
  int columns() const;
  uint32_t getNumMemTileRows() const;

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
std::string stringifyStrmSwPortType(StrmSwPortType val);

#endif  // IREE_AIE_RUNTIME_H
