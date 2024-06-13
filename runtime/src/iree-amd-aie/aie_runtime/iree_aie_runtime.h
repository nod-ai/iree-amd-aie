// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions. See
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: # Apache-2.0 WITH LLVM-exception

#ifndef IREE_AIE_RUNTIME_H
#define IREE_AIE_RUNTIME_H

#include <optional>

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

enum byte_ordering { Little_Endian, Big_Endian };
void startCDOFileStream(const char* cdoFileName);
void endCurrentCDOFileStream();
void FileHeader();
void EnAXIdebug();
void setEndianness(bool endianness);
void configureHeader();
}

struct AMDAIENPUTargetModel {
  int rows() { return 6; /* 1 Shim row, 1 memtile row, and 4 Core rows. */ }
  int columns() { return 5; }

  bool isCoreTile(int col, int row) { return row > 1; }

  bool isMemTile(int col, int row) { return row == 1; }

  uint32_t getNumLocks(int col, int row) {
    return isMemTile(col, row) ? 64 : 16;
  }

  bool isShimNOCTile(int col, int row) { return row == 0 && col > 0; }

  bool isShimPLTile(int col, int row) {
    // This isn't useful because it's not connected to anything.
    return row == 0 && col == 0;
  }

  uint32_t getNumMemTileRows() { return 1; }

  std::optional<XAie_LocType> getMemWest(XAie_LocType src);
  std::optional<XAie_LocType> getMemEast(XAie_LocType src);
  std::optional<XAie_LocType> getMemNorth(XAie_LocType src);
  std::optional<XAie_LocType> getMemSouth(XAie_LocType src);

  bool isMemWest(int srcCol, int srcRow, int dstCol, int dstRow);
  bool isMemEast(int srcCol, int srcRow, int dstCol, int dstRow);
  bool isMemNorth(int srcCol, int srcRow, int dstCol, int dstRow);
  bool isMemSouth(int srcCol, int srcRow, int dstCol, int dstRow);

  bool isLegalMemAffinity(int coreCol, int coreRow, int memCol, int memRow);

  static uint32_t getMemInternalBaseAddress() {
    return getMemEastBaseAddress();
  }

  static uint32_t getMemSouthBaseAddress() { return 0x00040000; }
  static uint32_t getMemWestBaseAddress() { return 0x00050000; }
  static uint32_t getMemNorthBaseAddress() { return 0x00060000; }
  static uint32_t getMemEastBaseAddress() { return 0x00070000; }
  static uint32_t getLocalMemorySize() { return 0x00010000; }

  uint32_t getNumBDs(int col, int row) { return isMemTile(col, row) ? 48 : 16; }

  uint32_t getMemTileSize() { return 0x00080000; }

  uint32_t getNumDestSwitchboxConnections(int col, int row,
                                          StrmSwPortType bundle);
  uint32_t getNumSourceSwitchboxConnections(int col, int row,
                                            StrmSwPortType bundle);
  uint32_t getNumDestShimMuxConnections(int col, int row,
                                        StrmSwPortType bundle);
  uint32_t getNumSourceShimMuxConnections(int col, int row,
                                          StrmSwPortType bundle);
  bool isLegalMemtileConnection(StrmSwPortType srcBundle, int srcChan,
                                StrmSwPortType dstBundle, int dstChan);
};

#endif  // IREE_AIE_RUNTIME_H
