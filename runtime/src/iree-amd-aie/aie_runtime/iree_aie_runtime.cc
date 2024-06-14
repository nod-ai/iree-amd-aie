// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions. See
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: # Apache-2.0 WITH LLVM-exception

#include "iree_aie_runtime.h"

#include <map>
#include <optional>

#define DEBUG_TYPE "iree-aie-runtime"

#define AIERC_STR(x) x, #x
static const std::map<AieRC, std::string> _AIERCTOSTR = {
    {AIERC_STR(XAIE_OK)},
    {AIERC_STR(XAIE_ERR)},
    {AIERC_STR(XAIE_INVALID_DEVICE)},
    {AIERC_STR(XAIE_INVALID_RANGE)},
    {AIERC_STR(XAIE_INVALID_ARGS)},
    {AIERC_STR(XAIE_INVALID_TILE)},
    {AIERC_STR(XAIE_ERR_STREAM_PORT)},
    {AIERC_STR(XAIE_INVALID_DMA_TILE)},
    {AIERC_STR(XAIE_INVALID_BD_NUM)},
    {AIERC_STR(XAIE_ERR_OUTOFBOUND)},
    {AIERC_STR(XAIE_INVALID_DATA_MEM_ADDR)},
    {AIERC_STR(XAIE_INVALID_ELF)},
    {AIERC_STR(XAIE_CORE_STATUS_TIMEOUT)},
    {AIERC_STR(XAIE_INVALID_CHANNEL_NUM)},
    {AIERC_STR(XAIE_INVALID_LOCK)},
    {AIERC_STR(XAIE_INVALID_DMA_DIRECTION)},
    {AIERC_STR(XAIE_INVALID_PLIF_WIDTH)},
    {AIERC_STR(XAIE_INVALID_LOCK_ID)},
    {AIERC_STR(XAIE_INVALID_LOCK_VALUE)},
    {AIERC_STR(XAIE_LOCK_RESULT_FAILED)},
    {AIERC_STR(XAIE_INVALID_DMA_DESC)},
    {AIERC_STR(XAIE_INVALID_ADDRESS)},
    {AIERC_STR(XAIE_FEATURE_NOT_SUPPORTED)},
    {AIERC_STR(XAIE_INVALID_BURST_LENGTH)},
    {AIERC_STR(XAIE_INVALID_BACKEND)},
    {AIERC_STR(XAIE_INSUFFICIENT_BUFFER_SIZE)},
    {AIERC_STR(XAIE_ERR_MAX)}};
#undef AIERC_STR

std::string AIERCTOSTR(AieRC rc) { return _AIERCTOSTR.at(rc); }

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const XAie_LocType &loc) {
  os << "XAie_LocType(col: " << std::to_string(loc.Col)
     << ", row: " << std::to_string(loc.Row) << ")";
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const XAie_Lock &lock) {
  os << "XAie_Lock(id: " << std::to_string(lock.LockId)
     << ", val: " << std::to_string(lock.LockVal) << ")";
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const XAie_Packet &packet) {
  os << "XAie_Packet(id: " << std::to_string(packet.PktId)
     << ", type: " << std::to_string(packet.PktType) << ")";
  return os;
}

bool isInternal(uint8_t srcCol, uint8_t srcRow, uint8_t dstCol,
                uint8_t dstRow) {
  return srcCol == dstCol && srcRow == dstRow;
}

bool isWest(uint8_t srcCol, uint8_t srcRow, uint8_t dstCol, uint8_t dstRow) {
  return srcCol == dstCol + 1 && srcRow == dstRow;
}

bool isEast(uint8_t srcCol, uint8_t srcRow, uint8_t dstCol, uint8_t dstRow) {
  return srcCol == dstCol - 1 && srcRow == dstRow;
}

bool isNorth(uint8_t srcCol, uint8_t srcRow, uint8_t dstCol, uint8_t dstRow) {
  return srcCol == dstCol && srcRow == dstRow - 1;
}

bool isSouth(uint8_t srcCol, uint8_t srcRow, uint8_t dstCol, uint8_t dstRow) {
  return srcCol == dstCol && srcRow == dstRow + 1;
}

AMDAIENPUDeviceModel::AMDAIENPUDeviceModel(
    uint8_t aieGen, uint64_t baseAddr, uint8_t colShift, uint8_t rowShift,
    uint8_t nColumns, uint8_t nRows, uint8_t memTileRowStart,
    uint8_t nMemTileRows, uint8_t nShimTileRows, uint8_t partitionStartCol,
    uint8_t partitionNumCols, bool aieSim, bool xaieDebug)
    : configPtr{.AieGen = aieGen,
                .BaseAddr = baseAddr,
                .ColShift = colShift,
                .RowShift = rowShift,
                .NumRows = nRows,
                .NumCols = nColumns,
                .ShimRowNum = 0,
                .MemTileRowStart = memTileRowStart,
                .MemTileNumRows = nMemTileRows,
                .AieTileRowStart =
                    static_cast<uint8_t>(memTileRowStart + nMemTileRows),
                .AieTileNumRows =
                    static_cast<uint8_t>(nRows - nMemTileRows - nShimTileRows),
                .PartProp = {},
                .Backend = XAIE_IO_BACKEND_CDO},
      devInst{} {
  TRY_XAIE_API_FATAL_ERROR(XAie_SetupPartitionConfig, &devInst,
                           XAIE_PARTITION_BASE_ADDR, partitionStartCol,
                           partitionNumCols);
  TRY_XAIE_API_FATAL_ERROR(XAie_CfgInitialize, &devInst, &configPtr);
  if (aieSim) {
    TRY_XAIE_API_FATAL_ERROR(XAie_SetIOBackend, &devInst, XAIE_IO_BACKEND_SIM);
  } else if (xaieDebug)
    TRY_XAIE_API_FATAL_ERROR(XAie_SetIOBackend, &devInst,
                             XAIE_IO_BACKEND_DEBUG);
  else
    TRY_XAIE_API_FATAL_ERROR(XAie_SetIOBackend, &devInst, XAIE_IO_BACKEND_CDO);

  TRY_XAIE_API_FATAL_ERROR(XAie_UpdateNpiAddr, &devInst, NPI_ADDR);
}

int AMDAIENPUDeviceModel::rows() const { return configPtr.NumRows; }

int AMDAIENPUDeviceModel::columns() const { return configPtr.NumCols; }

uint32_t AMDAIENPUDeviceModel::getNumMemTileRows() const {
  return configPtr.MemTileNumRows;
}

// TODO(max): these are buried somewhere in aie-rt...
uint32_t AMDAIENPUDeviceModel::getMemSouthBaseAddress() { return 0x00040000; }
uint32_t AMDAIENPUDeviceModel::getMemWestBaseAddress() { return 0x00050000; }
uint32_t AMDAIENPUDeviceModel::getMemNorthBaseAddress() { return 0x00060000; }
uint32_t AMDAIENPUDeviceModel::getMemEastBaseAddress() { return 0x00070000; }

bool AMDAIENPUDeviceModel::isCoreTile(uint8_t col, uint8_t row) {
  return devInst.DevOps->GetTTypefromLoc(&devInst, {.Row = row, .Col = col}) ==
         XAIEGBL_TILE_TYPE_AIETILE;
}

bool AMDAIENPUDeviceModel::isMemTile(uint8_t col, uint8_t row) {
  return devInst.DevOps->GetTTypefromLoc(&devInst, {.Row = row, .Col = col}) ==
         XAIEGBL_TILE_TYPE_MEMTILE;
}

bool AMDAIENPUDeviceModel::isShimNOCTile(uint8_t col, uint8_t row) {
  return devInst.DevOps->GetTTypefromLoc(&devInst, {.Row = row, .Col = col}) ==
         XAIEGBL_TILE_TYPE_SHIMNOC;
}

bool AMDAIENPUDeviceModel::isShimPLTile(uint8_t col, uint8_t row) {
  return devInst.DevOps->GetTTypefromLoc(&devInst, {.Row = row, .Col = col}) ==
         XAIEGBL_TILE_TYPE_SHIMPL;
}

uint32_t AMDAIENPUDeviceModel::getNumLocks(uint8_t col, uint8_t row) {
  uint8_t tileType =
      devInst.DevOps->GetTTypefromLoc(&devInst, {.Row = row, .Col = col});
  assert(tileType != XAIEGBL_TILE_TYPE_MAX && "invalid tile");
  return devInst.DevProp.DevMod[tileType].LockMod->NumLocks;
}

uint32_t AMDAIENPUDeviceModel::getNumBDs(uint8_t col, uint8_t row) {
  uint8_t tileType =
      devInst.DevOps->GetTTypefromLoc(&devInst, {.Row = row, .Col = col});
  assert(tileType != XAIEGBL_TILE_TYPE_MAX && "invalid tile");
  const XAie_DmaMod *dmaMod = devInst.DevProp.DevMod[tileType].DmaMod;
  return dmaMod->NumBds;
}

std::optional<TileLoc> AMDAIENPUDeviceModel::getMemWest(TileLoc src) {
  XAie_LocType ret = XAie_TileLoc(src.col - 1, src.row);
  if (devInst.DevOps->GetTTypefromLoc(&devInst, ret) == XAIEGBL_TILE_TYPE_MAX)
    return std::nullopt;
  return ret;
}

std::optional<TileLoc> AMDAIENPUDeviceModel::getMemEast(TileLoc src) {
  // east is self
  return src;
}

std::optional<TileLoc> AMDAIENPUDeviceModel::getMemNorth(TileLoc src) {
  XAie_LocType ret = XAie_TileLoc(src.col, src.row + 1);
  if (devInst.DevOps->GetTTypefromLoc(&devInst, ret) == XAIEGBL_TILE_TYPE_MAX)
    return std::nullopt;
  return ret;
}

std::optional<TileLoc> AMDAIENPUDeviceModel::getMemSouth(TileLoc src) {
  XAie_LocType ret = XAie_TileLoc(src.col, src.row - 1);
  auto tt = devInst.DevOps->GetTTypefromLoc(&devInst, ret);
  // The first row doesn't have a tile memory south
  // Memtiles don't have memory adjacency to neighboring core tiles.
  if (tt == XAIEGBL_TILE_TYPE_MAX || ret.Row == 0 ||
      tt == XAIEGBL_TILE_TYPE_MEMTILE)
    return std::nullopt;
  return ret;
}

// I don't know why you don't need to check for memtile or core tile here
// but this repros what mlir-aie does
bool AMDAIENPUDeviceModel::hasMemWest(uint8_t srcCol, uint8_t srcRow,
                                      uint8_t dstCol, uint8_t dstRow) {
  return isWest(srcCol, srcRow, dstCol, dstRow);
}

bool AMDAIENPUDeviceModel::hasMemEast(uint8_t srcCol, uint8_t srcRow,
                                      uint8_t dstCol, uint8_t dstRow) {
  return isInternal(srcCol, srcRow, dstCol, dstRow);
}

bool AMDAIENPUDeviceModel::hasMemNorth(uint8_t srcCol, uint8_t srcRow,
                                       uint8_t dstCol, uint8_t dstRow) {
  return isNorth(srcCol, srcRow, dstCol, dstRow);
}

bool AMDAIENPUDeviceModel::hasMemSouth(uint8_t srcCol, uint8_t srcRow,
                                       uint8_t dstCol, uint8_t dstRow) {
  return isSouth(srcCol, srcRow, dstCol, dstRow);
}

uint32_t AMDAIENPUDeviceModel::getLocalMemorySize(uint8_t col, uint8_t row) {
  auto tileLoc = XAie_TileLoc(col, row);
  uint8_t tileType = devInst.DevOps->GetTTypefromLoc(&devInst, tileLoc);
  assert(tileType != XAIEGBL_TILE_TYPE_MAX && "invalid tile");
  return devInst.DevProp.DevMod[tileType].CoreMod->DataMemSize;
}

uint32_t AMDAIENPUDeviceModel::getMemInternalBaseAddress() {
  return getMemEastBaseAddress();
}

uint32_t AMDAIENPUDeviceModel::getMemTileSize(uint8_t col, uint8_t row) {
  auto tileLoc = XAie_TileLoc(col, row);
  uint8_t tileType = devInst.DevOps->GetTTypefromLoc(&devInst, tileLoc);
  assert(tileType != XAIEGBL_TILE_TYPE_MAX && "invalid tile");
  return devInst.DevProp.DevMod[tileType].MemMod->Size;
}

bool AMDAIENPUDeviceModel::hasLegalMemAffinity(uint8_t coreCol, uint8_t coreRow,
                                               uint8_t memCol, uint8_t memRow) {
  bool isMemWest = hasMemWest(coreCol, coreRow, memCol, memRow);
  bool isMemEast = hasMemEast(coreCol, coreRow, memCol, memRow);
  bool isMemNorth = hasMemNorth(coreCol, coreRow, memCol, memRow);
  bool isMemSouth = hasMemSouth(coreCol, coreRow, memCol, memRow);

  if (isMemTile(coreCol, coreRow))
    return isEast(coreCol, coreRow, memCol, memRow) ||
           isInternal(coreCol, coreRow, memCol, memRow) ||
           isWest(coreCol, coreRow, memCol, memRow);
  return (isMemSouth && !isMemTile(memCol, memRow)) || isMemNorth ||
         isMemWest || isMemEast;
}

bool AMDAIENPUDeviceModel::isLegalMemtileConnection(uint8_t col, uint8_t row,
                                                    StrmSwPortType srcBundle,
                                                    uint8_t srcChan,
                                                    StrmSwPortType dstBundle,
                                                    uint8_t dstChan) {
  // this isn't correct but for agreement with mlir-aie...
  if (srcBundle == dstBundle and srcBundle != DMA) return true;
  assert(isMemTile(col, row) && "expected memtile");
  auto tileLoc = XAie_TileLoc(col, row);
  uint8_t tileType = devInst.DevOps->GetTTypefromLoc(&devInst, tileLoc);
  assert(tileType != XAIEGBL_TILE_TYPE_MAX && "invalid tile");
  const XAie_StrmMod *strmMod = devInst.DevProp.DevMod[tileType].StrmSw;
  AieRC RC = strmMod->PortVerify(/*slave*/ srcBundle, srcChan,
                                 /*master*/ dstBundle, dstChan);
  if (RC != XAIE_OK) {
    LLVM_DEBUG(llvm::dbgs() << "PortVerify failed with " << AIERCTOSTR(RC));
    LLVM_DEBUG(SHOW_ARGS(llvm::dbgs(), col, row, srcBundle, srcChan, dstBundle,
                         dstChan));
    return false;
  }
  return true;
}

// source <-> slave and dest <-> master
uint32_t AMDAIENPUDeviceModel::getNumSourceSwitchboxConnections(
    uint8_t col, uint8_t row, StrmSwPortType bundle) {
  // not sure if this makes sense but agrees with mlir-aie
  if ((bundle == NORTH && row == rows() - 1) || (bundle == WEST && col == 0) ||
      (bundle == EAST && col == columns() - 1))
    return 0;
  uint8_t tileType =
      devInst.DevOps->GetTTypefromLoc(&devInst, {.Row = row, .Col = col});
  assert(tileType != XAIEGBL_TILE_TYPE_MAX && "invalid tile");
  const XAie_StrmMod *strmMod = devInst.DevProp.DevMod[tileType].StrmSw;
  return strmMod->SlvConfig[bundle].NumPorts;
}

uint32_t AMDAIENPUDeviceModel::getNumDestSwitchboxConnections(
    uint8_t col, uint8_t row, StrmSwPortType bundle) {
  // not sure if this makes sense but agrees with mlir-aie
  if ((bundle == NORTH && row == rows() - 1) || (bundle == WEST && col == 0) ||
      (bundle == EAST && col == columns() - 1))
    return 0;

  uint8_t tileType =
      devInst.DevOps->GetTTypefromLoc(&devInst, {.Row = row, .Col = col});
  assert(tileType != XAIEGBL_TILE_TYPE_MAX && "invalid tile");
  const XAie_StrmMod *strmMod = devInst.DevProp.DevMod[tileType].StrmSw;
  return strmMod->MstrConfig[bundle].NumPorts;
}

struct AMDAIENPUDeviceModel mlir::iree_compiler::AMDAIE::getDeviceModel(
    AMDAIEDevice device) {
  switch (device) {
    case AMDAIEDevice::xcvc1902:
      return AMDAIENPUDeviceModel(XAIE_DEV_GEN_AIE, XAIE1_BASE_ADDR,
                                  XAIE1_COL_SHIFT, XAIE1_ROW_SHIFT,
                                  /*numCols*/ 50, /*numRows*/ 9,
                                  /*memTileRowStart*/ 0, /*nMemTileRows*/ 0,
                                  /*nShimTileRows*/ 0, /*partitionStartCol*/ 0,
                                  /*partitionNumCols*/ 50);
    case AMDAIEDevice::xcve2302:
      return AMDAIENPUDeviceModel(XAIE_DEV_GEN_AIEML, XAIE2_BASE_ADDR,
                                  XAIE2_COL_SHIFT, XAIE2_ROW_SHIFT,
                                  /*numCols*/ 17, /*numRows*/ 4,
                                  /*memTileRowStart*/ 1, /*nMemTileRows*/ 1,
                                  /*nShimTileRows*/ 1, /*partitionStartCol*/ 0,
                                  /*partitionNumCols*/ 17);
    case AMDAIEDevice::xcve2802:
      return AMDAIENPUDeviceModel(XAIE_DEV_GEN_AIEML, XAIE2_BASE_ADDR,
                                  XAIE2_COL_SHIFT, XAIE2_ROW_SHIFT,
                                  /*numCols*/ 38, /*numRows*/ 11,
                                  /*memTileRowStart*/ 2, /*nMemTileRows*/ 1,
                                  /*nShimTileRows*/ 1, /*partitionStartCol*/ 0,
                                  /*partitionNumCols*/ 38);
    case AMDAIEDevice::npu:
    case AMDAIEDevice::npu1_1col:
    case AMDAIEDevice::npu1_2col:
    case AMDAIEDevice::npu1_3col:
    case AMDAIEDevice::npu1_4col:
      return AMDAIENPUDeviceModel(XAIE_DEV_GEN_AIEML, XAIE2_BASE_ADDR,
                                  XAIE2_COL_SHIFT, XAIE2_ROW_SHIFT,
                                  /*numCols*/ 5, /*numRows*/ 6,
                                  /*memTileRowStart*/ 1, /*nMemTileRows*/ 1,
                                  /*nShimTileRows*/ 1, /*partitionStartCol*/ 0,
                                  /*partitionNumCols*/ 5);
  }
}

StrmSwPortType getConnectingStrmSwPortType(StrmSwPortType dir) {
  switch (dir) {
    case StrmSwPortType::NORTH:
      return StrmSwPortType::SOUTH;
    case StrmSwPortType::SOUTH:
      return StrmSwPortType::NORTH;
    case StrmSwPortType::EAST:
      return StrmSwPortType::WEST;
    case StrmSwPortType::WEST:
      return StrmSwPortType::EAST;
    default:
      return dir;
  }
}

std::string stringifyStrmSwPortType(StrmSwPortType val) {
  switch (val) {
    case StrmSwPortType::CORE:
      return "Core";
    case StrmSwPortType::DMA:
      return "DMA";
    case StrmSwPortType::FIFO:
      return "FIFO";
    case StrmSwPortType::SOUTH:
      return "South";
    case StrmSwPortType::WEST:
      return "West";
    case StrmSwPortType::NORTH:
      return "North";
    case StrmSwPortType::EAST:
      return "East";
    case StrmSwPortType::TRACE:
      return "Trace";
    case StrmSwPortType::CTRL:
      return "Ctrl";
    default:
      return "UNSUPPORTED";
  }
}