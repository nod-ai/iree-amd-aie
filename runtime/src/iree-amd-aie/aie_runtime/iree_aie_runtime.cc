// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions. See
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: # Apache-2.0 WITH LLVM-exception

#include "iree_aie_runtime.h"

#include <cstdint>
#include <numeric>

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"

extern "C" {
#include "xaiengine.h"
#undef s8
#undef u8
#undef u16
#undef s32
#undef u32
#undef u64
}

#define DEBUG_TYPE "iree-aie-runtime"

namespace MLIRAIELegacy {
extern uint32_t getNumDestSwitchBoxConnections(
    int col, int row, mlir::iree_compiler::AMDAIE::StrmSwPortType bundle,
    const mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel &deviceModel);
extern uint32_t getNumSourceSwitchBoxConnections(
    int col, int row, mlir::iree_compiler::AMDAIE::StrmSwPortType bundle,
    const mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel &deviceModel);
extern bool isLegalTileConnection(
    int col, int row, mlir::iree_compiler::AMDAIE::StrmSwPortType srcBundle,
    int srcChan, mlir::iree_compiler::AMDAIE::StrmSwPortType dstBundle,
    int dstChan,
    const mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel &deviceModel);
extern bool isShimTile(
    int col, int row,
    const mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel &deviceModel);
int rows(const mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel &deviceModel);
int columns(const mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel &deviceModel);
}  // namespace MLIRAIELegacy

namespace {
bool isSame(uint8_t srcCol, uint8_t srcRow, uint8_t dstCol, uint8_t dstRow) {
  return srcCol == dstCol && srcRow == dstRow;
}

// WARNING: these don't actually make sense (that's why they're here in this
// anon namespace)
// https://github.com/Xilinx/mlir-aie/issues/1021 but remain for compat with
// mlir-aie
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

bool isAieRtCompatStrmSwPortType(
    mlir::iree_compiler::AMDAIE::StrmSwPortType t) {
  return static_cast<uint8_t>(t) <= ::StrmSwPortType::SS_PORT_TYPE_MAX;
}

::StrmSwPortType checkedAieRtCompatStrmSwPortType(
    mlir::iree_compiler::AMDAIE::StrmSwPortType t, const char *file,
    unsigned int line, const char *function) {
#ifndef NDEBUG
  if (!isAieRtCompatStrmSwPortType(t)) {
    llvm::report_fatal_error(llvm::formatv(
        "{0}:{1}:{2}: StrmSwPortType  incompatible with aie-rt: {3}", file,
        line, function, to_string(t)));
  }
#endif
  return static_cast<::StrmSwPortType>(t);
}

#ifdef _WIN32
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

// macro so that line numbers are preserved for where the check fails
#define CheckedAieRtCompatStrmSwPortType(t) \
  checkedAieRtCompatStrmSwPortType(t, __FILE__, __LINE__, __PRETTY_FUNCTION__)
}  // namespace

namespace mlir::iree_compiler::AMDAIE {
StrmSwPortType getConnectingBundle(StrmSwPortType dir) {
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

bool isNPUDevice(AMDAIEDevice d) {
  return d == AMDAIEDevice::npu1 || d == AMDAIEDevice::npu1_1col ||
         d == AMDAIEDevice::npu1_2col || d == AMDAIEDevice::npu1_3col ||
         d == AMDAIEDevice::npu1_4col || d == AMDAIEDevice::npu4;
}

AMDAIEDeviceModel::AMDAIEDeviceModel(
    uint8_t aieGen, uint64_t baseAddr, uint8_t colShift, uint8_t rowShift,
    uint8_t devNColumns, uint8_t devNRows, uint8_t memTileRowStart,
    uint8_t nMemTileRows, uint8_t nShimTileRows, int partitionNumCols,
    int partitionStartCol, uint64_t partBaseAddr, uint64_t npiAddr, bool aieSim,
    bool xaieDebug, AMDAIEDevice device, AMDAIEDeviceConfig deviceConfig)
    : configPtr{/*AieGen*/ aieGen,
                /*BaseAddr*/ baseAddr,
                /*ColShift*/ colShift,
                /*RowShift*/ rowShift,
                /*NumRows*/ devNRows,
                /*NumCols*/ devNColumns,
                /*ShimRowNum*/ 0,
                /*MemTileRowStart*/ memTileRowStart,
                /*MemTileNumRows*/ nMemTileRows,
                // TODO(max): use XAIE*_AIE_TILE_ROW_START here
                // instead of this (once we eliminate legacy devices)
                /*AieTileRowStart*/
                static_cast<uint8_t>(memTileRowStart + nMemTileRows),
                /*AieTileNumRows*/
                static_cast<uint8_t>(devNRows - nMemTileRows - nShimTileRows),
                /*PartProp*/ {}},
      devInst{},
      deviceConfig(std::move(deviceConfig)),
      device(std::move(device)) {
  TRY_XAIE_API_FATAL_ERROR(XAie_SetupPartitionConfig, &devInst, partBaseAddr,
                           partitionStartCol, partitionNumCols);
  TRY_XAIE_API_FATAL_ERROR(XAie_CfgInitialize, &devInst, &configPtr);
  if (aieSim) {
    TRY_XAIE_API_FATAL_ERROR(XAie_SetIOBackend, &devInst, XAIE_IO_BACKEND_SIM);
  } else if (xaieDebug)
    TRY_XAIE_API_FATAL_ERROR(XAie_SetIOBackend, &devInst,
                             XAIE_IO_BACKEND_DEBUG);
  else
    TRY_XAIE_API_FATAL_ERROR(XAie_SetIOBackend, &devInst, XAIE_IO_BACKEND_CDO);

  TRY_XAIE_API_FATAL_ERROR(XAie_UpdateNpiAddr, &devInst, npiAddr);

  // TODO(max): this prevents some (most?) elfs from returning values?
  TRY_XAIE_API_FATAL_ERROR(XAie_TurnEccOff, &devInst);
}

int AMDAIEDeviceModel::rows() const {
  if (device == AMDAIEDevice::xcvc1902 || device == AMDAIEDevice::xcve2802)
    return MLIRAIELegacy::rows(*this);
  assert(isNPUDevice(device) && "expected NPU device");
  return devInst.NumRows;
}

int AMDAIEDeviceModel::columns() const {
  if (device == AMDAIEDevice::xcvc1902 || device == AMDAIEDevice::xcve2802)
    return MLIRAIELegacy::columns(*this);
  assert(isNPUDevice(device) && "expected NPU device");
  return devInst.NumCols;
}

// TODO(max): these are buried somewhere in aie-rt...
uint32_t AMDAIEDeviceModel::getMemSouthBaseAddress() const {
  return 0x00040000;
}

uint32_t AMDAIEDeviceModel::getMemWestBaseAddress() const { return 0x00050000; }

uint32_t AMDAIEDeviceModel::getMemNorthBaseAddress() const {
  return 0x00060000;
}

uint32_t AMDAIEDeviceModel::getMemEastBaseAddress() const { return 0x00070000; }

AMDAIETileType AMDAIEDeviceModel::getTileType(uint8_t col, uint8_t row) const {
  uint8_t tt = devInst.DevOps->GetTTypefromLoc(
      const_cast<XAie_DevInst *>(&devInst), XAie_TileLoc(col, row));
  assert(tt < XAIEGBL_TILE_TYPE_MAX && "expected valid tile type");
  return static_cast<AMDAIETileType>(tt);
}

bool AMDAIEDeviceModel::isCoreTile(uint8_t col, uint8_t row) const {
  return getTileType(col, row) == AMDAIETileType::AIETILE;
}

bool AMDAIEDeviceModel::isMemTile(uint8_t col, uint8_t row) const {
  return getTileType(col, row) == AMDAIETileType::MEMTILE;
}

bool AMDAIEDeviceModel::isShimNOCTile(uint8_t col, uint8_t row) const {
  return getTileType(col, row) == AMDAIETileType::SHIMNOC;
}

bool AMDAIEDeviceModel::isShimPLTile(uint8_t col, uint8_t row) const {
  return getTileType(col, row) == AMDAIETileType::SHIMPL;
}

bool AMDAIEDeviceModel::isShimNOCorPLTile(uint8_t col, uint8_t row) const {
  return isShimNOCTile(col, row) || isShimPLTile(col, row);
}

bool AMDAIEDeviceModel::isShimTile(uint8_t col, uint8_t row) const {
  if (device == AMDAIEDevice::xcvc1902 || device == AMDAIEDevice::xcve2302 ||
      device == AMDAIEDevice::xcve2802)
    return MLIRAIELegacy::isShimTile(col, row, *this);
  assert(isNPUDevice(device) && "expected NPU device");
  return row == configPtr.ShimRowNum;
}

// TODO(max): these should be optionals instead of returning 0.
uint32_t AMDAIEDeviceModel::getNumLocks(uint8_t col, uint8_t row) const {
  AMDAIETileType tileType = getTileType(col, row);
  if (tileType == AMDAIETileType::SHIMPL || tileType == AMDAIETileType::MAX)
    return 0;
  return devInst.DevProp.DevMod[static_cast<uint8_t>(tileType)]
      .LockMod->NumLocks;
}

uint32_t AMDAIEDeviceModel::getNumBDs(uint8_t col, uint8_t row) const {
  AMDAIETileType tileType = getTileType(col, row);
  if (tileType == AMDAIETileType::SHIMPL || tileType == AMDAIETileType::MAX)
    return 0;
  const XAie_DmaMod *dmaMod =
      devInst.DevProp.DevMod[static_cast<uint8_t>(tileType)].DmaMod;
  return dmaMod->NumBds;
}

std::optional<TileLoc> AMDAIEDeviceModel::getMemWest(TileLoc src) const {
  if (src.col - 1 < 0) return std::nullopt;
  XAie_LocType ret = XAie_TileLoc(src.col - 1, src.row);
  if (getTileType(ret.Col, ret.Row) == AMDAIETileType::MAX) return std::nullopt;
  return ret;
}

std::optional<TileLoc> AMDAIEDeviceModel::getMemEast(TileLoc src) const {
  // east is self
  return src;
}

std::optional<TileLoc> AMDAIEDeviceModel::getMemNorth(TileLoc src) const {
  if (src.row + 1 >= rows()) return std::nullopt;
  XAie_LocType ret = XAie_TileLoc(src.col, src.row + 1);
  if (getTileType(ret.Col, ret.Row) == AMDAIETileType::MAX) return std::nullopt;
  return ret;
}

std::optional<TileLoc> AMDAIEDeviceModel::getMemSouth(TileLoc src) const {
  if (src.row - 1 < 0) return std::nullopt;
  XAie_LocType ret = XAie_TileLoc(src.col, src.row - 1);
  auto tt = getTileType(ret.Col, ret.Row);
  // The first row doesn't have a tile memory south
  // Memtiles don't have memory adjacency to neighboring core tiles.
  if (ret.Row == 0 || tt == AMDAIETileType::MAX ||
      tt == AMDAIETileType::MEMTILE) {
    return std::nullopt;
  }
  return ret;
}

// TODO(max): I don't know why you don't need to check for memtile or core tile
// here but this repros what mlir-aie does
bool AMDAIEDeviceModel::hasMemWest(uint8_t srcCol, uint8_t srcRow,
                                   uint8_t dstCol, uint8_t dstRow) const {
  return isWest(srcCol, srcRow, dstCol, dstRow);
}

bool AMDAIEDeviceModel::hasMemEast(uint8_t srcCol, uint8_t srcRow,
                                   uint8_t dstCol, uint8_t dstRow) const {
  return isSame(srcCol, srcRow, dstCol, dstRow);
}

bool AMDAIEDeviceModel::hasMemNorth(uint8_t srcCol, uint8_t srcRow,
                                    uint8_t dstCol, uint8_t dstRow) const {
  return isNorth(srcCol, srcRow, dstCol, dstRow);
}

bool AMDAIEDeviceModel::hasMemSouth(uint8_t srcCol, uint8_t srcRow,
                                    uint8_t dstCol, uint8_t dstRow) const {
  return isSouth(srcCol, srcRow, dstCol, dstRow);
}

uint32_t AMDAIEDeviceModel::getLocalMemorySize(uint8_t col, uint8_t row) const {
  AMDAIETileType tileType = getTileType(col, row);
  assert(tileType != AMDAIETileType::MAX && "invalid tile");
  return devInst.DevProp.DevMod[static_cast<uint8_t>(tileType)]
      .CoreMod->DataMemSize;
}

uint32_t AMDAIEDeviceModel::getCoreTileLocalMemorySize() const {
  return devInst.DevProp.DevMod[XAIEGBL_TILE_TYPE_AIETILE].CoreMod->DataMemSize;
}

uint32_t AMDAIEDeviceModel::getMemInternalBaseAddress() const {
  return getMemEastBaseAddress();
}

uint32_t AMDAIEDeviceModel::getMemTileSize(uint8_t col, uint8_t row) const {
  AMDAIETileType tileType = getTileType(col, row);
  assert(tileType == AMDAIETileType::MEMTILE && "expected memtile");
  return devInst.DevProp.DevMod[static_cast<uint8_t>(tileType)].MemMod->Size;
}

bool AMDAIEDeviceModel::hasLegalMemAffinity(uint8_t coreCol, uint8_t coreRow,
                                            uint8_t memCol,
                                            uint8_t memRow) const {
  bool isMemWest = hasMemWest(coreCol, coreRow, memCol, memRow);
  bool isMemEast = hasMemEast(coreCol, coreRow, memCol, memRow);
  bool isMemNorth = hasMemNorth(coreCol, coreRow, memCol, memRow);
  bool isMemSouth = hasMemSouth(coreCol, coreRow, memCol, memRow);

  if (isMemTile(coreCol, coreRow)) {
    return isEast(coreCol, coreRow, memCol, memRow) ||
           isSame(coreCol, coreRow, memCol, memRow) ||
           isWest(coreCol, coreRow, memCol, memRow);
  }
  return (isMemSouth && !isMemTile(memCol, memRow)) || isMemNorth ||
         isMemWest || isMemEast;
}

bool AMDAIEDeviceModel::isLegalTileConnection(uint8_t col, uint8_t row,
                                              StrmSwPortType srcBundle,
                                              uint8_t srcChan,
                                              StrmSwPortType dstBundle,
                                              uint8_t dstChan) const {
  if (device == AMDAIEDevice::xcvc1902 || device == AMDAIEDevice::xcve2802)
    return MLIRAIELegacy::isLegalTileConnection(col, row, srcBundle, srcChan,
                                                dstBundle, dstChan, *this);
  assert(isNPUDevice(device) && "expected NPU device");

  AMDAIETileType tileType = getTileType(col, row);
  const XAie_StrmMod *strmMod =
      devInst.DevProp.DevMod[static_cast<uint8_t>(tileType)].StrmSw;
  if (!isAieRtCompatStrmSwPortType(srcBundle) ||
      !isAieRtCompatStrmSwPortType(dstBundle)) {
    return false;
  }
  if (srcChan >= strmMod->SlvConfig[CheckedAieRtCompatStrmSwPortType(srcBundle)]
                     .NumPorts) {
    return false;
  }
  if (dstChan >=
      strmMod->MstrConfig[CheckedAieRtCompatStrmSwPortType(dstBundle)]
          .NumPorts) {
    return false;
  }
  AieRC RC = strmMod->PortVerify(
      /*slave*/ CheckedAieRtCompatStrmSwPortType(srcBundle), srcChan,
      /*master*/ CheckedAieRtCompatStrmSwPortType(dstBundle), dstChan);
  if (RC != XAIE_OK) {
    LLVM_DEBUG(llvm::dbgs() << "PortVerify failed with " << RC << "\n");
    LLVM_DEBUG(SHOW_ARGS(llvm::dbgs(), col, row, srcBundle, (int)srcChan,
                         dstBundle, (int)dstChan));
    LLVM_DEBUG(llvm::dbgs() << "\n");
    return false;
  }
  return true;
}

// source <-> slave and dest <-> master
uint32_t AMDAIEDeviceModel::getNumSourceSwitchBoxConnections(
    uint8_t col, uint8_t row, StrmSwPortType bundle) const {
  if (device == AMDAIEDevice::xcvc1902 || device == AMDAIEDevice::xcve2802) {
    return MLIRAIELegacy::getNumSourceSwitchBoxConnections(col, row, bundle,
                                                           *this);
  }
  assert(isNPUDevice(device) && "expected NPU device");

  AMDAIETileType tileType = getTileType(col, row);
  // not sure if this makes sense but agrees with mlir-aie
  if ((bundle == StrmSwPortType::NORTH && row == rows() - 1) ||
      (bundle == StrmSwPortType::WEST && col == 0) ||
      (bundle == StrmSwPortType::EAST && col == columns() - 1) ||
      !isAieRtCompatStrmSwPortType(bundle) || tileType == AMDAIETileType::MAX) {
    return 0;
  }
  const XAie_StrmMod *strmMod =
      devInst.DevProp.DevMod[static_cast<uint8_t>(tileType)].StrmSw;
  return strmMod->SlvConfig[CheckedAieRtCompatStrmSwPortType(bundle)].NumPorts;
}

uint32_t AMDAIEDeviceModel::getNumDestSwitchBoxConnections(
    uint8_t col, uint8_t row, StrmSwPortType bundle) const {
  if (device == AMDAIEDevice::xcvc1902 || device == AMDAIEDevice::xcve2802) {
    return MLIRAIELegacy::getNumDestSwitchBoxConnections(col, row, bundle,
                                                         *this);
  }
  assert(isNPUDevice(device) && "expected NPU device");

  AMDAIETileType tileType = getTileType(col, row);
  // not sure if this makes sense but agrees with mlir-aie
  if ((bundle == StrmSwPortType::NORTH && row == rows() - 1) ||
      (bundle == StrmSwPortType::WEST && col == 0) ||
      (bundle == StrmSwPortType::EAST && col == columns() - 1) ||
      !isAieRtCompatStrmSwPortType(bundle) || tileType == AMDAIETileType::MAX) {
    return 0;
  }

  const XAie_StrmMod *strmMod =
      devInst.DevProp.DevMod[static_cast<uint8_t>(tileType)].StrmSw;
  return strmMod->MstrConfig[CheckedAieRtCompatStrmSwPortType(bundle)].NumPorts;
}

// the difference between this fn and the one it calls is SwitchBox vs Switchbox
uint32_t AMDAIEDeviceModel::getNumDestSwitchboxConnections(
    int col, int row, StrmSwPortType bundle) const {
  return getNumDestSwitchBoxConnections(static_cast<uint8_t>(col),
                                        static_cast<uint8_t>(row), bundle);
}

uint32_t AMDAIEDeviceModel::getColumnShift() const {
  return configPtr.ColShift;
}

uint32_t AMDAIEDeviceModel::getRowShift() const { return configPtr.RowShift; }

uint8_t AMDAIEDeviceModel::getStreamSwitchArbiterMax(uint8_t col,
                                                     uint8_t row) const {
  assert(isCoreTile(col, row) || isMemTile(col, row) || isShimTile(col, row));
  if (isCoreTile(col, row)) return deviceConfig.streamSwitchCoreArbiterMax;
  if (isMemTile(col, row)) return deviceConfig.streamSwitchMemTileArbiterMax;
  if (isShimTile(col, row)) return deviceConfig.streamSwitchShimArbiterMax;
  return 0;
}

uint8_t AMDAIEDeviceModel::getStreamSwitchMSelMax(uint8_t col,
                                                  uint8_t row) const {
  assert(isCoreTile(col, row) || isMemTile(col, row) || isShimTile(col, row));
  if (isCoreTile(col, row)) return deviceConfig.streamSwitchCoreMSelMax;
  if (isMemTile(col, row)) return deviceConfig.streamSwitchMemTileMSelMax;
  if (isShimTile(col, row)) return deviceConfig.streamSwitchShimMSelMax;
  return 0;
}

DenseMap<uint32_t, SmallVector<uint32_t>>
AMDAIEDeviceModel::getChannelToValidBdIds(AMDAIETileType tileType) const {
  switch (tileType) {
    case AMDAIETileType::MEMTILE: {
      SmallVector<uint32_t> evenRange(24);
      std::iota(evenRange.begin(), evenRange.end(), 0);
      SmallVector<uint32_t> oddRange(24);
      std::iota(oddRange.begin(), oddRange.end(), 24);
      DenseMap<uint32_t, SmallVector<uint32_t>> channelToValidBdIds = {
          {0, evenRange}, {1, oddRange},  {2, evenRange},
          {3, oddRange},  {4, evenRange}, {5, oddRange}};
      return channelToValidBdIds;
    }
    case AMDAIETileType::SHIMNOC: {
      SmallVector<uint32_t> range(16);
      std::iota(range.begin(), range.end(), 0);
      DenseMap<uint32_t, SmallVector<uint32_t>> channelToValidBdIds = {
          {0, range}, {1, range}};
      return channelToValidBdIds;
    }
    default:
      break;
  }
  llvm::report_fatal_error("Unhandled AMDAIETileType case");
}

struct AMDAIEDeviceModel getDeviceModel(AMDAIEDevice device) {
  switch (device) {
    case AMDAIEDevice::xcvc1902: {
      AMDAIEDeviceModel::AMDAIEDeviceConfig deviceConfig;
      deviceConfig.streamSwitchCoreArbiterMax = XAIE1_SS_ARBITER_MAX;
      deviceConfig.streamSwitchCoreMSelMax = XAIE1_SS_MSEL_MAX;
      deviceConfig.streamSwitchMemTileArbiterMax = XAIE1_SS_ARBITER_MAX;
      deviceConfig.streamSwitchMemTileMSelMax = XAIE1_SS_MSEL_MAX;
      deviceConfig.streamSwitchShimArbiterMax = XAIE1_SS_ARBITER_MAX;
      deviceConfig.streamSwitchShimMSelMax = XAIE1_SS_MSEL_MAX;
      return AMDAIEDeviceModel(XAIE_DEV_GEN_AIE, XAIE1_BASE_ADDR,
                               XAIE1_COL_SHIFT, XAIE1_ROW_SHIFT, XAIE1_NUM_COLS,
                               XAIE1_NUM_ROWS, XAIE1_MEM_TILE_ROW_START,
                               XAIE1_MEM_TILE_NUM_ROWS,
                               // mlir-aie disagrees with aie-rt here
                               /*nShimTileRows*/ 0,
                               /*partitionNumCols*/ 50,
                               /*partitionStartCol*/ 0,
                               /*partBaseAddr*/ XAIE1_PARTITION_BASE_ADDR,
                               /*npiAddr*/ XAIE1_NPI_BASEADDR,
                               /*aieSim*/ false,
                               /*xaieDebug*/ false,
                               /*device*/ device,
                               /*deviceConfig*/ std::move(deviceConfig));
    }
    case AMDAIEDevice::xcve2302: {
      AMDAIEDeviceModel::AMDAIEDeviceConfig deviceConfig;
      deviceConfig.streamSwitchCoreArbiterMax = XAIEML_SS_ARBITER_MAX;
      deviceConfig.streamSwitchCoreMSelMax = XAIEML_SS_MSEL_MAX;
      deviceConfig.streamSwitchMemTileArbiterMax = XAIEML_SS_ARBITER_MAX;
      deviceConfig.streamSwitchMemTileMSelMax = XAIEML_SS_MSEL_MAX;
      deviceConfig.streamSwitchShimArbiterMax = XAIEML_SS_ARBITER_MAX;
      deviceConfig.streamSwitchShimMSelMax = XAIEML_SS_MSEL_MAX;
      return AMDAIEDeviceModel(XAIE_DEV_GEN_AIEML, XAIEML_BASE_ADDR,
                               XAIEML_COL_SHIFT, XAIEML_ROW_SHIFT,
                               /*numCols*/ 17, /*numRows*/ 4,
                               /*memTileRowStart*/ 1, /*nMemTileRows*/ 1,
                               /*nShimTileRows*/ 1,
                               /*partitionNumCols*/ 17,
                               /*partitionStartCol*/ 0,
                               /*partBaseAddr*/ XAIEML_PARTITION_BASE_ADDR,
                               /*npiAddr*/ XAIEML_NPI_BASEADDR,
                               /*aieSim*/ false,
                               /*xaieDebug*/ false,
                               /*device*/ device,
                               /*deviceConfig*/ std::move(deviceConfig));
    }
    case AMDAIEDevice::xcve2802: {
      AMDAIEDeviceModel::AMDAIEDeviceConfig deviceConfig;
      deviceConfig.streamSwitchCoreArbiterMax = XAIEML_SS_ARBITER_MAX;
      deviceConfig.streamSwitchCoreMSelMax = XAIEML_SS_MSEL_MAX;
      deviceConfig.streamSwitchMemTileArbiterMax = XAIEML_SS_ARBITER_MAX;
      deviceConfig.streamSwitchMemTileMSelMax = XAIEML_SS_MSEL_MAX;
      deviceConfig.streamSwitchShimArbiterMax = XAIEML_SS_ARBITER_MAX;
      deviceConfig.streamSwitchShimMSelMax = XAIEML_SS_MSEL_MAX;
      return AMDAIEDeviceModel(XAIE_DEV_GEN_AIEML, XAIEML_BASE_ADDR,
                               XAIEML_COL_SHIFT, XAIEML_ROW_SHIFT,
                               XAIEML_NUM_COLS, XAIEML_NUM_ROWS,
                               XAIEML_MEM_TILE_ROW_START,
                               XAIEML_MEM_TILE_NUM_ROWS, XAIEML_SHIM_NUM_ROWS,
                               /*partitionNumCols*/ 38,
                               /*partitionStartCol*/ 0,
                               /*partBaseAddr*/ XAIEML_PARTITION_BASE_ADDR,
                               /*npiAddr*/ XAIEML_NPI_BASEADDR,
                               /*aieSim*/ false,
                               /*xaieDebug*/ false,
                               /*device*/ device,
                               /*deviceConfig*/ std::move(deviceConfig));
    }
    case AMDAIEDevice::npu1:
    case AMDAIEDevice::npu1_1col:
    case AMDAIEDevice::npu1_2col:
    case AMDAIEDevice::npu1_3col:
    case AMDAIEDevice::npu1_4col: {
      AMDAIEDeviceModel::AMDAIEDeviceConfig deviceConfig;
      deviceConfig.streamSwitchCoreArbiterMax = XAIE2IPU_SS_ARBITER_MAX;
      deviceConfig.streamSwitchCoreMSelMax = XAIE2IPU_SS_MSEL_MAX;
      deviceConfig.streamSwitchMemTileArbiterMax = XAIE2IPU_SS_ARBITER_MAX;
      deviceConfig.streamSwitchMemTileMSelMax = XAIE2IPU_SS_MSEL_MAX;
      deviceConfig.streamSwitchShimArbiterMax = XAIE2IPU_SS_ARBITER_MAX;
      deviceConfig.streamSwitchShimMSelMax = XAIE2IPU_SS_MSEL_MAX;
      int partitionNumCols, partitionStartCol;
      switch (device) {
        case AMDAIEDevice::npu1:
          partitionNumCols = 5;
          partitionStartCol = 0;
          break;
        case AMDAIEDevice::npu1_1col:
          partitionNumCols = 1;
          partitionStartCol = 1;
          break;
        case AMDAIEDevice::npu1_2col:
          partitionNumCols = 2;
          partitionStartCol = 1;
          break;
        case AMDAIEDevice::npu1_3col:
          partitionNumCols = 3;
          partitionStartCol = 1;
          break;
        case AMDAIEDevice::npu1_4col:
          partitionNumCols = 4;
          partitionStartCol = 1;
          break;
        default:
          llvm::report_fatal_error("unhandled NPU partitioning.\n");
      }
      return AMDAIEDeviceModel(
          XAIE_DEV_GEN_AIE2IPU, XAIE2IPU_BASE_ADDR, XAIE2IPU_COL_SHIFT,
          XAIE2IPU_ROW_SHIFT, XAIE2IPU_NUM_COLS, XAIE2IPU_NUM_ROWS,
          XAIE2IPU_MEM_TILE_ROW_START, XAIE2IPU_MEM_TILE_NUM_ROWS,
          XAIE2IPU_SHIM_NUM_ROWS, partitionNumCols, partitionStartCol,
          /*partBaseAddr*/ XAIE2IPU_PARTITION_BASE_ADDR,
          /*npiAddr*/ XAIE2IPU_NPI_BASEADDR,
          /*aieSim*/ false,
          /*xaieDebug*/ false,
          /*device*/ device,
          /*deviceConfig*/ std::move(deviceConfig));
    }
    case AMDAIEDevice::npu4: {
      AMDAIEDeviceModel::AMDAIEDeviceConfig deviceConfig;
      deviceConfig.streamSwitchCoreArbiterMax = XAIE_STRIXB0_SS_ARBITER_MAX;
      deviceConfig.streamSwitchCoreMSelMax = XAIE_STRIXB0_SS_MSEL_MAX;
      deviceConfig.streamSwitchMemTileArbiterMax = XAIE_STRIXB0_SS_ARBITER_MAX;
      deviceConfig.streamSwitchMemTileMSelMax = XAIE_STRIXB0_SS_MSEL_MAX;
      deviceConfig.streamSwitchShimArbiterMax = XAIE_STRIXB0_SS_ARBITER_MAX;
      deviceConfig.streamSwitchShimMSelMax = XAIE_STRIXB0_SS_MSEL_MAX;
      return AMDAIEDeviceModel(
          XAIE_DEV_GEN_AIE2P_STRIX_B0, XAIE_STRIXB0_BASE_ADDR,
          XAIE_STRIXB0_COL_SHIFT, XAIE_STRIXB0_ROW_SHIFT, XAIE_STRIXB0_NUM_COLS,
          XAIE_STRIXB0_NUM_ROWS, XAIE_STRIXB0_MEM_TILE_ROW_START,
          XAIE_STRIXB0_MEM_TILE_NUM_ROWS, XAIE_STRIXB0_SHIM_NUM_ROWS,
          /*partitionNumCols*/ 8,
          /*partitionStartCol*/ 0,
          /*partBaseAddr*/ XAIE_STRIXB0_PARTITION_BASE_ADDR,
          /*npiAddr*/ XAIE_STRIXB0_NPI_BASEADDR,
          /*aieSim*/ false,
          /*xaieDebug*/ false,
          /*device*/ device,
          /*deviceConfig*/ std::move(deviceConfig));
    }
  }

  llvm::report_fatal_error("Unhandled AMDAIEDevice case");
}

/// ============================= BEGIN ==================================
/// ================== stringification utils =============================
/// ======================================================================

std::string to_string(const int &value) { return std::to_string(value); }
std::string to_string(const uint32_t &value) { return std::to_string(value); }
std::string to_string(const uint64_t &value) { return std::to_string(value); }

std::string to_string(const StrmSwPortType &value) {
  switch (value) {
    STRINGIFY_ENUM_CASE(StrmSwPortType::CORE)
    STRINGIFY_ENUM_CASE(StrmSwPortType::DMA)
    STRINGIFY_ENUM_CASE(StrmSwPortType::CTRL)
    STRINGIFY_ENUM_CASE(StrmSwPortType::FIFO)
    STRINGIFY_ENUM_CASE(StrmSwPortType::SOUTH)
    STRINGIFY_ENUM_CASE(StrmSwPortType::WEST)
    STRINGIFY_ENUM_CASE(StrmSwPortType::NORTH)
    STRINGIFY_ENUM_CASE(StrmSwPortType::EAST)
    STRINGIFY_ENUM_CASE(StrmSwPortType::TRACE)
    STRINGIFY_ENUM_CASE(StrmSwPortType::UCTRLR)
    STRINGIFY_ENUM_CASE(StrmSwPortType::SS_PORT_TYPE_MAX)
    STRINGIFY_ENUM_CASE(StrmSwPortType::NOC)
  }

  llvm::report_fatal_error("Unhandled StrmSwPortType case");
}

std::string to_string(const ::StrmSwPortType &value) {
  using StrmSwPortType = ::StrmSwPortType;
  switch (value) {
    STRINGIFY_ENUM_CASE(StrmSwPortType::CORE)
    STRINGIFY_ENUM_CASE(StrmSwPortType::DMA)
    STRINGIFY_ENUM_CASE(StrmSwPortType::CTRL)
    STRINGIFY_ENUM_CASE(StrmSwPortType::FIFO)
    STRINGIFY_ENUM_CASE(StrmSwPortType::SOUTH)
    STRINGIFY_ENUM_CASE(StrmSwPortType::WEST)
    STRINGIFY_ENUM_CASE(StrmSwPortType::NORTH)
    STRINGIFY_ENUM_CASE(StrmSwPortType::EAST)
    STRINGIFY_ENUM_CASE(StrmSwPortType::TRACE)
    STRINGIFY_ENUM_CASE(StrmSwPortType::UCTRLR)
    STRINGIFY_ENUM_CASE(::StrmSwPortType::SS_PORT_TYPE_MAX)
  }

  llvm::report_fatal_error("Unhandled StrmSwPortType case");
}

std::string to_string(const AieRC &value) {
  switch (value) {
    STRINGIFY_ENUM_CASE(AieRC::XAIE_OK)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_ERR)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_DEVICE)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_RANGE)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_ARGS)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_TILE)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_ERR_STREAM_PORT)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_DMA_TILE)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_BD_NUM)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_ERR_OUTOFBOUND)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_DATA_MEM_ADDR)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_ELF)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_CORE_STATUS_TIMEOUT)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_CHANNEL_NUM)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_LOCK)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_DMA_DIRECTION)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_PLIF_WIDTH)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_LOCK_ID)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_LOCK_VALUE)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_LOCK_RESULT_FAILED)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_DMA_DESC)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_ADDRESS)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_FEATURE_NOT_SUPPORTED)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_BURST_LENGTH)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_BACKEND)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INSUFFICIENT_BUFFER_SIZE)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_API_POINTER)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_ERR_MAX)
  }
  // TODO(max): Don't understand why putting this under a default case doesn't
  // work/solve
  // TODO(max): We need to enable -Wswitch-enum as well
  llvm::report_fatal_error("Unhandled AieRC case");
}

std::string to_string(const AMDAIETileType &value) {
  switch (value) {
    STRINGIFY_ENUM_CASE(AMDAIETileType::AIETILE)
    STRINGIFY_ENUM_CASE(AMDAIETileType::SHIMNOC)
    STRINGIFY_ENUM_CASE(AMDAIETileType::SHIMPL)
    STRINGIFY_ENUM_CASE(AMDAIETileType::MEMTILE)
    STRINGIFY_ENUM_CASE(AMDAIETileType::MAX)
  }

  llvm::report_fatal_error("Unhandled AMDAIETileType case");
}

std::string to_string(const XAie_TxnOpcode &value) {
  switch (value) {
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_WRITE)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_BLOCKWRITE)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_BLOCKSET)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_MASKWRITE)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_MASKPOLL)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_CONFIG_SHIMDMA_BD)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_CONFIG_SHIMDMA_DMABUF_BD)
    //    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_BEGIN)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_TCT)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_DDR_PATCH)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_READ_REGS)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_RECORD_TIMER)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_MERGE_SYNC)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_NEXT)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_MAX)
  }

  llvm::report_fatal_error("Unhandled XAie_TxnOpcode case");
}

std::string to_string(const DMAChannelDir &value) {
  switch (value) {
    STRINGIFY_ENUM_CASE(DMAChannelDir::MM2S)
    STRINGIFY_ENUM_CASE(DMAChannelDir::S2MM)
  }

  llvm::report_fatal_error("Unhandled AMDAIETileType case");
}

STRINGIFY_2TUPLE_STRUCT(SwitchDMAConnection, direction, channel)
STRINGIFY_2TUPLE_STRUCT(TileLoc, col, row)
STRINGIFY_2TUPLE_STRUCT(XAie_LocType, Col, Row)
STRINGIFY_2TUPLE_STRUCT(XAie_Lock, LockId, LockVal)
STRINGIFY_2TUPLE_STRUCT(XAie_Packet, PktId, PktType)

std::string to_string(const XAie_OpHdr &t) {
  std::string s =
      "XAie_OpHdr(Op: " + to_string(static_cast<XAie_TxnOpcode>(t.Op));
  s += ", Col: " + to_string(t.Col);
  s += ", Row: " + to_string(t.Row) + ")";
  return s;
}

STRINGIFY_5TUPLE_STRUCT(XAie_TxnCmd, Opcode, Mask, RegOff, Value, Size)
STRINGIFY_4TUPLE_STRUCT(XAie_Write32Hdr, OpHdr, RegOff, Value, Size)
STRINGIFY_5TUPLE_STRUCT(XAie_MaskWrite32Hdr, OpHdr, RegOff, Value, Mask, Size)
STRINGIFY_4TUPLE_STRUCT(XAie_MaskPoll32Hdr, OpHdr, RegOff, Value, Size)
STRINGIFY_5TUPLE_STRUCT(XAie_BlockWrite32Hdr, OpHdr, Col, Row, RegOff, Size)
STRINGIFY_2TUPLE_STRUCT(XAie_CustomOpHdr, OpHdr, Size)

BOTH_OSTREAM_OPS_FORALL_TYPES(OSTREAM_OP_DEFN, BOTH_OSTREAM_OP)
}  // namespace mlir::iree_compiler::AMDAIE
