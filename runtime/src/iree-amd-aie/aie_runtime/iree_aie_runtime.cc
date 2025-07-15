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

uint8_t AMDAIEDeviceModel::getMinStrideBitWidth() const {
  return deviceConfig.minStrideBitWidth;
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

uint8_t AMDAIEDeviceModel::getDmaMaxQueueSize(uint8_t col, uint8_t row) const {
  uint8_t maxQueueSize = 0;
  TRY_XAIE_API_FATAL_ERROR(XAie_DmaGetMaxQueueSize,
                           const_cast<XAie_DevInst *>(&devInst),
                           XAie_TileLoc(col, row), &maxQueueSize);
  return maxQueueSize;
}

// TODO(max): these should be optionals instead of returning 0.
uint32_t AMDAIEDeviceModel::getNumLocks(uint8_t col, uint8_t row) const {
  AMDAIETileType tileType = getTileType(col, row);
  if (tileType == AMDAIETileType::SHIMPL || tileType == AMDAIETileType::MAX)
    return 0;
  return devInst.DevProp.DevMod[static_cast<uint8_t>(tileType)]
      .LockMod->NumLocks;
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

void setOddParityBit(uint32_t &word) {
  // Mask to keep the lower 31 bits (bits 30:0).
  uint32_t lower31Bits = word & 0x7FFFFFFF;
  // Compute the odd parity bit. It is set to 1 if the number of 1's in
  // lower31Bits is even, ensuring the total count (including this bit) becomes
  // odd. Otherwise, it is set to 0.
  uint32_t parity = (llvm::popcount(lower31Bits) + 1) % 2;
  // Set the parity bit in the most significant bit (bit 31).
  word = (parity << 31) | lower31Bits;
}

FailureOr<uint32_t> AMDAIEDeviceModel::getPacketHeader(uint32_t streamId,
                                                       uint32_t packetType,
                                                       uint32_t srcRow,
                                                       uint32_t srcCol) const {
  if (srcRow >= rows() || srcCol >= columns()) {
    llvm::errs() << "source tile out of range\n";
    return failure();
  }
  if (streamId > getPacketIdMaxIdx()) {
    llvm::errs() << "streamId out of range\n";
    return failure();
  }
  if (packetType > getpacketTypeMax()) {
    llvm::errs() << "packetType out of range\n";
    return failure();
  }
  // Construct the header by shifting and combining the individual fields.
  uint32_t header = (srcCol << packetHeaderFormat.srcColShift) |
                    (srcRow << packetHeaderFormat.srcRowShift) |
                    (packetType << packetHeaderFormat.packetTypeShift) |
                    (streamId << packetHeaderFormat.streamIdShift);
  setOddParityBit(header);
  return header;
}

FailureOr<uint32_t> AMDAIEDeviceModel::getControlHeader(
    uint32_t address, uint32_t length, uint32_t opcode,
    uint32_t streamId) const {
  if (address > getCtrlPktMaxAddress()) {
    llvm::errs() << "address out of range\n";
    return failure();
  }
  if (length > getCtrlPktMaxLength() || length == 0) {
    llvm::errs() << "length out of range\n";
    return failure();
  }
  if (opcode > getCtrlPktMaxOpcode()) {
    llvm::errs() << "opcode out of range\n";
    return failure();
  }
  if (streamId > getPacketIdMaxIdx()) {
    llvm::errs() << "streamId out of range\n";
    return failure();
  }
  // Construct the header by shifting and combining the individual fields.
  // Note that length `i` is encoded in the header as `i - 1`.
  uint32_t header = (streamId << controlHeaderFormat.streamIdShift) |
                    (opcode << controlHeaderFormat.operationShift) |
                    ((length - 1) << controlHeaderFormat.beatShift) |
                    (address << controlHeaderFormat.addressShift);
  setOddParityBit(header);
  return header;
}

uint32_t AMDAIEDeviceModel::getCtrlPktMaxAddress() const {
  return (1 << (controlHeaderFormat.beatShift -
                controlHeaderFormat.addressShift)) -
         1;
}

uint32_t AMDAIEDeviceModel::getCtrlPktMaxLength() const {
  return (1 << (controlHeaderFormat.operationShift -
                controlHeaderFormat.beatShift));
}

uint32_t AMDAIEDeviceModel::getCtrlPktMaxOpcode() const {
  return (1 << (controlHeaderFormat.streamIdShift -
                controlHeaderFormat.operationShift)) -
         1;
}

uint32_t AMDAIEDeviceModel::getTileMemorySizeInBytes(uint8_t col,
                                                     uint8_t row) const {
  AMDAIETileType tileType = getTileType(col, row);
  switch (tileType) {
    case AMDAIETileType::AIETILE:
      return getLocalMemorySize(col, row);
    case AMDAIETileType::MEMTILE:
      return getMemTileSize(col, row);
    case AMDAIETileType::SHIMNOC:
    case AMDAIETileType::SHIMPL:
      return std::numeric_limits<uint32_t>::max();
    default:
      return 0;
  }
}

uint32_t AMDAIEDeviceModel::getMemInternalBaseAddress() const {
  return getMemEastBaseAddress();
}

uint32_t AMDAIEDeviceModel::getMemTileSizeInBytes() const {
  return devInst.DevProp.DevMod[static_cast<uint8_t>(AMDAIETileType::MEMTILE)]
      .MemMod->Size;
}

uint32_t AMDAIEDeviceModel::getMemTileSize(uint8_t col, uint8_t row) const {
  AMDAIETileType tileType = getTileType(col, row);
  assert(tileType == AMDAIETileType::MEMTILE && "expected memtile");
  return devInst.DevProp.DevMod[static_cast<uint8_t>(tileType)].MemMod->Size;
}

SmallVector<uint32_t> AMDAIEDeviceModel::getMemSpaceRows(
    uint8_t memSpace) const {
  SmallVector<uint32_t> res;
  if (memSpace == 0) {
    res.resize(deviceConfig.shimTileNumRows);
    std::iota(res.begin(), res.end(), configPtr.ShimRowNum);
  } else if (memSpace == 1) {
    res.resize(configPtr.MemTileNumRows);
    std::iota(res.begin(), res.end(), configPtr.MemTileRowStart);
  } else if (memSpace == 2) {
    res.resize(configPtr.AieTileNumRows);
    std::iota(res.begin(), res.end(), configPtr.AieTileRowStart);
  }
  return res;
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

const llvm::SmallDenseMap<std::pair<StrmSwPortType, uint8_t>,
                          std::pair<StrmSwPortType, uint8_t>>
    AMDAIEDeviceModel::mm2sDmaNocToSpecialShimPortMap = {
        {{StrmSwPortType::DMA, 0}, {StrmSwPortType::NORTH, 3}},
        {{StrmSwPortType::DMA, 1}, {StrmSwPortType::NORTH, 7}},
        {{StrmSwPortType::NOC, 0}, {StrmSwPortType::NORTH, 2}},
        {{StrmSwPortType::NOC, 1}, {StrmSwPortType::NORTH, 3}},
        {{StrmSwPortType::NOC, 2}, {StrmSwPortType::NORTH, 6}},
        {{StrmSwPortType::NOC, 3}, {StrmSwPortType::NORTH, 7}}};

const llvm::SmallDenseMap<std::pair<StrmSwPortType, uint8_t>,
                          std::pair<StrmSwPortType, uint8_t>>
    AMDAIEDeviceModel::s2mmDmaNocToSpecialShimPortMap = {
        {{StrmSwPortType::DMA, 0}, {StrmSwPortType::NORTH, 2}},
        {{StrmSwPortType::DMA, 1}, {StrmSwPortType::NORTH, 3}},
        {{StrmSwPortType::NOC, 0}, {StrmSwPortType::NORTH, 2}},
        {{StrmSwPortType::NOC, 1}, {StrmSwPortType::NORTH, 3}},
        {{StrmSwPortType::NOC, 2}, {StrmSwPortType::NORTH, 4}},
        {{StrmSwPortType::NOC, 3}, {StrmSwPortType::NORTH, 5}}};

std::optional<std::pair<StrmSwPortType, uint8_t>>
AMDAIEDeviceModel::getShimMuxPortMappingForDmaOrNoc(
    StrmSwPortType port, uint8_t channel, DMAChannelDir direction) const {
  auto key = std::make_pair(port, channel);
  if (direction == DMAChannelDir::MM2S &&
      mm2sDmaNocToSpecialShimPortMap.count(key)) {
    return mm2sDmaNocToSpecialShimPortMap.at(key);
  } else if (direction == DMAChannelDir::S2MM &&
             s2mmDmaNocToSpecialShimPortMap.count(key)) {
    return s2mmDmaNocToSpecialShimPortMap.at(key);
  }
  return std::nullopt;
}

std::optional<std::pair<StrmSwPortType, uint8_t>>
AMDAIEDeviceModel::getDmaFromShimMuxPortMapping(StrmSwPortType port,
                                                uint8_t channel,
                                                DMAChannelDir direction) const {
  auto key = std::make_pair(port, channel);
  if (direction == DMAChannelDir::MM2S) {
    for (auto &entry : mm2sDmaNocToSpecialShimPortMap) {
      if (entry.first.first == StrmSwPortType::DMA && entry.second == key)
        return entry.first;
    }
  } else if (direction == DMAChannelDir::S2MM) {
    for (auto &entry : s2mmDmaNocToSpecialShimPortMap) {
      if (entry.first.first == StrmSwPortType::DMA && entry.second == key)
        return entry.first;
    }
  }
  return std::nullopt;
}

std::optional<std::string> AMDAIEDeviceModel::getNPUVersionString() const {
  switch (configPtr.AieGen) {
    case XAIE_DEV_GEN_AIE2IPU:
      return "npu1";
    case XAIE_DEV_GEN_AIE2P_STRIX_B0:
      return "npu4";
    default:
      return std::nullopt;
  }
}

std::optional<std::string> AMDAIEDeviceModel::getTargetArchString() const {
  switch (configPtr.AieGen) {
    case XAIE_DEV_GEN_AIE:
      return "AIE";
    case XAIE_DEV_GEN_AIE2IPU:
      return "AIE2";
    case XAIE_DEV_GEN_AIE2P_STRIX_B0:
      return "AIE2P";
    default:
      return std::nullopt;
  }
}

uint32_t AMDAIEDeviceModel::getColumnShift() const {
  return configPtr.ColShift;
}

uint32_t AMDAIEDeviceModel::getRowShift() const { return configPtr.RowShift; }

uint32_t AMDAIEDeviceModel::getCoreTileRowStart() const {
  return configPtr.AieTileRowStart;
}

uint32_t AMDAIEDeviceModel::getColumnFromAddress(uint32_t address) const {
  uint32_t columnMask = (1 << (getColumnShift() - getRowShift())) - 1;
  return (address >> getColumnShift()) & columnMask;
}

uint32_t AMDAIEDeviceModel::getRowFromAddress(uint32_t address) const {
  uint32_t rowMask = (1 << (getColumnShift() - getRowShift())) - 1;
  return (address >> getRowShift()) & rowMask;
}

uint32_t AMDAIEDeviceModel::getOffsetFromAddress(uint32_t address) const {
  uint32_t offsetMask = (1 << getRowShift()) - 1;
  return address & offsetMask;
}

uint8_t AMDAIEDeviceModel::getPacketIdMaxIdx() const {
  return deviceConfig.packetIdMaxIdx;
}

uint8_t AMDAIEDeviceModel::getpacketTypeMax() const {
  return deviceConfig.packetTypeMax;
}

uint8_t AMDAIEDeviceModel::getPacketIdMaskWidth() const {
  return deviceConfig.packetIdMaskWidth;
}

uint8_t AMDAIEDeviceModel::getNumPacketRuleSlots(uint8_t col,
                                                 uint8_t row) const {
  AMDAIETileType tileType = getTileType(col, row);
  const XAie_StrmMod *strmMod =
      devInst.DevProp.DevMod[static_cast<uint8_t>(tileType)].StrmSw;
  return strmMod->NumSlaveSlots;
}

bool AMDAIEDeviceModel::getCtrlPktTlastErrorDisabled() const {
  return deviceConfig.ctrlPktTlastErrorDisabled;
}

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
      deviceConfig.shimTileNumRows = XAIE1_SHIM_NUM_ROWS;
      deviceConfig.packetIdMaxIdx = XAIE1_PACKET_ID_MAX;
      deviceConfig.packetTypeMax = XAIE1_PACKET_TYPE_MAX;
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
      deviceConfig.shimTileNumRows = XAIEML_SHIM_NUM_ROWS;
      deviceConfig.packetIdMaxIdx = XAIEML_PACKET_ID_MAX;
      deviceConfig.packetTypeMax = XAIEML_PACKET_TYPE_MAX;
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
      deviceConfig.shimTileNumRows = XAIEML_SHIM_NUM_ROWS;
      deviceConfig.packetIdMaxIdx = XAIEML_PACKET_ID_MAX;
      deviceConfig.packetTypeMax = XAIEML_PACKET_TYPE_MAX;
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
      deviceConfig.shimTileNumRows = XAIE2IPU_SHIM_NUM_ROWS;
      deviceConfig.packetIdMaxIdx = XAIE2IPU_PACKET_ID_MAX;
      deviceConfig.packetTypeMax = XAIE2IPU_PACKET_TYPE_MAX;
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
      deviceConfig.shimTileNumRows = XAIE_STRIXB0_MEM_TILE_NUM_ROWS;
      deviceConfig.packetIdMaxIdx = XAIE_STRIXB0_PACKET_ID_MAX;
      deviceConfig.packetTypeMax = XAIE_STRIXB0_PACKET_TYPE_MAX;
      deviceConfig.ctrlPktTlastErrorDisabled = true;
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

/// Generate a DenseMap key we can use for the element types (alternatives
/// considered: implement tombstone for std::array, or use std::map instead of
/// DenseMap).
static constexpr uint32_t getElementTypeKey(uint32_t a, uint32_t b,
                                            uint32_t c) {
  return a + (b << 8) + (c << 16);
}

/// Map from (LHS bitwidth, RHS bitwidth, Accumulator bitwidth) to the NPU1
/// (AIE2) instruction size (m, n, k) for the integer types with those
/// bitwidths. This function is based on the following table pulled from the
/// AIEVec_MatMulOp documentation in
/// mlir-aie/include/aie/Dialect/AIEVec/IR/AIEVecOps.td
///
///   lhs                | rhs                | accumulator
///  :------------------:|:------------------:|:-----------------:
///   `vector<4x16xi8>`  | `vector<16x8xi4>`  | `vector<4x8xi32>`
///   `vector<4x8xi8>`   | `vector<8x8xi8>`   | `vector<4x8xi32>`
///   `vector<4x4xi16>`  | `vector<4x8xi8>`   | `vector<4x8xi32>`
///   `vector<4x2xi16>`  | `vector<2x8xi16>`  | `vector<4x8xi32>`
///   `vector<2x8xi16>`  | `vector<8x8xi8>`   | `vector<2x8xi64>`
///   `vector<4x8xi16>`  | `vector<8x4xi8>`   | `vector<4x4xi64>`
///   `vector<2x4xi16>`  | `vector<4x8xi16>`  | `vector<2x8xi64>`
///   `vector<4x4xi16>`  | `vector<4x4xi16>`  | `vector<4x4xi64>`
///   `vector<4x2xi32>`  | `vector<2x4xi16>`  | `vector<4x4xi64>`
///   `vector<4x8xbf16>` | `vector<8x4xbf16>` | `vector<4x4xf32>`
///
/// An instruction size (m, n, k) is returned for each combination of element
/// type in the table. Combinations of element type that are not covered by the
/// table return failure.
///
/// Example: consider the first line of the table:
///   `vector<4x16xi8>`  | `vector<16x8xi4>`  | `vector<4x8xi32>`
///
/// This first line says that if 'lhs' is an i8 tensor, 'rhs' is an i4 tensor
/// and 'accumulator' is an i32 tensor, then there is an AIE instruction for
/// matmul with m = 4, n = 8, k = 16.
static llvm::DenseMap<uint32_t, std::array<uint32_t, 3>> &
getNpu1IntegerMatmulInstructionSizeMap() {
  // Sanity check.
  static_assert(getElementTypeKey(1, 2, 3) == 1 + 2 * 256 + 3 * 65536);

  static llvm::DenseMap<uint32_t, std::array<uint32_t, 3>> matmulIntSizes{

      // `vector<4x16xi8>`  | `vector<16x8xi4>`  | `vector<4x8xi32>`
      {getElementTypeKey(8, 4, 32), {4, 8, 16}},

      // `vector<4x8xi8>`   | `vector<8x8xi8>`   | `vector<4x8xi32>`
      {getElementTypeKey(8, 8, 32), {4, 8, 8}},

      // `vector<4x4xi16>`  | `vector<4x8xi8>`   | `vector<4x8xi32>`
      {getElementTypeKey(16, 8, 32), {4, 8, 4}},

      // `vector<4x2xi16>`  | `vector<2x8xi16>`  | `vector<4x8xi32>`
      {getElementTypeKey(16, 16, 32), {4, 8, 2}},

      // `vector<2x8xi16>`  | `vector<8x8xi8>`   | `vector<2x8xi64>`
      // `vector<4x8xi16>`  | `vector<8x4xi8>`   | `vector<4x4xi64>`
      //   choosing the first i16 x i8 -> i64 instruction (arbitrarily)
      {getElementTypeKey(16, 8, 64), {2, 8, 8}},

      // `vector<2x4xi16>`  | `vector<4x8xi16>`  | `vector<2x8xi64>`
      // `vector<4x4xi16>`  | `vector<4x4xi16>`  | `vector<4x4xi64>`
      //   choosing the first i16 x i16 -> i64 instruction (arbitrarily)
      {getElementTypeKey(16, 16, 64), {2, 8, 4}},

      // `vector<4x2xi32>`  | `vector<2x4xi16>`  | `vector<4x4xi64>`
      {getElementTypeKey(32, 16, 64), {4, 4, 2}},
  };
  return matmulIntSizes;
}

/// Map from (LHS bitwidth, RHS bitwidth, Accumulator bitwidth) to the NPU4
/// (AIE2P) instruction size (m, n, k) for the integer types with those
/// bitwidths.
///
///   lhs                | rhs                | accumulator
///  :------------------:|:------------------:|:-----------------:
///   `vector<8x8xi8>`   | `vector<8x8xi8>`   | `vector<8x8xi32>`
///
/// An instruction size (m, n, k) is returned for each combination of element
/// type in the table. Combinations of element type that are not covered by the
/// table return failure.
///
/// Example: consider the line of the table:
///   `vector<8x8xi8>`  | `vector<8x8xi8>`  | `vector<8x8xi32>`
///
/// This first line says that if 'lhs' is an i8 tensor, 'rhs' is an i8 tensor
/// and 'accumulator' is an i32 tensor, then there is an AIE instruction for
/// matmul with m = 8, n = 8, k = 8.
static llvm::DenseMap<uint32_t, std::array<uint32_t, 3>> &
getNpu4IntegerMatmulInstructionSizeMap() {
  // Sanity check.
  static_assert(getElementTypeKey(1, 2, 3) == 1 + 2 * 256 + 3 * 65536);

  static llvm::DenseMap<uint32_t, std::array<uint32_t, 3>> matmulIntSizes{

      // `vector<8x8xi8>`   | `vector<8x8xi8>`   | `vector<8x8xi32>`
      {getElementTypeKey(8, 8, 32), {8, 8, 8}},

  };
  return matmulIntSizes;
}

/// Return the AIE instruction size (m, n, k) for the integer types with
/// bitwidths nBitsLhs, nBitsRhs, and nBitsAcc. Based on the table above.
static llvm::FailureOr<std::array<uint32_t, 3>> getIntegerMatmulInstructionSize(
    uint32_t nBitsLhs, uint32_t nBitsRhs, uint32_t nBitsAcc,
    const llvm::DenseMap<uint32_t, std::array<uint32_t, 3>> &mapForIntTypes) {
  auto it =
      mapForIntTypes.find(getElementTypeKey(nBitsLhs, nBitsRhs, nBitsAcc));
  if (it == mapForIntTypes.end()) {
    return failure();
  }
  return it->second;
}

llvm::FailureOr<std::array<uint32_t, 3>>
AMDAIEDeviceModel::getAIEMatmulInstructionSize(Type elTypeLhs, Type elTypeRhs,
                                               Type elTypeAcc) const {
  bool allFloatingPoint = isa<FloatType>(elTypeLhs) &&
                          isa<FloatType>(elTypeRhs) &&
                          isa<FloatType>(elTypeAcc);

  bool allInteger = isa<IntegerType>(elTypeLhs) &&
                    isa<IntegerType>(elTypeRhs) && isa<IntegerType>(elTypeAcc);

  if (!allInteger && !allFloatingPoint) {
    return failure();
  }

  auto nBitsLhs = elTypeLhs.getIntOrFloatBitWidth();
  auto nBitsRhs = elTypeRhs.getIntOrFloatBitWidth();
  auto nBitsAcc = elTypeAcc.getIntOrFloatBitWidth();

  if (allFloatingPoint) {
    if (nBitsLhs == 16 && nBitsRhs == 16 && nBitsAcc == 32) {
      if (device == AMDAIEDevice::npu4) {
        // Strix npu4 intrinsics.
        return std::array<uint32_t, 3>{8, 8, 8};
      } else {
        // Phoenix npu1_4col intrinsics.
        return std::array<uint32_t, 3>{4, 4, 8};
      }
    }
    // There is only 1 floating point case in the table (handled above).
    return failure();
  }

  assert(allInteger &&
         "expected all element types to either be all float types or all "
         "integer types");
  if (device == AMDAIEDevice::npu1_4col || device == AMDAIEDevice::npu1_3col ||
      device == AMDAIEDevice::npu1_2col || device == AMDAIEDevice::npu1_1col ||
      device == AMDAIEDevice::npu1) {
    return getIntegerMatmulInstructionSize(
        nBitsLhs, nBitsRhs, nBitsAcc, getNpu1IntegerMatmulInstructionSizeMap());
  } else if (device == AMDAIEDevice::npu4) {
    return getIntegerMatmulInstructionSize(
        nBitsLhs, nBitsRhs, nBitsAcc, getNpu4IntegerMatmulInstructionSizeMap());
  }
  return failure();
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
    STRINGIFY_ENUM_CASE(AieRC::XAIE_NOT_SUPPORTED)
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
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_NOOP)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_PREEMPT)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_MASKPOLL_BUSY)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_LOADPDI)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_LOAD_PM_START)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_CREATE_SCRATCHPAD)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_UPDATE_STATE_TABLE)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_UPDATE_REG)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_UPDATE_SCRATCH)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_CONFIG_SHIMDMA_BD)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_CONFIG_SHIMDMA_DMABUF_BD)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_TCT)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_DDR_PATCH)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_READ_REGS)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_RECORD_TIMER)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_MERGE_SYNC)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_NEXT)
    STRINGIFY_ENUM_CASE(XAie_TxnOpcode::XAIE_IO_LOAD_PM_END_INTERNAL)
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
