// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cassert>
#include <cstdint>  // uint
#include <cstdlib>  // calloc
#include <filesystem>
#include <map>
#include <optional>
#include <string>

#include "AMDAIETargets.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIEEnums.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "aie-generate-cdo"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace mlir::iree_compiler;

StrmSwPortType toStrmT(WireBundle w) {
  switch (w) {
    case WireBundle::Core:
      return StrmSwPortType::CORE;
    case WireBundle::DMA:
      return StrmSwPortType::DMA;
    case WireBundle::FIFO:
      return StrmSwPortType::FIFO;
    case WireBundle::South:
      return StrmSwPortType::SOUTH;
    case WireBundle::West:
      return StrmSwPortType::WEST;
    case WireBundle::North:
      return StrmSwPortType::NORTH;
    case WireBundle::East:
      return StrmSwPortType::EAST;
    case WireBundle::PLIO:
      llvm::report_fatal_error("unhandled PLIO");
    case WireBundle::NOC:
      llvm::report_fatal_error("unhandled NOC");
    case WireBundle::Trace:
      return StrmSwPortType::TRACE;
    case WireBundle::Ctrl:
      return StrmSwPortType::CTRL;
    default:
      llvm::report_fatal_error("unhandled WireBundle");
  }
}

auto ps = std::filesystem::path::preferred_separator;

#define NUM_LOCKS 16
#define MEM_TILE_LOCK_ID_INCR 64
#define BASE_ADDR_A_INCR 0x80000

namespace mlir::iree_compiler::AMDAIE {

LogicalResult configureLocksInBdBlock(XAie_DmaDesc &dmaTileBd, Block &block,
                                      AMDAIENPUDeviceModel deviceModel,
                                      XAie_LocType &tileLoc) {
  LLVM_DEBUG(llvm::dbgs() << "\nstart configuring bds\n");
  std::optional<int> acqValue, relValue, acqLockId, relLockId;
  bool acqEn;
  // switch (lock->getAc)
  for (auto op : block.getOps<UseLockOp>()) {
    // Only dyn_cast if you are going to check if it was of the type
    // expected; if you aren't checking use cast instead as it will at
    // least assert in debug mode with an easier to understand error than
    // dereferencing.
    LockOp lock = cast<LockOp>(op.getLock().getDefiningOp());
    switch (op.getAction()) {
      case LockAction::Acquire:
      case LockAction::AcquireGreaterEqual:
        acqEn = op.getAcqEn();
        acqLockId = lock.getLockIDValue();
        acqValue = op.getLockValue();
        if (op.acquireGE()) acqValue.value() = -acqValue.value();
        break;
      case LockAction::Release:
        relLockId = lock.getLockIDValue();
        relValue = op.getLockValue();
        break;
    }
  }

  assert(acqValue && relValue && acqLockId && relLockId &&
         "expected both use_lock(acquire) and use_lock(release) with bd");

  if (deviceModel.isMemTile(tileLoc.Col, tileLoc.Row)) {
    if (acqLockId) acqLockId.value() += MEM_TILE_LOCK_ID_INCR;
    if (relLockId) relLockId.value() += MEM_TILE_LOCK_ID_INCR;
  }

  // no RelEn in the arch spec even though the API requires you to set it?
  bool relEn = false;
  XAie_Lock acqLock = XAie_LockInit(acqLockId.value(), acqValue.value());
  XAie_Lock relLock = XAie_LockInit(relLockId.value(), relValue.value());
  TRY_XAIE_API_EMIT_ERROR((*block.getOps<UseLockOp>().begin()),
                          dmaTileBd.DmaMod->SetLock, &dmaTileBd, acqLock,
                          relLock, acqEn, relEn);
  return success();
}

LogicalResult configureBdInBlock(XAie_DmaDesc &dmaTileBd, Block &block,
                                 AMDAIENPUDeviceModel deviceModel,
                                 XAie_LocType &tileLoc, int bdId,
                                 std::optional<int> nextBdId) {
  std::optional<int> packetType;
  std::optional<int> packetID;
  auto maybePacketOps = block.getOps<DMABDPACKETOp>();
  if (!maybePacketOps.empty()) {
    assert(llvm::range_size(maybePacketOps) == 1 &&
           "expected only one dma_bd_packet");
    auto packetOp = *maybePacketOps.begin();
    packetType = packetOp.getPacketType();
    packetID = packetOp.getPacketID();
  }

  auto bdOp = *block.getOps<DMABDOp>().begin();

  if (deviceModel.isShimNOCTile(tileLoc.Col, tileLoc.Row)) {
    // write them out like this so they show up with names in debug prints
    size_t smid = 0;
    size_t burstLen = 16;  // (10):BLEN=16 (256Byte) (corresponds to
                           // 0x800000000 from target)
    size_t qOs = 0;
    size_t cache = 0;
    size_t secure = 0;
    TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaSetAxi, &dmaTileBd, smid, burstLen,
                            qOs, cache, secure);
  }

  // StringRef FifoMode = disable; // FIXME: when to enable FIFO mode?
  int baseAddr = 0;
  if (!deviceModel.isShimNOCTile(tileLoc.Col, tileLoc.Row)) {
    auto bufferOp = cast<AIE::BufferOp>(bdOp.getBuffer().getDefiningOp());
    if (!bufferOp.getAddress())
      return bufferOp.emitError("buffer must have address assigned");
    baseAddr = bufferOp.getAddress().value();
    if (deviceModel.isMemTile(tileLoc.Col, tileLoc.Row))
      baseAddr += BASE_ADDR_A_INCR;
  }

  std::optional<llvm::ArrayRef<BDDimLayoutAttr>> dims = bdOp.getDimensions();
  int lenInBytes = bdOp.getLenInBytes();
  int basePlusOffsetInBytes = baseAddr + bdOp.getOffsetInBytes();
  if (!dims) {
    TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaSetAddrLen, &dmaTileBd,
                            basePlusOffsetInBytes, lenInBytes);
  } else {
    XAie_DmaTensor dmaTileBdTensor = {};
    dmaTileBdTensor.NumDim = dims->size();
    dmaTileBdTensor.Dim = static_cast<XAie_DmaDimDesc *>(
        calloc(dmaTileBdTensor.NumDim, sizeof(XAie_DmaDimDesc)));
    if (!dmaTileBdTensor.Dim)
      return bdOp.emitError("couldn't allocate array of XAie_DmaDimDesc");
    // libxaie requires stride in multiples of 32b
    double elementWidthIn32bWords =
        static_cast<double>(bdOp.getBufferElementTypeWidthInBytes()) / 4.0;
    for (size_t i = 0; i < dims->size(); i++) {
      // Pass down dimensions in reverse order; in the MLIR, this allows
      // us to specify step sizes/wraps in the same order as we would
      // access a multi-dim C array, with the highest dimension first.
      int j = dims->size() - i - 1;
      uint16_t size;
      uint32_t stride;
      if (j > 0) {
        stride = static_cast<uint32_t>(dims.value()[i].getStride() *
                                       elementWidthIn32bWords);
        size = dims.value()[i].getSize();
      } else {
        stride = dims.value()[i].getStride();
        size = static_cast<uint16_t>(dims.value()[i].getSize() *
                                     elementWidthIn32bWords);
      }
      stride = stride > 0 ? stride : 1;
      // Assume AIE-ML architecture (ie use AieMlDimDesc instead of AieDimDesc);
      // asserted in AIETranslateToCDODirect).
      dmaTileBdTensor.Dim[j].AieMlDimDesc = {stride, size};
    }
    TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaSetMultiDimAddr, &dmaTileBd,
                            &dmaTileBdTensor, basePlusOffsetInBytes,
                            lenInBytes);
  }

  if (nextBdId) {
    auto enableNextBd = 1;
    TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaSetNextBd, &dmaTileBd,
                            nextBdId.value(), enableNextBd);
  }

  if (packetID) {
    if (!packetType)
      return bdOp.emitError("must have packetType with packetID");
    if (bdOp.getLen() == 0)
      return bdOp.emitOpError(
          "For MM2S channels, if Buffer_Length=0 then Enable_Packet must be "
          "set to 0, otherwise behavior is undefined (3.7.8 arch spec)");
    TRY_XAIE_API_EMIT_ERROR(
        bdOp, XAie_DmaSetPkt, &dmaTileBd,
        XAie_PacketInit(packetID.value(), packetType.value()));
  }
  TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaEnableBd, &dmaTileBd);
  TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaWriteBd, &deviceModel.devInst,
                          &dmaTileBd, tileLoc, bdId);
  LLVM_DEBUG(llvm::dbgs() << "\nend configuring bds\n");
  return success();
};

LogicalResult pushToBdQueueAndEnable(AMDAIENPUDeviceModel &deviceModel,
                                     Operation &op, XAie_LocType &tileLoc,
                                     int chNum, const DMAChannelDir &channelDir,
                                     int bdId, int repeatCount) {
  XAie_DmaDirection direction =
      channelDir == DMAChannelDir::S2MM ? DMA_S2MM : DMA_MM2S;
  auto enTokenIssue = tileLoc.Row == 0 && direction == DMA_S2MM;
  // in english repeat_count==0 means "do it once" and don't repeat but
  // libxaie treats repeat_count=1 as do it once.
  repeatCount += 1;
  TRY_XAIE_API_EMIT_ERROR(op, XAie_DmaChannelSetStartQueue,
                          &deviceModel.devInst, tileLoc, chNum, direction, bdId,
                          repeatCount, enTokenIssue);
  TRY_XAIE_API_EMIT_ERROR(op, XAie_DmaChannelEnable, &deviceModel.devInst,
                          tileLoc, chNum, direction);
  return success();
};

LogicalResult configureLocksAndBd(Block &block, XAie_LocType tileLoc,
                                  AMDAIENPUDeviceModel &deviceModel) {
  DMABDOp bd = *block.getOps<DMABDOp>().begin();
  assert(bd.getBdId().has_value() &&
         "DMABDOp must have assigned bd_id; did you forget to run "
         "aie-assign-bd-ids?");
  XAie_DmaDesc dmaTileBd;
  TRY_XAIE_API_EMIT_ERROR(bd, XAie_DmaDescInit, &deviceModel.devInst,
                          &dmaTileBd, tileLoc);
  if (!block.getOps<UseLockOp>().empty() &&
      failed(configureLocksInBdBlock(dmaTileBd, block, deviceModel, tileLoc)))
    return failure();
  if (!block.getOps<DMABDOp>().empty() &&
      failed(configureBdInBlock(dmaTileBd, block, deviceModel, tileLoc,
                                bd.getBdId().value(), bd.getNextBdId())))
    return failure();
  return success();
};

struct AIEControl {
  AMDAIENPUDeviceModel deviceModel;
  AIEControl(AMDAIENPUDeviceModel dm) : deviceModel(dm) {}

  LogicalResult addAieElfToCDO(uint8_t col, uint8_t row,
                               const StringRef elfPath, bool aieSim) {
    // loadSym: Load symbols from .map file. This argument is not used when
    // __AIESIM__ is not defined.
    TRY_XAIE_API_LOGICAL_RESULT(XAie_LoadElf, &deviceModel.devInst,
                                XAie_TileLoc(col, row), elfPath.str().c_str(),
                                /*loadSym*/ aieSim);
    return success();
  }

  LogicalResult addAieElfsToCDO(DeviceOp &device, const StringRef workDirPath,
                                bool aieSim) {
    for (auto tileOp : device.getOps<TileOp>())
      if (tileOp.isShimNOCorPLTile()) {
        // Resets no needed with V2 kernel driver
      } else {
        int col = tileOp.colIndex();
        int row = tileOp.rowIndex();
        if (auto coreOp = tileOp.getCoreOp()) {
          std::string fileName;
          if (auto fileAttr = coreOp.getElfFile())
            fileName = fileAttr->str();
          else
            fileName = (llvm::Twine("core_") + std::to_string(col) + "_" +
                        std::to_string(row) + ".elf")
                           .str();
          if (failed(addAieElfToCDO(
                  col, row,
                  (llvm::Twine(workDirPath) + std::string(1, ps) + fileName)
                      .str(),
                  aieSim)))
            return failure();
        }
      }
    return success();
  }

  LogicalResult addInitConfigToCDO(DeviceOp &device) {
    for (auto tileOp : device.getOps<TileOp>()) {
      auto tileLoc = XAie_TileLoc(tileOp.colIndex(), tileOp.rowIndex());
      if (!tileOp.isShimTile() && tileOp.getCoreOp()) {
        TRY_XAIE_API_EMIT_ERROR(tileOp, XAie_CoreReset, &deviceModel.devInst,
                                tileLoc);
        TRY_XAIE_API_EMIT_ERROR(tileOp, XAie_CoreUnreset, &deviceModel.devInst,
                                tileLoc);
        // Set locks to zero
        for (uint8_t l = 0; l < NUM_LOCKS; l++) {
          auto locInit = XAie_LockInit(l, 0);
          TRY_XAIE_API_EMIT_ERROR(tileOp, XAie_LockSetValue,
                                  &deviceModel.devInst, tileLoc, locInit);
        }
      }
    }

    // Set locks with explicit initializers
    device.walk<WalkOrder::PreOrder>([&](LockOp lockOp) {
      if (lockOp.getLockID() && lockOp.getInit()) {
        auto tileLoc = XAie_TileLoc(lockOp.getTileOp().colIndex(),
                                    lockOp.getTileOp().rowIndex());
        auto locInit = XAie_LockInit(*lockOp.getLockID(), *lockOp.getInit());
        TRY_XAIE_API_FATAL_ERROR(XAie_LockSetValue, &deviceModel.devInst,
                                 tileLoc, locInit);
      } else
        LLVM_DEBUG(llvm::dbgs()
                   << "lock op missing either id or init" << lockOp << "\n");
    });

    auto memOps = llvm::to_vector_of<TileElement>(device.getOps<MemOp>());
    llvm::append_range(memOps, device.getOps<MemTileDMAOp>());
    llvm::append_range(memOps, device.getOps<ShimDMAOp>());
    for (TileElement memOp : memOps) {
      int col = memOp.getTileID().col;
      int row = memOp.getTileID().row;
      XAie_LocType tileLoc = XAie_TileLoc(col, row);

      // handle DMA ops separately
      for (Block &block : memOp.getOperation()->getRegion(0)) {
        if (block.getOps<DMABDOp>().empty()) continue;
        if (failed(configureLocksAndBd(block, tileLoc, deviceModel)))
          return failure();
      }

      for (Block &block : memOp.getOperation()->getRegion(0)) {
        for (auto op : block.getOps<DMAStartOp>()) {
          DMABDOp bd = *op.getDest()->getOps<DMABDOp>().begin();
          int chNum = op.getChannelIndex();
          auto channelDir = op.getChannelDir();
          if (failed(pushToBdQueueAndEnable(
                  deviceModel, *bd.getOperation(), tileLoc, chNum, channelDir,
                  bd.getBdId().value(), op.getRepeatCount())))
            return failure();
        }
      }
    }

    // StreamSwitch (switchbox) configuration
    for (auto switchboxOp : device.getOps<SwitchboxOp>()) {
      int32_t col = switchboxOp.colIndex();
      int32_t row = switchboxOp.rowIndex();
      XAie_LocType tileLoc = XAie_TileLoc(col, row);
      //      assert(device.getDevice() == AIEDevice::npu &&
      //             "Only NPU currently supported");
      if (row == 0) {
        // FIXME hack for TCT routing
        // TODO Support both channels
        auto slvPortNum = 0;
        auto mstrPortNum = 0;
        TRY_XAIE_API_EMIT_ERROR(switchboxOp, XAie_StrmConnCctEnable,
                                &deviceModel.devInst, tileLoc, CTRL, slvPortNum,
                                SOUTH, mstrPortNum);
      }

      Block &b = switchboxOp.getConnections().front();
      for (auto connectOp : b.getOps<ConnectOp>())
        TRY_XAIE_API_EMIT_ERROR(
            switchboxOp, XAie_StrmConnCctEnable, &deviceModel.devInst, tileLoc,
            toStrmT(connectOp.getSourceBundle()), connectOp.sourceIndex(),
            toStrmT(connectOp.getDestBundle()), connectOp.destIndex());

      for (auto connectOp : b.getOps<MasterSetOp>()) {
        int mask = 0;
        int arbiter = -1;

        for (auto val : connectOp.getAmsels()) {
          AMSelOp amsel = cast<AMSelOp>(val.getDefiningOp());
          arbiter = amsel.arbiterIndex();
          int msel = amsel.getMselValue();
          mask |= (1 << msel);
        }

        bool isdma = connectOp.getDestBundle() == WireBundle::DMA;
        // assume a connection going south from row zero gets wired to shimdma
        // by a shimmux. TODO: fix the assumption
        if (!isdma && (switchboxOp.rowIndex() == 0))
          isdma = connectOp.getDestBundle() == WireBundle::South;
        // Flag for overriding DROP_HEADER. TODO: Formalize this in tablegen
        isdma &= !connectOp->hasAttr("keep_pkt_header");
        auto dropHeader =
            isdma ? XAIE_SS_PKT_DROP_HEADER : XAIE_SS_PKT_DONOT_DROP_HEADER;
        TRY_XAIE_API_EMIT_ERROR(
            connectOp, XAie_StrmPktSwMstrPortEnable, &deviceModel.devInst,
            tileLoc, toStrmT(connectOp.getDestBundle()), connectOp.destIndex(),
            dropHeader, arbiter, mask);
      }

      for (auto connectOp : b.getOps<PacketRulesOp>()) {
        int slot = 0;
        Block &block = connectOp.getRules().front();
        for (auto slotOp : block.getOps<PacketRuleOp>()) {
          AMSelOp amselOp = cast<AMSelOp>(slotOp.getAmsel().getDefiningOp());
          int arbiter = amselOp.arbiterIndex();
          int msel = amselOp.getMselValue();
          TRY_XAIE_API_EMIT_ERROR(connectOp, XAie_StrmPktSwSlavePortEnable,
                                  &deviceModel.devInst, tileLoc,
                                  toStrmT(connectOp.getSourceBundle()),
                                  connectOp.sourceIndex());
          auto packetInit = XAie_PacketInit(slotOp.valueInt(), /*PktType*/ 0);
          // TODO Need to better define packet id,type used here
          TRY_XAIE_API_EMIT_ERROR(connectOp, XAie_StrmPktSwSlaveSlotEnable,
                                  &deviceModel.devInst, tileLoc,
                                  toStrmT(connectOp.getSourceBundle()),
                                  connectOp.sourceIndex(), slot, packetInit,
                                  slotOp.maskInt(), msel, arbiter);
          slot++;
        }
      }
    }

    for (auto muxOp : device.getOps<ShimMuxOp>()) {
      // NOTE ShimMux always connects from the south as directions are
      // defined relative to the tile stream switch.
      auto tileLoc =
          XAie_TileLoc(muxOp.getTileOp().getCol(), muxOp.getTileOp().getRow());
      Block &b = muxOp.getConnections().front();
      for (auto connectOp : b.getOps<ConnectOp>()) {
        // demux!
        if (connectOp.getSourceBundle() == WireBundle::North)
          TRY_XAIE_API_EMIT_ERROR(muxOp, XAie_EnableAieToShimDmaStrmPort,
                                  &deviceModel.devInst, tileLoc,
                                  connectOp.sourceIndex());
        // mux
        if (connectOp.getDestBundle() == WireBundle::North)
          TRY_XAIE_API_EMIT_ERROR(muxOp, XAie_EnableShimDmaToAieStrmPort,
                                  &deviceModel.devInst, tileLoc,
                                  connectOp.destIndex());
      }
    }

    for (auto switchboxOp : device.getOps<ShimSwitchboxOp>()) {
      Block &b = switchboxOp.getConnections().front();
      auto tileLoc = XAie_TileLoc(switchboxOp.getCol(), 0);
      for (auto connectOp : b.getOps<ConnectOp>())
        TRY_XAIE_API_EMIT_ERROR(
            switchboxOp, XAie_StrmConnCctEnable, &deviceModel.devInst, tileLoc,
            toStrmT(connectOp.getSourceBundle()), connectOp.sourceIndex(),
            toStrmT(connectOp.getDestBundle()), connectOp.destIndex());
    }

    // Cascade configuration
    for (auto configOp : device.getOps<ConfigureCascadeOp>()) {
      TileOp tile = cast<TileOp>(configOp.getTile().getDefiningOp());
      auto tileLoc = XAie_TileLoc(tile.getCol(), tile.getRow());
      TRY_XAIE_API_EMIT_ERROR(
          device, XAie_CoreConfigAccumulatorControl, &deviceModel.devInst,
          tileLoc, toStrmT(static_cast<WireBundle>(configOp.getInputDir())),
          toStrmT(static_cast<WireBundle>(configOp.getOutputDir())));
    }

    return success();
  }

  LogicalResult addCoreEnableToCDO(DeviceOp &device) {
    // Start execution of all the cores.
    for (auto tileOp : device.getOps<TileOp>()) {
      auto tileLoc = XAie_TileLoc(tileOp.colIndex(), tileOp.rowIndex());
      if (!tileOp.isShimTile() && tileOp.getCoreOp())
        TRY_XAIE_API_EMIT_ERROR(device, XAie_CoreEnable, &deviceModel.devInst,
                                tileLoc);
    }
    return success();
  }

  void dmaUpdateBdAddr(DeviceOp &device, int col, int row, size_t addr,
                       size_t bdId) {
    auto tileLoc = XAie_TileLoc(col, row);
    TRY_XAIE_API_FATAL_ERROR(XAie_DmaUpdateBdAddr, &deviceModel.devInst,
                             tileLoc, addr, bdId);
  }
};

}  // namespace mlir::iree_compiler::AMDAIE

void initializeCDOGenerator(byte_ordering endianness, bool cdoDebug) {
  // Enables AXI-MM prints for configs being added in CDO
  if (cdoDebug) EnAXIdebug();
  setEndianness(endianness);
};

LogicalResult generateCDOBinary(const StringRef outputPath,
                                const std::function<LogicalResult()> &cb) {
  // Never generate a completely empty CDO file.  If the file only contains a
  // header, then bootgen flags it as invalid.
  startCDOFileStream(outputPath.str().c_str());
  FileHeader();
  insertNoOpCommand(4);
  if (failed(cb())) return failure();
  configureHeader();
  endCurrentCDOFileStream();
  return success();
}

LogicalResult generateCDOBinariesSeparately(
    mlir::iree_compiler::AMDAIE::AIEControl &ctl, const StringRef workDirPath,
    DeviceOp &device, bool aieSim, bool enableCores) {
  if (failed(generateCDOBinary(
          (llvm::Twine(workDirPath) + std::string(1, ps) + "aie_cdo_elfs.bin")
              .str(),
          [&ctl, &device, &workDirPath, &aieSim] {
            return ctl.addAieElfsToCDO(device, workDirPath, aieSim);
          })))
    return failure();

  if (failed(generateCDOBinary(
          (llvm::Twine(workDirPath) + std::string(1, ps) + "aie_cdo_init.bin")
              .str(),
          [&ctl, &device] { return ctl.addInitConfigToCDO(device); })))
    return failure();

  if (enableCores && !device.getOps<CoreOp>().empty() &&
      failed(generateCDOBinary(
          (llvm::Twine(workDirPath) + std::string(1, ps) + "aie_cdo_enable.bin")
              .str(),
          [&ctl, &device] { return ctl.addCoreEnableToCDO(device); })))
    return failure();

  return success();
}

LogicalResult generateCDOUnified(mlir::iree_compiler::AMDAIE::AIEControl &ctl,
                                 const StringRef workDirPath, DeviceOp &device,
                                 bool aieSim, bool enableCores) {
  return generateCDOBinary(
      (llvm::Twine(workDirPath) + std::string(1, ps) + "aie_cdo.bin").str(),
      [&ctl, &device, &workDirPath, &aieSim, &enableCores] {
        if (!device.getOps<CoreOp>().empty() &&
            failed(ctl.addAieElfsToCDO(device, workDirPath, aieSim)))
          return failure();
        if (failed(ctl.addInitConfigToCDO(device))) return failure();
        if (enableCores && !device.getOps<CoreOp>().empty() &&
            failed(ctl.addCoreEnableToCDO(device)))
          return failure();
        return success();
      });
}

namespace mlir::iree_compiler::AMDAIE {
LogicalResult AIETranslateToCDODirect(ModuleOp m, llvm::StringRef workDirPath,
                                      bool bigEndian, bool emitUnified,
                                      bool cdoDebug, bool aieSim,
                                      bool enableCores) {
  auto devOps = m.getOps<DeviceOp>();
  assert(llvm::range_size(devOps) == 1 &&
         "only exactly 1 device op supported.");
  DeviceOp device = *devOps.begin();
  mlir::iree_compiler::AMDAIE::AIEControl ctl(
      mlir::iree_compiler::AMDAIE::getDeviceModel(
          static_cast<AMDAIEDevice>(device.getDevice())));
  byte_ordering endianness =
      bigEndian ? byte_ordering::Big_Endian : byte_ordering::Little_Endian;
  initializeCDOGenerator(endianness, cdoDebug);
  if (emitUnified)
    return generateCDOUnified(ctl, workDirPath, device, aieSim, enableCores);
  return generateCDOBinariesSeparately(ctl, workDirPath, device, aieSim,
                                       enableCores);
}
}  // namespace mlir::iree_compiler::AMDAIE