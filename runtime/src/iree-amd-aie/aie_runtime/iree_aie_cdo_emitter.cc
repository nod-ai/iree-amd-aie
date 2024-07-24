// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions. See
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: # Apache-2.0 WITH LLVM-exception

#include "iree_aie_cdo_emitter.h"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>

#include "iree_aie_router.h"
#include "iree_aie_runtime.h"

#define DEBUG_TYPE "iree-aie-cdo-emitter"

using Path = std::filesystem::path;
auto ps = Path::preferred_separator;

namespace mlir::iree_compiler::AMDAIE {

LogicalResult configureLocksInBdBlock(
    const AMDAIEDeviceModel &deviceModel, XAie_DmaDesc &dmaTileBd,
    const TileLoc &tileLoc, std::optional<uint8_t> acqValue,
    std::optional<uint8_t> relValue, std::optional<uint8_t> acqLockId,
    std::optional<uint8_t> relLockId, bool acqEn) {
  assert(acqValue && relValue && acqLockId && relLockId &&
         "expected both use_lock(acquire) and use_lock(release) with bd");
  if (deviceModel.isMemTile(tileLoc.col, tileLoc.row)) {
    if (acqLockId) acqLockId.value() += XAIE2IPU_MEM_TILE_LOCK_ID_INCR;
    if (relLockId) relLockId.value() += XAIE2IPU_MEM_TILE_LOCK_ID_INCR;
  }

  // no RelEn in the arch spec even though the API requires you to set it?
  bool relEn = false;
  XAie_Lock acqLock = XAie_LockInit(acqLockId.value(), acqValue.value());
  XAie_Lock relLock = XAie_LockInit(relLockId.value(), relValue.value());
  TRY_XAIE_API_LOGICAL_RESULT(dmaTileBd.DmaMod->SetLock, &dmaTileBd, acqLock,
                              relLock, acqEn, relEn);
  return success();
}

LogicalResult configureBdInBlock(
    const AMDAIEDeviceModel &deviceModel, XAie_DmaDesc &dmaTileBd,
    const TileLoc &tileLoc, uint8_t bdId, std::optional<uint8_t> nextBdId,
    std::optional<uint8_t> packetType, std::optional<uint8_t> packetID,
    uint64_t baseAddr, uint64_t lenInBytes, uint64_t offsetInBytes,
    uint32_t bufferElementTypeWidthInBytes,
    const std::optional<std::vector<BDDimLayout>> &maybeDims,
    const std::optional<std::vector<BDPadLayout>> &maybePadDims) {
  if (deviceModel.isShimNOCTile(tileLoc.col, tileLoc.row)) {
    // write them out like this so they show up with names in debug prints
    size_t smid = 0;
    size_t burstLen = 16;  // (10):BLEN=16 (256Byte) (corresponds to
                           // 0x800000000 from target)
    size_t qOs = 0;
    size_t cache = 0;
    size_t secure = 0;
    TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaSetAxi, &dmaTileBd, smid, burstLen, qOs,
                                cache, secure);
  }

  // std::string& FifoMode = disable; // FIXME: when to enable FIFO mode?
  if (!deviceModel.isShimNOCTile(tileLoc.col, tileLoc.row)) {
    if (deviceModel.isMemTile(tileLoc.col, tileLoc.row))
      baseAddr += XAIE2IPU_ADDR_ARRAY_OFF;
  }
  uint64_t basePlusOffsetInBytes = baseAddr + offsetInBytes;

  // aie-rt expects multiples of 32b words (see docstring on
  // XAie_DmaSetMultiDimAddr). Thus, elementWidthIn32bWords is possibly a
  // fraction, e.g. bf16 => elementWidthIn32bWords == 0.5 so that size = 10 => 5
  // 32b words
  double elementWidthIn32bWords =
      static_cast<double>(bufferElementTypeWidthInBytes) / 4.0;

  if (const auto &dims = maybeDims) {
    TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaSetAddrLen, &dmaTileBd,
                                basePlusOffsetInBytes, lenInBytes);
  } else {
    XAie_DmaTensor dmaTileBdTensor = {};
    dmaTileBdTensor.NumDim = dims->size();
    dmaTileBdTensor.Dim = new XAie_DmaDimDesc[dmaTileBdTensor.NumDim];
    for (size_t i = 0; i < dims->size(); i++) {
      // Pass down dimensions in reverse order; in the MLIR, this allows
      // us to specify step sizes/strides in the same order as we would for
      // RankedTensorType/MemRefType.
      uint16_t size = dims->at(i).size;
      uint32_t stride = dims->at(i).stride;
      size_t j = dims->size() - i - 1;
      if (j > 0) {
        if (stride * bufferElementTypeWidthInBytes % 4 != 0) {
          llvm::errs() << "`stride` on dim " << i
                       << ", times element width (in bytes), should "
                          "be a multiple of 4 bytes";
          return failure();
        }
        stride = static_cast<uint32_t>(stride * elementWidthIn32bWords);
      } else {
        if (size * bufferElementTypeWidthInBytes % 4 != 0) {
          llvm::errs() << "`size` on dim " << i
                       << ", times element width (in bytes), should "
                          "be a multiple of 4 bytes";
          return failure();
        }
        size = static_cast<uint16_t>(size * elementWidthIn32bWords);
      }
      stride = stride > 0 ? stride : 1;
      // Assume AIE-ML architecture (ie use AieMlDimDesc instead of AieDimDesc);
      // asserted in AIETranslateToCDODirect).
      dmaTileBdTensor.Dim[j].AieMlDimDesc = {stride, size};
    }
    TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaSetMultiDimAddr, &dmaTileBd,
                                &dmaTileBdTensor, basePlusOffsetInBytes,
                                lenInBytes);
  }

  // ND zero padding.
  if (const auto &padDims = maybePadDims) {
    XAie_DmaPadTensor dmaPadTensor = {};
    dmaPadTensor.NumDim = padDims->size();
    dmaPadTensor.PadDesc = new XAie_PadDesc[dmaPadTensor.NumDim];
    for (size_t i = 0; i < padDims->size(); i++) {
      uint8_t before = padDims->at(i).const_pad_before;
      uint8_t after = padDims->at(i).const_pad_after;
      size_t j = padDims->size() - i - 1;
      if (j == 0) {
        if (before * bufferElementTypeWidthInBytes % 4 != 0) {
          llvm::errs()
              << "`before` padding on inner-most dim, times element width (in "
                 "bytes), should be a multiple of 4 bytes";
          return failure();
        }
        if (after * bufferElementTypeWidthInBytes % 4 != 0) {
          llvm::errs()
              << "`after` padding on inner-most dim, times element width (in "
                 "bytes), should be a multiple of 4 bytes";
          return failure();
        }
        before = static_cast<uint8_t>(before * elementWidthIn32bWords);
        after = static_cast<uint8_t>(after * elementWidthIn32bWords);
      }
      dmaPadTensor.PadDesc[j] = {before, after};
    }
    TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaSetPadding, &dmaTileBd, &dmaPadTensor);
  }

  if (nextBdId) {
    auto enableNextBd = 1;
    TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaSetNextBd, &dmaTileBd, nextBdId.value(),
                                enableNextBd);
  }

  if (packetID) {
    if (!packetType) {
      llvm::errs() << "must have packetType with packetID";
      return failure();
    }
    if (lenInBytes == 0) {
      llvm::errs()
          << "For MM2S channels, if Buffer_Length=0 then Enable_Packet must be "
             "set to 0, otherwise behavior is undefined (3.7.8 arch spec)";
      return failure();
    }

    TRY_XAIE_API_LOGICAL_RESULT(
        XAie_DmaSetPkt, &dmaTileBd,
        XAie_PacketInit(packetID.value(), packetType.value()));
  }
  TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaEnableBd, &dmaTileBd);
  TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaWriteBd,
                              const_cast<XAie_DevInst *>(&deviceModel.devInst),
                              &dmaTileBd, tileLoc, bdId);
  return success();
};

LogicalResult pushToBdQueueAndEnable(const AMDAIEDeviceModel &deviceModel,
                                     const TileLoc &tileLoc, uint8_t chNum,
                                     const DMAChannelDir &channelDir,
                                     uint8_t bdId, uint32_t repeatCount) {
  XAie_DmaDirection direction =
      channelDir == DMAChannelDir::S2MM ? DMA_S2MM : DMA_MM2S;
  auto enTokenIssue = tileLoc.row == 0 && direction == DMA_S2MM;
  // in english repeat_count==0 means "do it once" and don't repeat but
  // libxaie treats repeat_count=1 as do it once.
  repeatCount += 1;
  TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaChannelSetStartQueue,
                              const_cast<XAie_DevInst *>(&deviceModel.devInst),
                              tileLoc, chNum, direction, bdId, repeatCount,
                              enTokenIssue);
  TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaChannelEnable,
                              const_cast<XAie_DevInst *>(&deviceModel.devInst),
                              tileLoc, chNum, direction);
  return success();
};

LogicalResult configureLocksAndBd(
    const AMDAIEDeviceModel &deviceModel, const TileLoc &tileLoc,
    std::optional<uint8_t> bdId, std::optional<uint8_t> nextBdId,
    std::optional<uint8_t> acqValue, std::optional<uint8_t> relValue,
    std::optional<uint8_t> acqLockId, std::optional<uint8_t> relLockId,
    bool acqEn, std::optional<uint8_t> packetType,
    std::optional<uint8_t> packetID, uint32_t baseAddr, uint32_t lenInBytes,
    uint32_t offsetInBytes, uint32_t bufferElementTypeWidthInBytes,
    const std::optional<std::vector<BDDimLayout>> &maybeDims,
    const std::optional<std::vector<BDPadLayout>> &maybePadDims) {
  assert(bdId.has_value() &&
         "DMABDOp must have assigned bd_id; did you forget to run "
         "aie-assign-bd-ids?");
  XAie_DmaDesc dmaTileBd;
  TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaDescInit,
                              const_cast<XAie_DevInst *>(&deviceModel.devInst),
                              &dmaTileBd, tileLoc);
  if (failed(configureLocksInBdBlock(deviceModel, dmaTileBd, tileLoc, acqValue,
                                     relValue, acqLockId, relLockId, acqEn)))
    return failure();
  if (failed(configureBdInBlock(
          deviceModel, dmaTileBd, tileLoc, *bdId, nextBdId, packetType,
          packetID, baseAddr, lenInBytes, offsetInBytes,
          bufferElementTypeWidthInBytes, maybeDims, maybePadDims)))
    return failure();
  return success();
};

LogicalResult addAieElfToCDO(const AMDAIEDeviceModel &deviceModel, uint8_t col,
                             uint8_t row, const std::string &elfPath,
                             bool aieSim) {
  // loadSym: Load symbols from .map file. This argument is not used when
  // __AIESIM__ is not defined.
  TRY_XAIE_API_LOGICAL_RESULT(XAie_LoadElf,
                              const_cast<XAie_DevInst *>(&deviceModel.devInst),
                              XAie_TileLoc(col, row), elfPath.c_str(),
                              /*loadSym*/ aieSim);
  return success();
}

LogicalResult addAieElfsToCDO(
    const AMDAIEDeviceModel &deviceModel,
    const std::vector<std::pair<TileLoc, std::optional<std::string>>> &tiles,
    Path &workDirPath, bool aieSim) {
  for (const auto &[tile, elfFile] : tiles) {
    std::string fileName;
    if (elfFile)
      fileName = *elfFile;
    else
      fileName = "core_" + std::to_string(tile.col) + "_" +
                 std::to_string(tile.row) + ".elf";
    Path path = workDirPath / fileName;
    if (failed(addAieElfToCDO(deviceModel, tile.col, tile.row, path, aieSim)))
      return failure();
  }
  return success();
}

LogicalResult initializeLocks(const AMDAIEDeviceModel &deviceModel,
                              const std::vector<TileLoc> &tileLocs,
                              const std::vector<Lock> &locks) {
  for (const auto &tileLoc : tileLocs) {
    if (!deviceModel.isShimNOCorPLTile(tileLoc.col, tileLoc.row)) {
      TRY_XAIE_API_LOGICAL_RESULT(
          XAie_CoreReset, const_cast<XAie_DevInst *>(&deviceModel.devInst),
          tileLoc);
      TRY_XAIE_API_LOGICAL_RESULT(
          XAie_CoreUnreset, const_cast<XAie_DevInst *>(&deviceModel.devInst),
          tileLoc);
      // Set locks to zero
      for (uint8_t l = 0; l < deviceModel.getNumLocks(tileLoc.col, tileLoc.row);
           l++) {
        auto locInit = XAie_LockInit(l, 0);
        TRY_XAIE_API_LOGICAL_RESULT(
            XAie_LockSetValue, const_cast<XAie_DevInst *>(&deviceModel.devInst),
            tileLoc, locInit);
      }
    }
  }

  // Set locks with explicit initializers
  for (const auto &lock : locks) {
    auto locInit = XAie_LockInit(lock.id, *lock.init);
    TRY_XAIE_API_FATAL_ERROR(XAie_LockSetValue,
                             const_cast<XAie_DevInst *>(&deviceModel.devInst),
                             lock.tileLoc, locInit);
  }
  return success();
}

LogicalResult configureDMAs(const AMDAIEDeviceModel &deviceModel,
                            const std::vector<DMAConfig> &dmaConfigs,
                            const std::vector<DMAStart> &dmaStarts) {
  for (const auto &[tileLoc, bdId, nextBdId, acqValue, relValue, acqLockId,
                    relLockId, acqEn, packetType, packetID, baseAddr,
                    lenInBytes, offsetInBytes, bufferElementTypeWidthInBytes,
                    maybeDims, maybePadDims] : dmaConfigs) {
    if (failed(configureLocksAndBd(
            deviceModel, tileLoc, bdId, nextBdId, acqValue, relValue, acqLockId,
            relLockId, acqEn, packetType, packetID, baseAddr, lenInBytes,
            offsetInBytes, bufferElementTypeWidthInBytes, maybeDims,
            maybePadDims)))
      return failure();
  }

  for (const auto &[tileLoc, chNum, channelDir, bdId, repeatCount] :
       dmaStarts) {
    if (failed(pushToBdQueueAndEnable(deviceModel, tileLoc, chNum, channelDir,
                                      bdId, repeatCount)))
      return failure();
  }
  return success();
}

LogicalResult configureSwitches(
    const AMDAIEDeviceModel &deviceModel,
    const std::map<SwitchBox, std::vector<Connect>> &switchConnects) {
  auto *devInst = const_cast<XAie_DevInst *>(&deviceModel.devInst);
  // StreamSwitch (switchbox) configuration
  for (const auto &[switchBox, connects] : switchConnects) {
    TileLoc tileLoc(switchBox.col, switchBox.row);
    // FIXME hack for TCT routing
    // TODO Support both channels
    if (switchBox.row == 0) {
      auto slvPortNum = 0;
      auto mstrPortNum = 0;
      TRY_XAIE_API_LOGICAL_RESULT(XAie_StrmConnCctEnable, devInst, tileLoc,
                                  CTRL, slvPortNum, SOUTH, mstrPortNum);
    }

    for (Connect connect : connects) {
      TileLoc connTileLoc(connect.col, connect.row);
      assert(tileLoc == connTileLoc &&
             "expected conn.tileLoc to be the same as src.tileLoc");
      if (connect.interconnect == Connect::Interconnect::SWB) {
        TRY_XAIE_API_LOGICAL_RESULT(
            XAie_StrmConnCctEnable, devInst, connTileLoc,
            strmTtoStrmT(connect.src.bundle), connect.src.channel,
            strmTtoStrmT(connect.dst.bundle), connect.dst.channel);
      } else if (connect.interconnect == Connect::Interconnect::SHIMMUX) {
        // NOTE ShimMux always connects from the south as directions are
        // defined relative to the tile stream switch.
        // demux!
        if (connect.src.bundle == StrmSwPortType::NORTH) {
          TRY_XAIE_API_LOGICAL_RESULT(XAie_EnableAieToShimDmaStrmPort, devInst,
                                      tileLoc, connect.src.channel);
        }
        // mux
        if (connect.dst.bundle == StrmSwPortType::NORTH) {
          TRY_XAIE_API_LOGICAL_RESULT(XAie_EnableShimDmaToAieStrmPort, devInst,
                                      tileLoc, connect.dst.channel);
        }
      }
    }

    //    for (auto connectOp : b.getOps<MasterSetOp>()) {
    //      int mask = 0;
    //      int arbiter = -1;
    //
    //      for (auto val : connectOp.getAmsels()) {
    //        AMSelOp amsel = cast<AMSelOp>(val.getDefiningOp());
    //        arbiter = amsel.arbiterIndex();
    //        int msel = amsel.getMselValue();
    //        mask |= (1 << msel);
    //      }
    //
    //      bool isdma = connectOp.getDestBundle() == WireBundle::DMA;
    //      // assume a connection going south from row zero gets wired to
    //      shimdma
    //      // by a shimmux. TODO: fix the assumption
    //      if (!isdma && (switchboxOp.rowIndex() == 0))
    //        isdma = connectOp.getDestBundle() == WireBundle::South;
    //      // Flag for overriding DROP_HEADER. TODO: Formalize this in tablegen
    //      isdma &= !connectOp->hasAttr("keep_pkt_header");
    //      auto dropHeader =
    //          isdma ? XAIE_SS_PKT_DROP_HEADER : XAIE_SS_PKT_DONOT_DROP_HEADER;
    //      TRY_XAIE_API_LOGICAL_RESULT(
    //          connectOp, XAie_StrmPktSwMstrPortEnable, &deviceModel.devInst,
    //          tileLoc, toStrmT(connectOp.getDestBundle()),
    //          connectOp.destIndex(), dropHeader, arbiter, mask);
    //    }
    //
    //    for (auto connectOp : b.getOps<PacketRulesOp>()) {
    //      int slot = 0;
    //      Block &block = connectOp.getRules().front();
    //      for (auto slotOp : block.getOps<PacketRuleOp>()) {
    //        AMSelOp amselOp =
    //        cast<AMSelOp>(slotOp.getAmsel().getDefiningOp()); int arbiter =
    //        amselOp.arbiterIndex(); int msel = amselOp.getMselValue();
    //        TRY_XAIE_API_LOGICAL_RESULT(connectOp,
    //        XAie_StrmPktSwSlavePortEnable,
    //                                    &deviceModel.devInst, tileLoc,
    //                                    toStrmT(connectOp.getSourceBundle()),
    //                                    connectOp.sourceIndex());
    //        auto packetInit = XAie_PacketInit(slotOp.valueInt(), /*PktType*/
    //        0);
    //        // TODO Need to better define packet id,type used here
    //        TRY_XAIE_API_LOGICAL_RESULT(connectOp,
    //        XAie_StrmPktSwSlaveSlotEnable,
    //                                    &deviceModel.devInst, tileLoc,
    //                                    toStrmT(connectOp.getSourceBundle()),
    //                                    connectOp.sourceIndex(), slot,
    //                                    packetInit, slotOp.maskInt(), msel,
    //                                    arbiter);
    //        slot++;
    //      }
    //    }
  }
}

LogicalResult configureCascades() {
  // Cascade configuration
  for (auto configOp : device.getOps<ConfigureCascadeOp>()) {
    TileOp tile = cast<TileOp>(configOp.getTile().getDefiningOp());
    auto tileLoc = XAie_TileLoc(tile.getCol(), tile.getRow());
    TRY_XAIE_API_LOGICAL_RESULT(
        device, XAie_CoreConfigAccumulatorControl, &deviceModel.devInst,
        tileLoc, toStrmT(static_cast<WireBundle>(configOp.getInputDir())),
        toStrmT(static_cast<WireBundle>(configOp.getOutputDir())));
  }
}

LogicalResult addInitConfigToCDO(const AMDAIEDeviceModel &deviceModel,
                                 const std::vector<TileLoc> &tiles,
                                 const std::vector<Lock> &locks) {
  if (failed(initializeLocks(deviceModel, tiles, locks))) return failure();

  return success();
}

LogicalResult addCoreEnableToCDO(DeviceOp &device) {
  // Start execution of all the cores.
  for (auto tileOp : device.getOps<TileOp>()) {
    auto tileLoc = XAie_TileLoc(tileOp.colIndex(), tileOp.rowIndex());
    if (!tileOp.isShimTile() && tileOp.getCoreOp())
      TRY_XAIE_API_LOGICAL_RESULT(device, XAie_CoreEnable, &deviceModel.devInst,
                                  tileLoc);
  }
  return success();
}

void dmaUpdateBdAddr(DeviceOp &device, int col, int row, size_t addr,
                     size_t bdId) {
  auto tileLoc = XAie_TileLoc(col, row);
  TRY_XAIE_API_FATAL_ERROR(XAie_DmaUpdateBdAddr, &deviceModel.devInst, tileLoc,
                           addr, bdId);
}

void initializeCDOGenerator(byte_ordering endianness, bool cdoDebug) {
  // Enables AXI-MM prints for configs being added in CDO
  if (cdoDebug) EnAXIdebug();
  setEndianness(endianness);
};

LogicalResult generateCDOBinary(const std::string &outputPath,
                                const std::function<LogicalResult()> &cb) {
  startCDOFileStream(outputPath.str().c_str());
  FileHeader();
  // Never generate a completely empty CDO file.  If the file only contains a
  // header, then bootgen flags it as invalid.
  insertNoOpCommand(4);
  if (failed(cb())) return failure();
  configureHeader();
  endCurrentCDOFileStream();
  return success();
}

LogicalResult generateCDOBinariesSeparately(
    mlir::iree_compiler::AMDAIE::AIEControl &ctl,
    const std::string &workDirPath, DeviceOp &device, bool aieSim,
    bool enableCores) {
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
                                 const std::string &workDirPath,
                                 DeviceOp &device, bool aieSim,
                                 bool enableCores) {
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
}  // namespace mlir::iree_compiler::AMDAIE
