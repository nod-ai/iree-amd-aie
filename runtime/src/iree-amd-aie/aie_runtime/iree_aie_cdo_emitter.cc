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
#include "llvm/ADT/StringExtras.h"

#define DEBUG_TYPE "iree-aie-cdo-emitter"

using Path = std::filesystem::path;
auto ps = Path::preferred_separator;

namespace mlir::iree_compiler::AMDAIE {

FailureOr<XAie_DmaDesc> initDMADesc(const AMDAIEDeviceModel &deviceModel,
                                    const TileLoc &tileLoc) {
  XAie_DmaDesc dmaTileBd;
  auto devInst = const_cast<XAie_DevInst *>(&deviceModel.devInst);
  TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaDescInit, devInst, &dmaTileBd, tileLoc);
  return dmaTileBd;
}

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
    std::optional<uint8_t> packetType, std::optional<uint8_t> packetId,
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

    LLVM_DEBUG(llvm::dbgs() << dmaTileBdTensor << "\n");
    TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaSetMultiDimAddr, &dmaTileBd,
                                &dmaTileBdTensor, basePlusOffsetInBytes,
                                lenInBytes);
  } else {
    TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaSetAddrLen, &dmaTileBd,
                                basePlusOffsetInBytes, lenInBytes);
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
    LLVM_DEBUG(llvm::dbgs() << dmaPadTensor << "\n");
    TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaSetPadding, &dmaTileBd, &dmaPadTensor);
  }

  if (nextBdId) {
    auto enableNextBd = 1;
    TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaSetNextBd, &dmaTileBd, nextBdId.value(),
                                enableNextBd);
  }

  if (packetId) {
    if (!packetType) {
      llvm::errs() << "must have packetType with packetId";
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
        XAie_PacketInit(packetId.value(), packetType.value()));
  }
  TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaEnableBd, &dmaTileBd);
  auto devInst = const_cast<XAie_DevInst *>(&deviceModel.devInst);
  TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaWriteBd, devInst, &dmaTileBd, tileLoc,
                              bdId);
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
  auto devInst = const_cast<XAie_DevInst *>(&deviceModel.devInst);
  TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaChannelSetStartQueue, devInst, tileLoc,
                              chNum, direction, bdId, repeatCount,
                              enTokenIssue);
  TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaChannelEnable, devInst, tileLoc, chNum,
                              direction);
  return success();
};

LogicalResult addElfToCDO(const AMDAIEDeviceModel &deviceModel,
                          const Path &workDirPath, const TileLoc &tileLoc,
                          std::optional<std::string> elfFile, bool aieSim) {
  // loadSym: Load symbols from .map file. This argument is not used when
  // __AIESIM__ is not defined.
  std::string fileName;
  if (elfFile)
    fileName = *elfFile;
  else
    fileName = "core_" + std::to_string(tileLoc.col) + "_" +
               std::to_string(tileLoc.row) + ".elf";
  Path elfPath = workDirPath / fileName;
  auto devInst = const_cast<XAie_DevInst *>(&deviceModel.devInst);
  TRY_XAIE_API_LOGICAL_RESULT(XAie_LoadElf, devInst, tileLoc, elfPath.c_str(),
                              /*loadSym*/ aieSim);
  return success();
}

LogicalResult resetUnresetCore(const AMDAIEDeviceModel &deviceModel,
                               const TileLoc &tileLoc) {
  auto devInst = const_cast<XAie_DevInst *>(&deviceModel.devInst);
  TRY_XAIE_API_LOGICAL_RESULT(XAie_CoreReset, devInst, tileLoc);
  TRY_XAIE_API_LOGICAL_RESULT(XAie_CoreUnreset, devInst, tileLoc);
  return success();
}

LogicalResult initializeLock(const AMDAIEDeviceModel &deviceModel,
                             const Lock &lock) {
  auto devInst = const_cast<XAie_DevInst *>(&deviceModel.devInst);
  // Set locks with explicit initializers
  auto locInit = XAie_LockInit(lock.id, *lock.init);
  TRY_XAIE_API_FATAL_ERROR(XAie_LockSetValue, devInst, lock.tileLoc, locInit);
  return success();
}

LogicalResult configureStreamSwitch(const AMDAIEDeviceModel &deviceModel,
                                    const SwitchBox &tileLoc,
                                    const std::vector<Connect> &connects) {
  auto devInst = const_cast<XAie_DevInst *>(&deviceModel.devInst);
  // FIXME hack for TCT routing
  // TODO Support both channels
  if (tileLoc.row == 0) {
    auto slvPortNum = 0;
    auto mstrPortNum = 0;
    TRY_XAIE_API_LOGICAL_RESULT(XAie_StrmConnCctEnable, devInst, tileLoc, CTRL,
                                slvPortNum, SOUTH, mstrPortNum);
  }

  for (Connect connect : connects) {
    if (connect.interconnect == Connect::Interconnect::SWB) {
      TRY_XAIE_API_LOGICAL_RESULT(
          XAie_StrmConnCctEnable, devInst, tileLoc,
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
  return success();
}

LogicalResult configureMasterSet(const AMDAIEDeviceModel &deviceModel,
                                 const SwitchBox &tileLoc,
                                 const StrmSwPortType &destBundle,
                                 uint8_t destChannel,
                                 const std::vector<AMSel> &amSels,
                                 bool keepPktHeader) {
  uint8_t mSelEn = 0;
  // TODO(max): is this negative a bit pattern or a sentinel value?
  int arbiter = -1;
  for (auto amsel : amSels) {
    // TODO(max): this is very weird...
    arbiter = amsel.arbiterId;
    mSelEn |= (1 << amsel.msel);
  }

  bool isdma = destBundle == StrmSwPortType::DMA;
  // assume a connection going south from row zero gets wired to shimdma
  // by a shimmux. TODO copy-pasted: fix the assumption
  if (!isdma && (tileLoc.row == 0)) isdma = destBundle == StrmSwPortType::SOUTH;
  // Flag for overriding DROP_HEADER. TODO: Formalize this in tablegen
  isdma &= !keepPktHeader;
  auto dropHeader =
      isdma ? XAIE_SS_PKT_DROP_HEADER : XAIE_SS_PKT_DONOT_DROP_HEADER;
  auto devInst = const_cast<XAie_DevInst *>(&deviceModel.devInst);
  TRY_XAIE_API_LOGICAL_RESULT(XAie_StrmPktSwMstrPortEnable, devInst, tileLoc,
                              strmTtoStrmT(destBundle), destChannel, dropHeader,
                              arbiter, mSelEn);
  return success();
}

LogicalResult configurePacketRule(const AMDAIEDeviceModel &deviceModel,
                                  const SwitchBox &tileLoc,
                                  const StrmSwPortType &srcBundle,
                                  uint8_t srcChannel, const AMSel &amsel,
                                  uint8_t packetId, uint8_t mask,
                                  uint8_t slotNum) {
  auto devInst = const_cast<XAie_DevInst *>(&deviceModel.devInst);
  TRY_XAIE_API_LOGICAL_RESULT(XAie_StrmPktSwSlavePortEnable, devInst, tileLoc,
                              strmTtoStrmT(srcBundle), srcChannel);
  XAie_Packet packetInit = XAie_PacketInit(packetId, /*PktType*/ 0);
  // TODO Need to better define packet id,type used here
  TRY_XAIE_API_LOGICAL_RESULT(XAie_StrmPktSwSlaveSlotEnable, devInst, tileLoc,
                              strmTtoStrmT(srcBundle), srcChannel, slotNum,
                              packetInit, mask, amsel.msel, amsel.arbiterId);
  return success();
}

LogicalResult configureCascade(const AMDAIEDeviceModel &deviceModel,
                               const Cascade &casc) {
  auto devInst = const_cast<XAie_DevInst *>(&deviceModel.devInst);
  TRY_XAIE_API_LOGICAL_RESULT(
      XAie_CoreConfigAccumulatorControl, devInst, casc.tileLoc,
      strmTtoStrmT(static_cast<StrmSwPortType>(casc.inputDir)),
      strmTtoStrmT(static_cast<StrmSwPortType>(casc.outputDir)));
  return success();
}

LogicalResult coreEnable(const AMDAIEDeviceModel &deviceModel,
                         const TileLoc &tileLoc) {
  auto devInst = const_cast<XAie_DevInst *>(&deviceModel.devInst);
  TRY_XAIE_API_LOGICAL_RESULT(XAie_CoreEnable, devInst, tileLoc);
  return success();
}

void dmaUpdateBdAddr(const AMDAIEDeviceModel &deviceModel, int col, int row,
                     size_t addr, size_t bdId) {
  auto tileLoc = XAie_TileLoc(col, row);
  auto devInst = const_cast<XAie_DevInst *>(&deviceModel.devInst);
  TRY_XAIE_API_FATAL_ERROR(XAie_DmaUpdateBdAddr, devInst, tileLoc, addr, bdId);
}

void initializeCDOGenerator(byte_ordering endianness, bool cdoDebug) {
  // Enables AXI-MM prints for configs being added in CDO
  if (cdoDebug) EnAXIdebug();
  setEndianness(endianness);
};

LogicalResult generateCDOBinary(const Path &outputPath,
                                const std::function<LogicalResult()> &cb) {
  startCDOFileStream(outputPath.c_str());
  FileHeader();
  // Never generate a completely empty CDO file.  If the file only contains a
  // header, then bootgen flags it as invalid.
  insertNoOpCommand(4);
  if (failed(cb())) return failure();
  configureHeader();
  endCurrentCDOFileStream();
  return success();
}

STRINGIFY_2TUPLE_STRUCT(XAie_AieMlDmaDimDesc, StepSize, Wrap);

std::string to_string(const XAie_DmaDimDesc &v) {
  return to_string(v.AieMlDimDesc);
}

std::string to_string(const XAie_DmaTensor &v) {
  std::vector<XAie_DmaDimDesc> dims;
  dims.reserve(v.NumDim);
  for (int p = 0; p < v.NumDim; ++p) dims.push_back(v.Dim[p]);

  return "XAie_DmaTensor(" +
         llvm::join(llvm::map_range(
                        llvm::make_range(dims.begin(), dims.end()),
                        [](const XAie_DmaDimDesc &p) { return to_string(p); }),
                    ", ") +
         ")";
}

STRINGIFY_2TUPLE_STRUCT(XAie_PadDesc, Before, After);

std::string to_string(const XAie_DmaPadTensor &v) {
  std::vector<XAie_PadDesc> pads;
  pads.reserve(v.NumDim);
  for (int p = 0; p < v.NumDim; ++p) pads.push_back(v.PadDesc[p]);

  return "XAie_DmaPadTensor(" +
         llvm::join(llvm::map_range(
                        llvm::make_range(pads.begin(), pads.end()),
                        [](const XAie_PadDesc &p) { return to_string(p); }),
                    ", ") +
         ")";
}

BOTH_OSTREAM_OPS_FORALL_CDO_TYPES(OSTREAM_OP_DEFN, BOTH_OSTREAM_OP)

}  // namespace mlir::iree_compiler::AMDAIE
