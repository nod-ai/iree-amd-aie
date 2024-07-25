// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cassert>
#include <cstdint>  // uint
#include <filesystem>
#include <optional>
#include <string>

#include "AMDAIETargets.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIEEnums.h"
#include "iree-amd-aie/aie_runtime/iree_aie_configure.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "aie-generate-cdo"

using namespace mlir;
using xilinx::AIE::AMSelOp;
using xilinx::AIE::BufferOp;
using xilinx::AIE::CascadeDir;
using xilinx::AIE::ConfigureCascadeOp;
using xilinx::AIE::ConnectOp;
using xilinx::AIE::CoreOp;
using xilinx::AIE::DeviceOp;
using xilinx::AIE::DMABDOp;
using xilinx::AIE::DMABDPACKETOp;
using xilinx::AIE::DMAChannelDir;
using xilinx::AIE::DMAStartOp;
using xilinx::AIE::Interconnect;
using xilinx::AIE::LockOp;
using xilinx::AIE::MasterSetOp;
using xilinx::AIE::MemOp;
using xilinx::AIE::MemTileDMAOp;
using xilinx::AIE::PacketRuleOp;
using xilinx::AIE::PacketRulesOp;
using xilinx::AIE::ShimDMAOp;
using xilinx::AIE::ShimMuxOp;
using xilinx::AIE::SwitchboxOp;
using xilinx::AIE::TileElement;
using xilinx::AIE::TileOp;
using xilinx::AIE::UseLockOp;
using xilinx::AIE::WireBundle;

using Path = std::filesystem::path;

namespace {

mlir::iree_compiler::AMDAIE::StrmSwPortType toAMDAIEStrmT(WireBundle w) {
  using mlir::iree_compiler::AMDAIE::StrmSwPortType;
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
      return StrmSwPortType::NOC;
    case WireBundle::Trace:
      return StrmSwPortType::TRACE;
    case WireBundle::Ctrl:
      return StrmSwPortType::CTRL;
    default:
      llvm::report_fatal_error("unhandled WireBundle");
  }
}

mlir::iree_compiler::AMDAIE::Cascade::Direction toDir(CascadeDir dir) {
  switch (dir) {
    case xilinx::AIE::CascadeDir::South:
      return mlir::iree_compiler::AMDAIE::Cascade::Direction::SOUTH;
    case xilinx::AIE::CascadeDir::North:
      return mlir::iree_compiler::AMDAIE::Cascade::Direction::NORTH;
    case xilinx::AIE::CascadeDir::West:
      return mlir::iree_compiler::AMDAIE::Cascade::Direction::WEST;
    case xilinx::AIE::CascadeDir::East:
      return mlir::iree_compiler::AMDAIE::Cascade::Direction::EAST;
  }
  llvm::report_fatal_error("unhandled cascade dir");
}

mlir::iree_compiler::AMDAIE::Lock::Action toLock(xilinx::AIE::LockAction l) {
  switch (l) {
    case xilinx::AIE::LockAction::Acquire:
      return mlir::iree_compiler::AMDAIE::Lock::Action::Acquire;
    case xilinx::AIE::LockAction::AcquireGreaterEqual:
      return mlir::iree_compiler::AMDAIE::Lock::Action::AcquireGreaterEqual;
    case xilinx::AIE::LockAction::Release:
      return mlir::iree_compiler::AMDAIE::Lock::Action::Release;
  }
  llvm::report_fatal_error("unhandled lock action");
}

}  // namespace

namespace mlir::iree_compiler::AMDAIE {
LogicalResult configureLocksAndBd(Block &block, const TileLoc &tileLoc,
                                  const AMDAIEDeviceModel &deviceModel) {
  FailureOr<XAie_DmaDesc> dmaTileBd = initDMADesc(deviceModel, tileLoc);
  if (failed(dmaTileBd)) return failure();
  assert(!block.getOps<UseLockOp>().empty() && "BD block has no lock-usage");
  std::optional<int> acqValue, relValue, acqLockId, relLockId;
  bool acqEn;
  // switch (lock->getAc)
  for (auto op : block.getOps<UseLockOp>()) {
    // Only dyn_cast if you are going to check if it was of the type
    // expected; if you aren't checking use cast instead as it will at
    // least assert in debug mode with an easier to understand error than
    // dereferencing.
    LockOp lock = cast<LockOp>(op.getLock().getDefiningOp());
    switch (toLock(op.getAction())) {
      case Lock::Action::Acquire:
      case Lock::Action::AcquireGreaterEqual:
        acqEn = op.getAcqEn();
        acqLockId = lock.getLockIDValue();
        acqValue = op.getLockValue();
        if (op.acquireGE()) acqValue.value() = -acqValue.value();
        break;
      case Lock::Action::Release:
        relLockId = lock.getLockIDValue();
        relValue = op.getLockValue();
        break;
    }
  }
  assert(acqValue && relValue && acqLockId && relLockId &&
         "expected both use_lock(acquire) and use_lock(release) with bd");
  if (failed(configureDMALocks(deviceModel, dmaTileBd.value(), tileLoc,
                               *acqValue, *relValue, *acqLockId, *relLockId,
                               acqEn))) {
    return failure();
  }

  // pull metadata related to packet routing, bdid, buffer length, size, stride
  // to pass to aie-rt
  DMABDOp bdOp = *block.getOps<DMABDOp>().begin();
  assert(bdOp.getBdId().has_value() &&
         "DMABDOp must have assigned bd_id; did you forget to run "
         "aie-assign-bd-ids?");
  std::optional<uint8_t> packetType;
  std::optional<uint8_t> packetID;
  auto maybePacketOps = block.getOps<DMABDPACKETOp>();
  if (!maybePacketOps.empty()) {
    assert(llvm::range_size(maybePacketOps) == 1 &&
           "expected only one dma_bd_packet");
    auto packetOp = *maybePacketOps.begin();
    packetType = packetOp.getPacketType();
    packetID = packetOp.getPacketID();
  }

  BufferOp bufferOp = cast<BufferOp>(bdOp.getBuffer().getDefiningOp());
  if (!bufferOp.getAddress())
    return bufferOp.emitError("buffer must have address assigned");
  std::optional<std::vector<BDDimLayout>> maybeDims;
  if (auto dims = bdOp.getDimensions()) {
    maybeDims = std::vector<BDDimLayout>{};
    for (const auto &dim : (*dims)) {
      maybeDims->emplace_back(BDDimLayout{dim.getSize(), dim.getStride()});
    }
  }

  std::optional<std::vector<BDPadLayout>> maybePadDims;
  if (auto dims = bdOp.getPadDimensions()) {
    maybePadDims = std::vector<BDPadLayout>{};
    for (const auto &dim : (*dims)) {
      maybePadDims->emplace_back(
          BDPadLayout{dim.getConstPadBefore(), dim.getConstPadAfter()});
    }
  }
  if (failed(configureDMABD(deviceModel, dmaTileBd.value(), tileLoc,
                            static_cast<uint8_t>(*bdOp.getBdId()),
                            bdOp.getNextBdId().has_value()
                                ? std::optional<uint8_t>{static_cast<uint8_t>(
                                      *bdOp.getNextBdId())}
                                : std::nullopt,
                            packetType, packetID, *bufferOp.getAddress(),
                            bdOp.getLenInBytes(), bdOp.getOffsetInBytes(),
                            bdOp.getBufferElementTypeWidthInBytes(), maybeDims,
                            maybePadDims))) {
    return failure();
  }
  return success();
}

LogicalResult addAieElfsToCDO(const AMDAIEDeviceModel &deviceModel,
                              DeviceOp &device, const Path &workDirPath,
                              bool aieSim) {
  for (auto tileOp : device.getOps<TileOp>()) {
    TileLoc tileLoc{tileOp.getCol(), tileOp.getRow()};
    if (deviceModel.isShimNOCorPLTile(tileLoc.col, tileLoc.row)) continue;
    if (auto coreOp = tileOp.getCoreOp()) {
      std::string fileName;
      if (auto elfFile = coreOp.getElfFile())
        fileName = *elfFile;
      else
        fileName = "core_" + std::to_string(tileLoc.col) + "_" +
                   std::to_string(tileLoc.row) + ".elf";
      if (failed(addElfToTile(deviceModel, tileLoc, workDirPath / fileName,
                              aieSim))) {
        return failure();
      }
    }
  }
  return success();
}

LogicalResult addInitConfigToCDO(const AMDAIEDeviceModel &deviceModel,
                                 DeviceOp &device) {
  for (auto tileOp : device.getOps<TileOp>()) {
    TileLoc tileLoc = {tileOp.getCol(), tileOp.getRow()};
    if (deviceModel.isShimTile(tileOp.getCol(), tileOp.getRow())) {
      continue;
    }
    if (auto coreOp = tileOp.getCoreOp();
        coreOp && failed(resetUnResetCore(deviceModel, tileLoc))) {
      return failure();
    }
  }

  // Set locks with explicit initializers
  WalkResult r = device.walk<WalkOrder::PreOrder>([&](LockOp lockOp) {
    if (lockOp.getLockID() && lockOp.getInit()) {
      TileLoc tileLoc = {lockOp.getTileOp().getCol(),
                         lockOp.getTileOp().getRow()};
      Lock lock{tileLoc, static_cast<uint8_t>(*lockOp.getLockID()),
                static_cast<int8_t>(*lockOp.getInit())};
      if (failed(initializeLock(deviceModel, lock)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (r.wasInterrupted()) return failure();

  auto memOps = llvm::to_vector_of<TileElement>(device.getOps<MemOp>());
  llvm::append_range(memOps, device.getOps<MemTileDMAOp>());
  llvm::append_range(memOps, device.getOps<ShimDMAOp>());
  for (TileElement memOp : memOps) {
    TileLoc tileLoc = {memOp.getTileID().col, memOp.getTileID().row};
    if (deviceModel.isShimNOCorPLTile(tileLoc.col, tileLoc.row)) {
      continue;
    }

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
                deviceModel, tileLoc, chNum,
                static_cast<DMAChannelDir>(channelDir), bd.getBdId().value(),
                op.getRepeatCount())))
          return failure();
      }
    }
  }

  // StreamSwitch (switchbox) configuration
  for (auto switchboxOp : device.getOps<SwitchboxOp>()) {
    SwitchBox swb = {switchboxOp.colIndex(), switchboxOp.rowIndex()};
    std::vector<Connect> connects;
    for (auto connectOp : switchboxOp.getOps<ConnectOp>()) {
      connects.emplace_back(Port{toAMDAIEStrmT(connectOp.getSourceBundle()),
                                 connectOp.getSourceChannel()},
                            Port{toAMDAIEStrmT(connectOp.getDestBundle()),
                                 connectOp.getDestChannel()},
                            Connect::Interconnect::SWB, swb.col, swb.row);
    }
    if (failed(configureStreamSwitch(deviceModel, swb, connects))) {
      return failure();
    }

    Block &b = switchboxOp.getConnections().front();
    for (auto masterSetOp : b.getOps<MasterSetOp>()) {
      std::vector<AMSel> amSels;
      for (auto val : masterSetOp.getAmsels()) {
        AMSelOp amsel = cast<AMSelOp>(val.getDefiningOp());
        amSels.push_back({static_cast<uint8_t>(amsel.getArbiterID()),
                          static_cast<uint8_t>(amsel.getMsel())});
      }
      if (failed(configureSwitchPacketMasters(
              deviceModel, swb, toAMDAIEStrmT(masterSetOp.getDestBundle()),
              masterSetOp.getDestChannel(), amSels,
              masterSetOp->hasAttr("keep_pkt_header"))))
        return failure();
    }

    for (auto packetRulesOp : b.getOps<PacketRulesOp>()) {
      int slot = 0;
      Block &block = packetRulesOp.getRules().front();
      for (auto packetRuleOp : block.getOps<PacketRuleOp>()) {
        AMSelOp amselOp =
            cast<AMSelOp>(packetRuleOp.getAmsel().getDefiningOp());
        if (failed(configureSwitchPacketSlaves(
                deviceModel, swb,
                toAMDAIEStrmT(packetRulesOp.getSourceBundle()),
                packetRulesOp.getSourceChannel(),
                AMSel{static_cast<uint8_t>(amselOp.arbiterIndex()),
                      static_cast<uint8_t>(amselOp.getMselValue())},
                packetRuleOp.valueInt(), packetRuleOp.maskInt(), slot)))
          return failure();
        slot++;
      }
    }
  }

  for (auto muxOp : device.getOps<ShimMuxOp>()) {
    SwitchBox swb = {muxOp.getTileOp().getCol(), muxOp.getTileOp().getRow()};
    std::vector<Connect> connects;
    for (auto connectOp : muxOp.getOps<ConnectOp>()) {
      connects.emplace_back(Port{toAMDAIEStrmT(connectOp.getSourceBundle()),
                                 connectOp.getSourceChannel()},
                            Port{toAMDAIEStrmT(connectOp.getDestBundle()),
                                 connectOp.getDestChannel()},
                            Connect::Interconnect::SHIMMUX, swb.col, swb.row);
    }
    if (failed(configureStreamSwitch(deviceModel, swb, connects))) {
      return failure();
    }
  }

  // Cascade configuration
  for (auto configOp : device.getOps<ConfigureCascadeOp>()) {
    TileOp tile = cast<TileOp>(configOp.getTile().getDefiningOp());
    TileLoc tileLoc = {tile.getCol(), tile.getCol()};
    Cascade casc = {tileLoc, toDir(configOp.getInputDir()),
                    toDir(configOp.getOutputDir())};
    if (failed(configureCascade(deviceModel, casc))) return failure();
  }

  return success();
}

LogicalResult addCoreEnableToCDO(const AMDAIEDeviceModel &deviceModel,
                                 DeviceOp &device) {
  // Start execution of all the cores.
  for (auto tileOp : device.getOps<TileOp>()) {
    TileLoc tileLoc = {tileOp.getCol(), tileOp.getRow()};
    if (auto coreOp = tileOp.getCoreOp();
        coreOp && failed(coreEnable(deviceModel, tileLoc)))
      return failure();
  }
  return success();
}

LogicalResult generateCDOBinariesSeparately(
    const AMDAIEDeviceModel &deviceModel, const Path &workDirPath,
    DeviceOp &device, bool aieSim, bool enableCores) {
  if (failed(generateCDOBinary(workDirPath / "aie_cdo_elfs.bin",
                               [&deviceModel, &device, &workDirPath, &aieSim] {
                                 return addAieElfsToCDO(deviceModel, device,
                                                        workDirPath, aieSim);
                               })))
    return failure();

  if (failed(generateCDOBinary(workDirPath / "aie_cdo_init.bin",
                               [&deviceModel, &device] {
                                 return addInitConfigToCDO(deviceModel, device);
                               })))
    return failure();

  if (enableCores && !device.getOps<CoreOp>().empty() &&
      failed(generateCDOBinary(workDirPath / "aie_cdo_enable.bin",
                               [&deviceModel, &device] {
                                 return addCoreEnableToCDO(deviceModel, device);
                               })))
    return failure();

  return success();
}

LogicalResult AIETranslateToCDODirect(ModuleOp m, llvm::StringRef workDirPath,
                                      bool bigEndian, bool emitUnified,
                                      bool cdoDebug, bool aieSim,
                                      bool enableCores) {
  auto devOps = m.getOps<DeviceOp>();
  assert(llvm::range_size(devOps) == 1 &&
         "only exactly 1 device op supported.");
  DeviceOp device = *devOps.begin();
  AMDAIEDeviceModel deviceModel =
      AMDAIE::getDeviceModel(static_cast<AMDAIEDevice>(device.getDevice()));
  byte_ordering endianness =
      bigEndian ? byte_ordering::Big_Endian : byte_ordering::Little_Endian;
  DEBUG_WITH_TYPE("aie-cdo-driver-debug", cdoDebug = true);
  initializeCDOGenerator(endianness, cdoDebug);
  return generateCDOBinariesSeparately(deviceModel, Path(workDirPath.str()),
                                       device, aieSim, enableCores);
}
}  // namespace mlir::iree_compiler::AMDAIE
