// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AMDAIERT.h"

#include "iree-amd-aie/aie_runtime/iree_aie_configure.h"

using namespace mlir;

using xilinx::AIE::AMSelOp;
using xilinx::AIE::BDDimLayoutAttr;
using xilinx::AIE::BDPadLayoutAttr;
using xilinx::AIE::BufferOp;
using xilinx::AIE::ConnectOp;
using xilinx::AIE::CoreOp;
using xilinx::AIE::DeviceOp;
using xilinx::AIE::DMABDOp;
using xilinx::AIE::DMABDPACKETOp;
using xilinx::AIE::DMAChannelDir;
using xilinx::AIE::DMAStartOp;
using xilinx::AIE::LockAction;
using xilinx::AIE::LockOp;
using xilinx::AIE::MasterSetOp;
using xilinx::AIE::MemOp;
using xilinx::AIE::MemTileDMAOp;
using xilinx::AIE::PacketRuleOp;
using xilinx::AIE::PacketRulesOp;
using xilinx::AIE::ShimMuxOp;
using xilinx::AIE::SwitchboxOp;
using xilinx::AIE::TileOp;
using xilinx::AIE::UseLockOp;

using Path = std::filesystem::path;

namespace mlir::iree_compiler::AMDAIE {

LogicalResult addAllAieElfs(const AMDAIEDeviceModel &deviceModel,
                            DeviceOp &device, const Path &workDirPath,
                            bool aieSim) {
  for (auto tileOp : device.getOps<TileOp>()) {
    TileLoc tileLoc{tileOp.getCol(), tileOp.getRow()};
    if (deviceModel.isShimNOCorPLTile(tileLoc.col, tileLoc.row)) continue;
    if (CoreOp coreOp = getCoreOp(tileOp)) {
      std::string fileName;
      std::optional<StringRef> elfFile = coreOp.getElfFile();
      if (elfFile.has_value()) {
        fileName = *elfFile;
      } else {
        fileName = "core_" + std::to_string(tileLoc.col) + "_" +
                   std::to_string(tileLoc.row) + ".elf";
      }
      if (failed(addElfToTile(deviceModel, tileLoc, workDirPath / fileName,
                              aieSim))) {
        return failure();
      }
    }
  }
  return success();
}

LogicalResult addAllCoreEnable(const AMDAIEDeviceModel &deviceModel,
                               DeviceOp &device) {
  // Start execution of all the cores.
  for (auto tileOp : device.getOps<TileOp>()) {
    TileLoc tileLoc = {tileOp.getCol(), tileOp.getRow()};
    if (auto coreOp = getCoreOp(tileOp);
        coreOp && failed(coreEnable(deviceModel, tileLoc)))
      return failure();
  }
  return success();
}

Lock::Action toLock(LockAction l) {
  // Convert `xilinx::AIE::LockAction` to
  // `mlir::iree_compiler::AMDAIE::LockAction`.
  switch (l) {
    case LockAction::Acquire:
      return Lock::Action::Acquire;
    case LockAction::AcquireGreaterEqual:
      return Lock::Action::AcquireGreaterEqual;
    case LockAction::Release:
      return Lock::Action::Release;
  }
  llvm::report_fatal_error("unhandled lock action");
}

LogicalResult configureLocksAndBd(Block &block, const TileLoc &tileLoc,
                                  const AMDAIEDeviceModel &deviceModel) {
  FailureOr<XAie_DmaDesc> dmaTileBd = initDMADesc(deviceModel, tileLoc);
  if (failed(dmaTileBd)) return failure();
  std::optional<int> acqValue, relValue, acqLockId, relLockId;
  bool acqEn;
  for (UseLockOp op : block.getOps<UseLockOp>()) {
    // Only dyn_cast if you are going to check if it was of the type
    // expected; if you aren't checking use cast instead as it will at
    // least assert in debug mode with an easier to understand error than
    // dereferencing.
    LockOp lock = cast<LockOp>(op.getLock().getDefiningOp());
    switch (toLock(op.getAction())) {
      case Lock::Action::Acquire:
      case Lock::Action::AcquireGreaterEqual:
        acqEn = op.getAcqEn();
        acqLockId = lock.getLockID();
        acqValue = op.getValue().value_or(1);
        if (op.getAction() == LockAction::AcquireGreaterEqual)
          acqValue.value() = -acqValue.value();
        break;
      case Lock::Action::Release:
        relLockId = lock.getLockID();
        relValue = op.getValue().value_or(1);
        break;
    }
  }
  // Disable acquire and release locks if not set.
  if (!acqLockId) {
    acqLockId = 0;
    acqValue = 0;
    acqEn = false;
  }
  if (!relLockId) {
    relLockId = 0;
    relValue = 0;
  }
  assert(acqValue && relValue && acqLockId && relLockId &&
         "expected both use_lock(acquire) and use_lock(release) with bd");
  if (failed(configureDMALocks(deviceModel, dmaTileBd.value(), tileLoc,
                               *acqValue, *relValue, *acqLockId, *relLockId,
                               acqEn))) {
    return failure();
  }

  // Pull metadata related to packet routing, bdId, buffer length, size, stride
  // to pass to aie-rt.
  DMABDOp bdOp = *block.getOps<DMABDOp>().begin();
  assert(bdOp.getBdId().has_value() &&
         "DMABDOp must have assigned bd_id; did you forget to run "
         "aie-assign-bd-ids?");
  bool validBd = true;
  std::optional<uint8_t> packetType;
  std::optional<uint8_t> packetID;
  bool enablePacket = false;
  auto maybePacketOps = block.getOps<DMABDPACKETOp>();
  if (!maybePacketOps.empty()) {
    assert(llvm::range_size(maybePacketOps) == 1 &&
           "expected only one dma_bd_packet");
    DMABDPACKETOp packetOp = *maybePacketOps.begin();
    packetType = packetOp.getPacketType();
    packetID = packetOp.getPacketId();
    enablePacket = true;
  }

  BufferOp bufferOp = cast<BufferOp>(bdOp.getBuffer().getDefiningOp());
  if (!bufferOp.getAddress())
    return bufferOp.emitError("buffer must have address assigned");
  // Convert `xilinx::AIE::BDDimLayoutAttr` to
  // `mlir::iree_compiler::AMDAIE::BDDimLayout`.
  std::optional<std::vector<BDDimLayout>> maybeDims;
  if (std::optional<std::vector<BDDimLayoutAttr>> dims = bdOp.getDimensions()) {
    maybeDims = std::vector<BDDimLayout>{};
    for (const BDDimLayoutAttr &dim : (*dims)) {
      maybeDims->emplace_back(BDDimLayout{dim.getSize(), dim.getStride()});
    }
  }

  // Convert `xilinx::AIE::BDPadLayoutAttr` to
  // `mlir::iree_compiler::AMDAIE::BDPadLayout`.
  std::optional<std::vector<BDPadLayout>> maybePadDims;
  if (std::optional<std::vector<BDPadLayoutAttr>> dims =
          bdOp.getPadDimensions()) {
    maybePadDims = std::vector<BDPadLayout>{};
    for (const BDPadLayoutAttr &dim : (*dims)) {
      maybePadDims->emplace_back(
          BDPadLayout{dim.getConstPadBefore(), dim.getConstPadAfter()});
    }
  }

  bool enableNextBd = bdOp.getNextBdId().has_value();
  std::optional<uint8_t> nextBdId =
      enableNextBd
          ? std::optional<uint8_t>{static_cast<uint8_t>(*bdOp.getNextBdId())}
          : std::nullopt;
  std::optional<BDIterLayout> maybeIter = std::nullopt;
  if (failed(configureDMABD(deviceModel, dmaTileBd.value(), tileLoc, validBd,
                            static_cast<uint8_t>(*bdOp.getBdId()), enableNextBd,
                            nextBdId, enablePacket, packetType, packetID,
                            *bufferOp.getAddress(), getLenInBytes(bdOp),
                            getOffsetInBytes(bdOp),
                            getBufferElementTypeWidthInBytes(bdOp), maybeDims,
                            maybePadDims, maybeIter))) {
    return failure();
  }
  return success();
}

LogicalResult addInitConfig(const AMDAIEDeviceModel &deviceModel,
                            DeviceOp &device) {
  // Reset and unreset all cores.
  for (auto tileOp : device.getOps<TileOp>()) {
    TileLoc tileLoc = {tileOp.getCol(), tileOp.getRow()};
    if (deviceModel.isShimTile(tileOp.getCol(), tileOp.getRow())) {
      continue;
    }
    if (auto coreOp = getCoreOp(tileOp);
        coreOp && failed(resetUnResetCore(deviceModel, tileLoc))) {
      return failure();
    }
  }

  // Set locks with explicit initializers.
  WalkResult r;
  r = device.walk<WalkOrder::PreOrder>([&](LockOp lockOp) {
    if (lockOp.getLockID() && lockOp.getInit()) {
      TileOp t = xilinx::AIE::getTileOp(*lockOp.getOperation());
      TileLoc tileLoc = {t.getCol(), t.getRow()};
      Lock lock{tileLoc, static_cast<uint8_t>(*lockOp.getLockID()),
                static_cast<int8_t>(*lockOp.getInit())};
      if (failed(initializeLock(deviceModel, lock)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (r.wasInterrupted()) return failure();

  // Set up the memory operations.
  auto memOps = llvm::to_vector_of<Operation *>(device.getOps<MemOp>());
  llvm::append_range(memOps, device.getOps<MemTileDMAOp>());
  for (Operation *memOp : memOps) {
    TileOp t = xilinx::AIE::getTileOp(*memOp);
    TileLoc tileLoc = {t.getCol(), t.getRow()};
    if (deviceModel.isShimNOCorPLTile(tileLoc.col, tileLoc.row)) {
      continue;
    }

    // Handle DMA ops separately.
    for (Block &block : memOp->getRegion(0)) {
      if (block.getOps<DMABDOp>().empty()) continue;
      if (failed(configureLocksAndBd(block, tileLoc, deviceModel)))
        return failure();
    }

    for (Block &block : memOp->getRegion(0)) {
      for (auto op : block.getOps<DMAStartOp>()) {
        DMABDOp bd = *op.getDest()->getOps<DMABDOp>().begin();
        int chNum = op.getChannelIndex();
        auto channelDir = static_cast<DMAChannelDir>(op.getChannelDir());
        bool issueToken = tileLoc.row == 0 && channelDir == DMAChannelDir::MM2S;
        bool setChannelEnable = true;
        if (failed(configurePushToBdQueue(
                deviceModel, tileLoc, chNum, channelDir, bd.getBdId().value(),
                op.getRepeatCount(), issueToken, setChannelEnable)))
          return failure();
      }
    }
  }

  // StreamSwitch (switchbox) configuration.
  for (auto switchboxOp : device.getOps<SwitchboxOp>()) {
    TileOp t = xilinx::AIE::getTileOp(*switchboxOp.getOperation());
    TileLoc tileLoc = {t.getCol(), t.getRow()};
    std::vector<Connect> connects;
    for (auto connectOp : switchboxOp.getOps<ConnectOp>()) {
      connects.emplace_back(
          Port{connectOp.getSourceBundle(), connectOp.getSourceChannel()},
          Port{connectOp.getDestBundle(), connectOp.getDestChannel()},
          Connect::Interconnect::SWB, tileLoc.col, tileLoc.row);
    }
    if (failed(configureStreamSwitch(deviceModel, tileLoc, connects))) {
      return failure();
    }

    Block &b = switchboxOp.getConnections().front();
    for (auto masterSetOp : b.getOps<MasterSetOp>()) {
      std::vector<AMSel> amSels;
      for (Value val : masterSetOp.getAmsels()) {
        AMSelOp amsel = cast<AMSelOp>(val.getDefiningOp());
        amSels.push_back({static_cast<uint8_t>(amsel.getArbiterID()),
                          static_cast<uint8_t>(amsel.getMsel())});
      }
      if (failed(configureSwitchPacketMasters(
              deviceModel, tileLoc, masterSetOp.getDestBundle(),
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
                deviceModel, tileLoc, packetRulesOp.getSourceBundle(),
                packetRulesOp.getSourceChannel(),
                AMSel{amselOp.getArbiterID(), amselOp.getMsel()},
                packetRuleOp.getValue(), packetRuleOp.getMask(), slot)))
          return failure();
        slot++;
      }
    }
  }

  for (auto muxOp : device.getOps<ShimMuxOp>()) {
    TileOp t = xilinx::AIE::getTileOp(*muxOp.getOperation());
    TileLoc tileLoc = {t.getCol(), t.getRow()};
    std::vector<Connect> connects;
    for (auto connectOp : muxOp.getOps<ConnectOp>()) {
      connects.emplace_back(
          Port{connectOp.getSourceBundle(), connectOp.getSourceChannel()},
          Port{connectOp.getDestBundle(), connectOp.getDestChannel()},
          Connect::Interconnect::SHIMMUX, tileLoc.col, tileLoc.row);
    }
    if (failed(configureStreamSwitch(deviceModel, tileLoc, connects))) {
      return failure();
    }
  }

  return success();
}

}  // namespace mlir::iree_compiler::AMDAIE
