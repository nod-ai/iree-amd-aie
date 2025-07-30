// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AMDAIETransactionBuilder.h"

#include "AMDAIEUtils.h"

#define DEBUG_TYPE "iree-amdaie-transaction-builder"

namespace mlir::iree_compiler::AMDAIE {

void TransactionBuilder::clearAndInitialize() {
  instructions.clear();
  // Setup txn header.
  TRY_XAIE_API_FATAL_ERROR(XAie_StartTransaction, &deviceModel.devInst,
                           XAIE_TRANSACTION_DISABLE_AUTO_FLUSH);
}

size_t TransactionBuilder::getInstructionSize() const {
  return instructions.size();
}

ArrayRef<uint32_t> TransactionBuilder::finalizeAndReturnInstructions() {
  std::unique_ptr<uint8_t, decltype(&free)> txn_ptr(
      XAie_ExportSerializedTransaction(&deviceModel.devInst, 0, 0), &free);
  // Extract transaction size.
  auto *hdr = reinterpret_cast<XAie_TxnHeader *>(txn_ptr.get());
  size_t sizeInBytes = hdr->TxnSize;
  size_t instructionCount = sizeInBytes / sizeof(uint32_t);
  // Resize instructions and copy data.
  instructions.resize(instructionCount);
  memcpy(instructions.data(), txn_ptr.get(), sizeInBytes);
  LLVM_DEBUG(llvm::dbgs() << "Instruction size: " << getInstructionSize()
                          << "\n");
  // Clear the transaction.
  TRY_XAIE_API_FATAL_ERROR(XAie_ClearTransaction, &deviceModel.devInst);
  return ArrayRef<uint32_t>(instructions.data(), instructions.size());
}

void TransactionBuilder::dumpTransactionAsHex() const {
  llvm::outs() << "Transaction: \n";
  for (uint32_t word : instructions) {
    // Write hex as 0xXXXXXXXX
    llvm::outs() << utohexstr(word, 8) << "\n";
  }
}

LogicalResult TransactionBuilder::appendAddressPatch(uint32_t addr,
                                                     uint32_t argIdx,
                                                     uint32_t offset) {
  std::array<uint32_t, 10> words = {0};

  words[4] = addr;
  words[5] = 0;
  words[6] = argIdx;
  words[7] = 0;
  words[8] = offset;
  words[9] = 0;

  uint8_t opCode =
      static_cast<uint8_t>(XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_DDR_PATCH);
  uint32_t *data = &words[0];
  uint32_t size = words.size() * sizeof(uint32_t);
  return configureCustomTxnOp(deviceModel, opCode, data, size);
}

LogicalResult TransactionBuilder::appendLockOp(AMDAIE::LockOp lockOp) {
  auto tile = lockOp.getTile().getDefiningOp<AMDAIE::TileOp>();
  std::optional<int64_t> maybeCol = getConstantIntValue(tile.getCol());
  std::optional<int64_t> maybeRow = getConstantIntValue(tile.getRow());
  if (!maybeCol || !maybeRow) {
    return tile->emitOpError()
           << "expected column and row integer value/constant";
  }
  XAie_LocType tileLoc = XAie_TileLoc(*maybeCol, *maybeRow);
  Lock lock{tileLoc, static_cast<uint8_t>(lockOp.getValue()),
            static_cast<int8_t>(*lockOp.getInitValue())};
  if (failed(initializeLock(deviceModel, lock))) return failure();
  return success();
}

LogicalResult TransactionBuilder::appendDmaStartOp(
    AMDAIE::DMAStartOp dmaStartOp) {
  // Get tile location.
  auto channelOp = dmaStartOp.getChannel().getDefiningOp<AMDAIE::ChannelOp>();
  AMDAIE::TileOp tile = channelOp.getTileOp();
  std::optional<int64_t> maybeCol = getConstantIntValue(tile.getCol());
  std::optional<int64_t> maybeRow = getConstantIntValue(tile.getRow());
  if (!maybeCol || !maybeRow) {
    return tile->emitOpError()
           << "expected column and row integer value/constant";
  }
  XAie_LocType tileLoc = XAie_TileLoc(*maybeCol, *maybeRow);
  // Get channel op.
  int chNum = channelOp.getValue();
  auto channelDir = static_cast<DMAChannelDir>(channelOp.getDirection());
  bool enableOutOfOrder = dmaStartOp.getEnableOutOfOrder().value_or(false);
  // Configure the DMA as in-order or out-of-order mode.
  if (failed(configureOutofOrderMode(deviceModel, tileLoc, chNum, channelDir,
                                     enableOutOfOrder))) {
    return failure();
  }
  // Initialize the DMA descriptor.
  FailureOr<XAie_DmaDesc> dmaDesc = initDMADesc(deviceModel, tileLoc);
  if (failed(dmaDesc)) return failure();
  // Configure DMA BD ops within DMA Start op.
  SmallVector<AMDAIE::DMABDOp> dmaBdOps;
  dmaStartOp.walk(
      [&](AMDAIE::DMABDOp dmaBdOp) { dmaBdOps.push_back(dmaBdOp); });
  for (AMDAIE::DMABDOp dmaBdOp : dmaBdOps) {
    Block *parentBlock = dmaBdOp->getBlock();
    // Configure DMA Locks.
    std::optional<int> acqValue, relValue, acqLockId, relLockId;
    for (AMDAIE::UseLockOp useLockOp :
         parentBlock->getOps<AMDAIE::UseLockOp>()) {
      auto lockOp = useLockOp.getLock().getDefiningOp<AMDAIE::LockOp>();
      if (useLockOp.getAction() == AMDAIE::LockAction::AcquireGreaterOrEqual ||
          useLockOp.getAction() == AMDAIE::LockAction::Acquire) {
        acqValue = useLockOp.getValue();
        if (useLockOp.getAction() == AMDAIE::LockAction::AcquireGreaterOrEqual)
          acqValue.value() = -acqValue.value();
        acqLockId = lockOp.getValue();
      } else if (useLockOp.getAction() == AMDAIE::LockAction::Release) {
        relValue = useLockOp.getValue();
        relLockId = lockOp.getValue();
      }
    }
    // Disable acquire and release locks if not set.
    if (!acqLockId) {
      acqLockId = 0;
      acqValue = 0;
    }
    if (!relLockId) {
      relLockId = 0;
      relValue = 0;
    }
    assert(acqValue && relValue && acqLockId && relLockId &&
           "expected both use_lock(acquire) and use_lock(release) with bd");
    if (failed(configureDMALocks(deviceModel, dmaDesc.value(), tileLoc,
                                 *acqValue, *relValue, *acqLockId, *relLockId,
                                 /*acqEn=*/true))) {
      return failure();
    }
    // Get BD ID.
    std::optional<uint32_t> maybeBdId = dmaBdOp.getBdId();
    if (!maybeBdId) return failure();
    bool validBd = true;
    // Get packet metadata.
    std::optional<uint8_t> maybePacketType;
    std::optional<uint8_t> maybePacketID;
    std::optional<uint8_t> maybeOutOfOrderBdId;
    bool enablePacket = false;
    auto maybePacketOps = parentBlock->getOps<AMDAIE::DmaBdPacketOp>();
    if (!maybePacketOps.empty()) {
      assert(llvm::range_size(maybePacketOps) == 1 &&
             "expected only one dma_bd_packet");
      AMDAIE::DmaBdPacketOp packetOp = *maybePacketOps.begin();
      maybePacketType = packetOp.getPacketType();
      maybePacketID = packetOp.getPacketId();
      maybeOutOfOrderBdId = packetOp.getOutOfOrderBdId();
      enablePacket = true;
    }
    // Get base address.
    auto bufferOp = dmaBdOp.getBuffer().getDefiningOp<AMDAIE::BufferOp>();
    if (!bufferOp) return dmaBdOp.emitError("buffer op not found");
    std::optional<uint32_t> baseAddr = bufferOp.getAddress();
    if (!baseAddr) return bufferOp.emitError("buffer address not found");
    // Get dimensions.
    std::optional<SmallVector<BDDimLayout>> maybeDims;
    if (auto maybeDimsAttr = dmaBdOp.getDimensions()) {
      maybeDims =
          llvm::map_to_vector(*maybeDimsAttr, [](const BDDimLayoutAttr &attr) {
            return BDDimLayout{attr.getSize(), attr.getStride()};
          });
    }
    std::optional<SmallVector<BDPadLayout>> maybePadDims;
    std::optional<BDIterLayout> maybeIter = std::nullopt;
    // Get next BD ID.
    bool enableNextBd = dmaBdOp.getNextBdId().has_value();
    std::optional<uint8_t> nextBdId =
        enableNextBd ? std::optional<uint8_t>{static_cast<uint8_t>(
                           *dmaBdOp.getNextBdId())}
                     : std::nullopt;
    if (failed(configureDMABD(
            deviceModel, dmaDesc.value(), tileLoc, validBd,
            static_cast<uint8_t>(*maybeBdId), enableNextBd, nextBdId,
            enablePacket, maybePacketType, maybePacketID, maybeOutOfOrderBdId,
            *baseAddr, dmaBdOp.getLenInBytes(), dmaBdOp.getOffsetInBytes(),
            dmaBdOp.getBufferElementTypeWidthInBytes(), maybeDims, maybePadDims,
            maybeIter))) {
      return failure();
    }
  }

  // Configure push to BD queue.
  // TODO: Generalize it as this is currently hardcoded to only shim side for
  // now.
  AMDAIE::DMABDOp dmaBdOp =
      *dmaStartOp.getBody().getOps<AMDAIE::DMABDOp>().begin();
  bool issueToken = tileLoc.Row == 0 && channelDir == DMAChannelDir::MM2S;
  if (failed(configurePushToBdQueue(
          deviceModel, tileLoc, chNum, channelDir, dmaBdOp.getBdId().value(),
          dmaStartOp.getRepeatCount(), issueToken, enableOutOfOrder))) {
    return failure();
  }
  return success();
}

LogicalResult TransactionBuilder::appendTCTSync(uint32_t col, uint32_t row,
                                                uint32_t direction,
                                                uint32_t rowNum,
                                                uint32_t colNum,
                                                uint32_t channel) {
  std::array<uint32_t, 2> words = {0};

  words[0] |= direction & 0xff;
  words[0] |= (row & 0xff) << 8;
  words[0] |= (col & 0xff) << 16;

  words[1] |= (rowNum & 0xff) << 8;
  words[1] |= (colNum & 0xff) << 16;
  words[1] |= (channel & 0xff) << 24;

  uint8_t opCode = static_cast<uint8_t>(XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_TCT);
  uint32_t *data = &words[0];
  uint32_t size = words.size() * sizeof(uint32_t);
  return configureCustomTxnOp(deviceModel, opCode, data, size);
}

LogicalResult TransactionBuilder::appendPushToQueueOp(
    uint32_t col, uint32_t row, AMDAIE::DMAChannelDir direction,
    uint32_t channel, uint32_t bdId, uint32_t repeatCount, bool issueToken) {
  auto tileLoc = XAie_TileLoc(col, row);
  // Npu push to queue is always in-order.
  return configurePushToBdQueue(deviceModel, tileLoc, channel, direction, bdId,
                                repeatCount, issueToken,
                                /*enableOutofOrder=*/false);
}

LogicalResult TransactionBuilder::appendWriteBdOp(
    uint32_t col, uint32_t row, uint32_t bdId, uint32_t bufferLength,
    uint32_t bufferOffset, bool enablePacket, uint32_t packetId,
    uint32_t packetType, uint32_t outOfOrderBdId, ArrayRef<int32_t> sizes,
    SmallVector<int32_t> strides, uint32_t iterationCurrent,
    uint32_t iterationSize, uint32_t iterationStride, uint32_t nextBd,
    bool useNextBd, bool validBd, int32_t lockRelVal, uint32_t lockRelId,
    bool lockAcqEnable, int32_t lockAcqVal, uint32_t lockAcqId) {
  // Configure DMA Locks.
  auto tileLoc = XAie_TileLoc(col, row);
  FailureOr<XAie_DmaDesc> dmaTileBd = initDMADesc(deviceModel, tileLoc);
  if (failed(dmaTileBd)) return failure();
  if (failed(configureDMALocks(deviceModel, dmaTileBd.value(), tileLoc,
                               lockAcqVal, lockRelVal, lockAcqId, lockRelId,
                               lockAcqEnable))) {
    return failure();
  }
  // The aie-rt API expects `strides`, `iterationStride`, and `iterationSize` to
  // be clamped to at least 1, so that they can be encoded as (value - 1) in the
  // hardware.
  std::for_each(strides.begin(), strides.end(),
                [](int32_t &stride) { stride = std::max(stride, 1); });
  iterationSize = std::max(iterationSize, 1U);
  iterationStride = std::max(iterationStride, 1U);
  // Configure DMA BD.
  uint32_t minStrideBitWidth = deviceModel.getMinStrideBitWidth();
  uint32_t bufferElementTypeWidthInBytes = minStrideBitWidth / 8;
  uint32_t bufferLengthInBytes = bufferLength * bufferElementTypeWidthInBytes;
  SmallVector<BDDimLayout> dims = {
      {static_cast<uint16_t>(sizes[0]), static_cast<uint32_t>(strides[0])},
      {static_cast<uint16_t>(sizes[1]), static_cast<uint32_t>(strides[1])},
      {static_cast<uint16_t>(sizes[2]), static_cast<uint32_t>(strides[2])}};
  std::optional<SmallVector<BDPadLayout>> pads = std::nullopt;
  BDIterLayout iter = {iterationStride, static_cast<uint8_t>(iterationSize),
                       static_cast<uint8_t>(iterationCurrent)};
  return configureDMABD(deviceModel, dmaTileBd.value(), tileLoc, validBd, bdId,
                        useNextBd, nextBd, enablePacket, packetType, packetId,
                        outOfOrderBdId, deviceModel.devInst.BaseAddr,
                        bufferLengthInBytes, bufferOffset,
                        bufferElementTypeWidthInBytes, dims, pads, iter);
}

}  // namespace mlir::iree_compiler::AMDAIE
