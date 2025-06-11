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

// configureDMABD
//     const AMDAIEDeviceModel &deviceModel, XAie_DmaDesc &dmaDesc,
//     const TileLoc &tileLoc, bool validBd, uint8_t bdId, bool enableNextBd,
//     std::optional<uint8_t> nextBdId, bool enablePacket,
//     std::optional<uint8_t> packetType, std::optional<uint8_t> packetId,
//     uint64_t baseAddr, uint64_t lenInBytes, uint64_t offsetInBytes,
//     uint32_t bufferElementTypeWidthInBytes,
//     const std::optional<std::vector<BDDimLayout>> &maybeDims,
//     const std::optional<std::vector<BDPadLayout>> &maybePadDims,
//     const std::optional<BDIterLayout> &maybeIter)
LogicalResult TransactionBuilder::appendDmaStartOp(
    AMDAIE::DMAStartOp dmaStartOp) {
  // Configure DMA Locks.
  
  auto tile = dmaStartOp.getTile().getDefiningOp<AMDAIE::TileOp>();
  // BdIdGenerator gen(col, row, deviceModel);
  std::optional<int64_t> col = getConstantIntValue(tile.getCol());
  std::optional<int64_t> row = getConstantIntValue(tile.getRow());
  if (!col || !row) {
      return tile->emitOpError()
                  << "expected column and row integer value/constant";
  }
  XAie_LocType tileLoc = XAie_TileLoc(col, row);
  FailureOr<XAie_DmaDesc> dmaTileBd = initDMADesc(deviceModel, tileLoc);
  if (failed(dmaTileBd)) return failure();
  for (auto& region : dmaStartOp->getRegions()) {
    for (auto& block : region.getBlocks()) {
      for (Operation& operation : block.getOperations()) {
        llvm::outs()<<"==== > "<<(operation)<<"\n";
        llvm::outs().flush();
        break;
        deviceModel, /*dmaDesc=*/dmaTileBd, tileLoc, /*validBd=*/true, bdId
      }
      // break;
    }
    // break;
  }
  return success();
  // dmaStartOp->walk<WalkOrder::PreOrder>([&](AMDAIE::DMABDOp bd) {

  // });
  // if (failed(configureDMALocks(deviceModel, dmaTileBd.value(), tileLoc,
  //                              lockAcqVal, lockRelVal, lockAcqId, lockRelId,
  //                              lockAcqEnable))) {
  //   return failure();
  // }
  // // The aie-rt API expects `strides`, `iterationStride`, and `iterationSize` to
  // // be clamped to at least 1, so that they can be encoded as (value - 1) in the
  // // hardware.
  // std::for_each(strides.begin(), strides.end(),
  //               [](int32_t &stride) { stride = std::max(stride, 1); });
  // iterationSize = std::max(iterationSize, 1U);
  // iterationStride = std::max(iterationStride, 1U);
  // // Configure DMA BD.
  // uint32_t minStrideBitWidth = deviceModel.getMinStrideBitWidth();
  // uint32_t bufferElementTypeWidthInBytes = minStrideBitWidth / 8;
  // uint32_t bufferLengthInBytes = bufferLength * bufferElementTypeWidthInBytes;
  // std::vector<BDDimLayout> dims = {
  //     {static_cast<uint16_t>(sizes[0]), static_cast<uint32_t>(strides[0])},
  //     {static_cast<uint16_t>(sizes[1]), static_cast<uint32_t>(strides[1])},
  //     {static_cast<uint16_t>(sizes[2]), static_cast<uint32_t>(strides[2])}};
  // std::optional<std::vector<BDPadLayout>> pads = std::nullopt;
  // BDIterLayout iter = {iterationStride, static_cast<uint8_t>(iterationSize),
  //                      static_cast<uint8_t>(iterationCurrent)};
  // return configureDMABD(deviceModel, dmaTileBd.value(), tileLoc, validBd, bdId,
  //                       useNextBd, nextBd, enablePacket, packetType, packetId,
  //                       deviceModel.devInst.BaseAddr, bufferLengthInBytes,
  //                       bufferOffset, bufferElementTypeWidthInBytes, dims, pads,
  //                       iter);
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
  // Assume channel is enabled by default.
  bool setChannelEnable = false;
  auto tileLoc = XAie_TileLoc(col, row);
  return configurePushToBdQueue(deviceModel, tileLoc, channel, direction, bdId,
                                repeatCount, issueToken, setChannelEnable);
}

LogicalResult TransactionBuilder::appendWriteBdOp(
    uint32_t col, uint32_t row, uint32_t bdId, uint32_t bufferLength,
    uint32_t bufferOffset, bool enablePacket, uint32_t packetId,
    uint32_t packetType, ArrayRef<int32_t> sizes, SmallVector<int32_t> strides,
    uint32_t iterationCurrent, uint32_t iterationSize, uint32_t iterationStride,
    uint32_t nextBd, bool useNextBd, bool validBd, int32_t lockRelVal,
    uint32_t lockRelId, bool lockAcqEnable, int32_t lockAcqVal,
    uint32_t lockAcqId) {
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
  std::vector<BDDimLayout> dims = {
      {static_cast<uint16_t>(sizes[0]), static_cast<uint32_t>(strides[0])},
      {static_cast<uint16_t>(sizes[1]), static_cast<uint32_t>(strides[1])},
      {static_cast<uint16_t>(sizes[2]), static_cast<uint32_t>(strides[2])}};
  std::optional<std::vector<BDPadLayout>> pads = std::nullopt;
  BDIterLayout iter = {iterationStride, static_cast<uint8_t>(iterationSize),
                       static_cast<uint8_t>(iterationCurrent)};
  return configureDMABD(deviceModel, dmaTileBd.value(), tileLoc, validBd, bdId,
                        useNextBd, nextBd, enablePacket, packetType, packetId,
                        deviceModel.devInst.BaseAddr, bufferLengthInBytes,
                        bufferOffset, bufferElementTypeWidthInBytes, dims, pads,
                        iter);
}

}  // namespace mlir::iree_compiler::AMDAIE
