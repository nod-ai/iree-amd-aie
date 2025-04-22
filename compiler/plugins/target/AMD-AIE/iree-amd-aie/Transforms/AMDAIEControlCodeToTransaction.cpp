// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/iree_aie_configure.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "iree-amdaie-controlcode-to-transaction"

#define TXN_OPC_TCT XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_TCT
#define TXN_OPC_DDR_PATCH XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_DDR_PATCH

namespace mlir::iree_compiler::AMDAIE {

class TransactionBuilder {
 public:
  AMDAIE::AMDAIEDeviceModel deviceModel;
  TransactionBuilder(AMDAIE::AMDAIEDeviceModel deviceModel)
      : deviceModel(std::move(deviceModel)) {}

  void clearAndInitialize() {
    instructions.clear();
    // Setup txn header.
    TRY_XAIE_API_FATAL_ERROR(XAie_StartTransaction, &deviceModel.devInst,
                             XAIE_TRANSACTION_DISABLE_AUTO_FLUSH);
  }

  size_t getInstructionSize() const { return instructions.size(); }

  ArrayRef<uint32_t> finalizeAndReturnInstructions() {
    std::unique_ptr<uint8_t, decltype(&free)> txn_ptr(
        XAie_ExportSerializedTransaction(&deviceModel.devInst, 0, 0), &free);
    // Extract transaction size.
    auto *hdr = reinterpret_cast<XAie_TxnHeader *>(txn_ptr.get());
    size_t sizeInBytes = hdr->TxnSize;
    size_t instructionCount = sizeInBytes / sizeof(uint32_t);
    // Resize instructions and copy data.
    instructions.resize(instructionCount);
    memcpy(instructions.data(), txn_ptr.get(), sizeInBytes);
    LLVM_DEBUG(llvm::dbgs()
               << "Instruction size: " << getInstructionSize() << "\n");
    // Clear the transaction.
    TRY_XAIE_API_FATAL_ERROR(XAie_ClearTransaction, &deviceModel.devInst);
    return ArrayRef<uint32_t>(instructions.data(), instructions.size());
  }

  void dumpTransactionAsHex() const {
    llvm::outs() << "Transaction: \n";
    for (uint32_t word : instructions) {
      // Write hex as 0xXXXXXXXX
      llvm::outs() << utohexstr(word, 8) << "\n";
    }
  }

  LogicalResult appendAddressPatch(uint32_t addr, uint32_t argIdx,
                                   uint32_t offset) {
    std::array<uint32_t, 10> words = {0};

    words[4] = addr;
    words[5] = 0;
    words[6] = argIdx;
    words[7] = 0;
    words[8] = offset;
    words[9] = 0;

    uint8_t opCode = static_cast<uint8_t>(TXN_OPC_DDR_PATCH);
    uint32_t *data = &words[0];
    uint32_t size = words.size() * sizeof(uint32_t);
    return configureCustomTxnOp(deviceModel, opCode, data, size);
  }

  LogicalResult appendTCTSync(uint32_t col, uint32_t row, uint32_t direction,
                              uint32_t rowNum, uint32_t colNum,
                              uint32_t channel) {
    std::array<uint32_t, 2> words = {0};

    words[0] |= direction & 0xff;
    words[0] |= (row & 0xff) << 8;
    words[0] |= (col & 0xff) << 16;

    words[1] |= (rowNum & 0xff) << 8;
    words[1] |= (colNum & 0xff) << 16;
    words[1] |= (channel & 0xff) << 24;

    uint8_t opCode = static_cast<uint8_t>(TXN_OPC_TCT);
    uint32_t *data = &words[0];
    uint32_t size = words.size() * sizeof(uint32_t);
    return configureCustomTxnOp(deviceModel, opCode, data, size);
  }

  LogicalResult appendPushToQueueOp(uint32_t col, uint32_t row,
                                    AMDAIE::DMAChannelDir direction,
                                    uint32_t channel, uint32_t bdId,
                                    uint32_t repeatCount, bool issueToken) {
    // Assume channel is enabled by default.
    bool setChannelEnable = false;
    auto tileLoc = XAie_TileLoc(col, row);
    return configurePushToBdQueue(deviceModel, tileLoc, channel, direction,
                                  bdId, repeatCount, issueToken,
                                  setChannelEnable);
  }

  LogicalResult appendWriteBdOp(
      uint32_t col, uint32_t row, uint32_t bdId, uint32_t bufferLength,
      uint32_t bufferOffset, bool enablePacket, uint32_t packetId,
      uint32_t packetType, ArrayRef<int32_t> sizes, ArrayRef<int32_t> strides,
      uint32_t iterationCurrent, uint32_t iterationSize,
      uint32_t iterationStride, uint32_t nextBd, bool useNextBd, bool validBd,
      int32_t lockRelVal, uint32_t lockRelId, bool lockAcqEnable,
      int32_t lockAcqVal, uint32_t lockAcqId) {
    // Configure DMA Locks.
    auto tileLoc = XAie_TileLoc(col, row);
    FailureOr<XAie_DmaDesc> dmaTileBd = initDMADesc(deviceModel, tileLoc);
    if (failed(dmaTileBd)) return failure();
    if (failed(configureDMALocks(deviceModel, dmaTileBd.value(), tileLoc,
                                 lockAcqVal, lockRelVal, lockAcqId, lockRelId,
                                 lockAcqEnable))) {
      return failure();
    }
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
    return configureDMABD(deviceModel, dmaTileBd.value(), tileLoc, validBd,
                          bdId, useNextBd, nextBd, enablePacket, packetType,
                          packetId, deviceModel.devInst.BaseAddr,
                          bufferLengthInBytes, bufferOffset,
                          bufferElementTypeWidthInBytes, dims, pads, iter);
  }

 private:
  std::vector<uint32_t> instructions;
};

LogicalResult convertOp(AMDAIE::NpuAddressPatchOp op,
                        TransactionBuilder &builder) {
  uint32_t col = op.getCol();
  uint32_t bdId = op.getBdId();
  uint32_t colShift = builder.deviceModel.getColumnShift();
  uint32_t addr = (col << colShift) | (0x1D004 + bdId * 0x20);
  if (failed(builder.appendAddressPatch(addr, op.getArgIdx(), op.getOffset())))
    return failure();
  return success();
}

LogicalResult convertOp(AMDAIE::NpuTctSyncOp op, TransactionBuilder &builder) {
  if (failed(builder.appendTCTSync(
          op.getCol(), op.getRow(), static_cast<uint32_t>(op.getDirection()),
          op.getRowNum(), op.getColNum(), op.getChannel()))) {
    return failure();
  }
  return success();
}

LogicalResult convertOp(AMDAIE::NpuPushToQueueOp op,
                        TransactionBuilder &builder) {
  if (failed(builder.appendPushToQueueOp(
          op.getCol(), op.getRow(), op.getDirection(), op.getChannel(),
          op.getBdId(), op.getRepeatCount(),
          static_cast<bool>(op.getAsyncToken())))) {
    return failure();
  }
  return success();
}

LogicalResult convertOp(AMDAIE::NpuWriteBdOp op, TransactionBuilder &builder) {
  uint32_t col = op.getCol();
  uint32_t row = op.getRow();
  uint32_t bdId = op.getBdId();
  ArrayRef<int32_t> sizes = op.getSizes();
  SmallVector<int32_t> strides(op.getStrides());
  if (sizes.size() != 3) return op.emitOpError() << "expected 3 sizes";
  if (strides.size() != 3) return op.emitOpError() << "expected 3 strides";
  // Strides and iteration_size will be encoded as `actual - 1`, so we need to
  // ensure they are at least 1.
  std::for_each(strides.begin(), strides.end(),
                [](int32_t &stride) { stride = std::max(stride, int32_t(1)); });
  uint32_t iterationSize = std::max(op.getIterationSize(), uint32_t(1));
  uint32_t iterationStride = std::max(op.getIterationStride(), uint32_t(1));
  if (failed(builder.appendWriteBdOp(
          col, row, bdId, op.getBufferLength(), op.getBufferOffset(),
          op.getEnablePacket(), op.getPacketId(), op.getPacketType(), sizes,
          strides, op.getIterationCurrent(), iterationSize, iterationStride,
          op.getNextBd(), op.getUseNextBd(), op.getValidBd(),
          op.getLockRelVal(), op.getLockRelId(), op.getLockAcqEnable(),
          op.getLockAcqVal(), op.getLockAcqId()))) {
    return failure();
  }
  return success();
}

LogicalResult controlCodeToTransaction(IRRewriter &rewriter,
                                       AMDAIE::ControlCodeOp controlCodeOp,
                                       TransactionBuilder &builder) {
  SmallVector<Operation *> toBeErased;
  WalkResult res = controlCodeOp->walk([&](Operation *op) {
    LogicalResult switchResult =
        TypeSwitch<Operation *, LogicalResult>(op)
            .Case<AMDAIE::NpuAddressPatchOp, AMDAIE::NpuTctSyncOp,
                  AMDAIE::NpuPushToQueueOp, AMDAIE::NpuWriteBdOp>(
                [&](auto npuOp) {
                  if (failed(convertOp(npuOp, builder))) return failure();
                  toBeErased.push_back(npuOp);
                  return success();
                })
            .Default([&](Operation *) { return success(); });
    if (failed(switchResult)) return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return failure();
  for (Operation *op : toBeErased) {
    op->dropAllUses();
    rewriter.eraseOp(op);
  }
  return success();
}

namespace {

class AMDAIEControlCodeToTransactionPass
    : public impl::AMDAIEControlCodeToTransactionBase<
          AMDAIEControlCodeToTransactionPass> {
 public:
  AMDAIEControlCodeToTransactionPass(
      const AMDAIEControlCodeToTransactionOptions &options)
      : AMDAIEControlCodeToTransactionBase(options) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }
  void runOnOperation() override;
};

void AMDAIEControlCodeToTransactionPass::runOnOperation() {
  Operation *parentOp = getOperation();
  MLIRContext *context = &getContext();

  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(parentOp);
  std::optional<AMDAIEDevice> maybeDevice = getConfigAMDAIEDevice(targetAttr);
  if (!maybeDevice) {
    parentOp->emitOpError()
        << "has no AMDAIEDevice in the target attribute configuration. This "
           "device-specific information is required to lower control code "
           "ops.";
    return signalPassFailure();
  }
  AMDAIE::AMDAIEDeviceModel deviceModel =
      AMDAIE::getDeviceModel(maybeDevice.value());
  TransactionBuilder transactionBuilder(std::move(deviceModel));

  IRRewriter rewriter(context);
  WalkResult res = parentOp->walk([&](AMDAIE::WorkgroupOp workgroupOp) {
    transactionBuilder.clearAndInitialize();
    if (failed(controlCodeToTransaction(rewriter, workgroupOp.getControlCode(),
                                        transactionBuilder))) {
      return WalkResult::interrupt();
    }
    ArrayRef<uint32_t> instructions =
        transactionBuilder.finalizeAndReturnInstructions();
    workgroupOp.setNpuInstructionsAttr(DenseUI32ResourceElementsAttr::get(
        RankedTensorType::get(
            transactionBuilder.getInstructionSize(),
            IntegerType::get(&getContext(), 32, IntegerType::Unsigned)),
        "npu_instructions",
        HeapAsmResourceBlob::allocateAndCopyInferAlign(instructions)));
    if (dumpTransaction) transactionBuilder.dumpTransactionAsHex();
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEControlCodeToTransactionPass(
    AMDAIEControlCodeToTransactionOptions options) {
  return std::make_unique<AMDAIEControlCodeToTransactionPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
