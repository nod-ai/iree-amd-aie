// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "iree-amdaie-controlcode-to-transaction"

#define TXN_OPC_WRITE 0x0
#define TXN_OPC_BLOCKWRITE 0x1
#define TXN_OPC_TCT 0x80
#define TXN_OPC_DDR_PATCH 0x81

namespace mlir::iree_compiler::AMDAIE {

class TransactionBuilder {
 public:
  AMDAIE::AMDAIEDeviceModel deviceModel;
  TransactionBuilder(AMDAIE::AMDAIEDeviceModel deviceModel)
      : deviceModel(std::move(deviceModel)) {}

  void clearAndInitialize() {
    instructions.clear();
    llvm::MutableArrayRef<uint32_t> words = reserveAndGetTail(4);
    // setup txn header
    words[0] = 0x06030100;
    words[1] = 0x00000105;
  }

  size_t getInstructionSize() const { return instructions.size(); }

  ArrayRef<uint32_t> finalizeAndReturnInstructions() {
    finalizeHeader();
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
    llvm::MutableArrayRef<uint32_t> words = reserveAndGetTail(12);
    words[0] = TXN_OPC_DDR_PATCH;
    words[1] = words.size() * sizeof(uint32_t);  // Operation Size

    words[6] = addr;
    words[7] = 0;
    words[8] = argIdx;
    words[9] = 0;
    words[10] = offset;
    words[11] = 0;
    instructionCounter++;
    return success();
  }

  LogicalResult appendTCTSync(uint32_t col, uint32_t row, uint32_t direction,
                              uint32_t rowNum, uint32_t colNum,
                              uint32_t channel) {
    llvm::MutableArrayRef<uint32_t> words = reserveAndGetTail(4);
    words[0] = TXN_OPC_TCT;
    words[1] = words.size() * sizeof(uint32_t);  // Operation Size

    words[2] |= direction & 0xff;
    words[2] |= (row & 0xff) << 8;
    words[2] |= (col & 0xff) << 16;

    words[3] |= (rowNum & 0xff) << 8;
    words[3] |= (colNum & 0xff) << 16;
    words[3] |= (channel & 0xff) << 24;
    instructionCounter++;
    return success();
  }

  LogicalResult appendPushToQueueOp(uint32_t col, uint32_t row,
                                    AMDAIE::DMAChannelDir direction,
                                    uint32_t channel, uint32_t bdId,
                                    uint32_t repeatCount, bool issueToken) {
    uint32_t colShift = deviceModel.getColumnShift();
    uint32_t rowShift = deviceModel.getRowShift();
    uint32_t addr =
        direction == AMDAIE::DMAChannelDir::MM2S ? 0x1D214 : 0x1D204;
    if (channel == 1) addr += 0x8;
    // TODO(jornt): use aie-rt's transaction serializer instead to avoid these
    // indiscrepancies between this file and aie-rt.
    addr = ((col & 0xff) << colShift) | ((row & 0xff) << rowShift) |
           (addr & 0xFFFFF);
    uint32_t value = 0;
    value |= bdId & 0xF;
    value |= (repeatCount & 0xFF) << 16;
    if (issueToken) value |= 0x80000000;
    return appendWrite32Op(addr, value);
  }

  LogicalResult appendWrite32Op(uint32_t addr, uint32_t value) {
    llvm::MutableArrayRef<uint32_t> words = reserveAndGetTail(6);
    // XAIE_IO_WRITE
    words[0] = TXN_OPC_WRITE;
    words[1] = 0;
    words[2] = addr;
    words[3] = 0;
    words[4] = value;                            // Value
    words[5] = words.size() * sizeof(uint32_t);  // Operation Size
    instructionCounter++;
    return success();
  }

  LogicalResult appendWriteBdOp(
      uint32_t bdAddr, uint32_t bufferLength, uint32_t bufferOffset,
      bool enablePacket, uint32_t outOfOrderId, uint32_t packetId,
      uint32_t packetType, uint32_t d0Size, uint32_t d0Stride, uint32_t d1Size,
      uint32_t d1Stride, uint32_t d2Stride, uint32_t iterationCurrent,
      uint32_t iterationSize, uint32_t iterationStride, uint32_t nextBd,
      bool useNextBd, bool validBd, int32_t lockRelVal, uint32_t lockRelId,
      bool lockAcqEnable, int32_t lockAcqVal, uint32_t lockAcqId) {
    llvm::MutableArrayRef<uint32_t> words = reserveAndGetTail(12);
    words[0] = TXN_OPC_BLOCKWRITE;
    words[1] = 0;
    // RegOff
    words[2] = bdAddr;                           // ADDR
    words[3] = words.size() * sizeof(uint32_t);  // Operation Size
    // DMA_BDX_0
    words[4] = bufferLength;
    // DMA_BDX_1
    words[5] = bufferOffset;
    // DMA_BDX_2
    // En Packet , OoO BD ID , Packet ID , Packet Type
    words[6] |= ((int)enablePacket & 0x1) << 30;
    words[6] |= (outOfOrderId & 0x3f) << 24;
    words[6] |= (packetId & 0x1f) << 19;
    words[6] |= (packetType & 0x7) << 16;
    // DMA_BDX_3
    // TODO: Secure Access
    words[7] |= (d0Size & 0x3ff) << 20;
    words[7] |= d0Stride & 0xfffff;
    // DMA_BDX_4
    words[8] = 0x80000000;  // burst length;
    words[8] |= (d1Size & 0x3ff) << 20;
    words[8] |= d1Stride & 0xfffff;
    // DMA_BDX_5
    // TODO: SIMID, AxCache, AXQoS
    words[9] = d2Stride & 0xfffff;
    // DMA_BDX_6
    words[10] |= (iterationCurrent & 0x3f) << 26;
    words[10] |= (iterationSize & 0x3f) << 20;
    words[10] |= iterationStride & 0xfffff;
    // DMA_BDX_7
    // TODO: TLAST Suppress
    words[11] |= (nextBd & 0xf) << 27;
    words[11] |= ((int)useNextBd & 0x1) << 26;
    words[11] |= ((int)validBd & 0x1) << 25;
    words[11] |= (lockRelVal & 0xef) << 18;
    words[11] |= (lockRelId & 0xf) << 13;
    words[11] |= ((int)lockAcqEnable & 0x1) << 12;
    words[11] |= (lockAcqVal & 0xef) << 5;
    words[11] |= lockAcqId & 0xf;
    instructionCounter++;
    return success();
  }

 private:
  void finalizeHeader() {
    // Finalize txn header.
    instructions[2] = instructionCounter;
    instructions[3] = instructions.size() * sizeof(uint32_t);
  }

  llvm::MutableArrayRef<uint32_t> reserveAndGetTail(size_t tailSize) {
    auto oldSize = instructions.size();
    auto newSize = oldSize + tailSize;
    instructions.resize(newSize, 0);
    return llvm::MutableArrayRef<uint32_t>(instructions.data() + oldSize,
                                           tailSize);
  }
  size_t instructionCounter{0};
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

LogicalResult convertOp(AMDAIE::NpuDmaWaitOp op, TransactionBuilder &builder) {
  for (Value token : op.getAsyncTokens()) {
    auto pushToQueueOp =
        dyn_cast_if_present<AMDAIE::NpuPushToQueueOp>(token.getDefiningOp());
    if (!pushToQueueOp) {
      return op.emitOpError()
             << "should operate on an `amdaie.push_to_queue` op";
    }
    if (failed(builder.appendTCTSync(
            pushToQueueOp.getCol(), pushToQueueOp.getRow(),
            static_cast<uint32_t>(pushToQueueOp.getDirection()), 1, 1,
            pushToQueueOp.getChannel()))) {
      return failure();
    }
  }
  return success();
}

LogicalResult convertOp(AMDAIE::NpuPushToQueueOp op,
                        TransactionBuilder &builder) {
  uint32_t repeatCount = op.getRepeatCount() - 1;
  if (failed(builder.appendPushToQueueOp(op.getCol(), op.getRow(),
                                         op.getDirection(), op.getChannel(),
                                         op.getBdId(), repeatCount, true))) {
    return failure();
  }
  return success();
}

LogicalResult convertOp(AMDAIE::NpuWriteBdOp op, TransactionBuilder &builder) {
  uint32_t col = op.getCol();
  uint32_t row = op.getRow();
  uint32_t bdId = op.getBdId();
  uint32_t colShift = builder.deviceModel.getColumnShift();
  uint32_t rowShift = builder.deviceModel.getRowShift();
  uint32_t bdAddr =
      (col << colShift) | (row << rowShift) | (0x1D000 + bdId * 0x20);
  ArrayRef<int32_t> sizes = op.getSizes();
  ArrayRef<int32_t> strides = op.getStrides();
  if (sizes.size() != 3) return op.emitOpError() << "expected 3 sizes";
  if (strides.size() != 3) return op.emitOpError() << "expected 3 strides";
  uint32_t d0Size = sizes[sizes.size() - 1];
  uint32_t d1Size = sizes[sizes.size() - 2];
  // Strides and iteration_size are encoded as `actual - 1`, but `0` should stay
  // `0` as it's not supported;
  uint32_t d0Stride =
      std::max((int64_t)strides[strides.size() - 1] - 1, (int64_t)0);
  uint32_t d1Stride =
      std::max((int64_t)strides[strides.size() - 2] - 1, (int64_t)0);
  uint32_t d2Stride =
      std::max((int64_t)strides[strides.size() - 3] - 1, (int64_t)0);
  uint32_t iterationSize =
      std::max((int64_t)op.getIterationSize() - 1, (int64_t)0);
  uint32_t iterationStride =
      std::max((int64_t)op.getIterationStride() - 1, (int64_t)0);
  if (failed(builder.appendWriteBdOp(
          bdAddr, op.getBufferLength(), op.getBufferOffset(),
          op.getEnablePacket(), op.getOutOfOrderId(), op.getPacketId(),
          op.getPacketType(), d0Size, d0Stride, d1Size, d1Stride, d2Stride,
          op.getIterationCurrent(), iterationSize, iterationStride,
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
            .Case<AMDAIE::NpuAddressPatchOp, AMDAIE::NpuDmaWaitOp,
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
    LLVM_DEBUG(llvm::dbgs() << "Instruction size: "
                            << transactionBuilder.getInstructionSize() << "\n");
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
