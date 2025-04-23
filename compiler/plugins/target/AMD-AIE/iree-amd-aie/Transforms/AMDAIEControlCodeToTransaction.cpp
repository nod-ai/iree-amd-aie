// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIETransactionBuilder.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "iree-amdaie-controlcode-to-transaction"

namespace mlir::iree_compiler::AMDAIE {

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
  if (failed(builder.appendWriteBdOp(
          op.getCol(), op.getRow(), op.getBdId(), op.getBufferLength(),
          op.getBufferOffset(), op.getEnablePacket(), op.getPacketId(),
          op.getPacketType(), op.getSizes(),
          SmallVector<int32_t>(op.getStrides()), op.getIterationCurrent(),
          op.getIterationSize(), op.getIterationStride(), op.getNextBd(),
          op.getUseNextBd(), op.getValidBd(), op.getLockRelVal(),
          op.getLockRelId(), op.getLockAcqEnable(), op.getLockAcqVal(),
          op.getLockAcqId()))) {
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
