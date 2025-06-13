// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AIEDialect.h"
#include "AIEXDialect.h"
#include "Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIETransactionBuilder.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::iree_compiler::AMDAIE;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

#define DEBUG_TYPE "amdaie-dma-to-npu"

LogicalResult appendSync(NpuSyncOp op, TransactionBuilder &transactionBuilder) {
  if (failed(transactionBuilder.appendTCTSync(
          op.getColumn(), op.getRow(), static_cast<uint32_t>(op.getDirection()),
          op.getRowNum(), op.getColumnNum(), op.getChannel()))) {
    return failure();
  }
  return success();
}

LogicalResult appendAddressPatch(xilinx::AIEX::NpuAddressPatchOp op,
                                 TransactionBuilder &transactionBuilder) {
  if (failed(transactionBuilder.appendAddressPatch(op.getAddr(), op.getArgIdx(),
                                                   op.getArgPlus()))) {
    return failure();
  }
  return success();
}

LogicalResult appendPushToQueue(NpuPushQueueOp op,
                                TransactionBuilder &transactionBuilder) {
  if (failed(transactionBuilder.appendPushToQueueOp(
          op.getColumn(), op.getRow(), op.getDirection(), op.getChannel(),
          op.getBdId(), op.getRepeatCount(), op.getIssueToken()))) {
    return failure();
  }
  return success();
}

LogicalResult appendWriteBd(xilinx::AIEX::NpuWriteBdOp op,
                            TransactionBuilder &transactionBuilder) {
  SmallVector<int32_t> sizes{static_cast<int32_t>(op.getD2Size()),
                             static_cast<int32_t>(op.getD1Size()),
                             static_cast<int32_t>(op.getD0Size())};
  SmallVector<int32_t> strides{static_cast<int32_t>(op.getD2Stride()),
                               static_cast<int32_t>(op.getD1Stride()),
                               static_cast<int32_t>(op.getD0Stride())};
  if (failed(transactionBuilder.appendWriteBdOp(
          op.getColumn(), op.getRow(), op.getBdId(), op.getBufferLength(),
          op.getBufferOffset(), op.getEnablePacket(), op.getPacketId(),
          op.getPacketType(), sizes, strides, op.getIterationCurrent(),
          op.getIterationSize(), op.getIterationStride(), op.getNextBd(),
          op.getUseNextBd(), op.getValidBd(), op.getLockRelVal(),
          op.getLockRelId(), op.getLockAcqEnable(), op.getLockAcqVal(),
          op.getLockAcqId()))) {
    return failure();
  }
  return success();
}

template <typename SourceOp>
class ConvertNpuOp : public OpConversionPattern<SourceOp> {
 public:
  using AppenderTy =
      function_ref<LogicalResult(SourceOp, TransactionBuilder &)>;
  AppenderTy appender;
  TransactionBuilder &transactionBuilder;
  ConvertNpuOp(MLIRContext *ctx, AppenderTy appender,
               TransactionBuilder &transactionBuilder)
      : OpConversionPattern<SourceOp>(ctx),
        appender(appender),
        transactionBuilder(transactionBuilder) {}

  LogicalResult matchAndRewrite(
      SourceOp op, typename SourceOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (failed(appender(op, transactionBuilder))) return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

// Helper class to get a ShimDMAAllocationOp for a given <device, symbol name>
// pair. An object of this class is invalidated if, for any symbol_name, a
// ShimDMAAllocationOp that uses it changes, as the cache is not updated in
// this case.
struct ShimDMAllocationGetter {
 public:
  // Return the first ShimDMAAllocationOp nested inside the DeviceOp 'dev'
  // that uses the symbol 'sym_name'
  std::optional<AIE::ShimDMAAllocationOp> get(AIE::DeviceOp dev,
                                              StringRef sym_name) {
    auto key = std::make_pair(dev, sym_name);
    auto it = allocGetter.find(key);
    if (it != allocGetter.end()) return it->second;

    auto allocOp = cachelessGet(dev, sym_name);
    allocGetter[key] = allocOp;
    return allocOp;
  }

 private:
  llvm::DenseMap<std::pair<AIE::DeviceOp, StringRef>,
                 std::optional<AIE::ShimDMAAllocationOp>>
      allocGetter;

  // Finding the ShimDMAAllocationOp for a given <DeviceOp, symbol_name> pair
  // can be slow when the symbol is used in many places. This version of the
  // function is only called when the cache does not have a
  // ShimDMAAllocationOp stored from a previous lookup.
  std::optional<AIE::ShimDMAAllocationOp> cachelessGet(AIE::DeviceOp dev,
                                                       StringRef sym_name) {
    auto *sym = dev.lookupSymbol(sym_name);
    if (!sym) return std::nullopt;

    auto uses = SymbolTable::getSymbolUses(sym, dev);
    for (auto use : *uses)
      if (auto infoOp = dyn_cast<AIE::ShimDMAAllocationOp>(use.getUser()))
        return infoOp;

    return std::nullopt;
  }
};

struct DmaToNpuPattern : OpConversionPattern<NpuDmaMemcpyNdOp> {
  using OpConversionPattern::OpConversionPattern;
  ShimDMAllocationGetter &allocGetter;

  DmaToNpuPattern(MLIRContext *context, ShimDMAllocationGetter &getter,
                  PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), allocGetter(getter) {}

  LogicalResult matchAndRewrite(
      NpuDmaMemcpyNdOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto *ctx = op->getContext();
    auto i32ty = IntegerType::get(ctx, 32);
    auto zero = IntegerAttr::get(i32ty, 0);
    auto memref = adaptor.getMemref();

    auto dev = op->getParentOfType<AIE::DeviceOp>();
    if (!dev) return failure();

    auto infoOp = allocGetter.get(dev, op.getMetadata());
    if (!infoOp)
      return op->emitOpError("couldn't find shim_dma_allocation op.");

    auto channelDir = infoOp->getChannelDir();
    bool isMM2S = channelDir == AIE::DMAChannelDir::MM2S;
    int col = infoOp->getCol();

    // initialize fields to zero
    auto column = zero;
    auto bd_id = zero;
    auto buffer_length = zero;
    auto buffer_offset = zero;
    auto enable_packet = zero;
    auto out_of_order_id = zero;
    auto packet_id = zero;
    auto packet_type = zero;
    auto d0_size = zero;
    auto d0_stride = zero;
    auto d1_size = zero;
    auto d1_stride = zero;
    auto d2_size = zero;
    auto d2_stride = zero;
    auto iteration_current = zero;
    auto iteration_size = zero;
    auto iteration_stride = zero;
    auto next_bd = zero;
    auto row = zero;
    auto use_next_bd = zero;
    auto valid_bd = zero;
    auto lock_rel_val = zero;
    auto lock_rel_id = zero;
    auto lock_acq_enable = zero;
    auto lock_acq_val = zero;
    auto lock_acq_id = zero;
    auto d0_zero_before = zero;
    auto d1_zero_before = zero;
    auto d2_zero_before = zero;
    auto d0_zero_after = zero;
    auto d1_zero_after = zero;
    auto d2_zero_after = zero;

    auto issue_token = BoolAttr::get(ctx, false);
    auto repeat_count = zero;

    llvm::SmallVector<int64_t, 4> strides = op.getStridesInAddressGranularity();
    llvm::SmallVector<int64_t, 4> sizes = op.getSizesInAddressGranularity();
    int64_t offset = op.getOffsetInBytes();

    // column
    column = IntegerAttr::get(i32ty, col);

    // arg_idx
    RuntimeSequenceOp seq_op = op->getParentOfType<RuntimeSequenceOp>();
    assert(seq_op &&
           "NpuDmaMemcpyNdOp must be inside a RuntimeSequenceOp; "
           "verify() should have ensured this.");
    Block &entryBB = seq_op.getBody().front();
    int arg_idx = -1;
    for (int i = 0, e = entryBB.getNumArguments(); i < e; i++) {
      if (entryBB.getArgument(i) == memref) {
        arg_idx = i;
        break;
      }
    }
    if (arg_idx < 0) return failure();

    // bd_id
    bd_id = IntegerAttr::get(i32ty, op.getId());

    // buffer_length
    buffer_length = IntegerAttr::get(i32ty, sizes[2] * sizes[1] * sizes[0]);

    // buffer_offset
    buffer_offset = IntegerAttr::get(i32ty, offset);

    // d0_size
    if (strides[1]) d0_size = IntegerAttr::get(i32ty, sizes[0]);

    // d0_stride
    if (strides[0]) d0_stride = IntegerAttr::get(i32ty, strides[0]);

    // d1_size
    if (strides[2]) d1_size = IntegerAttr::get(i32ty, sizes[1]);

    // d1_stride
    if (strides[1]) d1_stride = IntegerAttr::get(i32ty, strides[1]);

    // d2_size
    if (strides[3]) d2_size = IntegerAttr::get(i32ty, sizes[2]);

    // d2_stride
    if (strides[2]) d2_stride = IntegerAttr::get(i32ty, strides[2]);

    // iteration_size
    if (strides[3]) iteration_size = IntegerAttr::get(i32ty, sizes[3]);

    // iteration_stride
    if (strides[3]) iteration_stride = IntegerAttr::get(i32ty, strides[3]);

    // valid_bd
    valid_bd = IntegerAttr::get(i32ty, 1);

    // repeat_count
    repeat_count = IntegerAttr::get(i32ty, sizes[3]);

    // enable_packet
    if (auto packetInfo = op.getPacket()) {
      enable_packet = IntegerAttr::get(i32ty, 1);
      packet_type = IntegerAttr::get(i32ty, packetInfo->getPktType());
      packet_id = IntegerAttr::get(i32ty, packetInfo->getPktId());
    }

    // Set the issue_token
    issue_token = BoolAttr::get(ctx, op.getIssueToken());
    // Earlier, all S2MM channels were implicitly assumed to issue a token.
    // This logic is kept for now for backward compatibility.
    if (!isMM2S) issue_token = BoolAttr::get(ctx, true);

    // d0_zero_before
    d0_zero_before = IntegerAttr::get(i32ty, op.getD0ZeroBefore());

    // d1_zero_before
    d1_zero_before = IntegerAttr::get(i32ty, op.getD1ZeroBefore());

    // d2_zero_before
    d2_zero_before = IntegerAttr::get(i32ty, op.getD2ZeroBefore());

    // d0_zero_after
    d0_zero_after = IntegerAttr::get(i32ty, op.getD0ZeroAfter());

    // d1_zero_after
    d1_zero_after = IntegerAttr::get(i32ty, op.getD1ZeroAfter());

    // d2_zero_after
    d2_zero_after = IntegerAttr::get(i32ty, op.getD2ZeroAfter());

    rewriter.create<xilinx::AIEX::NpuWriteBdOp>(
        op->getLoc(), column, bd_id, buffer_length, buffer_offset,
        enable_packet, out_of_order_id, packet_id, packet_type, d0_size,
        d0_stride, d1_size, d1_stride, d2_size, d2_stride, iteration_current,
        iteration_size, iteration_stride, next_bd, row, use_next_bd, valid_bd,
        lock_rel_val, lock_rel_id, lock_acq_enable, lock_acq_val, lock_acq_id,
        d0_zero_before, d1_zero_before, d2_zero_before, d0_zero_after,
        d1_zero_after, d2_zero_after);

    AMDAIEDeviceModel tm =
        getDeviceModel(static_cast<AMDAIEDevice>(dev.getDevice()));

    uint32_t addr =
        (col << tm.getColumnShift()) | (0x1D004 + op.getId() * 0x20);
    rewriter.create<xilinx::AIEX::NpuAddressPatchOp>(op->getLoc(), addr, arg_idx, offset);

    rewriter.create<NpuPushQueueOp>(
        op->getLoc(), column, row, infoOp->getChannelDirAttr(),
        infoOp->getChannelIndexAttr(), issue_token, repeat_count, bd_id);

    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert NpuDmaWaitOp into NpuSyncOp by retrieving the necessary
/// information from the ShimDMAAllocationOp referenced through the
/// symbol argument of this op.
struct DmaWaitToNpuPattern : OpConversionPattern<xilinx::AIEX::NpuDmaWaitOp> {
 private:
  ShimDMAllocationGetter &allocGetter;

 public:
  using OpConversionPattern::OpConversionPattern;

  DmaWaitToNpuPattern(MLIRContext *context, ShimDMAllocationGetter &getter,
                      PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), allocGetter(getter) {}

  LogicalResult matchAndRewrite(
      xilinx::AIEX::NpuDmaWaitOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    AIE::DeviceOp dev = op->getParentOfType<AIE::DeviceOp>();
    if (!dev) return op->emitError("couldn't find parent of type DeviceOp");

    std::optional<AIE::ShimDMAAllocationOp> shimDmaAllocOp =
        allocGetter.get(dev, op.getSymbol());
    if (!shimDmaAllocOp) {
      return op->emitError("couldn't find shim_dma_allocation op");
    }

    // Create with `column_num == 1` and `row_num == 1` to check for a single
    // column and row. Row is always 0 for shim tiles.
    (void)rewriter.replaceOpWithNewOp<NpuSyncOp>(
        op, shimDmaAllocOp->getCol(), /* row */ 0,
        static_cast<uint32_t>(shimDmaAllocOp->getChannelDir()),
        shimDmaAllocOp->getChannelIndex(), 1, 1);
    return success();
  }
};

namespace mlir::iree_compiler::AMDAIE {
struct AMDAIEDmaToNpuPass : mlir::OperationPass<DeviceOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AMDAIEDmaToNpuPass)

  AMDAIEDmaToNpuPass() : mlir::OperationPass<DeviceOp>(resolveTypeID()) {}

  llvm::StringRef getArgument() const override { return "amdaie-dma-to-npu"; }

  llvm::StringRef getName() const override { return "AMDAIEDmaToNpuPass"; }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    return std::make_unique<AMDAIEDmaToNpuPass>(
        *static_cast<const AMDAIEDmaToNpuPass *>(this));
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<xilinx::AIE::AIEDialect>();
    registry.insert<xilinx::AIEX::AIEXDialect>();
  }

  void runOnOperation() override {
    // Lower NpuDmaMemcpyNdOp to "NpuWriteBdOp + NpuAddressPatchOp +
    // NpuPushQueueOp".
    // Lower NpuDmaWaitOp to NpuSyncOp.
    ShimDMAllocationGetter cachingGetter;

    AIE::DeviceOp device = getOperation();

    ConversionTarget target(getContext());
    target.addLegalDialect<AIEXDialect>();
    target.addLegalOp<AIE::BufferOp>();
    target.addLegalOp<AIE::ShimDMAAllocationOp>();
    target.addIllegalOp<NpuDmaMemcpyNdOp>();
    target.addIllegalOp<xilinx::AIEX::NpuDmaWaitOp>();

    RewritePatternSet patterns(&getContext());
    patterns.insert<DmaToNpuPattern>(&getContext(), cachingGetter);
    patterns.insert<DmaWaitToNpuPattern>(&getContext(), cachingGetter);

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      signalPassFailure();

    // Convert NpuWriteBdOp, NpuAddressPatchOp, NpuPushQueueOp, and NpuSyncOp to
    // transactions using aie-rt.
    patterns.clear();
    target.addIllegalOp<NpuSyncOp>();
    target.addIllegalOp<NpuPushQueueOp>();
    target.addIllegalOp<xilinx::AIEX::NpuAddressPatchOp>();
    target.addIllegalOp<xilinx::AIEX::NpuWriteBdOp>();

    TransactionBuilder transactionBuilder(
        getDeviceModel(static_cast<AMDAIEDevice>(device.getDevice())));
    transactionBuilder.clearAndInitialize();

    patterns.insert<ConvertNpuOp<NpuSyncOp>>(&getContext(), appendSync,
                                             transactionBuilder);
    patterns.insert<ConvertNpuOp<NpuPushQueueOp>>(
        &getContext(), appendPushToQueue, transactionBuilder);
    patterns.insert<ConvertNpuOp<xilinx::AIEX::NpuAddressPatchOp>>(
        &getContext(), appendAddressPatch, transactionBuilder);
    patterns.insert<ConvertNpuOp<xilinx::AIEX::NpuWriteBdOp>>(&getContext(), appendWriteBd,
                                                transactionBuilder);

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      signalPassFailure();

    // Finalize the instructions and set as an attribute.
    ArrayRef<uint32_t> instructions =
        transactionBuilder.finalizeAndReturnInstructions();
    device->setAttr(
        "npu_instructions",
        DenseUI32ResourceElementsAttr::get(
            RankedTensorType::get(
                instructions.size(),
                IntegerType::get(&getContext(), 32, IntegerType::Unsigned)),
            "npu_instructions",
            HeapAsmResourceBlob::allocateAndCopyInferAlign(instructions)));

    SmallVector<RuntimeSequenceOp> seqOps;
    device->walk([&](RuntimeSequenceOp seqOp) { seqOps.push_back(seqOp); });

    if (seqOps.size() > 1) {
      device->emitOpError("has ")
          << seqOps.size()
          << " aiex.runtime_sequence ops. Expected no more than 1.";
      signalPassFailure();
    }

    if (seqOps.size() == 1) {
      auto seqOp = seqOps[0];
      StringRef name = seqOp.getSymName().value();
      device->setAttr("runtime_sequence_name",
                      StringAttr::get(&getContext(), name));
      seqOp.erase();
    }
  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>> createAMDAIEDmaToNpuPass() {
  return std::make_unique<AMDAIEDmaToNpuPass>();
}

void registerAMDAIEDmaToNpu() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createAMDAIEDmaToNpuPass();
  });
}

}  // namespace mlir::iree_compiler::AMDAIE
