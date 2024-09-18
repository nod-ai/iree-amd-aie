// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AIEDialect.h"
#include "AIEXDialect.h"
#include "Passes.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
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

#define TXN_OPC_WRITE 0x0
#define TXN_OPC_BLOCKWRITE 0x1
#define TXN_OPC_TCT 0x80
#define TXN_OPC_DDR_PATCH 0x81

// Example:
// - instructions = {3,4,5}
// - tailSize = 2
// instructions becomes {3,4,5,0,0} and
// a mutable reference to the tail {0,0} is returned.
llvm::MutableArrayRef<uint32_t> reserveAndGetTail(
    std::vector<uint32_t> &instructions, uint64_t tailSize) {
  auto oldSize = instructions.size();
  auto newSize = oldSize + tailSize;
  instructions.resize(newSize, 0);
  return llvm::MutableArrayRef<uint32_t>(instructions.data() + oldSize,
                                         tailSize);
}

void appendSync(std::vector<uint32_t> &instructions, NpuSyncOp op) {
  auto words = reserveAndGetTail(instructions, 4);

  // XAIE_IO_CUSTOM_OP_TCT
  words[0] = TXN_OPC_TCT;

  words[1] = words.size() * sizeof(uint32_t);  // Operation Size

  words[2] |= static_cast<uint32_t>(op.getDirection()) & 0xff;
  words[2] |= (op.getRow() & 0xff) << 8;
  words[2] |= (op.getColumn() & 0xff) << 16;

  words[3] |= (op.getRowNum() & 0xff) << 8;
  words[3] |= (op.getColumnNum() & 0xff) << 16;
  words[3] |= (op.getChannel() & 0xff) << 24;
}

void appendWrite32(std::vector<uint32_t> &instructions, NpuWrite32Op op) {
  auto words = reserveAndGetTail(instructions, 6);
  DeviceOp device = op->getParentOfType<DeviceOp>();
  AMDAIEDeviceModel tm =
      getDeviceModel(static_cast<AMDAIEDevice>(device.getDevice()));

  // XAIE_IO_WRITE
  words[0] = TXN_OPC_WRITE;
  words[1] = 0;
  words[2] = op.getAddress();
  auto col = op.getColumn();
  auto row = op.getRow();
  if (col && row)
    words[2] = ((*col & 0xff) << tm.getColumnShift()) |
               ((*row & 0xff) << tm.getRowShift()) | (words[2] & 0xFFFFF);
  words[3] = 0;
  words[4] = op.getValue();                    // Value
  words[5] = words.size() * sizeof(uint32_t);  // Operation Size
}

void appendAddressPatch(std::vector<uint32_t> &instructions,
                        NpuAddressPatchOp op) {
  auto words = reserveAndGetTail(instructions, 12);

  // XAIE_IO_CUSTOM_OP_DDR_PATCH
  words[0] = TXN_OPC_DDR_PATCH;
  words[1] = words.size() * sizeof(uint32_t);  // Operation Size

  words[6] = op.getAddr();
  words[7] = 0;

  words[8] = op.getArgIdx();
  words[9] = 0;

  words[10] = op.getArgPlus();
  words[11] = 0;
}

void appendWriteBdShimTile(std::vector<uint32_t> &instructions,
                           NpuWriteBdOp op) {
  auto words = reserveAndGetTail(instructions, 12);
  DeviceOp device = op->getParentOfType<DeviceOp>();
  AMDAIEDeviceModel tm =
      getDeviceModel(static_cast<AMDAIEDevice>(device.getDevice()));

  // XAIE_IO_BLOCKWRITE
  words[0] = TXN_OPC_BLOCKWRITE;
  words[1] = 0;

  // RegOff
  auto bd_id = op.getBdId();
  uint32_t bd_addr = (op.getColumn() << tm.getColumnShift()) |
                     (op.getRow() << tm.getRowShift()) |
                     (0x1D000 + bd_id * 0x20);
  words[2] = bd_addr;                          // ADDR
  words[3] = words.size() * sizeof(uint32_t);  // Operation Size

  // DMA_BDX_0
  words[4] = op.getBufferLength();

  // DMA_BDX_1
  words[5] = op.getBufferOffset();

  // DMA_BDX_2
  // En Packet , OoO BD ID , Packet ID , Packet Type
  words[6] |= (op.getEnablePacket() & 0x1) << 30;
  words[6] |= (op.getOutOfOrderId() & 0x3f) << 24;
  words[6] |= (op.getPacketId() & 0x1f) << 19;
  words[6] |= (op.getPacketType() & 0x7) << 16;

  // DMA_BDX_3
  // TODO: Secure Access
  words[7] |= (op.getD0Size() & 0x3ff) << 20;
  words[7] |= op.getD0Stride() & 0xfffff;

  // DMA_BDX_4
  words[8] = 0x80000000;  // burst length;
  words[8] |= (op.getD1Size() & 0x3ff) << 20;
  words[8] |= op.getD1Stride() & 0xfffff;

  // DMA_BDX_5
  // TODO: SIMID, AxCache, AXQoS
  words[9] = op.getD2Stride() & 0xfffff;

  // DMA_BDX_6
  words[10] |= (op.getIterationCurrent() & 0x3f) << 26;
  words[10] |= (op.getIterationSize() & 0x3f) << 20;
  words[10] |= op.getIterationStride() & 0xfffff;

  // DMA_BDX_7
  // TODO: TLAST Suppress
  words[11] |= (op.getNextBd() & 0xf) << 27;
  words[11] |= (op.getUseNextBd() & 0x1) << 26;
  words[11] |= (op.getValidBd() & 0x1) << 25;
  words[11] |= (op.getLockRelVal() & 0xef) << 18;
  words[11] |= (op.getLockRelId() & 0xf) << 13;
  words[11] |= (op.getLockAcqEnable() & 0x1) << 12;
  words[11] |= (op.getLockAcqVal() & 0xef) << 5;
  words[11] |= op.getLockAcqId() & 0xf;
}

template <typename SourceOp>
class ConvertNpuOp : public OpConversionPattern<SourceOp> {
 public:
  std::vector<uint32_t> &instructions;
  using AppenderTy = function_ref<void(std::vector<uint32_t> &, SourceOp)>;
  AppenderTy appender;
  uint32_t &count;
  ConvertNpuOp(MLIRContext *ctx, std::vector<uint32_t> &instructions,
               AppenderTy appender, uint32_t &count)
      : OpConversionPattern<SourceOp>(ctx),
        instructions(instructions),
        appender(appender),
        count(count) {}

  LogicalResult matchAndRewrite(
      SourceOp op, typename SourceOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    appender(instructions, op);
    rewriter.eraseOp(op);
    count++;
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

struct PushToNpuPattern : OpConversionPattern<NpuPushQueueOp> {
  using OpConversionPattern::OpConversionPattern;

  PushToNpuPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit) {}

  LogicalResult matchAndRewrite(
      NpuPushQueueOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // the offset of the task queue register in the tile
    uint32_t queue_offset;
    if (op.getDirection() == AIE::DMAChannelDir::MM2S)
      queue_offset = 0x1D214;
    else
      queue_offset = 0x1D204;
    if (op.getChannel() == 1) queue_offset += 0x8;

    // the value to write
    uint32_t bd_id = op.getBdId();
    uint32_t repeat_cnt = op.getRepeatCount();
    uint32_t cmd = 0;
    cmd |= bd_id & 0xF;
    cmd |= (repeat_cnt & 0xFF) << 16;
    if (op.getIssueToken()) cmd |= 0x80000000;

    auto i32ty = IntegerType::get(op->getContext(), 32);
    auto column = IntegerAttr::get(i32ty, op.getColumn());
    auto row = IntegerAttr::get(i32ty, 0);
    rewriter.create<NpuWrite32Op>(op->getLoc(), queue_offset, cmd, nullptr,
                                  column, row);
    rewriter.eraseOp(op);

    return success();
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
    if (strides[0]) d0_stride = IntegerAttr::get(i32ty, strides[0] - 1);

    // d1_size
    if (strides[2]) d1_size = IntegerAttr::get(i32ty, sizes[1]);

    // d1_stride
    if (strides[1]) d1_stride = IntegerAttr::get(i32ty, strides[1] - 1);

    // d2_stride
    if (strides[2]) d2_stride = IntegerAttr::get(i32ty, strides[2] - 1);

    // iteration_size
    if (strides[3]) iteration_size = IntegerAttr::get(i32ty, sizes[3] - 1);

    // iteration_stride
    if (strides[3]) iteration_stride = IntegerAttr::get(i32ty, strides[3] - 1);

    // valid_bd
    valid_bd = IntegerAttr::get(i32ty, 1);

    // repeat_count
    repeat_count = IntegerAttr::get(i32ty, sizes[3] - 1);

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

    rewriter.create<NpuWriteBdOp>(
        op->getLoc(), column, bd_id, buffer_length, buffer_offset,
        enable_packet, out_of_order_id, packet_id, packet_type, d0_size,
        d0_stride, d1_size, d1_stride, d2_stride, iteration_current,
        iteration_size, iteration_stride, next_bd, row, use_next_bd, valid_bd,
        lock_rel_val, lock_rel_id, lock_acq_enable, lock_acq_val, lock_acq_id);

    AMDAIEDeviceModel tm =
        getDeviceModel(static_cast<AMDAIEDevice>(dev.getDevice()));

    uint32_t addr =
        (col << tm.getColumnShift()) | (0x1D004 + op.getId() * 0x20);
    rewriter.create<NpuAddressPatchOp>(op->getLoc(), addr, arg_idx, offset);

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
struct DmaWaitToNpuPattern : OpConversionPattern<NpuDmaWaitOp> {
 private:
  ShimDMAllocationGetter &allocGetter;

 public:
  using OpConversionPattern::OpConversionPattern;

  DmaWaitToNpuPattern(MLIRContext *context, ShimDMAllocationGetter &getter,
                      PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), allocGetter(getter) {}

  LogicalResult matchAndRewrite(
      NpuDmaWaitOp op, OpAdaptor adaptor,
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
    ShimDMAllocationGetter cachingGetter;

    AIE::DeviceOp device = getOperation();

    ConversionTarget target(getContext());
    target.addLegalDialect<AIEXDialect>();
    target.addLegalOp<AIE::BufferOp>();
    target.addLegalOp<AIE::ShimDMAAllocationOp>();
    target.addIllegalOp<NpuDmaMemcpyNdOp>();
    target.addIllegalOp<NpuDmaWaitOp>();
    target.addIllegalOp<NpuPushQueueOp>();

    RewritePatternSet patterns(&getContext());
    patterns.insert<DmaToNpuPattern>(&getContext(), cachingGetter);
    patterns.insert<DmaWaitToNpuPattern>(&getContext(), cachingGetter);
    patterns.insert<PushToNpuPattern>(&getContext());

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      signalPassFailure();

    patterns.clear();
    target.addIllegalOp<NpuSyncOp>();
    target.addIllegalOp<NpuWrite32Op>();
    target.addIllegalOp<NpuAddressPatchOp>();
    target.addIllegalOp<NpuWriteBdOp>();

    std::vector<uint32_t> instructions;
    auto words = reserveAndGetTail(instructions, 4);

    // setup txn header
    words[0] = 0x06030100;
    words[1] = 0x00000105;
    uint32_t count = 0;

    patterns.insert<ConvertNpuOp<NpuSyncOp>>(&getContext(), instructions,
                                             appendSync, count);
    patterns.insert<ConvertNpuOp<NpuWrite32Op>>(&getContext(), instructions,
                                                appendWrite32, count);
    patterns.insert<ConvertNpuOp<NpuAddressPatchOp>>(
        &getContext(), instructions, appendAddressPatch, count);
    patterns.insert<ConvertNpuOp<NpuWriteBdOp>>(&getContext(), instructions,
                                                appendWriteBdShimTile, count);

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      signalPassFailure();

    instructions[2] = count;
    instructions[3] = instructions.size() * sizeof(uint32_t);

    ArrayRef<uint32_t> instsArrRef(instructions.data(), instructions.size());
    device->setAttr(
        "npu_instructions",
        DenseUI32ResourceElementsAttr::get(
            RankedTensorType::get(
                instsArrRef.size(),
                IntegerType::get(&getContext(), 32, IntegerType::Unsigned)),
            "npu_instructions",
            HeapAsmResourceBlob::allocateAndCopyInferAlign(instsArrRef)));

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
