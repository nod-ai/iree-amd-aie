//===- AMDAIEDmaToNpu.cpp ------------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//===- AIETargetNPU.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <vector>

#include "AIETargets.h"
#include "Passes.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Format.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

#define TXN_OPC_WRITE 0x0
#define TXN_OPC_BLOCKWRITE 0x1
#define TXN_OPC_TCT 0x80
#define TXN_OPC_DDR_PATCH 0x81

namespace {

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
  const AIETargetModel &tm = op->getParentOfType<DeviceOp>().getTargetModel();

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
  const AIETargetModel &tm = op->getParentOfType<DeviceOp>().getTargetModel();

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

}  // namespace

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

namespace {
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
}  // namespace

struct PushToNpuPattern : OpConversionPattern<NpuPushQueueOp> {
 public:
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
    rewriter.create<NpuWrite32Op>(op->getLoc(), queue_offset, cmd, column, row);
    rewriter.eraseOp(op);
    return success();
  }
};

struct DmaToNpuPattern : OpConversionPattern<NpuDmaMemcpyNdOp> {
  using OpConversionPattern::OpConversionPattern;

 private:
  ShimDMAllocationGetter &allocGetter;

 public:
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
    if (!infoOp) {
      return op->emitOpError("couldn't find shim_dma_allocation op.");
    }

    auto channelDir = infoOp->getChannelDir();
    bool isMM2S = channelDir == AIE::DMAChannelDir::MM2S;
    int col = infoOp->getCol();

    // initialize fields to zero
    auto column = zero;
    auto ddr_id = zero;
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

    llvm::SmallVector<int64_t, 3> strides = llvm::map_to_vector(
        llvm::reverse(op.getMixedStrides()),
        [](OpFoldResult s) { return getConstantIntValue(s).value(); });
    llvm::SmallVector<int64_t, 4> sizes = llvm::map_to_vector(
        llvm::reverse(op.getMixedSizes()),
        [](OpFoldResult s) { return getConstantIntValue(s).value(); });
    llvm::SmallVector<int64_t, 4> offsets = llvm::map_to_vector(
        llvm::reverse(op.getMixedOffsets()),
        [](OpFoldResult s) { return getConstantIntValue(s).value(); });

    // column
    column = IntegerAttr::get(i32ty, col);

    // ddr_id
    Block &entryBB = op->getParentOfType<func::FuncOp>().getBody().front();
    int arg_idx = -1;
    for (int i = 0, e = entryBB.getNumArguments(); i < e; i++) {
      if (entryBB.getArgument(i) == memref) {
        arg_idx = i;
        break;
      }
    }
    if (arg_idx < 0) return failure();
    ddr_id = IntegerAttr::get(i32ty, arg_idx);

    // bd_id
    bd_id = IntegerAttr::get(i32ty, op.getId());

    // buffer_length
    buffer_length = IntegerAttr::get(i32ty, sizes[2] * sizes[1] * sizes[0]);

    // buffer_offset
    size_t stride = 1;
    size_t offset = 0;
    MemRefType my_memref = op.getMemref().getType();
    auto shape = my_memref.getShape();
    size_t R = shape.size();
    size_t el_bit_width = my_memref.getElementTypeBitWidth();
    assert(el_bit_width % 8 == 0 &&
           "Expected Memref element bitwidth to be multiple of 8.");
    size_t S = el_bit_width / 8;
    for (size_t i = 0; i < R; i++) {
      offset += offsets[i] * stride * S;
      stride *= shape[R - i - 1];
    }
    buffer_offset = IntegerAttr::get(i32ty, offset);

    // enable_packet

    // out_of_order_id

    // packet_id

    // packet_type

    // d0_size
    if (strides[0]) d0_size = IntegerAttr::get(i32ty, sizes[0]);

    // d0_stride
    d0_stride = IntegerAttr::get(i32ty, 0);

    // d1_size
    if (strides[1]) d1_size = IntegerAttr::get(i32ty, sizes[1]);

    // d1_stride
    if (strides[0]) d1_stride = IntegerAttr::get(i32ty, strides[0] - 1);

    // d2_stride
    if (strides[1]) d2_stride = IntegerAttr::get(i32ty, strides[1] - 1);

    // iteration_current

    // iteration_size
    if (strides[2]) iteration_size = IntegerAttr::get(i32ty, sizes[3] - 1);

    // iteration_stride
    if (strides[2]) iteration_stride = IntegerAttr::get(i32ty, strides[2] - 1);

    // next_bd

    // use_next_bd

    // valid_bd
    valid_bd = IntegerAttr::get(i32ty, 1);

    // lock_rel_val

    // lock_rel_id

    // lock_acq_enable

    // lock_acq_val

    // lock_acq_id

    // repeat_count
    repeat_count = IntegerAttr::get(i32ty, sizes[3] - 1);

    // Set the issue_token
    issue_token = BoolAttr::get(ctx, op.getIssueToken());
    // Earlier, all S2MM channels were implicitly assumed to issue a token.
    // This logic is kept for now for backward compatibility.
    if (!isMM2S) issue_token = BoolAttr::get(ctx, true);

    rewriter.create<NpuWriteBdOp>(
        op->getLoc(), column, ddr_id, bd_id, buffer_length, buffer_offset,
        enable_packet, out_of_order_id, packet_id, packet_type, d0_size,
        d0_stride, d1_size, d1_stride, d2_stride, iteration_current,
        iteration_size, iteration_stride, next_bd, row, use_next_bd, valid_bd,
        lock_rel_val, lock_rel_id, lock_acq_enable, lock_acq_val, lock_acq_id);

    const AIE::AIETargetModel &tm =
        op->getParentOfType<AIE::DeviceOp>().getTargetModel();

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

struct AIEDmaToNpuPass : xilinx::AIEX::impl::AIEDmaToNpuBase<AIEDmaToNpuPass> {
  void runOnOperation() override {
    ShimDMAllocationGetter cachingGetter;

    AIE::DeviceOp device = getOperation();

    ConversionTarget target(getContext());
    target.addLegalDialect<AIEXDialect>();
    target.addLegalOp<AIE::BufferOp>();
    target.addLegalOp<AIE::ShimDMAAllocationOp>();
    target.addIllegalOp<NpuWriteRTPOp>();
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
    std::vector<int32_t> signedInstructions(instructions.begin(),
                                            instructions.end());
    device->setAttr("npu_instructions",
                    DenseI32ArrayAttr::get(&getContext(), signedInstructions));
  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>> AIEX::createAIEDmaToNpuPass() {
  return std::make_unique<AIEDmaToNpuPass>();
}