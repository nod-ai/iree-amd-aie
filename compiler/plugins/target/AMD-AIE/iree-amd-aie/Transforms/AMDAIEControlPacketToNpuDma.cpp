// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-control-packet-to-half-dma-cpy-nd"

namespace mlir::iree_compiler::AMDAIE {

namespace {

struct ControlPacketDmaBuilder {
  AMDAIE::AMDAIEDeviceModel deviceModel;
  ControlPacketDmaBuilder(AMDAIE::AMDAIEDeviceModel deviceModel)
      : deviceModel(std::move(deviceModel)) {}

  std::vector<uint32_t> ctrlPktSequence;

  llvm::MutableArrayRef<uint32_t> reserveAndGetTail(size_t tailSize) {
    size_t oldSize = ctrlPktSequence.size();
    size_t newSize = oldSize + tailSize;
    ctrlPktSequence.resize(newSize, 0);
    return llvm::MutableArrayRef<uint32_t>(ctrlPktSequence.data() + oldSize,
                                           tailSize);
  }

  void dumpSequenceAsHex() const {
    llvm::outs() << "Control Packet Sequence: \n";
    // Write hex as 0xXXXXXXXX
    for (uint32_t word : ctrlPktSequence)
      llvm::outs() << utohexstr(word, 8) << "\n";
  }

  LogicalResult convert(IRRewriter &rewriter, AMDAIE::WorkgroupOp workgroupOp) {
    ctrlPktSequence.clear();

    // Get all the `ConnectionOp` whose target is a `CTRL` port.
    DenseMap<TileLoc, AMDAIE::ConnectionOp> tileLocToCtrlConnect;
    DenseMap<TileLoc, AMDAIE::TileOp> tileLocToTileOp;
    auto res = workgroupOp->walk([&](AMDAIE::ConnectionOp connectionOp) {
      for (Value target : connectionOp.getTargetChannels()) {
        AMDAIE::ChannelOp targetChannelOp =
            dyn_cast<AMDAIE::ChannelOp>(target.getDefiningOp());
        if (!targetChannelOp) {
          connectionOp.emitOpError() << "expected a `amdaie.channel` op target";
          return WalkResult::interrupt();
        }
        if (targetChannelOp.getPortType() == StrmSwPortType::CTRL) {
          TileOp tileOp = targetChannelOp.getTileOp();
          TileLoc tileLoc = {
              static_cast<int>(getConstantIndexOrAssert(tileOp.getCol())),
              static_cast<int>(getConstantIndexOrAssert(tileOp.getRow()))};
          tileLocToCtrlConnect[tileLoc] = connectionOp;
          tileLocToTileOp[tileLoc] = tileOp;
        }
      }
      return WalkResult::advance();
    });
    if (res.wasInterrupted()) return failure();

    std::vector<AMDAIE::NpuControlPacketOp> ctrlPktOps;
    // Convert `NpuControlPacketOp` to `NpuDmaCpyNdOp` + `NpuDmaWaitOp`.
    res = workgroupOp->walk([&](AMDAIE::NpuControlPacketOp ctrlPktOp) {
      ctrlPktOps.push_back(ctrlPktOp);
      // Get `ConnectionOp` for the `CTRL` port.
      uint32_t address = ctrlPktOp.getAddress();
      uint32_t addrOffset = deviceModel.getOffsetFromAddress(address);
      int32_t col = deviceModel.getColumnFromAddress(address);
      int32_t row = deviceModel.getRowFromAddress(address);
      if (!tileLocToCtrlConnect.count({col, row})) {
        ctrlPktOp.emitOpError()
            << "tries to write to tile (col=" << col << ", row=" << row
            << "), but it's `CTRL` port is not routed.";
        return WalkResult::interrupt();
      }
      AMDAIE::ConnectionOp connectionOp = tileLocToCtrlConnect[{col, row}];

      // Get the source offsets, sizes, and strides.
      uint32_t dataLength = ctrlPktOp.getLength();
      int64_t headerAndDataLength = dataLength + 1;
      SmallVector<int64_t> dmaSourceOffsets{
          0, 0, 0, static_cast<long>(ctrlPktSequence.size())};
      SmallVector<int64_t> dmaSourceSizes{1, 1, 1, headerAndDataLength};
      SmallVector<int64_t> dmaSourceStrides{0, 0, 0, 1};
      // Target offsets, sizes, and strides are left empty.
      SmallVector<int64_t> dmaTargetOffsets;
      SmallVector<int64_t> dmaTargetSizes;
      SmallVector<int64_t> dmaTargetStrides;

      // Store the control packet header.
      llvm::MutableArrayRef<uint32_t> words =
          reserveAndGetTail(headerAndDataLength);
      FailureOr<uint32_t> header = deviceModel.getCtrlPktHeader(
          addrOffset, dataLength, static_cast<uint32_t>(ctrlPktOp.getOpcode()),
          ctrlPktOp.getStreamId());
      if (failed(header)) {
        ctrlPktOp.emitOpError() << "failed to get control packet header.";
        return WalkResult::interrupt();
      }

      words[0] = *header;
      // Store the control packet data.
      std::optional<ArrayRef<int32_t>> maybeData =
          ctrlPktOp.getDataFromArrayOrResource();
      if (maybeData.has_value()) {
        for (uint32_t i = 0; i < dataLength; ++i) {
          int32_t data = maybeData.value()[i];
          words[i + 1] = reinterpret_cast<uint32_t &>(data);
        }
      }

      rewriter.setInsertionPoint(ctrlPktOp);
      // Create token.
      SmallVector<Type> resultTypes = {
          rewriter.getType<AMDAIE::AsyncSourceTokenType>()};
      TypeRange sourceResultTypes = TypeRange{resultTypes};

      // Create `NpuDmaCpyNdOp` and `NpuDmaWaitOp`.
      auto dmaOp = rewriter.create<AMDAIE::NpuDmaCpyNdOp>(
          rewriter.getUnknownLoc(), sourceResultTypes, connectionOp, nullptr,
          dmaTargetOffsets, dmaTargetSizes, dmaTargetStrides,
          /*target_bd_id=*/nullptr, connectionOp.getSource(), dmaSourceOffsets,
          dmaSourceSizes, dmaSourceStrides, /*source_bd_id=*/nullptr);
      rewriter.create<AMDAIE::NpuDmaWaitOp>(rewriter.getUnknownLoc(),
                                            dmaOp.getResult(0));

      return WalkResult::advance();
    });
    if (res.wasInterrupted()) return failure();

    // Erase all the `NpuControlPacketOp`.
    for (AMDAIE::NpuControlPacketOp ctrlPktOp : ctrlPktOps)
      rewriter.eraseOp(ctrlPktOp);

    // Store the control packet sequence in the `WorkgroupOp`.
    workgroupOp.setCtrlpktSequenceAttr(DenseUI32ResourceElementsAttr::get(
        RankedTensorType::get(
            ctrlPktSequence.size(),
            IntegerType::get(rewriter.getContext(), 32, IntegerType::Unsigned)),
        "ctrlpkt_sequence",
        HeapAsmResourceBlob::allocateAndCopyInferAlign(
            ArrayRef<uint32_t>(ctrlPktSequence))));
    return success();
  }
};

class AMDAIEControlPacketToNpuDmaPass
    : public impl::AMDAIEControlPacketToNpuDmaBase<
          AMDAIEControlPacketToNpuDmaPass> {
 public:
  AMDAIEControlPacketToNpuDmaPass(
      const AMDAIEControlPacketToNpuDmaOptions &options)
      : AMDAIEControlPacketToNpuDmaBase(options) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  void runOnOperation() override;
};

void AMDAIEControlPacketToNpuDmaPass::runOnOperation() {
  Operation *parentOp = getOperation();
  IRRewriter rewriter(parentOp->getContext());

  // Get `AMDAIEDeviceModel`.
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(parentOp);
  std::optional<AMDAIEDevice> maybeDevice = getConfigAMDAIEDevice(targetAttr);
  if (!maybeDevice) {
    parentOp->emitOpError() << "has no AMDAIEDevice in the target "
                               "attribute configuration.";
    return signalPassFailure();
  }
  AMDAIE::AMDAIEDeviceModel deviceModel =
      AMDAIE::getDeviceModel(maybeDevice.value());
  ControlPacketDmaBuilder ctrlPktDmaBuilder(std::move(deviceModel));

  SmallVector<AMDAIE::WorkgroupOp> workgroupOps;

  WalkResult res = parentOp->walk([&](AMDAIE::WorkgroupOp workgroupOp) {
    if (failed(ctrlPktDmaBuilder.convert(rewriter, workgroupOp)))
      return WalkResult::interrupt();

    if (dumpSequence) ctrlPktDmaBuilder.dumpSequenceAsHex();

    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEControlPacketToNpuDmaPass(
    AMDAIEControlPacketToNpuDmaOptions options) {
  return std::make_unique<AMDAIEControlPacketToNpuDmaPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
