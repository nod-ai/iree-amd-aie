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
    auto oldSize = ctrlPktSequence.size();
    auto newSize = oldSize + tailSize;
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
      if (connectionOp.getTargetChannels().size() != 1) {
        connectionOp.emitOpError() << "expected a single target channel";
        return WalkResult::interrupt();
      }

      auto targetChannelOp = dyn_cast<AMDAIE::ChannelOp>(
          connectionOp.getTargetChannels()[0].getDefiningOp());
      if (targetChannelOp.getPortType() == StrmSwPortType::CTRL) {
        TileOp tileOp = targetChannelOp.getTileOp();
        TileLoc tileLoc = {
            static_cast<int>(getConstantIndexOrAssert(tileOp.getCol())),
            static_cast<int>(getConstantIndexOrAssert(tileOp.getRow()))};
        tileLocToCtrlConnect[tileLoc] = connectionOp;
        tileLocToTileOp[tileLoc] = tileOp;
      }
      return WalkResult::advance();
    });
    if (res.wasInterrupted()) return failure();

    std::vector<AMDAIE::NpuControlPacketOp> ctrlPktOps;
    // Convert `NpuControlPacketOp` to `NpuHalfDmaCpyNdOp` + `NpuDmaWaitOp`.
    res = workgroupOp->walk([&](AMDAIE::NpuControlPacketOp ctrlPktOp) {
      ctrlPktOps.push_back(ctrlPktOp);
      // Get `ConnectionOp` for the `CTRL` port.
      uint32_t colShift = deviceModel.getColumnShift();
      uint32_t rowShift = deviceModel.getRowShift();
      llvm::errs() << "colShift: " << colShift << "rowShift: " << rowShift
                   << "\n";
      uint32_t address = ctrlPktOp.getAddress() & 0xFFFFF;
      int32_t col = (ctrlPktOp.getAddress() >> colShift) & 0x1F;
      int32_t row = (ctrlPktOp.getAddress() >> rowShift) & 0x1F;
      if (!tileLocToCtrlConnect.count({col, row})) {
        ctrlPktOp.emitOpError()
            << "tries to write to tile (col=" << col << ", row=" << row
            << "), but it's `CTRL` port is not routed.";
        return WalkResult::interrupt();
      }
      AMDAIE::ConnectionOp connectionOp = tileLocToCtrlConnect[{col, row}];

      // Get `sourceChannelOp`.
      if (connectionOp.getSourceChannels().size() != 1) {
        connectionOp.emitOpError() << "expected a single source channel";
        return WalkResult::interrupt();
      }
      auto sourceChannelOp = dyn_cast<AMDAIE::ChannelOp>(
          connectionOp.getSourceChannels()[0].getDefiningOp());

      // Get `offsets`, `sizes`, and `strides`.
      uint32_t dataLength = ctrlPktOp.getLength();
      int64_t headerAndDataLength = dataLength + 1;
      SmallVector<int64_t> offsets{0, 0, 0,
                                   static_cast<long>(ctrlPktSequence.size())};
      SmallVector<int64_t> sizes{1, 1, 1, headerAndDataLength};
      SmallVector<int64_t> strides{0, 0, 0, 1};

      // Store the control packet header and data.
      llvm::MutableArrayRef<uint32_t> words =
          reserveAndGetTail(headerAndDataLength);
      // Subtract 1 from `dataLength` because the length `i` is encoded in the
      // header as `i - 1`.
      words[0] = deviceModel.getCtrlPktHeader(
          address, dataLength - 1, static_cast<uint32_t>(ctrlPktOp.getOpcode()),
          ctrlPktOp.getStreamId());
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
          rewriter.getType<AMDAIE::AsyncTokenType>()};
      TypeRange sourceResultTypes = TypeRange{resultTypes};

      // Get `bdId`, use `0` for now.
      // TODO (zhewen): let `AMDAIEAssignNpuDmaBdIdsPass` decide?
      auto constant = rewriter.create<arith::ConstantOp>(
          rewriter.getUnknownLoc(), rewriter.getIndexAttr(0));
      auto bdIdOp = rewriter.create<AMDAIE::BdIdOp>(rewriter.getUnknownLoc(),
                                                    sourceChannelOp.getTileOp(),
                                                    constant.getResult());

      // Create `NpuHalfDmaCpyNdOp` and `NpuDmaWaitOp`.
      auto dmaOp = rewriter.create<AMDAIE::NpuHalfDmaCpyNdOp>(
          rewriter.getUnknownLoc(), sourceResultTypes, connectionOp,
          connectionOp.getSource(), offsets, sizes, strides, bdIdOp,
          sourceChannelOp);
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

class AMDAIEControlPacketToHalfDmaCpyNdPass
    : public impl::AMDAIEControlPacketToHalfDmaCpyNdBase<
          AMDAIEControlPacketToHalfDmaCpyNdPass> {
 public:
  AMDAIEControlPacketToHalfDmaCpyNdPass(
      const AMDAIEControlPacketToHalfDmaCpyNdOptions &options)
      : AMDAIEControlPacketToHalfDmaCpyNdBase(options) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  void runOnOperation() override;
};

void AMDAIEControlPacketToHalfDmaCpyNdPass::runOnOperation() {
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

std::unique_ptr<Pass> createAMDAIEControlPacketToHalfDmaCpyNdPass(
    AMDAIEControlPacketToHalfDmaCpyNdOptions options) {
  return std::make_unique<AMDAIEControlPacketToHalfDmaCpyNdPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
