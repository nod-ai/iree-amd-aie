// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEDmaUtils.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-control-packet-to-half-dma-cpy-nd"

namespace mlir::iree_compiler::AMDAIE {

namespace {

struct CtrlPktBdTransfer {
  AMDAIE::ConnectionOp connectionOp;
  SmallVector<int64_t> offsets;
  SmallVector<int64_t> sizes;
  SmallVector<int64_t> strides;
  bool syncBarrier;
};

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
          connectionOp.emitOpError()
              << "expected an `amdaie.channel` op target";
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
    std::vector<CtrlPktBdTransfer> ctrlPktBdTransfers;
    AMDAIE::AMDAIETileType lastDestTileType = AMDAIE::AMDAIETileType::MAX;
    // Convert `NpuControlPacketOp` to BD transfers.
    res = workgroupOp->walk([&](AMDAIE::NpuControlPacketOp ctrlPktOp) {
      ctrlPktOps.push_back(ctrlPktOp);
      // Get `ConnectionOp` for the `CTRL` port.
      uint32_t address = ctrlPktOp.getAddress();
      uint32_t addrOffset = deviceModel.getOffsetFromAddress(address);
      int32_t destCol = deviceModel.getColumnFromAddress(address);
      int32_t destRow = deviceModel.getRowFromAddress(address);
      if (!tileLocToCtrlConnect.count({destCol, destRow})) {
        ctrlPktOp.emitOpError()
            << "tries to configure the tile (col=" << destCol
            << ", row=" << destRow << "), but it's `CTRL` port is not routed.";
        return WalkResult::interrupt();
      }
      AMDAIE::ConnectionOp connectionOp =
          tileLocToCtrlConnect[{destCol, destRow}];
      // Get the source tile location.
      SmallVector<Value> srcChannels = connectionOp.getSourceChannels();
      if (srcChannels.size() != 1) {
        ctrlPktOp.emitOpError() << "expected exactly one source channel";
        return WalkResult::interrupt();
      }
      AMDAIE::ChannelOp srcChannelOp =
          dyn_cast<AMDAIE::ChannelOp>(srcChannels[0].getDefiningOp());
      if (!srcChannelOp) {
        ctrlPktOp.emitOpError() << "expected an `amdaie.channel` op source";
        return WalkResult::interrupt();
      }
      AMDAIE::TileOp srcTileOp = srcChannelOp.getTileOp();
      int32_t srcCol = getConstantIndexOrAssert(srcTileOp.getCol());
      int32_t srcRow = getConstantIndexOrAssert(srcTileOp.getRow());
      // Get the control packet data length.
      uint32_t dataLength = ctrlPktOp.getLength();
      // Plus one for the control header, which is always present.
      int64_t headerAndDataLength = dataLength + 1;

      AMDAIE::AMDAIETileType destTileType =
          deviceModel.getTileType(destCol, destRow);
      // If the destination tile type is different from the last one, we need
      // to maintain the issuing order of control packets.
      if (lastDestTileType != destTileType && ctrlPktBdTransfers.size() > 0)
        ctrlPktBdTransfers.back().syncBarrier = true;

      // If the AIE device has the control packet TLAST error disabled,
      // multiple control packets can be packaged into a single BD transfer to
      // improve throughput. Otherwise, shim DMA can only issue one control
      // packet per BD transfer.
      bool packIntoLastBdTransfer = false;
      if (deviceModel.getCtrlPktTlastErrorDisabled() &&
          ctrlPktBdTransfers.size() > 0) {
        const CtrlPktBdTransfer &lastBdTransfer = ctrlPktBdTransfers.back();
        // Check if the same connection is used.
        if (lastBdTransfer.connectionOp == connectionOp) {
          if (!deviceModel.isShimTile(srcCol, srcRow)) {
            ctrlPktOp.emitOpError()
                << "expected the source tile to be a shim tile";
            return WalkResult::interrupt();
          }
          FailureOr<uint32_t> maybeMaxIntraSize =
              deviceModel.getDmaBdProp<uint16_t>(
                  AMDAIE::AMDAIETileType::SHIMNOC, 0,
                  AMDAIE::AMDAIEDmaBdProp::WrapMax);
          if (failed(maybeMaxIntraSize)) return WalkResult::interrupt();
          // Check if the new sizes are still valid.
          SmallVector<int64_t> newBdTransferSizes = lastBdTransfer.sizes;
          // Plus one for the extra packet header.
          newBdTransferSizes.back() += (headerAndDataLength + 1);
          // TODO(zhewen): use all dimensions available.
          packIntoLastBdTransfer =
              newBdTransferSizes.back() <= *maybeMaxIntraSize;
        }
      }
      if (packIntoLastBdTransfer) {
        // Pack into the last BD transfer.
        // Plus one for the extra packet header.
        headerAndDataLength++;
        ctrlPktBdTransfers.back().sizes.back() += headerAndDataLength;
      } else {
        // Create a new BD transfer.
        // TODO(zhewen): use all dimensions available.
        ctrlPktBdTransfers.push_back(
            {connectionOp,
             /*offsets=*/
             {0, 0, 0, static_cast<int64_t>(ctrlPktSequence.size())},
             /*sizes=*/{1, 1, 1, headerAndDataLength},
             /*strides=*/{0, 0, 0, 1}});
      }

      llvm::MutableArrayRef<uint32_t> words =
          reserveAndGetTail(headerAndDataLength);
      size_t idx = 0;
      // Store the optional packet header.
      if (packIntoLastBdTransfer) {
        // Each control packet requires a packet header, but only the
        // first control packet in a BD transfer has its header automatically
        // inserted by the shim DMA. For all subsequent control packets in the
        // same BD transfer, we must "manually" insert the packet header.
        std::optional<AMDAIE::FlowOp> maybeFlowOp = connectionOp.getFlowOp();
        if (!maybeFlowOp) {
          ctrlPktOp.emitOpError()
              << "expected a flow operation for the connection";
          return WalkResult::interrupt();
        }
        std::optional<uint8_t> maybePacketId = maybeFlowOp->getPacketId();
        if (!maybePacketId) {
          ctrlPktOp.emitOpError() << "expected a packet ID for the flow";
          return WalkResult::interrupt();
        }
        FailureOr<uint32_t> maybePacketHeader = deviceModel.getPacketHeader(
            *maybePacketId, /*packetType=*/0, srcRow, srcCol);
        if (failed(maybePacketHeader)) {
          ctrlPktOp.emitOpError() << "failed to get packet header.";
          return WalkResult::interrupt();
        }
        words[idx++] = *maybePacketHeader;
      }
      // Store the control header.
      FailureOr<uint32_t> maybeControlHeader = deviceModel.getControlHeader(
          addrOffset, dataLength, static_cast<uint32_t>(ctrlPktOp.getOpcode()),
          ctrlPktOp.getStreamId());
      if (failed(maybeControlHeader)) {
        ctrlPktOp.emitOpError() << "failed to get control header.";
        return WalkResult::interrupt();
      }
      words[idx++] = *maybeControlHeader;
      // Store the control packet data.
      std::optional<ArrayRef<int32_t>> maybeData =
          ctrlPktOp.getDataFromArrayOrResource();
      for (int32_t data : maybeData.value())
        words[idx++] = reinterpret_cast<uint32_t &>(data);
      // Update the last destination tile type.
      lastDestTileType = destTileType;
      return WalkResult::advance();
    });
    if (res.wasInterrupted()) return failure();

    // Convert each control packet BD transfer to a `NpuDmaCpyNdOp` and
    // `NpuDmaWaitOp`.
    Block *controlCodeBlock = workgroupOp.getControlCode().getBody();
    rewriter.setInsertionPoint(controlCodeBlock->getTerminator());
    for (CtrlPktBdTransfer &stream : ctrlPktBdTransfers) {
      // Target offsets, sizes, and strides are left empty.
      SmallVector<int64_t> dmaTargetOffsets;
      SmallVector<int64_t> dmaTargetSizes;
      SmallVector<int64_t> dmaTargetStrides;
      // Create token.
      SmallVector<Type> resultTypes = {
          rewriter.getType<AMDAIE::AsyncSourceTokenType>()};
      TypeRange sourceResultTypes = TypeRange{resultTypes};
      // Create `NpuDmaCpyNdOp` and `NpuDmaWaitOp`.
      auto dmaOp = rewriter.create<AMDAIE::NpuDmaCpyNdOp>(
          rewriter.getUnknownLoc(), sourceResultTypes, stream.connectionOp,
          nullptr, dmaTargetOffsets, dmaTargetSizes, dmaTargetStrides,
          /*target_bd_id=*/nullptr, stream.connectionOp.getSource(),
          stream.offsets, stream.sizes, stream.strides,
          /*source_bd_id=*/nullptr);
      rewriter.create<AMDAIE::NpuDmaWaitOp>(rewriter.getUnknownLoc(),
                                            dmaOp.getResult(0));
      if (stream.syncBarrier)
        rewriter.create<AMDAIE::NpuBarrierOp>(rewriter.getUnknownLoc());
    }

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
