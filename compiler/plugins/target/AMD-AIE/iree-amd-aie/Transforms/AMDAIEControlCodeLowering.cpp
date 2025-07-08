// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEDmaUtils.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-controlcode-lowering"

namespace mlir::iree_compiler::AMDAIE {

struct HalfDmaCpyNdToNpuConverter final
    : OpConversionPattern<AMDAIE::NpuHalfDmaCpyNdOp> {
  using OpConversionPattern::OpConversionPattern;

  HalfDmaCpyNdToNpuConverter(MLIRContext *context,
                             const AMDAIE::AMDAIEDeviceModel &deviceModel,
                             int32_t argIdxOffset)
      : OpConversionPattern(context),
        deviceModel(std::move(deviceModel)),
        argIdxOffset(argIdxOffset) {
    minStrideBitWidth = deviceModel.getMinStrideBitWidth();
  }

  /// Insert ops to write a BD, patch the address and push it to the queue. This
  /// is specific to Shim BDs for now.
  FailureOr<AMDAIE::NpuPushToQueueOp> insertWriteBdOps(
      AMDAIE::NpuHalfDmaCpyNdOp op, ConversionPatternRewriter &rewriter,
      AMDAIE::AMDAIETileType tileType, AMDAIE::BdIdOp bdIdOp,
      AMDAIE::ChannelOp channelOp, int64_t bufferLength, int64_t bufferOffset,
      int32_t enablePacket, int32_t packetId, int32_t packetType,
      SmallVector<OpFoldResult> sizes, SmallVector<OpFoldResult> strides,
      bool loweringCtrlpktDma) const {
    FailureOr<uint8_t> maybeNumIntraAddrDim = deviceModel.getDmaProp<uint8_t>(
        tileType, AMDAIE::AMDAIEDmaProp::NumAddrDim);
    if (failed(maybeNumIntraAddrDim)) return failure();
    uint8_t numIntraAddrDim = *maybeNumIntraAddrDim;
    uint8_t numAddrDim =
        numIntraAddrDim + deviceModel.deviceConfig.dmaNbInterDims;

    // Default values, used for control packet DMAs only.
    // The `argIdx` must match the driver and is assumed to be 0.
    // The `elemWidthInBits` is fixed at 32 for control packet data.
    // The `memSpace` is set to 0, indicating storage in global memory.
    int64_t argIdx = 0;
    int64_t elemWidthInBits = 32;
    uint8_t memSpace = 0;
    if (!loweringCtrlpktDma) {
      // Normal DMAs, update `argIdx`, `elemWidthInBits`, and `memSpace` based
      // on the memref.
      auto logicalObjFifo =
          dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromMemrefOp>(
              op.getInput().getDefiningOp());
      if (!logicalObjFifo) {
        return op.emitOpError() << "expected input to be an "
                                   "`amdaie.logicalobjectfifo.from_memref`";
      }
      auto assumeAlignmentOp = dyn_cast_if_present<memref::AssumeAlignmentOp>(
          logicalObjFifo.getMemref().getDefiningOp());

      IREE::HAL::InterfaceBindingSubspanOp subspanOp;
      if (assumeAlignmentOp) {
        subspanOp = dyn_cast_if_present<IREE::HAL::InterfaceBindingSubspanOp>(
            assumeAlignmentOp.getViewSource().getDefiningOp());
      } else {
        subspanOp = dyn_cast_if_present<IREE::HAL::InterfaceBindingSubspanOp>(
            logicalObjFifo.getMemref().getDefiningOp());
      }
      if (!subspanOp) {
        return logicalObjFifo.emitOpError()
               << "must operate on a `hal.interface.binding.subspan`";
      }
      argIdx = subspanOp.getBinding().getZExtValue();
      MemRefType memrefType = logicalObjFifo.getMemrefType();
      elemWidthInBits = memrefType.getElementTypeBitWidth();
      memSpace = logicalObjFifo.getMemorySpaceAsUInt();
    }
    // Add the optional offset to the argIdx.
    argIdx += argIdxOffset;
    if (argIdx < 0) return op.emitOpError() << "argIdx must be non-negative";

    std::optional<AMDAIE::DMAChannelDir> maybeDmaDirection =
        channelOp.getDirection();
    if (!maybeDmaDirection) {
      return channelOp.emitOpError()
             << "direction needed for lowering of NPU ops";
    }
    AMDAIE::TileOp tileOp =
        dyn_cast_if_present<AMDAIE::TileOp>(bdIdOp.getTile().getDefiningOp());
    if (!tileOp)
      return bdIdOp.emitOpError() << "must operate on an `amdaie.tile`";
    int64_t col = getConstantIndexOrAssert(tileOp.getCol());
    int64_t row = getConstantIndexOrAssert(tileOp.getRow());
    int32_t bdId = getConstantIndexOrAssert(bdIdOp.getValue());
    int32_t outOfOrderId{0};

    SmallVector<OpFoldResult> offsets(
        strides.size(), getAsIndexOpFoldResult(rewriter.getContext(), 0));
    (void)foldUnitDims(rewriter.getContext(), offsets, sizes, strides);

    DmaDimConfig dmaDimConfig(deviceModel, memSpace);
    SmallVector<int64_t> maxSizes = dmaDimConfig.getMaxSizes(offsets.size());
    SmallVector<OpFoldResult> linearOffsets, linearSizes, linearStrides;
    (void)foldLinearDims(
        rewriter.getContext(), offsets, sizes, strides, linearOffsets,
        linearSizes, linearStrides, [&](size_t idxFromEnd, int64_t size) {
          return idxFromEnd < maxSizes.size() &&
                 size <= maxSizes[maxSizes.size() - idxFromEnd - 1];
        });

    SmallVector<int32_t, 4> staticSizes;
    SmallVector<int32_t, 4> staticStrides;
    // Padding is unused for now.
    SmallVector<int32_t, 4> paddingsBefore;
    SmallVector<int32_t, 4> paddingsAfter;
    int32_t iterationCurrent{0};
    int32_t iterationSize{0};
    int32_t iterationStride{0};
    int32_t repeatCount{1};
    for (auto iter : llvm::enumerate(llvm::zip(linearSizes, linearStrides))) {
      int64_t size = getConstantIndexOrAssert(std::get<0>(iter.value()));
      int64_t stride = getConstantIndexOrAssert(std::get<1>(iter.value()));

      /// Map the outer dimension to the iteration dimension if intra dimensions
      /// are all used already or if the first stride == 0 as only the iteration
      /// dimension supports stride == 0.
      if (iter.index() == 0 &&
          (linearSizes.size() == numAddrDim || stride == 0)) {
        if (stride == 0) {
          repeatCount = size;
        } else {
          iterationStride = std::max(
              stride * elemWidthInBits / minStrideBitWidth, (int64_t)1);
          iterationSize = size;
          if (stride == 1) size = (size * elemWidthInBits) / minStrideBitWidth;
          repeatCount = iterationSize;
        }
      } else {
        staticStrides.push_back(
            std::max(stride * elemWidthInBits / minStrideBitWidth, (int64_t)1));
        // Innermost size needs to account for addressing granularity.
        if (iter.index() == (linearSizes.size() - 1)) {
          staticSizes.push_back(size * elemWidthInBits / minStrideBitWidth);
        } else {
          staticSizes.push_back(size);
        }
      }
    }
    // Make sure sizes/strides have the correct size based on the number from
    // intra addressing dimensions.
    assert(staticSizes.size() <= numIntraAddrDim &&
           "The number of dimensions in DMA sizes should not more than the "
           "number of `intra` addressing dimensions");
    assert(staticStrides.size() <= numIntraAddrDim &&
           "The number of dimensions in DMA strides should not more than the "
           "number of `intra` addressing dimensions");
    staticSizes.insert(staticSizes.begin(),
                       numIntraAddrDim - staticSizes.size(), 0);
    staticStrides.insert(staticStrides.begin(),
                         numIntraAddrDim - staticStrides.size(), 0);

    bool useNextBd = false;
    int32_t nextBd{0};
    if (std::optional<AMDAIE::BdIdOp> nextBdIdOp = op.getNextBdIdOp()) {
      nextBd = getConstantIndexOrAssert(nextBdIdOp.value().getValue());
      useNextBd = true;
    }

    bool validBd{true};
    int32_t lockRelVal{0};
    int32_t lockRelId{0};
    bool lockAcqEnable{false};
    int32_t lockAcqVal{0};
    int32_t lockAcqId{0};

    uint32_t bufferLengthInWords =
        bufferLength * elemWidthInBits / minStrideBitWidth;
    uint32_t innerBufferLength = bufferLengthInWords / repeatCount;
    uint32_t bufferOffsetInBytes = bufferOffset * elemWidthInBits / 8;

    // Offset set to zero for shim as the offset is embedded in the address
    // patch.
    rewriter.create<AMDAIE::NpuWriteBdOp>(
        op.getLoc(), col, row, bdId, innerBufferLength, 0, staticSizes,
        staticStrides, paddingsBefore, paddingsAfter, iterationCurrent,
        iterationSize, iterationStride, enablePacket, packetId, packetType,
        outOfOrderId, useNextBd, nextBd, validBd, lockAcqEnable, lockRelVal,
        lockRelId, lockAcqVal, lockAcqId);
    rewriter.create<AMDAIE::NpuAddressPatchOp>(op.getLoc(), col, bdId, argIdx,
                                               bufferOffsetInBytes);
    SmallVector<Type> resultTypes = {
        rewriter.getType<AMDAIE::AsyncTokenType>()};
    TypeRange resultTypeRange =
        op.getAsyncToken() ? TypeRange{resultTypes} : TypeRange{};
    auto npuPushToQueueOp = rewriter.create<AMDAIE::NpuPushToQueueOp>(
        op.getLoc(), resultTypeRange, col, row, maybeDmaDirection.value(),
        channelOp.getValue(), repeatCount, bdId);
    return npuPushToQueueOp;
  }

  LogicalResult matchAndRewrite(
      AMDAIE::NpuHalfDmaCpyNdOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "matchAndRewrite[AMDAIE::NpuHalfDmaCpyNdOp]\n");
    // First retrieve the connection and flow ops operated on.
    // NOTE(jornt): this will logic will simplify in the future when DMA ops can
    // operate directly on `amdaie.flow`.
    std::optional<AMDAIE::ConnectionOp> maybeConnectionOp =
        op.getConnectionOp();
    if (!maybeConnectionOp) {
      return op.emitOpError()
             << "expected to operate on an `amdaie.connection`";
    }
    std::optional<AMDAIE::FlowOp> maybeFlowOp = maybeConnectionOp->getFlowOp();
    if (!maybeFlowOp) {
      return maybeConnectionOp->emitOpError()
             << "expected to operate on an `amdaie.flow`";
    }
    bool enablePacket = maybeFlowOp->getIsPacketFlow();
    int32_t packetId{0};
    int32_t packetType{0};
    std::optional<uint8_t> maybePacketId = maybeFlowOp->getPacketId();
    if (enablePacket) {
      if (!maybePacketId) {
        return maybeFlowOp->emitOpError()
               << "packet flow enabled, but no packet ID is set";
      }
      packetId = maybePacketId.value();
    }
    FailureOr<bool> maybeIsControlFlow = maybeFlowOp->isControlFlow();
    if (failed(maybeIsControlFlow)) {
      return maybeFlowOp->emitOpError()
             << "failed to determine if it is control flow";
    }
    std::optional<AMDAIE::ChannelOp> maybeChannelOp = op.getChannelOp();
    if (!maybeChannelOp)
      return op.emitOpError() << "found non-`amdaie.channel` channel";
    // Lower only the half DMA op if its input originates from the shim;
    // otherwise, erase it. We must check both the memory space and port type,
    // as certain special ports (e.g., `CTRL`) have an undefined memory space
    // (currently set to 0), and we still want to exclude them.
    if (op.getMemorySpaceAsUInt() != 0 ||
        maybeChannelOp->getPortType() != StrmSwPortType::DMA) {
      rewriter.eraseOp(op);
      return success();
    }
    std::optional<AMDAIE::BdIdOp> maybeBdIdOp = op.getBdIdOp();
    if (!maybeBdIdOp) {
      return op.emitOpError() << "must have a BD ID op to lower to "
                                 "`amdaie.npu.write_bd`";
    }
    std::optional<int64_t> maybeSize = op.getAccessStaticSize();
    if (!maybeSize)
      return op.emitOpError() << "could not compute a static size";
    std::optional<int64_t> maybeOffset = op.getStaticBaseOffset();
    if (!maybeOffset)
      return op.emitOpError() << "could not compute a static source offset";
    SmallVector<OpFoldResult> sizes = op.getMixedSizes();
    SmallVector<OpFoldResult> strides = op.getMixedStrides();
    FailureOr<AMDAIE::NpuPushToQueueOp> npuPushToQueueOp = insertWriteBdOps(
        op, rewriter, AMDAIE::AMDAIETileType::SHIMNOC, maybeBdIdOp.value(),
        maybeChannelOp.value(), maybeSize.value(), maybeOffset.value(),
        enablePacket, packetId, packetType, sizes, strides,
        *maybeIsControlFlow);
    if (failed(npuPushToQueueOp)) return failure();
    rewriter.replaceOp(op, *npuPushToQueueOp);

    std::optional<AMDAIE::BdIdOp> nextBdIdOp = op.getNextBdIdOp();
    if (nextBdIdOp) {
      // `next_bd` is set, so either at the beginning or middle of a chain.
      // No need to push to the queue, just erase the op.
      rewriter.eraseOp(*npuPushToQueueOp);
    } else {
      std::optional<AMDAIE::BdIdOp> maybeStartBdIdOp = op.getStartBdIdOp();
      if (maybeStartBdIdOp) {
        // Update with the BD ID at the start of the chain.
        AMDAIE::BdIdOp startBdIdOp = maybeStartBdIdOp.value();
        uint32_t startBdId = getConstantIndexOrAssert(startBdIdOp.getValue());
        npuPushToQueueOp->setBdId(startBdId);
      }
    }
    return success();
  }

 private:
  const AMDAIE::AMDAIEDeviceModel &deviceModel;
  uint8_t minStrideBitWidth;
  // Offset to be added to the `argIdx` field of the write BD operation.
  int32_t argIdxOffset;
};

struct DmaWaitToTctSyncConverter final
    : OpConversionPattern<AMDAIE::NpuDmaWaitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AMDAIE::NpuDmaWaitOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "matchAndRewrite[AMDAIE::NpuDmaWaitOp]\n");
    // Collect all half DMA ops from the async tokens.
    SmallVector<AMDAIE::NpuPushToQueueOp> pushToQueueOps;
    for (Value asyncToken : op.getAsyncTokens()) {
      auto pushToQueueOp = dyn_cast_if_present<AMDAIE::NpuPushToQueueOp>(
          asyncToken.getDefiningOp());
      if (!pushToQueueOp) {
        return op.emitOpError()
               << "should operate on an `amdaie.push_to_queue` op async token";
      }
      pushToQueueOps.push_back(pushToQueueOp);
    }
    // Sort the half DMA ops by direction, channel, row, and column.
    std::sort(pushToQueueOps.begin(), pushToQueueOps.end(),
              [](AMDAIE::NpuPushToQueueOp a, AMDAIE::NpuPushToQueueOp b) {
                return std::make_tuple(a.getDirection(), a.getChannel(),
                                       a.getRow(), a.getCol()) <
                       std::make_tuple(b.getDirection(), b.getChannel(),
                                       b.getRow(), b.getCol());
              });
    // Batch DMA operations with the same row, channel, and direction into a
    // single TCT sync operation, as long as they have consecutive columns.
    llvm::MapVector<AMDAIE::NpuPushToQueueOp, uint32_t> columnBatches;
    for (auto pushToQueueOp : pushToQueueOps) {
      if (!columnBatches.empty()) {
        auto &[lastPushOp, lastColNum] = columnBatches.back();
        if (lastPushOp.getRow() == pushToQueueOp.getRow() &&
            lastPushOp.getCol() + lastColNum == pushToQueueOp.getCol() &&
            lastPushOp.getDirection() == pushToQueueOp.getDirection() &&
            lastPushOp.getChannel() == pushToQueueOp.getChannel()) {
          ++lastColNum;
          continue;
        }
      }
      columnBatches.insert({pushToQueueOp, 1});
    }
    // Convert to TCT sync ops.
    for (auto &[pushToQueueOp, colNum] : columnBatches) {
      uint32_t rowNum = 1;
      rewriter.create<AMDAIE::NpuTctSyncOp>(
          op.getLoc(), pushToQueueOp.getCol(), pushToQueueOp.getRow(),
          pushToQueueOp.getDirection(), pushToQueueOp.getChannel(), colNum,
          rowNum);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

namespace {
class AMDAIEControlCodeLoweringPass
    : public impl::AMDAIEControlCodeLoweringBase<
          AMDAIEControlCodeLoweringPass> {
 public:
  AMDAIEControlCodeLoweringPass(const AMDAIEControlCodeLoweringOptions &options)
      : AMDAIEControlCodeLoweringBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }
  void runOnOperation() override;
};

void AMDAIEControlCodeLoweringPass::runOnOperation() {
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

  // First conversion: HalfDmaCpyNdOp to WriteBdOp, AddressPatchOp and
  // PushToQueueOp.
  {
    AMDAIE::AMDAIEDeviceModel deviceModel =
        AMDAIE::getDeviceModel(maybeDevice.value());
    RewritePatternSet patterns(context);
    ConversionTarget conversionTarget(*context);
    conversionTarget.addLegalDialect<AMDAIEDialect>();
    conversionTarget.addIllegalOp<AMDAIE::NpuHalfDmaCpyNdOp>();
    patterns.insert<HalfDmaCpyNdToNpuConverter>(context, deviceModel,
                                                argIdxOffset);

    if (failed(applyPartialConversion(parentOp, conversionTarget,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  // Second conversion: DmaWaitOp to TctSyncOp.
  // The two conversions are separate to simplify the attribute handling, such
  // as col, row, direction, channel, etc.
  {
    RewritePatternSet patterns(context);
    ConversionTarget conversionTarget(*context);
    conversionTarget.addLegalDialect<AMDAIEDialect>();
    conversionTarget.addIllegalOp<AMDAIE::NpuDmaWaitOp>();
    patterns.insert<DmaWaitToTctSyncConverter>(context);
    if (failed(applyPartialConversion(parentOp, conversionTarget,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEControlCodeLoweringPass(
    AMDAIEControlCodeLoweringOptions options) {
  return std::make_unique<AMDAIEControlCodeLoweringPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
