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
                             const AMDAIE::AMDAIEDeviceModel &deviceModel)
      : OpConversionPattern(context), deviceModel(std::move(deviceModel)) {
    minStrideBitWidth = deviceModel.getMinStrideBitWidth();
  }

  /// Insert ops to write a BD, patch the address and push it to the queue. This
  /// is specific to Shim BDs for now.
  FailureOr<AMDAIE::NpuPushToQueueOp> insertWriteBdOps(
      AMDAIE::NpuHalfDmaCpyNdOp op, ConversionPatternRewriter &rewriter,
      AMDAIE::AMDAIETileType tileType,
      AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjFifo,
      AMDAIE::BdIdOp bdIdOp, AMDAIE::ChannelOp channelOp, int64_t bufferLength,
      int64_t bufferOffset, int32_t enablePacket, int32_t packetId,
      int32_t packetType, ArrayRef<OpFoldResult> sizes,
      ArrayRef<OpFoldResult> strides) const {
    uint8_t numIntraAddrDim = deviceModel.getDmaProp<uint8_t>(
        tileType, AMDAIE::AMDAIEDmaProp::NumAddrDim);
    uint8_t numAddrDim =
        numIntraAddrDim + deviceModel.deviceConfig.dmaNbInterDims;
    auto subspanOp = dyn_cast_if_present<IREE::HAL::InterfaceBindingSubspanOp>(
        logicalObjFifo.getMemref().getDefiningOp());
    if (!subspanOp) {
      return logicalObjFifo.emitOpError()
             << "must operate on an `hal.interface.binding.subspan`";
    }
    int64_t argIdx = subspanOp.getBinding().getZExtValue();
    MemRefType memrefType = logicalObjFifo.getMemrefType();
    int64_t elemWidthInBits = memrefType.getElementTypeBitWidth();
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

    SmallVector<int32_t, 4> staticSizes;
    SmallVector<int32_t, 4> staticStrides;
    // Padding is unused for now.
    SmallVector<int32_t, 4> paddingsBefore;
    SmallVector<int32_t, 4> paddingsAfter;
    int32_t iterationCurrent{0};
    int32_t iterationSize{0};
    int32_t iterationStride{0};
    int32_t repeatCount{1};
    for (auto iter : llvm::enumerate(llvm::zip(sizes, strides))) {
      int64_t size = getConstantIndexOrAssert(std::get<0>(iter.value()));
      int64_t stride = getConstantIndexOrAssert(std::get<1>(iter.value()));

      /// Map the outer dimension to the iteration dimension if intra dimensions
      /// are all used already or if the first stride == 0 as only the iteration
      /// dimension supports stride == 0.
      if (iter.index() == 0 && (sizes.size() == numAddrDim || stride == 0)) {
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
        if (iter.index() == (sizes.size() - 1)) {
          staticSizes.push_back(size * elemWidthInBits / minStrideBitWidth);
        } else {
          staticSizes.push_back(size);
        }
      }
    }
    // Make sure sizes/strides have the correct size based on the number from
    // intra addressing dimensions.
    staticSizes.insert(staticSizes.begin(),
                       numIntraAddrDim - staticSizes.size(), 0);
    staticStrides.insert(staticStrides.begin(),
                         numIntraAddrDim - staticStrides.size(), 0);

    bool useNextBd{false};
    int32_t nextBd{0};
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
    // Only support Shim for now.
    if (op.getMemorySpaceAsUInt() != 0) {
      rewriter.eraseOp(op);
      return success();
    }
    auto logicalObjFifo =
        dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromMemrefOp>(
            op.getInput().getDefiningOp());
    if (!logicalObjFifo) {
      return op.emitOpError() << "expected input to be an "
                                 "`amdaie.logicalobjectfifo.from_memref`";
    }
    std::optional<AMDAIE::BdIdOp> maybeBdIdOp = op.getBdIdOp();
    if (!maybeBdIdOp) {
      return op.emitOpError() << "must have a BD ID op to lower to "
                                 "`amdaie.npu.write_bd`";
    }
    std::optional<AMDAIE::ChannelOp> maybeChannelOp = op.getChannelOp();
    if (!maybeChannelOp)
      return op.emitOpError() << "found non-`amdaie.channel` channel";
    std::optional<int64_t> maybeSize = op.getAccessStaticSize();
    if (!maybeSize)
      return op.emitOpError() << "could not compute a static size";
    std::optional<int64_t> maybeOffset = op.getStaticBaseOffset();
    if (!maybeOffset)
      return op.emitOpError() << "could not compute a static source offset";
    SmallVector<OpFoldResult> sizes = op.getMixedSizes();
    SmallVector<OpFoldResult> strides = op.getMixedStrides();
    FailureOr<AMDAIE::NpuPushToQueueOp> npuPushToQueueOp = insertWriteBdOps(
        op, rewriter, AMDAIE::AMDAIETileType::SHIMNOC, logicalObjFifo,
        maybeBdIdOp.value(), maybeChannelOp.value(), maybeSize.value(),
        maybeOffset.value(), enablePacket, packetId, packetType, sizes,
        strides);
    if (failed(npuPushToQueueOp)) return failure();
    rewriter.replaceOp(op, *npuPushToQueueOp);
    return success();
  }

 private:
  const AMDAIE::AMDAIEDeviceModel &deviceModel;
  uint8_t minStrideBitWidth;
};

namespace {
class AMDAIEControlCodeLoweringPass
    : public impl::AMDAIEControlCodeLoweringBase<
          AMDAIEControlCodeLoweringPass> {
 public:
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
  AMDAIE::AMDAIEDeviceModel deviceModel =
      AMDAIE::getDeviceModel(maybeDevice.value());

  RewritePatternSet patterns(context);
  ConversionTarget conversionTarget(*context);
  conversionTarget.addLegalDialect<AMDAIEDialect>();
  conversionTarget.addIllegalOp<AMDAIE::NpuHalfDmaCpyNdOp>();
  patterns.insert<HalfDmaCpyNdToNpuConverter>(context, deviceModel);
  if (failed(applyPartialConversion(parentOp, conversionTarget,
                                    std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEControlCodeLoweringPass() {
  return std::make_unique<AMDAIEControlCodeLoweringPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
