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
      auto subspanOp =
          dyn_cast_if_present<IREE::HAL::InterfaceBindingSubspanOp>(
              logicalObjFifo.getMemref().getDefiningOp());
      if (!subspanOp) {
        return logicalObjFifo.emitOpError()
               << "must operate on an `hal.interface.binding.subspan`";
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
      // if (!pushToQueueOp) {
      //   return op.emitOpError()
      //          << "should operate on an `amdaie.push_to_queue` op async token";
      // }
      if (!pushToQueueOp) {
        return failure();
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


/// Compute the 'global' repetition count: the product over all dimensions with
/// zero stride of the size of the dimension.
///
/// The case where sizes and strides are empty is a special case, and '0' is
/// returned.
static int64_t getRepetitionCount(ArrayRef<OpFoldResult> sizes,
                                  ArrayRef<OpFoldResult> strides) {
  assert(sizes.size() == strides.size() &&
         "expected stride and size vectors of same size");
  if (strides.empty()) return 0;
  size_t repetitionCount{1};
  for (uint32_t i = 0; i < strides.size(); ++i) {
    if (!isConstantIntValue(strides[i], 0)) continue;
    std::optional<int64_t> maybeSize = getConstantIntValue(sizes[i]);
    assert(maybeSize.has_value() &&
           "expected constant size in this zero stride dimension");
    assert(maybeSize.value() >= 0 && "expected a non-negative size");
    repetitionCount *= maybeSize.value();
  }
  return repetitionCount;
}

/// Utility to retrieve the common repetition count from all producers and
/// consumers of a logical objectFifo.
static FailureOr<size_t> getRepetitionCount(LogicalObjFifoOpInterface op) {
  SmallVector<int64_t> repetitionCounts;
  auto appendRepetitionCount = [&](ArrayRef<OpFoldResult> sizes,
                                   ArrayRef<OpFoldResult> strides) {
    size_t repetitionCount = getRepetitionCount(sizes, strides);
    if (repetitionCount != 0) repetitionCounts.push_back(repetitionCount);
  };

  for (Operation *userOp : op->getUsers()) {
    if (auto connectionOp = dyn_cast<AMDAIE::ConnectionOp>(userOp)) {
      FailureOr<AMDAIE::NpuCircularDmaCpyNdOp> maybeNpuDmaUserOp =
          connectionOp.getNpuCircularDmaCpyNdUser();

      if (failed(maybeNpuDmaUserOp)) continue;

      AMDAIE::NpuCircularDmaCpyNdOp npuDma = maybeNpuDmaUserOp.value();

      if (connectionOp.getTarget() &&
          dyn_cast_if_present<LogicalObjFifoOpInterface>(
              connectionOp.getTarget().getDefiningOp()) == op) {
        appendRepetitionCount(npuDma.getTargetMixedSizes(),
                              npuDma.getTargetMixedStrides());
      }

      if (connectionOp.getSource() &&
          dyn_cast_if_present<LogicalObjFifoOpInterface>(
              connectionOp.getSource().getDefiningOp()) == op) {
        appendRepetitionCount(npuDma.getSourceMixedSizes(),
                              npuDma.getSourceMixedStrides());
      }
    }
  }

  // merge the repetition counts:
  if (repetitionCounts.empty()) return 1;
  int64_t combinedRepetitionCount =
      *std::min_element(repetitionCounts.begin(), repetitionCounts.end());

  // if any of the repetition counts are not divisible by the combined
  // repetition count, that's a problem:
  if (!std::all_of(
          repetitionCounts.begin(), repetitionCounts.end(),
          [&](size_t c) { return c % combinedRepetitionCount == 0; })) {
    return op.emitOpError()
           << " could not resolved a common repetition count based on the "
              "individual repetition counts: "
           << getArrayString<int64_t>(repetitionCounts);
  }
  return combinedRepetitionCount;
}

//===----------------------------------------------------------------------===//
// AIEDeviceBuilder utilities
//===----------------------------------------------------------------------===//

using BDDimLayoutAndLength = std::pair<AMDAIE::BDDimLayoutArrayAttr, int64_t>;

BDDimLayoutAndLength convertSizeStrideToBDDimLayoutArrayAttr(
    IRRewriter &rewriter, ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides) {
  assert(sizes.size() == strides.size() &&
         "expected stride and size vectors of same size");
  SmallVector<AMDAIE::BDDimLayoutAttr, 4> bdDimLayoutAttr;
  // If the access pattern (strides/sizes) have a single dimension, make it
  // implicit with an empty `BDDimLayoutAttr` as this is what the AIE dialect
  // expects.
  if (strides.size() == 1 && strides[0] == 1) {
    return std::make_pair(AMDAIE::BDDimLayoutArrayAttr::get(
                              rewriter.getContext(), ArrayRef(bdDimLayoutAttr)),
                          sizes[0]);
  }
  bdDimLayoutAttr.reserve(sizes.size());
  // Compute the length of the DMA transfer.
  int64_t transferLength =
      sizes.empty()
          ? 0
          : std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<>());
  for (auto [size, stride] : llvm::zip(sizes, strides)) {
    bdDimLayoutAttr.push_back(
        AMDAIE::BDDimLayoutAttr::get(rewriter.getContext(), size, stride));
  }
  return std::make_pair(AMDAIE::BDDimLayoutArrayAttr::get(
                            rewriter.getContext(), ArrayRef(bdDimLayoutAttr)),
                        transferLength);
}

/// Create a new `aie.dma_start` op with a sequence of DMA BD blocks within the
/// provided `memOp`.
///
/// Example of a S2MM DMA start op being created with two DMA blocks performing
/// a circular double buffering DMA operation:
///
///  %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
///    %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
///  ^bb1:  // 2 preds: ^bb0, ^bb2
///    aie.use_lock(%lock_0_1_51, AcquireGreaterOrEqual, 2)
///    aie.dma_bd(%buffer_0_1_49 : memref<2048xi32, 1 : i32>) {len = 2048 : i32}
///    aie.use_lock(%lock_0_1_52, Release, 2)
///    aie.next_bd ^bb2
///  ^bb2:  // pred: ^bb1
///    aie.use_lock(%lock_0_1_51, AcquireGreaterOrEqual, 2)
///    aie.dma_bd(%buffer_0_1_50 : memref<2048xi32, 1 : i32>) {len = 2048 : i32}
///    aie.use_lock(%lock_0_1_52, Release, 2)
///    aie.next_bd ^bb1
LogicalResult createDMABlocks(
    IRRewriter &rewriter, Operation *memOp, AMDAIE::DMAChannelDir channelDir, int channelIndex,
    ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, size_t acqNum,
    size_t relNum, int64_t offset, const SmallVector<AMDAIE::BufferOp> &bufferOps,
    const std::pair<AMDAIE::LockOp, AMDAIE::LockOp> &locks,
    std::optional<uint8_t> pktId) {
  OpBuilder::InsertionGuard g(rewriter);

  Block &endBlock = memOp->getRegion(0).getBlocks().back();
  assert(!endBlock.getOps<AMDAIE::EndOp>().empty() &&
         "expected last block to have aie.end");
  Block *lastDmaBlock = endBlock.getSinglePredecessor(),
        *dmaBlock = rewriter.createBlock(&endBlock),
        *bdBlock = rewriter.createBlock(&endBlock);

  // Create DMA channel.
  rewriter.setInsertionPointToStart(dmaBlock);
  rewriter.create<AMDAIE::DMAStartOp>(rewriter.getUnknownLoc(), channelDir,
                                   channelIndex, /*repeatCount=*/1, bdBlock,
                                   &endBlock);
  if (lastDmaBlock) lastDmaBlock->getTerminator()->setSuccessor(dmaBlock, 1);

  auto createDMAOps = [&](Block *succ, AMDAIE::BufferOp buff,
                          AMDAIE::BDDimLayoutArrayAttr dims, bool shouldAcqLock,
                          bool shouldRelLock, int64_t transferLength,
                          int64_t offset) {
    AMDAIE::LockOp acqLock = locks.first, relLock = locks.second;
    if (shouldAcqLock) {
      rewriter.create<AMDAIE::UseLockOp>(rewriter.getUnknownLoc(), acqLock,
                                      AMDAIE::LockAction::AcquireGreaterOrEqual,
                                      acqNum);
    }
    // Insert a packet op for MM2S DMAs if part of a packet flow. Only do
    // this for MM2S DMA ports as only those can insert packet headers.
    // if (channelDir == AIE::DMAChannelDir::MM2S && pktId.has_value()) {
    //   rewriter.create<AIE::DMABDPACKETOp>(rewriter.getUnknownLoc(),
    //                                       /*pkt_type*/ 0,
    //                                       /*pkt_id*/ pktId.value());
    // }
    if (!dims.getValue().empty()) {
      rewriter.create<AMDAIE::DMABDOp>(rewriter.getUnknownLoc(), buff, offset,
                                    transferLength, dims);
    } else {
      rewriter.create<AMDAIE::DMABDOp>(rewriter.getUnknownLoc(), buff, offset,
                                    transferLength);
    }
    if (shouldRelLock) {
      rewriter.create<AMDAIE::UseLockOp>(rewriter.getUnknownLoc(), relLock,
                                      AMDAIE::LockAction::Release, relNum);
    }
    rewriter.create<AMDAIE::NextBDOp>(rewriter.getUnknownLoc(), succ);
  };

  // Find the last index with a zero stride. All dimensions before and including
  // this one will be converted into separate DMA ops, while the dimensions
  // after this will be included in the access pattern within a DMA op. This is
  // needed becaused low-level DMA BD configurations currently don't support
  // zero stride and/or because more dimensions are needed than available.
  int64_t lastZeroStrideIndex{-1};
  for (size_t i = 0; i < strides.size(); i++)
    if (strides[i] == 0) lastZeroStrideIndex = i;

  // Convert all dimensions after the last index with zero stride to a
  // `BDDimLayoutArrayAttr` as these are the inner/intra DMA dimensions.
  auto [dims, transferLength] = convertSizeStrideToBDDimLayoutArrayAttr(rewriter,
      ArrayRef<int64_t>(sizes).drop_front(lastZeroStrideIndex + 1),
      ArrayRef<int64_t>(strides).drop_front(lastZeroStrideIndex + 1));

  SmallVector<size_t> indexRange(lastZeroStrideIndex + 1);
  std::iota(indexRange.begin(), indexRange.end(), 0);
  // Compute the total number of iterations of all dimensions up till
  // `lastZeroStrideIndex`.
  int64_t numIters = std::accumulate(
      sizes.begin(), sizes.begin() + indexRange.size(), 1, std::multiplies<>());
  // Compute the divisors to be used to get the indices for every dimension from
  // the total number of iterations (as if all dimensions are coalesced).
  SmallVector<int64_t> cartesianDivisors(indexRange.size(), 1);
  for (int64_t i = indexRange.size() - 2; i >= 0; i--)
    cartesianDivisors[i] = cartesianDivisors[i + 1] * sizes[i + 1];

  // Create blocks with DMA ops.
  Block *succ = nullptr, *curr = bdBlock;
  for (size_t blockIndex = 0; blockIndex < bufferOps.size(); ++blockIndex) {
    // Iterate through the cartesian product of all dimension up to the last
    // dimension with zero strides to create a DMA chain of `dma_bd` ops.
    for (int64_t index = 0; index < numIters; index++) {
      SmallVector<int64_t> indices = llvm::map_to_vector(
          indexRange,
          [&](size_t i) { return (index / cartesianDivisors[i]) % sizes[i]; });
      bool isFirst = llvm::all_of(indices, [](int64_t v) { return v == 0; });
      bool isLast = llvm::all_of(
          indexRange, [&](size_t i) { return indices[i] == (sizes[i] - 1); });
      if (blockIndex == bufferOps.size() - 1 && isLast) {
        succ = bdBlock;
      } else {
        succ = rewriter.createBlock(&endBlock);
      }
      rewriter.setInsertionPointToStart(curr);
      int64_t addOffset = 0;
      for (size_t i = 0; i < indexRange.size(); i++)
        addOffset += (indices[i] * strides[i]);
      createDMAOps(succ, bufferOps[blockIndex], dims, isFirst, isLast,
                   transferLength, offset + addOffset);
      curr = succ;
    }
  }
  return success();
}


LogicalResult foldDimsAndReturnAsStatic(
    IRRewriter &rewriter, AMDAIE::AMDAIEDeviceModel deviceModel, SmallVector<OpFoldResult> sizes, SmallVector<OpFoldResult> strides,
    SmallVector<int64_t> &newSizes, SmallVector<int64_t> &newStrides,
    size_t repetitionCount, uint8_t memSpace,
    function_ref<InFlightDiagnostic()> emitError) {
  if (failed(foldRepetitionCount(rewriter.getContext(), sizes, strides,
                                 repetitionCount))) {
    return emitError() << "could not fold repetition counts from sizes: "
                       << getConstantIntValuesString(sizes)
                       << " strides: " << getConstantIntValuesString(strides)
                       << " repetitionCount: " << repetitionCount << ".";
  }
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
  std::optional<SmallVector<int64_t>> maybeStaticSizes =
      getConstantIntValues(linearSizes);
  std::optional<SmallVector<int64_t>> maybeStaticStrides =
      getConstantIntValues(linearStrides);
  if (!maybeStaticSizes || !maybeStaticStrides) {
    return emitError()
           << "found dynamic sizes or strides which is not supported";
  }
  newSizes = std::move(maybeStaticSizes.value());
  newStrides = std::move(maybeStaticStrides.value());
  return success();
}

/// Convert the `amdaie.connection` operation into DMA operations. Depending on
/// the location of the source/target of the connection, different DMA ops are
/// created:
/// 1. Source/target on a Shim tile: iterate through producer/consumer channels
/// and create corresponding `aie.shim_dma_allocation` ops.
/// 2. Source/target on MemTile: iterate through producer/consumer channels,
/// lookup the correct `aie.memtile_dma` op and create new DMA BD blocks inside.
/// 3. Source/target on MemTile: iterate through producer/consumer channels,
/// lookup the correct `aie.mem` op and create new DMA BD blocks inside.
LogicalResult connectionToAIE(
    IRRewriter &rewriter, AMDAIE::AMDAIEDeviceModel deviceModel, AMDAIE::ConnectionOp connectionOp, Block *deviceBlock,
    int &connectionIndex,
    DenseMap<Value, Operation *> &tileToMemOpMap) {
  LLVM_DEBUG(llvm::dbgs() << "Convert [AMDAIE::ConnectionOp]\n");
  Value source = connectionOp.getSource();
  auto sourceObjFifoLikeOp =
      dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
          source.getDefiningOp());
  if (!sourceObjFifoLikeOp) {
    return connectionOp.emitOpError()
           << "expected source to be an logical objFifo-like op";
  }
  Value target = connectionOp.getTarget();
  auto targetObjFifoLikeOp =
      dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
          target.getDefiningOp());
  if (!targetObjFifoLikeOp) {
    return connectionOp.emitOpError()
           << "expected target to be an logical objFifo-like op";
  }
  // We will only deal with L2<->L1 (Memtile related DMAs)
  if (sourceObjFifoLikeOp.getMemorySpaceAsUInt() == 0 || targetObjFifoLikeOp.getMemorySpaceAsUInt() == 0 ) {
    return success();
  }
  // TODO(avarma): Need to set correct insertion point.
  rewriter.setInsertionPoint(deviceBlock->getTerminator());
  SmallVector<AMDAIE::ChannelOp> producerChannels;
  SmallVector<AMDAIE::ChannelOp> consumerChannels;
  for (Value producerChannel : connectionOp.getSourceChannels()) {
    auto channelOp =
        dyn_cast<AMDAIE::ChannelOp>(producerChannel.getDefiningOp());
    if (!channelOp) {
      return connectionOp.emitOpError()
             << "found non-`amdaie.channel` source channel";
    }
    producerChannels.push_back(channelOp);
  }
  for (Value consumerChannel : connectionOp.getTargetChannels()) {
    auto channelOp =
        dyn_cast<AMDAIE::ChannelOp>(consumerChannel.getDefiningOp());
    if (!channelOp) {
      return connectionOp.emitOpError()
             << "found non-`amdaie.channel` target channel";
    }
    consumerChannels.push_back(channelOp);
  }

  std::optional<AMDAIE::FlowOp> maybeFlowOp = connectionOp.getFlowOp();
  if (!maybeFlowOp) return connectionOp.emitOpError() << "has no flow op";

  FailureOr<bool> isCtrlFlow = maybeFlowOp->isControlFlow();
  if (failed(isCtrlFlow)) {
    return connectionOp.emitOpError()
           << "could not determine if flow is control";
  }
  // No DMA op needed for control flow.
  if (isCtrlFlow.value()) return success();

  std::optional<uint8_t> packetId = maybeFlowOp->getPacketId();

  FailureOr<AMDAIE::NpuCircularDmaCpyNdOp> maybeNpuDmaUserOp =
      connectionOp.getNpuCircularDmaCpyNdUser();
  if (failed(maybeNpuDmaUserOp))
    return connectionOp.emitOpError() << "has no circular NPU DMA op user";

  SmallVector<Operation *> sourceMemOps;
  // if (sourceObjFifoLikeOp.getMemorySpaceAsUInt() == 0) {
  //   for (AMDAIE::ChannelOp channel : producerChannels) {
  //     AIE::ShimDMAAllocationOp shimDmaAllocOp = createShimDmaAllocation(
  //         deviceBlock, channel.getTileOp(), AIE::DMAChannelDir::MM2S,
  //         channel.getValue(), sourceObjFifoLikeOp.getMemrefType(),
  //         connectionIndex);
  //     sourceMemOps.push_back(shimDmaAllocOp.getOperation());
  //   }
  // } else {
    auto sourceObjFifo =
        dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromBuffersOp>(
            source.getDefiningOp());
    if (!sourceObjFifo) {
      return connectionOp.emitOpError()
             << "expected source to be an "
                "`amdaie.logicalobjectfifo.from_buffers` op";
    }
    FailureOr<size_t> repetitionCount = getRepetitionCount(
        cast<LogicalObjFifoOpInterface>(sourceObjFifo.getOperation()));
    if (failed(repetitionCount)) {
      return sourceObjFifo->emitOpError()
             << "could not retrieve the repetition count";
    }
    std::optional<size_t> maybeOffset =
        maybeNpuDmaUserOp->getSourceStaticBaseOffset();
    if (!maybeOffset) {
      return maybeNpuDmaUserOp->emitOpError()
             << "could not compute a static base offset for source";
    }
    std::optional<uint8_t> maybeSourceMemSpace =
        maybeNpuDmaUserOp->getSourceMemorySpaceAsUInt();
    if (!maybeSourceMemSpace) {
      return maybeNpuDmaUserOp->emitOpError()
             << "expected to have a source memory space";
    }
    SmallVector<CopyOpInterface> objFifoProducers =
        sourceObjFifo.getCopyLikeProducers();
    SmallVector<CopyOpInterface> objFifoConsumers =
        sourceObjFifo.getCopyLikeConsumers();
    // Default acquire/release value is 1. Will be adjusted depending on number
    // of producers/consumers.
    int acqNum{1};
    if (objFifoConsumers.size() < objFifoProducers.size()) {
      assert(objFifoProducers.size() % objFifoConsumers.size() == 0);
      acqNum = objFifoProducers.size() / objFifoConsumers.size();
    }
    for (AMDAIE::ChannelOp channel : producerChannels) {
      Operation *memOp = tileToMemOpMap.at(channel.getTile());
      AMDAIE::TileOp tileOp = channel.getTileOp();
      SmallVector<AMDAIE::BufferOp> buffers = sourceObjFifo.getBuffersOnTile(tileOp);
      SmallVector<AMDAIE::LockOp> producerLocks = sourceObjFifo.getProducerLocksOnTile(tileOp);
      SmallVector<AMDAIE::LockOp> consumerLocks = sourceObjFifo.getConsumerLocksOnTile(tileOp);
      if (producerLocks.size() != 1) {
        return sourceObjFifo.emitOpError()
               << "expected a single producer lock for tile: "
               << channel.getTile() << ", channel: " << channel.getResult();
      }
      if (consumerLocks.size() != 1) {
        return sourceObjFifo.emitOpError()
               << "expected a single consumer lock for tile: "
               << channel.getTile() << ", channel: " << channel.getResult();
      }
      std::pair<AMDAIE::LockOp, AMDAIE::LockOp> lockPair =
          std::make_pair(consumerLocks[0], producerLocks[0]);
      SmallVector<int64_t> canonicalizedSizes, canonicalizedStrides;
      if (failed(foldDimsAndReturnAsStatic(rewriter, deviceModel,
              maybeNpuDmaUserOp->getSourceMixedSizes(),
              maybeNpuDmaUserOp->getSourceMixedStrides(), canonicalizedSizes,
              canonicalizedStrides, repetitionCount.value(),
              maybeSourceMemSpace.value(),
              [&]() { return maybeNpuDmaUserOp->emitOpError(); }))) {
        return failure();
      };
      rewriter.moveOpBefore(memOp, deviceBlock,
                            deviceBlock->without_terminator().end());
      if (failed(createDMABlocks(
              rewriter, memOp, AMDAIE::DMAChannelDir::MM2S, channel.getValue(),
              canonicalizedSizes, canonicalizedStrides, acqNum, acqNum,
              maybeOffset.value(), buffers, lockPair, packetId))) {
        return sourceObjFifo.emitOpError() << "could not create DMA operations";
      }
    }
  // }

  SmallVector<Operation *> targetMemOps;
  // if (targetObjFifoLikeOp.getMemorySpaceAsUInt() == 0) {
  //   for (AMDAIE::ChannelOp channel : consumerChannels) {
  //     AIE::ShimDMAAllocationOp shimDmaAllocOp = createShimDmaAllocation(
  //         deviceBlock, channel.getTileOp(), AIE::DMAChannelDir::S2MM,
  //         channel.getValue(), targetObjFifoLikeOp.getMemrefType(),
  //         connectionIndex);
  //     targetMemOps.push_back(shimDmaAllocOp.getOperation());
  //   }
  // } else {
    auto targetObjFifo =
        dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromBuffersOp>(
            target.getDefiningOp());
    if (!targetObjFifo) {
      return connectionOp.emitOpError()
             << "expected target to be an "
                "`amdaie.logicalobjectfifo.from_buffers` op";
    }
    repetitionCount = getRepetitionCount(
        cast<LogicalObjFifoOpInterface>(targetObjFifo.getOperation()));
    if (failed(repetitionCount)) {
      return targetObjFifo->emitOpError()
             << "could not retrieve the repetition count";
    }
    maybeOffset =
        maybeNpuDmaUserOp->getTargetStaticBaseOffset();
    if (!maybeOffset) {
      return maybeNpuDmaUserOp->emitOpError()
             << "could not compute a static base offset for source";
    }
    std::optional<uint8_t> maybeTargetMemSpace =
        maybeNpuDmaUserOp->getTargetMemorySpaceAsUInt();
    if (!maybeTargetMemSpace) {
      return maybeNpuDmaUserOp->emitOpError()
             << "expected to have a target memory space";
    }
    objFifoProducers =
        targetObjFifo.getCopyLikeProducers();
    objFifoConsumers =
        targetObjFifo.getCopyLikeConsumers();
    // Default acquire/release value is 1. Will be adjusted depending on number
    // of producers/consumers.
    acqNum = 1;
    if (objFifoProducers.size() < objFifoConsumers.size()) {
      assert(objFifoConsumers.size() % objFifoProducers.size() == 0);
      acqNum = objFifoConsumers.size() / objFifoProducers.size();
    }
    for (AMDAIE::ChannelOp channel : consumerChannels) {
      // TODO(avarma): This line will still be required for more than 1 AIE core perhaps.
      Operation *memOp = tileToMemOpMap.at(channel.getTile());
      AMDAIE::TileOp tileOp = channel.getTileOp();
      SmallVector<AMDAIE::BufferOp> buffers = targetObjFifo.getBuffersOnTile(tileOp);
      SmallVector<AMDAIE::LockOp> producerLocks = targetObjFifo.getProducerLocksOnTile(tileOp);
      SmallVector<AMDAIE::LockOp> consumerLocks = targetObjFifo.getConsumerLocksOnTile(tileOp);
      if (producerLocks.size() != 1) {
        return targetObjFifo.emitOpError()
               << "expected a single producer lock for tile: "
               << channel.getTile();
      }
      if (consumerLocks.size() != 1) {
        return targetObjFifo.emitOpError()
               << "expected a single consumer lock for tile: "
               << channel.getTile();
      }
      std::pair<AMDAIE::LockOp, AMDAIE::LockOp> lockPair =
          std::make_pair(producerLocks[0], consumerLocks[0]);
      SmallVector<int64_t> canonicalizedSizes, canonicalizedStrides;
      if (failed(foldDimsAndReturnAsStatic(rewriter, deviceModel,
              maybeNpuDmaUserOp->getTargetMixedSizes(),
              maybeNpuDmaUserOp->getTargetMixedStrides(), canonicalizedSizes,
              canonicalizedStrides, repetitionCount.value(),
              maybeTargetMemSpace.value(),
              [&]() { return maybeNpuDmaUserOp->emitOpError(); }))) {
        return failure();
      };
      rewriter.moveOpBefore(memOp, deviceBlock,
                            deviceBlock->without_terminator().end());
      if (failed(createDMABlocks(
              rewriter, memOp, AMDAIE::DMAChannelDir::S2MM, channel.getValue(),
              canonicalizedSizes, canonicalizedStrides, acqNum, acqNum,
              maybeOffset.value(), buffers, lockPair, packetId))) {
        return targetObjFifo.emitOpError() << "could not create DMA operations";
      }
    }
  // }

  // TODO(avarma): Keep track of source/target mem ops for this connection for later retrieval
  // to create NPU ops.
  // connectionToSourceTargetMemOps[connectionOp] =
  //     std::make_pair(sourceMemOps, targetMemOps);
  return success();
}


template <typename MemOp>
LogicalResult logicalObjFifoFromBuffersToMemOp(
    IRRewriter &rewriter, AMDAIE::LogicalObjectFifoFromBuffersOp logicalObjFifo,
    Block *deviceBlock,
    DenseMap<Value, Operation *> &tileToMemOpMap) {
  LLVM_DEBUG(
      llvm::dbgs() << "Convert [AMDAIE::LogicalObjectFifoFromBuffersOp]\n");
  OpBuilder::InsertionGuard guard(rewriter);
  SmallVector<CopyOpInterface> consumers =
      logicalObjFifo.getCopyLikeConsumers();
  SmallVector<CopyOpInterface> producers =
      logicalObjFifo.getCopyLikeProducers();
  if (producers.size() > 1 && consumers.size() > 1) {
    return logicalObjFifo.emitOpError()
           << "has a multi-producer, multi-consumer DMA "
              "pattern, which is currently not supported";
  }
  // Create a memory op for every unique tile and fill it with DMA ops.
  for (Value tile : logicalObjFifo.getTiles()) {
    if (tileToMemOpMap.contains(tile)) continue;
    // Value aieTile = mapper.lookup(tile);
    rewriter.setInsertionPoint(deviceBlock->getTerminator());
    auto newMemOp = rewriter.create<MemOp>(rewriter.getUnknownLoc(), tile);
    rewriter.setInsertionPointToStart(&newMemOp.getRegion().emplaceBlock());
    rewriter.create<AMDAIE::EndOp>(rewriter.getUnknownLoc());
    // Keep track of the MemOps on different tiles.
    tileToMemOpMap[tile] = newMemOp.getOperation();
  }
  return success();
}

LogicalResult logicalObjFifoFromBuffersToAIE(
    IRRewriter &rewriter, AMDAIE::LogicalObjectFifoFromBuffersOp logicalObjFifo, Block *deviceBlock,
    DenseMap<Value, Operation *> &tileToMemOpMap) {
  LLVM_DEBUG(
      llvm::dbgs() << "Convert [AMDAIE::LogicalObjectFifoFromBuffersOp]\n");
  uint8_t memSpaceUInt = logicalObjFifo.getMemorySpaceAsUInt();
  if (memSpaceUInt == 1) {
    // L2
    return logicalObjFifoFromBuffersToMemOp<AMDAIE::MemTileDMAOp>(
        rewriter, logicalObjFifo, deviceBlock, tileToMemOpMap);
  // }
  // else if (memSpaceUInt == 2) {
  //   // L1
  //   return logicalObjFifoFromBuffersToMemOp<AIE::MemOp>(
  //       rewriter, logicalObjFifo, mapper, deviceBlock, tileToMemOpMap);
  } else {
    return logicalObjFifo.emitOpError()
           << "has unsupported memory space for lowering to AIE: "
           << std::to_string(memSpaceUInt);
  }
  return success();
}

LogicalResult convertNpuDmaCpyToMemtileFunc(Operation* workgroupOp) {
  IRRewriter rewriter(workgroupOp->getContext());
  OpBuilder::InsertionGuard guard(rewriter);
  Block *deviceBlock = &workgroupOp->getRegion(0).front();
  DenseMap<Value, Operation *> tileToMemOpMap;
  // Create AMDAIE::Memtile_DMA
  WalkResult res = workgroupOp->walk<WalkOrder::PreOrder>(
    [&](AMDAIE::LogicalObjectFifoFromBuffersOp logicalObjFifo) {
    if (failed(logicalObjFifoFromBuffersToAIE(rewriter, logicalObjFifo,
                                              deviceBlock, tileToMemOpMap))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return failure();

  // Merge core operations into end of the device block
  // rewriter.inlineBlockBefore(deviceCoreBlock, deviceBlock,
  //                            deviceBlock->without_terminator().end());
  return success();
}

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
    // conversionTarget.addIllegalOp<AMDAIE::NpuDmaWaitOp>();
    patterns.insert<DmaWaitToTctSyncConverter>(context);
    if (failed(applyPartialConversion(parentOp, conversionTarget,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  {
    (void)convertNpuDmaCpyToMemtileFunc(parentOp);
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEControlCodeLoweringPass(
    AMDAIEControlCodeLoweringOptions options) {
  return std::make_unique<AMDAIEControlCodeLoweringPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
