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
      AMDAIE::AMDAIETileType tileType, AMDAIE::ConnectionOp connectionOp,
      AMDAIE::BdIdOp bdIdOp, AMDAIE::ChannelOp channelOp, int64_t bufferLength,
      int64_t bufferOffset, int32_t enablePacket, int32_t packetId,
      int32_t packetType, SmallVector<OpFoldResult> sizes,
      SmallVector<OpFoldResult> strides, bool loweringCtrlpktDma) const {
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
        op.getLoc(), connectionOp, col, row, bdId, innerBufferLength, 0,
        staticSizes, staticStrides, paddingsBefore, paddingsAfter,
        iterationCurrent, iterationSize, iterationStride, enablePacket,
        packetId, packetType, outOfOrderId, useNextBd, nextBd, validBd,
        lockAcqEnable, lockRelVal, lockRelId, lockAcqVal, lockAcqId);
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
        op, rewriter, AMDAIE::AMDAIETileType::SHIMNOC,
        maybeConnectionOp.value(), maybeBdIdOp.value(), maybeChannelOp.value(),
        maybeSize.value(), maybeOffset.value(), enablePacket, packetId,
        packetType, sizes, strides, *maybeIsControlFlow);
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

//===----------------------------------------------------------------------===//
// AIEDeviceBuilder utilities
//===----------------------------------------------------------------------===//
// TODO(avarma): Copied from LowerToAIE. Templatize it later because of a few
//               nuances:-
//               1. multiple dialect specific ops' usage within the same
//               function.
//               2. In AIE we have concept of memtile_dma which will contain
//                  sequences of dma_start op, but in case of AMDAIE dma_start
//                  can be a standalone op.
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

/// Create a new `amdaie.dma_start` op with a sequence of DMA BD blocks.
///
/// Example of a S2MM DMA start op being created with two DMA blocks performing
/// a circular double buffering DMA operation:
///
///  %0 = amdaie.dma_start(%tile_0_1, S2MM, 0) {
///       amdaie.use_lock(%lock_0_1_51, AcquireGreaterOrEqual(2))
///       amdaie.dma_bd(%buffer_0_1_49 : memref<2048xi32, 1 : i32>) {len = 2048
///       : i32} amdaie.use_lock(%lock_0_1_52, Release(2)) amdaie.next_bd ^bb1
///     ^bb1:  // pred: ^bb2
///       amdaie.use_lock(%lock_0_1_51, AcquireGreaterOrEqual(2))
///       amdaie.dma_bd(%buffer_0_1_50 : memref<2048xi32, 1 : i32>) {len = 2048
///       : i32} amdaie.use_lock(%lock_0_1_52, Release(2)) amdaie.next_bd ^bb2
///     ^bb2:
///       amdaie.end
///  }
LogicalResult createDMABlocks(
    IRRewriter &rewriter, AMDAIE::ChannelOp channelOp,
    AMDAIE::ConnectionOp connectionOp, ArrayRef<int64_t> sizes,
    ArrayRef<int64_t> strides, size_t acqNum, size_t relNum, int64_t offset,
    const SmallVector<AMDAIE::BufferOp> &bufferOps,
    const std::pair<AMDAIE::LockOp, AMDAIE::LockOp> &locks,
    std::optional<uint8_t> pktId) {
  OpBuilder::InsertionGuard g(rewriter);

  // Create DMA start on channel.
  auto dmaStartOp = rewriter.create<AMDAIE::DMAStartOp>(
      rewriter.getUnknownLoc(), channelOp, ValueRange{connectionOp},
      /*repeatCount=*/1);
  rewriter.setInsertionPointToStart(&dmaStartOp.getRegion().emplaceBlock());
  rewriter.create<AMDAIE::EndOp>(rewriter.getUnknownLoc());
  Block &endBlock = dmaStartOp->getRegion(0).getBlocks().back();
  Block *lastDmaBlock = endBlock.getSinglePredecessor();
  Block *bdBlock = rewriter.createBlock(&endBlock);
  if (lastDmaBlock) lastDmaBlock->getTerminator()->setSuccessor(bdBlock, 1);

  auto createDMAOps = [&](Block *succ, AMDAIE::BufferOp buff,
                          AMDAIE::BDDimLayoutArrayAttr dims, bool shouldAcqLock,
                          bool shouldRelLock, int64_t transferLength,
                          int64_t offset) {
    AMDAIE::LockOp acqLock = locks.first, relLock = locks.second;
    if (shouldAcqLock) {
      rewriter.create<AMDAIE::UseLockOp>(
          rewriter.getUnknownLoc(), acqLock,
          AMDAIE::LockAction::AcquireGreaterOrEqual, acqNum);
    }
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
  auto [dims, transferLength] = convertSizeStrideToBDDimLayoutArrayAttr(
      rewriter, ArrayRef<int64_t>(sizes).drop_front(lastZeroStrideIndex + 1),
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
        succ = &endBlock;
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

/// Form DmaStartOp block for the connection op's source using the access
/// patterns of the half dma op.
static LogicalResult processConnectionSourceForDmaStart(
    IRRewriter &rewriter, AMDAIE::ConnectionOp connectionOp,
    std::optional<uint8_t> packetId, Value source,
    AMDAIE::AMDAIEDeviceModel &deviceModel, size_t sourceOffset,
    uint8_t sourceMemSpace, SmallVector<OpFoldResult> sourceMixedSizes,
    SmallVector<OpFoldResult> sourceMixedStrides) {
  SmallVector<AMDAIE::ChannelOp> producerChannels;
  for (Value producerChannel : connectionOp.getSourceChannels()) {
    auto channelOp =
        dyn_cast<AMDAIE::ChannelOp>(producerChannel.getDefiningOp());
    if (!channelOp) {
      return connectionOp.emitOpError()
             << "found non-`amdaie.channel` source channel";
    }
    producerChannels.push_back(channelOp);
  }
  auto sourceObjFifo =
      dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromBuffersOp>(
          source.getDefiningOp());
  if (!sourceObjFifo) {
    return connectionOp.emitOpError()
           << "expected source to be an "
              "`amdaie.logicalobjectfifo.from_buffers` op";
  }
  FailureOr<size_t> repetitionCount = getRepetitionCount(
      cast<LogicalObjFifoOpInterface>(sourceObjFifo.getOperation()),
      /*reprogramDmas=*/true);
  if (failed(repetitionCount)) {
    return sourceObjFifo->emitOpError()
           << "could not retrieve the repetition count";
  }
  SmallVector<CopyOpInterface> objFifoProducers =
      sourceObjFifo.getCopyLikeProducers();
  SmallVector<CopyOpInterface> objFifoConsumers =
      sourceObjFifo.getCopyLikeConsumers();
  // Default acquire/release value is 1. Will be adjusted depending on number
  // of producers/consumers.
  int acqNum{1};
  if (objFifoConsumers.size() < objFifoProducers.size()) {
    assert(objFifoProducers.size() % objFifoConsumers.size() == 0 &&
           "expected total no. of producers of objFifo to be divisible by the "
           "total no. of consumers");
    acqNum = objFifoProducers.size() / objFifoConsumers.size();
  }
  for (AMDAIE::ChannelOp channel : producerChannels) {
    AMDAIE::TileOp tileOp = channel.getTileOp();
    SmallVector<AMDAIE::BufferOp> buffers =
        sourceObjFifo.getBuffersOnTile(tileOp);
    SmallVector<AMDAIE::LockOp> producerLocks =
        sourceObjFifo.getProducerLocksOnTile(tileOp);
    SmallVector<AMDAIE::LockOp> consumerLocks =
        sourceObjFifo.getConsumerLocksOnTile(tileOp);
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
    if (failed(foldDimsAndReturnAsStatic(
            rewriter, deviceModel, sourceMixedSizes, sourceMixedStrides,
            canonicalizedSizes, canonicalizedStrides, repetitionCount.value(),
            sourceMemSpace, [&]() { return connectionOp->emitOpError(); }))) {
      return failure();
    };
    if (failed(createDMABlocks(rewriter, channel, connectionOp,
                               canonicalizedSizes, canonicalizedStrides, acqNum,
                               acqNum, sourceOffset, buffers, lockPair,
                               packetId))) {
      return sourceObjFifo.emitOpError() << "could not create DMA operations";
    }
  }
  return success();
}

/// Form DmaStartOp block for the connection op's target using the access
/// patterns of the half dma op.
static LogicalResult processConnectionTargetForDmaStart(
    IRRewriter &rewriter, AMDAIE::ConnectionOp connectionOp,
    std::optional<uint8_t> packetId, Value target,
    AMDAIE::AMDAIEDeviceModel &deviceModel, size_t targetOffset,
    uint8_t targetMemSpace, SmallVector<OpFoldResult> targetMixedSizes,
    SmallVector<OpFoldResult> targetMixedStrides) {
  SmallVector<AMDAIE::ChannelOp> consumerChannels;
  for (Value consumerChannel : connectionOp.getTargetChannels()) {
    auto channelOp =
        dyn_cast<AMDAIE::ChannelOp>(consumerChannel.getDefiningOp());
    if (!channelOp) {
      return connectionOp.emitOpError()
             << "found non-`amdaie.channel` target channel";
    }
    consumerChannels.push_back(channelOp);
  }
  auto targetObjFifo =
      dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromBuffersOp>(
          target.getDefiningOp());
  if (!targetObjFifo) {
    return connectionOp.emitOpError()
           << "expected target to be an "
              "`amdaie.logicalobjectfifo.from_buffers` op";
  }
  FailureOr<size_t> repetitionCount = getRepetitionCount(
      cast<LogicalObjFifoOpInterface>(targetObjFifo.getOperation()),
      /*reprogramDmas=*/true);
  if (failed(repetitionCount)) {
    return targetObjFifo->emitOpError()
           << "could not retrieve the repetition count";
  }
  SmallVector<CopyOpInterface> objFifoProducers =
      targetObjFifo.getCopyLikeProducers();
  SmallVector<CopyOpInterface> objFifoConsumers =
      targetObjFifo.getCopyLikeConsumers();
  // Default acquire/release value is 1. Will be adjusted depending on number
  // of producers/consumers.
  int acqNum = 1;
  if (objFifoProducers.size() < objFifoConsumers.size()) {
    assert(objFifoConsumers.size() % objFifoProducers.size() == 0 &&
           "expected total no. of consumers of objFifo to be divisible by the "
           "total no. of producers");
    acqNum = objFifoConsumers.size() / objFifoProducers.size();
  }
  for (AMDAIE::ChannelOp channel : consumerChannels) {
    AMDAIE::TileOp tileOp = channel.getTileOp();
    SmallVector<AMDAIE::BufferOp> buffers =
        targetObjFifo.getBuffersOnTile(tileOp);
    SmallVector<AMDAIE::LockOp> producerLocks =
        targetObjFifo.getProducerLocksOnTile(tileOp);
    SmallVector<AMDAIE::LockOp> consumerLocks =
        targetObjFifo.getConsumerLocksOnTile(tileOp);
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
    if (failed(foldDimsAndReturnAsStatic(
            rewriter, deviceModel, targetMixedSizes, targetMixedStrides,
            canonicalizedSizes, canonicalizedStrides, repetitionCount.value(),
            targetMemSpace, [&]() { return connectionOp->emitOpError(); }))) {
      return failure();
    };
    if (failed(createDMABlocks(rewriter, channel, connectionOp,
                               canonicalizedSizes, canonicalizedStrides, acqNum,
                               acqNum, targetOffset, buffers, lockPair,
                               packetId))) {
      return targetObjFifo.emitOpError() << "could not create DMA operations";
    }
  }
  return success();
}

/// Convert the `amdaie.connection` operation into DMA operations.
LogicalResult halfDmaToDmaStartBlocks(
    IRRewriter &rewriter, AMDAIE::AMDAIEDeviceModel &deviceModel,
    AMDAIE::NpuHalfDmaCpyNdOp dmaOp, uint8_t &connectionIndex,
    DenseMap<Value, Operation *> &tileToMemOpMap) {
  LLVM_DEBUG(llvm::dbgs() << "Convert [AMDAIE::ConnectionOp]\n");
  std::optional<AMDAIE::ConnectionOp> maybeConnectionOp =
      dmaOp.getConnectionOp();
  if (!maybeConnectionOp) {
    return failure();
  }
  AMDAIE::ConnectionOp connectionOp = *maybeConnectionOp;
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
  rewriter.setInsertionPoint(dmaOp);

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

  std::optional<size_t> maybeOffset = dmaOp.getStaticBaseOffset();
  if (!maybeOffset) {
    return dmaOp->emitOpError() << "could not compute a static base offset";
  }
  std::optional<uint8_t> maybeMemSpace = dmaOp.getMemorySpaceAsUInt();
  if (!maybeMemSpace) {
    return dmaOp->emitOpError() << "expected to have a memory space in input";
  }
  if (dmaOp.getInput() == source) {
    if (failed(processConnectionSourceForDmaStart(
            rewriter, connectionOp, packetId, source, deviceModel, *maybeOffset,
            *maybeMemSpace, dmaOp.getMixedSizes(), dmaOp.getMixedStrides()))) {
      return failure();
    }
  } else if (dmaOp.getInput() == target) {
    if (failed(processConnectionTargetForDmaStart(
            rewriter, connectionOp, packetId, target, deviceModel, *maybeOffset,
            *maybeMemSpace, dmaOp.getMixedSizes(), dmaOp.getMixedStrides()))) {
      return failure();
    }
  } else {
    return dmaOp->emitOpError()
           << "expected input of amdaie.npu.half_dma_cpy_nd to be either a "
              "source or a target of a connection op";
  }
  return success();
}

LogicalResult lowerDmasForReprogramming(
    Operation *moduleOp, AMDAIE::AMDAIEDeviceModel &deviceModel) {
  IRRewriter rewriter(moduleOp->getContext());
  OpBuilder::InsertionGuard guard(rewriter);
  Operation *controlCodeOp = nullptr;
  moduleOp->walk(
      [&](AMDAIE::WorkgroupOp op) { controlCodeOp = op.getControlCode(); });
  DenseMap<Value, Operation *> tileToMemOpMap;
  uint8_t connectionIndex{0};
  llvm::SmallSetVector<Operation *, 16> toBeErased;
  WalkResult res = controlCodeOp->walk<WalkOrder::PreOrder>(
      [&](AMDAIE::NpuHalfDmaCpyNdOp halfDmaOp) {
        std::optional<AMDAIE::BdIdOp> bdIdOp = halfDmaOp.getBdIdOp();
        if (bdIdOp) return WalkResult::advance();
        if (failed(halfDmaToDmaStartBlocks(rewriter, deviceModel, halfDmaOp,
                                           connectionIndex, tileToMemOpMap))) {
          return WalkResult::interrupt();
        }
        // Add the op and it's users as candidates to be erased later.
        for (Operation *userOp : halfDmaOp->getUsers()) {
          toBeErased.insert(userOp);
        }
        toBeErased.insert(halfDmaOp);
        return WalkResult::advance();
      });
  if (res.wasInterrupted()) return failure();

  for (Operation *op : toBeErased) rewriter.eraseOp(op);
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

  AMDAIE::AMDAIEDeviceModel deviceModel =
      AMDAIE::getDeviceModel(maybeDevice.value());

  // In case of DMA reprogramming we will be converting all npu.half_dma_cpy_nd
  // ops to dma_start blocks.
  if (reprogramDmas &&
      failed(lowerDmasForReprogramming(parentOp, deviceModel))) {
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
