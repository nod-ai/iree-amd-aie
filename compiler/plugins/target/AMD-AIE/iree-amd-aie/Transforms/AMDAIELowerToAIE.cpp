// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering from the AMDAIE dialect to AIE and AIEX
// dialects.
//
//===----------------------------------------------------------------------===//

#include "AMDAIELowerToAIE.h"

#include <memory>
#include <numeric>

#include "aie/AIEDialect.h"
#include "aie/AIEXDialect.h"
#include "iree-amd-aie/IR/AMDAIEAttrs.h"
#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEDmaUtils.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"

#define DEBUG_TYPE "iree-amdaie-lower-to-aie"

using namespace xilinx;

namespace mlir::iree_compiler::AMDAIE {

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

BDDimLayoutAndLength AIEDeviceBuilder::convertSizeStrideToBDDimLayoutArrayAttr(
    ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides) {
  assert(sizes.size() == strides.size() &&
         "expected stride and size vectors of same size");
  SmallVector<AIE::BDDimLayoutAttr, 4> bdDimLayoutAttr;
  // If the access pattern (strides/sizes) have a single dimension, make it
  // implicit with an empty `BDDimLayoutAttr` as this is what the AIE dialect
  // expects.
  if (strides.size() == 1 && strides[0] == 1) {
    return std::make_pair(AIE::BDDimLayoutArrayAttr::get(
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
        AIE::BDDimLayoutAttr::get(rewriter.getContext(), size, stride));
  }
  return std::make_pair(AIE::BDDimLayoutArrayAttr::get(
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
///    aie.use_lock(%lock_0_1_51, AcquireGreaterEqual, 2)
///    aie.dma_bd(%buffer_0_1_49 : memref<2048xi32, 1 : i32>) {len = 2048 : i32}
///    aie.use_lock(%lock_0_1_52, Release, 2)
///    aie.next_bd ^bb2
///  ^bb2:  // pred: ^bb1
///    aie.use_lock(%lock_0_1_51, AcquireGreaterEqual, 2)
///    aie.dma_bd(%buffer_0_1_50 : memref<2048xi32, 1 : i32>) {len = 2048 : i32}
///    aie.use_lock(%lock_0_1_52, Release, 2)
///    aie.next_bd ^bb1
LogicalResult AIEDeviceBuilder::createDMABlocks(
    Operation *memOp, AIE::DMAChannelDir channelDir, int channelIndex,
    ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, size_t acqNum,
    size_t relNum, int64_t offset, const SmallVector<AIE::BufferOp> &bufferOps,
    const std::pair<AIE::LockOp, AIE::LockOp> &locks,
    std::optional<uint8_t> pktId) {
  OpBuilder::InsertionGuard g(rewriter);

  Block &endBlock = memOp->getRegion(0).getBlocks().back();
  assert(!endBlock.getOps<AIE::EndOp>().empty() &&
         "expected last block to have aie.end");
  Block *lastDmaBlock = endBlock.getSinglePredecessor(),
        *dmaBlock = rewriter.createBlock(&endBlock),
        *bdBlock = rewriter.createBlock(&endBlock);

  // Create DMA channel.
  rewriter.setInsertionPointToStart(dmaBlock);
  rewriter.create<AIE::DMAStartOp>(rewriter.getUnknownLoc(), channelDir,
                                   channelIndex, /*repeatCount*/ 0, bdBlock,
                                   &endBlock);
  if (lastDmaBlock) lastDmaBlock->getTerminator()->setSuccessor(dmaBlock, 1);

  auto createDMAOps = [&](Block *succ, AIE::BufferOp buff,
                          AIE::BDDimLayoutArrayAttr dims, bool shouldAcqLock,
                          bool shouldRelLock, int64_t transferLength,
                          int64_t offset) {
    AIE::LockOp acqLock = locks.first, relLock = locks.second;
    if (shouldAcqLock) {
      rewriter.create<AIE::UseLockOp>(rewriter.getUnknownLoc(), acqLock,
                                      AIE::LockAction::AcquireGreaterEqual,
                                      acqNum);
    }
    // Insert a packet op for MM2S DMAs if part of a packet flow. Only do
    // this for MM2S DMA ports as only those can insert packet headers.
    if (channelDir == AIE::DMAChannelDir::MM2S && pktId.has_value()) {
      rewriter.create<AIE::DMABDPACKETOp>(rewriter.getUnknownLoc(),
                                          /*pkt_type*/ 0,
                                          /*pkt_id*/ pktId.value());
    }
    if (!dims.getValue().empty()) {
      rewriter.create<AIE::DMABDOp>(rewriter.getUnknownLoc(), buff, offset,
                                    transferLength, dims);
    } else {
      rewriter.create<AIE::DMABDOp>(rewriter.getUnknownLoc(), buff, offset,
                                    transferLength);
    }
    if (shouldRelLock) {
      rewriter.create<AIE::UseLockOp>(rewriter.getUnknownLoc(), relLock,
                                      AIE::LockAction::Release, relNum);
    }
    rewriter.create<AIE::NextBDOp>(rewriter.getUnknownLoc(), succ);
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

SmallVector<Operation *> AIEDeviceBuilder::createFlowOps(
    AMDAIE::FlowOp flowOp, ArrayRef<AMDAIE::ChannelOp> producerChannels,
    ArrayRef<AMDAIE::ChannelOp> consumerChannels) {
  LLVM_DEBUG(llvm::dbgs() << "-- createFlowOps\n");
  OpBuilder::InsertionGuard g(rewriter);
  SmallVector<Operation *> flowOps;
  for (AMDAIE::ChannelOp producerChannel : producerChannels) {
    Value aieProducerTile = mapper.lookup(producerChannel.getTile());
    std::optional<uint8_t> pktId = flowOp.getPacketId();
    if (pktId) {
      OpBuilder::InsertionGuard gg(rewriter);
      AIE::PacketFlowOp pktFlow = rewriter.create<AIE::PacketFlowOp>(
          rewriter.getUnknownLoc(), pktId.value(), nullptr, nullptr);
      Region &r_pktFlow = pktFlow.getPorts();
      Block *b_pktFlow = rewriter.createBlock(&r_pktFlow);
      rewriter.setInsertionPointToStart(b_pktFlow);
      rewriter.create<AIE::PacketSourceOp>(
          rewriter.getUnknownLoc(), aieProducerTile,
          producerChannel.getPortType(), producerChannel.getValue());
      for (AMDAIE::ChannelOp consumerChannel : consumerChannels) {
        Value aieConsumerTile = mapper.lookup(consumerChannel.getTile());
        rewriter.create<AIE::PacketDestOp>(
            rewriter.getUnknownLoc(), aieConsumerTile,
            consumerChannel.getPortType(), consumerChannel.getValue());
      }
      rewriter.create<AIE::EndOp>(rewriter.getUnknownLoc());
      flowOps.push_back(pktFlow.getOperation());
    } else {
      for (AMDAIE::ChannelOp consumerChannel : consumerChannels) {
        Value aieConsumerTile = mapper.lookup(consumerChannel.getTile());
        AIE::FlowOp flowOp = rewriter.create<AIE::FlowOp>(
            rewriter.getUnknownLoc(), aieProducerTile,
            producerChannel.getPortType(), producerChannel.getValue(),
            aieConsumerTile, consumerChannel.getPortType(),
            consumerChannel.getValue());
        flowOps.push_back(flowOp.getOperation());
      }
    }
  }
  return flowOps;
}

AIE::ShimDMAAllocationOp AIEDeviceBuilder::createShimDmaAllocation(
    Block *deviceBlock, AMDAIE::TileOp tileOp, AIE::DMAChannelDir dmaChannelDir,
    uint8_t channel, MemRefType memrefType, int &connectionIndex) {
  OpBuilder::InsertionGuard g(rewriter);
  auto shimDmaAllocOp = rewriter.create<AIE::ShimDMAAllocationOp>(
      rewriter.getUnknownLoc(), "shim_" + std::to_string(connectionIndex++),
      dmaChannelDir, channel, getConstantIndexOrAssert(tileOp.getCol()));
  rewriter.setInsertionPointToStart(deviceBlock);
  StringRef symName = shimDmaAllocOp.getSymName();
  rewriter.create<memref::GlobalOp>(rewriter.getUnknownLoc(), symName,
                                    rewriter.getStringAttr("public"),
                                    memrefType, nullptr, false, nullptr);
  return shimDmaAllocOp;
}

void AIEDeviceBuilder::eraseOp(Operation *op) {
  for (Value result : op->getResults()) mapper.erase(result);
  mapper.erase(op);
  op->dropAllUses();
  rewriter.eraseOp(op);
}

LogicalResult AIEDeviceBuilder::foldDimsAndReturnAsStatic(
    SmallVector<OpFoldResult> sizes, SmallVector<OpFoldResult> strides,
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

void AIEDeviceBuilder::remapOperands(Operation *op) {
  for (int i = 0; i < op->getNumOperands(); ++i) {
    Value operand = op->getOperand(i);
    if (mapper.contains(operand)) {
      op->setOperand(i, mapper.lookup(operand));
    }
  }
}

//===----------------------------------------------------------------------===//
// Convert `amdaie.core` op to `aie.core` op.
//===----------------------------------------------------------------------===//

LogicalResult AIEDeviceBuilder::coreMemrefExtractStridedMetadataToAIE(
    memref::ExtractStridedMetadataOp extractStridedMetadataOp,
    SmallVector<Operation *> &toBeErased) {
  LLVM_DEBUG(llvm::dbgs() << "Convert [memref.extract_strided_metadata]\n");
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(extractStridedMetadataOp);
  Value newSource =
      mapper.lookupOrDefault(extractStridedMetadataOp.getSource());
  memref::ExtractStridedMetadataOp newExtractStridedMetadataOp =
      rewriter.create<memref::ExtractStridedMetadataOp>(
          extractStridedMetadataOp.getLoc(), newSource);
  // Map old op to new op.
  rewriter.replaceAllUsesWith(extractStridedMetadataOp->getResults(),
                              newExtractStridedMetadataOp->getResults());
  toBeErased.push_back(extractStridedMetadataOp);
  return success();
}

LogicalResult AIEDeviceBuilder::coreFuncCallOpToAIE(
    func::CallOp oldCallOp, SmallVector<Operation *> &toBeErased) {
  LLVM_DEBUG(llvm::dbgs() << "Convert [func.call / function declaration]\n");
  // Form new argument(s) and function type for the func.call op.
  SmallVector<Value> newArgs;
  SmallVector<Type> newArgTypes;
  SmallVector<Type> newResultTypes;
  for (Value operand : oldCallOp.getOperands()) {
    Value newOperand = mapper.lookupOrDefault(operand);
    newArgs.push_back(newOperand);
    newArgTypes.push_back(newOperand.getType());
  }
  FunctionType newFunctionType =
      rewriter.getFunctionType(newArgTypes, newResultTypes);
  // Fetch name of the ukernel function to look up its declaration in the
  // Symbol table.
  auto moduleOp = oldCallOp->getParentOfType<ModuleOp>();
  StringRef fnName = oldCallOp.getCallee();
  auto fnDecl = dyn_cast_if_present<func::FuncOp>(
      SymbolTable::lookupSymbolIn(moduleOp, fnName));
  assert(fnDecl && "expected function declaration");
  // Check the mapper to see if we've already created a new function declaration
  // with the new function type. If not, create the same. We need to create a
  // new function declaration because the caller's function type has changed by
  // this point.
  if (!mapper.contains(fnDecl.getOperation())) {
    OpBuilder::InsertionGuard g(rewriter);
    auto symbolTableOp = SymbolTable::getNearestSymbolTable(oldCallOp);
    rewriter.setInsertionPointToStart(&symbolTableOp->getRegion(0).front());
    auto newFnDecl =
        rewriter.create<func::FuncOp>(fnDecl.getLoc(), fnName, newFunctionType);
    SymbolTable::setSymbolVisibility(newFnDecl,
                                     SymbolTable::Visibility::Private);
    newFnDecl->setAttr("llvm.bareptr", rewriter.getBoolAttr(true));
    fnDecl.getBody().cloneInto(&(newFnDecl.getBody()), mapper);
    mapper.map(fnDecl.getOperation(), newFnDecl.getOperation());
    fnDecl = newFnDecl;
  }
  // Fetch the new function declaration and create the new func.call op.
  auto newFnDecl = cast<func::FuncOp>(mapper.lookupOrDefault(fnDecl));
  rewriter.create<func::CallOp>(oldCallOp->getLoc(), newFnDecl, newArgs);
  toBeErased.push_back(oldCallOp);
  return success();
}

LogicalResult AIEDeviceBuilder::coreUseLockToAIE(
    AMDAIE::UseLockOp useLockOp, SmallVector<Operation *> &toBeErased) {
  LLVM_DEBUG(llvm::dbgs() << "Convert [AMDAIE::UseLockOp]\n");
  OpBuilder::InsertionGuard guard(rewriter);
  AIE::LockAction lockAction;
  if (useLockOp.getAction() == AMDAIE::LockAction::AcquireGreaterOrEqual) {
    lockAction = AIE::LockAction::AcquireGreaterEqual;
  } else if (useLockOp.getAction() == AMDAIE::LockAction::Acquire) {
    lockAction = AIE::LockAction::Acquire;
  } else if (useLockOp.getAction() == AMDAIE::LockAction::Release) {
    lockAction = AIE::LockAction::Release;
  } else {
    useLockOp.emitOpError() << "unsupported lock action in lowering to AIE: "
                            << stringifyEnum(useLockOp.getAction());
  }
  Value aieLock = mapper.lookup(useLockOp.getLock());
  rewriter.create<AIE::UseLockOp>(useLockOp.getLoc(), aieLock, lockAction,
                                  useLockOp.getValue());
  toBeErased.push_back(useLockOp);
  return success();
}

/// Convert `amdaie.core` into `aie.core`.
LogicalResult AIEDeviceBuilder::coreToAIE(AMDAIE::CoreOp coreOp,
                                          AIE::DeviceOp deviceOp,
                                          Block *deviceCoreBlock) {
  LLVM_DEBUG(llvm::dbgs() << "Convert [AMDAIE::CoreOp]\n");
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(deviceCoreBlock);

  // Create the AIE::CoreOp, copy all operations from AMDAIE::CoreOp and then
  // walk the new core's operations to convert them to AIE dialect operations.
  Block *coreBlock = coreOp.getBody();
  auto tileOp =
      dyn_cast<AIE::TileOp>(mapper.lookup(coreOp.getTileOp().getOperation()));
  if (!tileOp) {
    return coreOp.emitError()
           << "couldn't look up input `aie.tile` operation in IR map";
  }
  auto aieCoreOp =
      rewriter.create<AIE::CoreOp>(rewriter.getUnknownLoc(), tileOp);
  Region &aieCoreRegion = aieCoreOp.getBody();
  Block *aieCoreBlock = rewriter.createBlock(&aieCoreRegion);
  auto insertIt = aieCoreBlock->begin();
  auto coreBlockBegin = coreBlock->begin();
  auto coreBlockEnd = coreBlock->getTerminator()->getIterator();
  aieCoreBlock->getOperations().splice(insertIt, coreBlock->getOperations(),
                                       coreBlockBegin, coreBlockEnd);
  // Set the optional `link_with` attribute for ukernel path.
  aieCoreOp.setLinkWith(coreOp.getLinkWith());
  rewriter.setInsertionPointToEnd(aieCoreBlock);
  rewriter.create<AIE::EndOp>(rewriter.getUnknownLoc());

  SmallVector<Operation *> toBeErased;
  WalkResult walkResult = aieCoreOp.walk([&](Operation *op) {
    rewriter.setInsertionPoint(op);
    if (TypeSwitch<Operation *, LogicalResult>(op)
            .Case<memref::ExtractStridedMetadataOp>(
                [&](auto extractStridedMetadataOp) {
                  return coreMemrefExtractStridedMetadataToAIE(
                      extractStridedMetadataOp, toBeErased);
                })
            .Case<func::CallOp>([&](auto oldCallOp) {
              return coreFuncCallOpToAIE(oldCallOp, toBeErased);
            })
            .Case<AMDAIE::UseLockOp>([&](auto useLockOp) {
              return coreUseLockToAIE(useLockOp, toBeErased);
            })
            .Default([&](Operation *op) {
              remapOperands(op);
              return success();
            })
            .failed()) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    coreOp.emitError("could not convert to AIEDialect ops");
    return failure();
  }
  for (Operation *op : toBeErased) eraseOp(op);

  mapper.map(coreOp.getResult(), aieCoreOp.getResult());
  mapper.map(coreOp.getOperation(), aieCoreOp.getOperation());
  return success();
}

//===----------------------------------------------------------------------===//
// Convert ops in Workgroup to AIE ops
//===----------------------------------------------------------------------===//

/// Convert `amdaie.buffer` to `aie.buffer`.
LogicalResult AIEDeviceBuilder::bufferToAIE(AMDAIE::BufferOp bufferOp,
                                            Block *deviceBlock, int &bufferId) {
  LLVM_DEBUG(llvm::dbgs() << "Convert [AMDAIE::BufferOp]\n");
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(deviceBlock->getTerminator());
  auto elemType = cast<MemRefType>(bufferOp.getType());
  Value tile = mapper.lookup(bufferOp.getTile());
  auto aieBufferOp = rewriter.create<AIE::BufferOp>(
      bufferOp.getLoc(), elemType, tile,
      rewriter.getStringAttr("buff_" + std::to_string(bufferId++)),
      /*address*/ bufferOp.getAddressAttr(),
      /*mem_bank*/ nullptr);
  mapper.map(bufferOp.getResult(), aieBufferOp.getResult());
  mapper.map(bufferOp.getOperation(), aieBufferOp.getOperation());
  return success();
}

/// Convert the `amdaie.connection` operation into `aie.flow` ops and DMA
/// operations. Depending on the location of the source/target of the
/// connection, different DMA ops are created:
/// 1. Source/target on a Shim tile: iterate through producer/consumer channels
/// and create corresponding `aie.shim_dma_allocation` ops.
/// 2. Source/target on MemTile: iterate through producer/consumer channels,
/// lookup the correct `aie.memtile_dma` op and create new DMA BD blocks inside.
/// 3. Source/target on MemTile: iterate through producer/consumer channels,
/// lookup the correct `aie.mem` op and create new DMA BD blocks inside.
LogicalResult AIEDeviceBuilder::connectionToAIE(
    AMDAIE::ConnectionOp connectionOp, Block *deviceBlock,
    int &connectionIndex) {
  LLVM_DEBUG(llvm::dbgs() << "Convert [AMDAIE::ConnectionOp]\n");
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
  std::optional<uint8_t> packetId =
      maybeFlowOp ? maybeFlowOp->getPacketId() : std::nullopt;

  FailureOr<AMDAIE::NpuCircularDmaCpyNdOp> maybeNpuDmaUserOp =
      connectionOp.getNpuCircularDmaCpyNdUser();
  if (failed(maybeNpuDmaUserOp))
    return connectionOp.emitOpError() << "has no circular NPU DMA op user";

  SmallVector<Operation *> sourceMemOps;
  Value source = connectionOp.getSource();
  auto sourceObjFifoLikeOp =
      dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
          source.getDefiningOp());
  if (!sourceObjFifoLikeOp) {
    return connectionOp.emitOpError()
           << "expected source to be an logical objFifo-like op";
  }
  if (sourceObjFifoLikeOp.getMemorySpaceAsUInt() == 0) {
    for (AMDAIE::ChannelOp channel : producerChannels) {
      AIE::ShimDMAAllocationOp shimDmaAllocOp = createShimDmaAllocation(
          deviceBlock, channel.getTileOp(), AIE::DMAChannelDir::MM2S,
          channel.getValue(), sourceObjFifoLikeOp.getMemrefType(),
          connectionIndex);
      sourceMemOps.push_back(shimDmaAllocOp.getOperation());
    }
  } else {
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
      SmallVector<AIE::BufferOp> buffers = llvm::map_to_vector(
          sourceObjFifo.getBuffersOnTile(tileOp),
          [&](AMDAIE::BufferOp bufferOp) {
            return cast<AIE::BufferOp>(mapper.lookup(bufferOp.getOperation()));
          });
      SmallVector<AIE::LockOp> producerLocks = llvm::map_to_vector(
          sourceObjFifo.getProducerLocksOnTile(tileOp),
          [&](AMDAIE::LockOp lockOp) {
            return cast<AIE::LockOp>(mapper.lookup(lockOp.getOperation()));
          });
      SmallVector<AIE::LockOp> consumerLocks = llvm::map_to_vector(
          sourceObjFifo.getConsumerLocksOnTile(tileOp),
          [&](AMDAIE::LockOp lockOp) {
            return cast<AIE::LockOp>(mapper.lookup(lockOp.getOperation()));
          });
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
      std::pair<AIE::LockOp, AIE::LockOp> lockPair =
          std::make_pair(consumerLocks[0], producerLocks[0]);
      SmallVector<int64_t> canonicalizedSizes, canonicalizedStrides;
      if (failed(foldDimsAndReturnAsStatic(
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
              memOp, AIE::DMAChannelDir::MM2S, channel.getValue(),
              canonicalizedSizes, canonicalizedStrides, acqNum, acqNum,
              maybeOffset.value(), buffers, lockPair, packetId))) {
        return sourceObjFifo.emitOpError() << "could not create DMA operations";
      }
    }
  }

  SmallVector<Operation *> targetMemOps;
  Value target = connectionOp.getTarget();
  auto targetObjFifoLikeOp =
      dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
          target.getDefiningOp());
  if (!targetObjFifoLikeOp) {
    return connectionOp.emitOpError()
           << "expected target to be an logical objFifo-like op";
  }
  if (targetObjFifoLikeOp.getMemorySpaceAsUInt() == 0) {
    for (AMDAIE::ChannelOp channel : consumerChannels) {
      AIE::ShimDMAAllocationOp shimDmaAllocOp = createShimDmaAllocation(
          deviceBlock, channel.getTileOp(), AIE::DMAChannelDir::S2MM,
          channel.getValue(), targetObjFifoLikeOp.getMemrefType(),
          connectionIndex);
      targetMemOps.push_back(shimDmaAllocOp.getOperation());
    }
  } else {
    auto targetObjFifo =
        dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromBuffersOp>(
            target.getDefiningOp());
    if (!targetObjFifo) {
      return connectionOp.emitOpError()
             << "expected target to be an "
                "`amdaie.logicalobjectfifo.from_buffers` op";
    }
    FailureOr<size_t> repetitionCount = getRepetitionCount(
        cast<LogicalObjFifoOpInterface>(targetObjFifo.getOperation()));
    if (failed(repetitionCount)) {
      return targetObjFifo->emitOpError()
             << "could not retrieve the repetition count";
    }
    std::optional<size_t> maybeOffset =
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
    SmallVector<CopyOpInterface> objFifoProducers =
        targetObjFifo.getCopyLikeProducers();
    SmallVector<CopyOpInterface> objFifoConsumers =
        targetObjFifo.getCopyLikeConsumers();
    // Default acquire/release value is 1. Will be adjusted depending on number
    // of producers/consumers.
    int acqNum{1};
    if (objFifoProducers.size() < objFifoConsumers.size()) {
      assert(objFifoConsumers.size() % objFifoProducers.size() == 0);
      acqNum = objFifoConsumers.size() / objFifoProducers.size();
    }
    for (AMDAIE::ChannelOp channel : consumerChannels) {
      Operation *memOp = tileToMemOpMap.at(channel.getTile());
      AMDAIE::TileOp tileOp = channel.getTileOp();
      SmallVector<AIE::BufferOp> buffers = llvm::map_to_vector(
          targetObjFifo.getBuffersOnTile(tileOp),
          [&](AMDAIE::BufferOp bufferOp) {
            return cast<AIE::BufferOp>(mapper.lookup(bufferOp.getOperation()));
          });
      SmallVector<AIE::LockOp> producerLocks = llvm::map_to_vector(
          targetObjFifo.getProducerLocksOnTile(tileOp),
          [&](AMDAIE::LockOp lockOp) {
            return cast<AIE::LockOp>(mapper.lookup(lockOp.getOperation()));
          });
      SmallVector<AIE::LockOp> consumerLocks = llvm::map_to_vector(
          targetObjFifo.getConsumerLocksOnTile(tileOp),
          [&](AMDAIE::LockOp lockOp) {
            return cast<AIE::LockOp>(mapper.lookup(lockOp.getOperation()));
          });
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
      std::pair<AIE::LockOp, AIE::LockOp> lockPair =
          std::make_pair(producerLocks[0], consumerLocks[0]);
      SmallVector<int64_t> canonicalizedSizes, canonicalizedStrides;
      if (failed(foldDimsAndReturnAsStatic(
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
              memOp, AIE::DMAChannelDir::S2MM, channel.getValue(),
              canonicalizedSizes, canonicalizedStrides, acqNum, acqNum,
              maybeOffset.value(), buffers, lockPair, packetId))) {
        return targetObjFifo.emitOpError() << "could not create DMA operations";
      }
    }
  }

  // Keep track of source/target mem ops for this connection for later retrieval
  // to create NPU ops.
  connectionToSourceTargetMemOps[connectionOp] =
      std::make_pair(sourceMemOps, targetMemOps);
  return success();
}

/// Convert the `amdaie.flow` ops into `aie.flow` ops.
LogicalResult AIEDeviceBuilder::flowToAIE(AMDAIE::FlowOp flowOp,
                                          Block *deviceBlock) {
  LLVM_DEBUG(llvm::dbgs() << "Convert [AMDAIE::ConnectionOp]\n");
  rewriter.setInsertionPoint(deviceBlock->getTerminator());
  SmallVector<AMDAIE::ChannelOp> producerChannels;
  SmallVector<AMDAIE::ChannelOp> consumerChannels;
  for (Value producerChannel : flowOp.getSources()) {
    auto channelOp =
        dyn_cast_if_present<AMDAIE::ChannelOp>(producerChannel.getDefiningOp());
    if (!channelOp) {
      return flowOp.emitOpError()
             << "found non-`amdaie.channel` source channel";
    }
    producerChannels.push_back(channelOp);
  }
  for (Value consumerChannel : flowOp.getTargets()) {
    auto channelOp =
        dyn_cast_if_present<AMDAIE::ChannelOp>(consumerChannel.getDefiningOp());
    if (!channelOp) {
      return flowOp.emitOpError()
             << "found non-`amdaie.channel` target channel";
    }
    consumerChannels.push_back(channelOp);
  }
  // Insert flow ops.
  rewriter.setInsertionPoint(deviceBlock->getTerminator());
  SmallVector<Operation *> flowOps =
      createFlowOps(flowOp, producerChannels, consumerChannels);
  return success();
}

LogicalResult AIEDeviceBuilder::lockToAIE(AMDAIE::LockOp lockOp,
                                          Block *deviceBlock, int &lockIndex) {
  LLVM_DEBUG(llvm::dbgs() << "Convert [AMDAIE::LockOp]\n");
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(deviceBlock->getTerminator());
  Value tile = mapper.lookup(lockOp.getTile());
  auto aieLockOp = rewriter.create<AIE::LockOp>(
      lockOp.getLoc(), tile, lockOp.getValueAttr(), lockOp.getInitValueAttr(),
      rewriter.getStringAttr("lock_" + std::to_string(lockIndex++)));
  mapper.map(lockOp.getResult(), aieLockOp.getResult());
  mapper.map(lockOp.getOperation(), aieLockOp.getOperation());
  return success();
}

template <typename MemOp>
LogicalResult logicalObjFifoFromBuffersToMemOp(
    IRRewriter &rewriter, AMDAIE::LogicalObjectFifoFromBuffersOp logicalObjFifo,
    IRMapping &mapper, Block *deviceBlock,
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
    Value aieTile = mapper.lookup(tile);
    rewriter.setInsertionPoint(deviceBlock->getTerminator());
    auto newMemOp = rewriter.create<MemOp>(rewriter.getUnknownLoc(), aieTile);
    rewriter.setInsertionPointToStart(&newMemOp.getRegion().emplaceBlock());
    rewriter.create<AIE::EndOp>(rewriter.getUnknownLoc());
    // Keep track of the MemOps on different tiles.
    tileToMemOpMap[tile] = newMemOp.getOperation();
  }
  return success();
}

LogicalResult AIEDeviceBuilder::logicalObjFifoFromBuffersToAIE(
    AMDAIE::LogicalObjectFifoFromBuffersOp logicalObjFifo, Block *deviceBlock) {
  LLVM_DEBUG(
      llvm::dbgs() << "Convert [AMDAIE::LogicalObjectFifoFromBuffersOp]\n");
  uint8_t memSpaceUInt = logicalObjFifo.getMemorySpaceAsUInt();
  if (memSpaceUInt == 1) {
    // L2
    return logicalObjFifoFromBuffersToMemOp<AIE::MemTileDMAOp>(
        rewriter, logicalObjFifo, mapper, deviceBlock, tileToMemOpMap);
  } else if (memSpaceUInt == 2) {
    // L1
    return logicalObjFifoFromBuffersToMemOp<AIE::MemOp>(
        rewriter, logicalObjFifo, mapper, deviceBlock, tileToMemOpMap);
  } else {
    return logicalObjFifo.emitOpError()
           << "has unsupported memory space for lowering to AIE: "
           << std::to_string(memSpaceUInt);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Convert `amdaie.tile` operation to `aie.tile`
//===----------------------------------------------------------------------===//

LogicalResult AIEDeviceBuilder::tileToAIE(AMDAIE::TileOp tileOp,
                                          Block *deviceBlock) {
  LLVM_DEBUG(llvm::dbgs() << "Convert [AMDAIE::TileOp]\n");
  OpBuilder::InsertionGuard guard(rewriter);
  int64_t col = getConstantIntValue(tileOp.getCol()).value();
  int64_t row = getConstantIntValue(tileOp.getRow()).value();
  rewriter.setInsertionPointToStart(deviceBlock);
  auto aieTileOp =
      rewriter.create<xilinx::AIE::TileOp>(rewriter.getUnknownLoc(), col, row);
  mapper.map(tileOp.getResult(), aieTileOp.getResult());
  mapper.map(tileOp.getOperation(), aieTileOp.getOperation());
  return success();
}

//===----------------------------------------------------------------------===//
// Convert amdaie.workgroup operation and insert into aie.device
//===----------------------------------------------------------------------===//

LogicalResult AIEDeviceBuilder::workgroupToAIE(AMDAIE::WorkgroupOp workgroupOp,
                                               xilinx::AIE::DeviceOp deviceOp) {
  OpBuilder::InsertionGuard guard(rewriter);
  Block *deviceBlock = &deviceOp.getRegion().front();
  Block *deviceCoreBlock = rewriter.createBlock(&deviceOp.getRegion());
  rewriter.setInsertionPoint(deviceBlock, deviceBlock->begin());

  // Walk all operations in the AIE region and convert to AIE ops
  int bufferId{0};
  int lockId{0};
  int connectionIndex{0};
  WalkResult res = workgroupOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
    return TypeSwitch<Operation *, WalkResult>(op)
        .Case<AMDAIE::BdIdOp>([&](auto bdIdOp) {
          // BD ID ops are purely used for retrieving information in other ops
          // so don't convert to AIE dialect.
          return WalkResult::advance();
        })
        .Case<AMDAIE::BufferOp>([&](auto bufferOp) {
          if (failed(bufferToAIE(bufferOp, deviceBlock, bufferId))) {
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        })
        .Case<AMDAIE::ChannelOp>([&](auto channelOp) {
          // Channel ops are purely used for retrieving information in other ops
          // so don't convert to AIE dialect.
          return WalkResult::advance();
        })
        .Case<AMDAIE::CircularDmaCpyNdOp>([&](auto dmaOp) {
          dmaOp.emitOpError()
              << "`amdaie.circular_dma_cpy_nd` unsupported in lowering to AIE";
          return WalkResult::interrupt();
        })
        .Case<AMDAIE::ConnectionOp>([&](auto dmaOp) {
          if (failed(connectionToAIE(dmaOp, deviceBlock, connectionIndex))) {
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        })
        .Case<AMDAIE::ControlCodeOp>([&](auto controlCodeOp) {
          // Skip control code as it should already be translated into firmware
          // code at this point.
          // TODO(jornt): currently, it still contains ops that are needed in
          // this translation, but don't have to be translated themselves.
          return WalkResult::skip();
        })
        .Case<AMDAIE::CoreOp>([&](auto coreOp) {
          if (failed(coreToAIE(coreOp, deviceOp, deviceCoreBlock))) {
            coreOp.emitError("could not convert to AIEDialect ops");
            return WalkResult::interrupt();
          }
          return WalkResult::skip();
        })
        .Case<AMDAIE::FlowOp>([&](auto flowOp) {
          if (failed(flowToAIE(flowOp, deviceBlock))) {
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        })
        .Case<AMDAIE::LockOp>([&](auto lockOp) {
          if (failed(lockToAIE(lockOp, deviceBlock, lockId))) {
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        })
        .Case<AMDAIE::LogicalObjectFifoFromBuffersOp>([&](auto logicalObjFifo) {
          if (failed(logicalObjFifoFromBuffersToAIE(logicalObjFifo,
                                                    deviceBlock))) {
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        })
        .Case<AMDAIE::LogicalObjectFifoPlaceholderOp>([&](auto logicalObjFifo) {
          // Skip placeholder ops as they don't have an equivalent in the
          // AIE dialect and shim dma allocations are created from
          // connections directly currently.
          return WalkResult::advance();
        })
        .Case<AMDAIE::TileOp>([&](auto tileOp) {
          if (failed(tileToAIE(tileOp, deviceBlock))) {
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        })
        .Case<AMDAIE::WorkgroupOp>([&](auto workgroupOp) {
          // Skip workgroup ops themselves.
          return WalkResult::advance();
        })
        .Default([&](Operation *op) {
          rewriter.setInsertionPoint(deviceBlock->getTerminator());
          if (!isa_and_present<AMDAIEDialect>(op->getDialect())) {
            rewriter.clone(*op, mapper);
          } else {
            op->emitOpError() << "is unsupported in lowering to AIE dialect";
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
  });
  if (res.wasInterrupted()) return failure();

  // Merge core operations into end of the device block
  rewriter.inlineBlockBefore(deviceCoreBlock, deviceBlock,
                             deviceBlock->without_terminator().end());
  return success();
}

//===----------------------------------------------------------------------===//
// Convert the module operation's contents to the AIE dialect
//===----------------------------------------------------------------------===//

/// Convert a `ModuleOp` contents to the AIE dialect by inserting a
/// `AIE::DeviceOp` into the module for every encountered `FuncOp`, and then
/// traverse the function build the AIE device operation and convert all AMDAIE
/// dialect operations to AIE dialect operations.
LogicalResult AIEDeviceBuilder::lowerToAIE(ModuleOp moduleOp) {
  Block *moduleBlock = &moduleOp->getRegion(0).front();
  xilinx::AIE::AIEDevice aieDevice = static_cast<xilinx::AIE::AIEDevice>(
      static_cast<uint32_t>(deviceModel.device));

  auto funcRes = moduleOp.walk([&](func::FuncOp funcOp) {
    if (funcOp.isPrivate()) {
      return WalkResult::advance();
    }

    // Create aie.device.
    rewriter.setInsertionPoint(moduleBlock, moduleBlock->begin());
    auto deviceOp = rewriter.create<xilinx::AIE::DeviceOp>(
        rewriter.getUnknownLoc(),
        xilinx::AIE::AIEDeviceAttr::get(rewriter.getContext(), aieDevice));
    xilinx::AIE::DeviceOp::ensureTerminator(deviceOp.getRegion(), rewriter,
                                            deviceOp.getLoc());
    Block *deviceBlock = deviceOp.getBody();
    rewriter.setInsertionPoint(deviceBlock, deviceBlock->begin());

    // Create aiex.runtime_sequence inside aie.device
    auto npuFuncOp = rewriter.create<xilinx::AIEX::RuntimeSequenceOp>(
        rewriter.getUnknownLoc(), rewriter.getStringAttr(funcOp.getSymName()));
    Region &body = npuFuncOp.getBody();
    body.emplaceBlock();

    // Walk the AIE regions ops and convert ops into pure AIEDialect ops.
    // IRMapping mapper;
    rewriter.setInsertionPointToStart(deviceBlock);
    WalkResult res = funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<func::FuncOp, func::ReturnOp>(op)) {
        return WalkResult::advance();
      } else if (auto workgroupOp = dyn_cast<AMDAIE::WorkgroupOp>(op)) {
        if (failed(workgroupToAIE(workgroupOp, deviceOp))) {
          return WalkResult::interrupt();
        }
        return WalkResult::skip();
      } else {
        if (!isa_and_present<AMDAIEDialect>(op->getDialect())) {
          rewriter.clone(*op, mapper);
        }
      }
      return WalkResult::advance();
    });
    if (res.wasInterrupted()) return WalkResult::interrupt();

    SmallVector<AMDAIE::WorkgroupOp> workgroupOps;
    funcOp->walk([&](AMDAIE::WorkgroupOp op) { workgroupOps.push_back(op); });
    // Only a single workgroup op is supported as only a single `aie.device` is
    // created.
    if (workgroupOps.size() > 1) {
      funcOp.emitOpError()
          << "multiple `amdaie.workgroup` ops is not supported";
      return WalkResult::interrupt();
    }
    if (workgroupOps.size() == 1) {
      AMDAIE::WorkgroupOp workgroupOp = workgroupOps[0];
      mlir::Attribute maybeNpuInstructions =
          workgroupOp.getNpuInstructionsAttr();
      // Only add attributes if the instructions attribute is found to
      // facilitate simplified tests.
      if (maybeNpuInstructions) {
        deviceOp->setAttr("npu_instructions", maybeNpuInstructions);
        deviceOp->setAttr("runtime_sequence_name",
                          rewriter.getStringAttr(funcOp.getSymName()));
      }
    }

    // Move NPU instruction function to the end of the device block.
    rewriter.moveOpBefore(npuFuncOp, deviceBlock,
                          deviceBlock->without_terminator().end());
    // After walking the FuncOp, it has been converted into a DeviceOp and can
    // safely be erased.
    eraseOp(funcOp);
    return WalkResult::advance();
  });
  if (funcRes.wasInterrupted()) return failure();

  // All Ukernel related function declarations will be within aie.device, so
  // delete the ones outside from the SymbolTable.
  SymbolTable symbolTable(moduleOp);
  moduleOp->walk([&](func::FuncOp funcOp) {
    if (funcOp.isPrivate() && !funcOp->getParentOfType<AIE::DeviceOp>()) {
      symbolTable.erase(funcOp);
    }
  });

  SmallVector<Operation *> opsToBeErased;
  moduleOp.walk([&](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
    opsToBeErased.push_back(subspanOp.getOperation());
    SmallVector<Operation *> userQueue(subspanOp->getUsers().begin(),
                                       subspanOp->getUsers().end());
    while (!userQueue.empty()) {
      Operation *current = userQueue.pop_back_val();
      opsToBeErased.push_back(current);
      userQueue.insert(userQueue.end(), current->getUsers().begin(),
                       current->getUsers().end());
    }
  });

  for (Operation *op : llvm::reverse(opsToBeErased)) rewriter.eraseOp(op);
  return success();
}

class AMDAIELowerToAIEPass
    : public impl::AMDAIELowerToAIEBase<AMDAIELowerToAIEPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect, AMDAIEDialect,
                    xilinx::AIE::AIEDialect, xilinx::AIEX::AIEXDialect>();
  }

  void runOnOperation() override {
    // Main function call to convert all operations into AIE dialect
    // operations inside an AIE device.
    ModuleOp moduleOp = getOperation();
    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(moduleOp);
    std::optional<AMDAIEDevice> maybeDevice = getConfigAMDAIEDevice(targetAttr);
    if (!maybeDevice.has_value()) {
      moduleOp->emitOpError(
          "No AMDAIEDevice found in the target attribute configuration. This "
          "is needed to lower to the AIE dialect.");
      return signalPassFailure();
    }
    AMDAIEDeviceModel deviceModel = getDeviceModel(maybeDevice.value());
    AIEDeviceBuilder builder(moduleOp.getContext(), std::move(deviceModel));
    if (failed(builder.lowerToAIE(moduleOp))) return signalPassFailure();
  }
};

std::unique_ptr<Pass> createAMDAIELowerToAIEPass() {
  return std::make_unique<AMDAIELowerToAIEPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
