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
#include "iree-amd-aie/Transforms/AMDAIEDmaUtils.h"
#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/PassManager.h"

#define DEBUG_TYPE "iree-amdaie-lower-to-aie"

using namespace xilinx;

namespace mlir::iree_compiler::AMDAIE {

//===----------------------------------------------------------------------===//
// AIEDeviceBuilder utilities
//===----------------------------------------------------------------------===//

AIE::BDDimLayoutArrayAttr
AIEDeviceBuilder::convertSizeStrideToBDDimLayoutArrayAttr(
    const SmallVector<OpFoldResult> &sizes,
    const SmallVector<OpFoldResult> &strides) {
  assert(sizes.size() == strides.size() &&
         "expected stride and size vectors of same size");
  // Fold remaining dimensions, assuming zero offsets as offsets should be taken
  // care of separately.
  SmallVector<OpFoldResult> offsets(
      strides.size(), getAsIndexOpFoldResult(rewriter.getContext(), 0));
  SmallVector<OpFoldResult> newOffsets;
  SmallVector<OpFoldResult> newSizes;
  SmallVector<OpFoldResult> newStrides;
  foldDims(offsets, sizes, strides, newOffsets, newSizes, newStrides);

  SmallVector<AIE::BDDimLayoutAttr, 4> bdDimLayoutAttr;
  // If the access pattern (strides/sizes) have a single dimension, make it
  // implicit with an empty `BDDimLayoutAttr` as this is what the AIE dialect
  // expects.
  if (newStrides.size() == 1) {
    std::optional<int64_t> stride = getConstantIntValue(newStrides[0]);
    if (stride && stride.value() == 1) {
      return AIE::BDDimLayoutArrayAttr::get(rewriter.getContext(),
                                            ArrayRef(bdDimLayoutAttr));
    }
  }
  bdDimLayoutAttr.reserve(newSizes.size());
  for (auto [size, stride] : llvm::zip(newSizes, newStrides)) {
    bdDimLayoutAttr.push_back(AIE::BDDimLayoutAttr::get(
        rewriter.getContext(), getConstantIntValue(size).value(),
        getConstantIntValue(stride).value()));
  }
  return AIE::BDDimLayoutArrayAttr::get(rewriter.getContext(),
                                        ArrayRef(bdDimLayoutAttr));
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
void AIEDeviceBuilder::createDMA(
    Operation *memOp, AIE::DMAChannelDir channelDir, int channelIndex,
    AIE::BDDimLayoutArrayAttr dims, size_t acqNum, size_t relNum, int64_t len,
    int64_t offset, const SmallVector<AIE::BufferOp> &bufferOps,
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

  auto createBdBlockOps = [&](AIE::BufferOp buff, Block *succ) {
    AIE::LockOp acqLock = locks.first, relLock = locks.second;
    rewriter.create<AIE::UseLockOp>(rewriter.getUnknownLoc(), acqLock,
                                    AIE::LockAction::AcquireGreaterEqual,
                                    acqNum);
    // Insert a packet op for MM2S DMAs if part of a packet flow. Only do this
    // for MM2S DMA ports as only those can insert packet headers.
    if (channelDir == AIE::DMAChannelDir::MM2S && pktId) {
      rewriter.create<AIE::DMABDPACKETOp>(rewriter.getUnknownLoc(),
                                          /*pkt_type*/ 0,
                                          /*pkt_id*/ pktId.value());
    }
    if (!dims.getValue().empty()) {
      rewriter.create<AIE::DMABDOp>(rewriter.getUnknownLoc(), buff, offset, len,
                                    dims);
    } else {
      rewriter.create<AIE::DMABDOp>(rewriter.getUnknownLoc(), buff, offset,
                                    len);
    }
    rewriter.create<AIE::UseLockOp>(rewriter.getUnknownLoc(), relLock,
                                    AIE::LockAction::Release, relNum);
    rewriter.create<AIE::NextBDOp>(rewriter.getUnknownLoc(), succ);
  };

  // Create Bd blocks.
  Block *succ = nullptr, *curr = bdBlock;
  for (size_t blockIndex = 0; blockIndex < bufferOps.size(); ++blockIndex) {
    if (blockIndex == bufferOps.size() - 1) {
      succ = bdBlock;
    } else {
      succ = rewriter.createBlock(&endBlock);
    }
    rewriter.setInsertionPointToStart(curr);
    createBdBlockOps(bufferOps[blockIndex], succ);
    curr = succ;
  }
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
            rewriter.getUnknownLoc(), aieProducerTile, AIE::WireBundle::DMA,
            producerChannel.getValue(), aieConsumerTile, AIE::WireBundle::DMA,
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

void AIEDeviceBuilder::foldDims(const SmallVector<OpFoldResult> &offsets,
                                const SmallVector<OpFoldResult> &sizes,
                                const SmallVector<OpFoldResult> &strides,
                                SmallVector<OpFoldResult> &newOffsets,
                                SmallVector<OpFoldResult> &newSizes,
                                SmallVector<OpFoldResult> &newStrides) {
  SmallVector<OpFoldResult> tmpOffsets;
  SmallVector<OpFoldResult> tmpSizes;
  SmallVector<OpFoldResult> tmpStrides;
  (void)foldUnitDims(offsets, sizes, strides, tmpOffsets, tmpSizes, tmpStrides);
  (void)foldLinearDims(rewriter.getContext(), tmpOffsets, tmpSizes, tmpStrides,
                       newOffsets, newSizes, newStrides);
  (void)foldSingleDim(newOffsets, newSizes, newStrides);
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
// Convert amdaie.controlcode operation to NPU instruction func
//===----------------------------------------------------------------------===//

/// Convert the `amdaie.npu.dma_cpy_nd` operation to `aiex.npu.dma_memcpy_nd`.
LogicalResult AIEDeviceBuilder::npuDmaCpyNdOpToAIE(
    AMDAIE::NpuDmaCpyNdOp dmaOp, SmallVector<Operation *> &toBeErased) {
  LLVM_DEBUG(llvm::dbgs() << "Convert [AMDAIE::NpuDmaCpyNdOp]\n");
  AMDAIE::ConnectionOp connectionOp = dmaOp.getConnectionOp();

  SmallVector<Value> offsets, sizes, strides;
  ArrayRef<int64_t> staticOffsets, staticSizes, staticStrides;
  AMDAIE::BdIdOp bdIdOp;
  LogicalObjectFifoFromMemrefOp logicalObjFifo;
  SmallVector<Operation *> memOps;
  AIE::PacketInfoAttr pktInfoAttr = nullptr;
  // Convert bidirectional `amdaie.npu.dma_cpy_nd` op into two halves.
  if (dmaOp.getSource()) {
    offsets = dmaOp.getSourceOffsets();
    sizes = dmaOp.getSourceSizes();
    strides = dmaOp.getSourceStrides();
    staticOffsets = dmaOp.getSourceStaticOffsets();
    staticSizes = dmaOp.getSourceStaticSizes();
    staticStrides = dmaOp.getSourceStaticStrides();
    bdIdOp = dmaOp.getSourceBdIdOp();
    if (!bdIdOp) {
      return dmaOp.emitOpError()
             << "must have a source BD ID op to lower to the AIE dialect.";
    }
    logicalObjFifo = dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromMemrefOp>(
        dmaOp.getSource().getDefiningOp());
    if (!logicalObjFifo) {
      return dmaOp.emitOpError() << "expected source to be an "
                                    "`amdaie.logicalobjectfifo.from_memref`";
    }
    memOps = connectionToSourceTargetMemOps[connectionOp].first;
    // Set the packet info attribute for MM2S DMAs, operating on a packet flow
    // connection.
    std::optional<AMDAIE::FlowOp> maybeFlowOp = connectionOp.getFlowOp();
    if (maybeFlowOp && maybeFlowOp->getPacketId()) {
      pktInfoAttr = AIE::PacketInfoAttr::get(
          rewriter.getContext(),
          /*pkt_type*/ 0, /*pkt_id*/ maybeFlowOp->getPacketId().value());
    }
  } else if (dmaOp.getTarget()) {
    offsets = dmaOp.getTargetOffsets();
    sizes = dmaOp.getTargetSizes();
    strides = dmaOp.getTargetStrides();
    staticOffsets = dmaOp.getTargetStaticOffsets();
    staticSizes = dmaOp.getTargetStaticSizes();
    staticStrides = dmaOp.getTargetStaticStrides();
    bdIdOp = dmaOp.getTargetBdIdOp();
    if (!bdIdOp) {
      return dmaOp.emitOpError()
             << "must have a target BD ID op to lower to the AIE dialect.";
    }
    logicalObjFifo = dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromMemrefOp>(
        dmaOp.getTarget().getDefiningOp());
    if (!logicalObjFifo) {
      return dmaOp.emitOpError() << "expected target to be an "
                                    "`amdaie.logicalobjectfifo.from_memref`";
    }
    memOps = connectionToSourceTargetMemOps[connectionOp].second;
  } else {
    return dmaOp.emitOpError()
           << "has neither source not target memory space as L3.";
  }

  Value memref = bindingsMapper.lookup(logicalObjFifo.getMemref());

  if (memOps.size() != 1) {
    return dmaOp.emitOpError() << "only a single connection op source expected";
  }
  auto shimDmaAllocOp = dyn_cast<AIE::ShimDMAAllocationOp>(memOps[0]);
  if (!shimDmaAllocOp) {
    return dmaOp.emitOpError() << "expected the source of the connection to "
                                  "be mapped to a `AIE::ShimDMAAllocationOp`";
  }

  if (!offsets.empty() || !sizes.empty() || !strides.empty()) {
    // Not doing now as better to just eliminate use of aiex dialect
    // altogether.
    return dmaOp.emitError()
           << "Expect all source offsets, sizes, and strides to be static at "
              "this point. Dynamic values can be supported, just need to "
              "cast from 'index' to 64-bit signless integer for "
              "aiex.npu.dma_memcpy_nd.";
  }

  uint32_t bdId = bdIdOp.getValue();
  bool issueToken = dmaOp.hasDmaWaitOpUser();

  rewriter.setInsertionPoint(dmaOp);
  rewriter.create<AIEX::NpuDmaMemcpyNdOp>(
      dmaOp.getLoc(), SmallVector<Type, 1>{}, 0, 0, memref, offsets, sizes,
      strides, staticOffsets, staticSizes, staticStrides, pktInfoAttr,
      shimDmaAllocOp.getSymName(), bdId, issueToken);

  toBeErased.push_back(dmaOp);
  return success();
}

/// Convert the `amdaie.npu.dma_wait` operation to `aiex.npu.dma_wait`.
LogicalResult AIEDeviceBuilder::npuDmaWaitToAIE(
    AMDAIE::NpuDmaWaitOp waitOp, SmallVector<Operation *> &toBeErased) {
  LLVM_DEBUG(llvm::dbgs() << "Convert [AMDAIE::NpuDmaWaitOp]\n");
  rewriter.setInsertionPoint(waitOp);
  for (Value asyncToken : waitOp.getAsyncTokens()) {
    auto npuDmaOp =
        dyn_cast_if_present<NpuDmaCpyNdOp>(asyncToken.getDefiningOp());
    if (!npuDmaOp) {
      return waitOp.emitOpError()
             << "should be operating on `amdaie.npu.dma_cpy_nd` for "
                "lowering";
    }
    AMDAIE::ConnectionOp connectionOp = npuDmaOp.getConnectionOp();
    if (!connectionToSourceTargetMemOps.contains(connectionOp)) {
      return connectionOp.emitOpError() << "should be found in the connection "
                                           "to source/target mem ops map";
    }
    SmallVector<Operation *> memOps =
        isa<AMDAIE::AsyncSourceTokenType>(asyncToken.getType())
            ? connectionToSourceTargetMemOps[connectionOp].first
            : connectionToSourceTargetMemOps[connectionOp].second;
    if (memOps.size() != 1) {
      return waitOp.emitOpError()
             << "only a single connection op source expected";
    }
    auto shimDmaAllocOp = dyn_cast<AIE::ShimDMAAllocationOp>(memOps[0]);
    if (!shimDmaAllocOp) {
      return waitOp.emitOpError()
             << "expected the source of the connection to "
                "be mapped to a `AIE::ShimDMAAllocationOp`";
    }
    rewriter.create<AIEX::NpuDmaWaitOp>(rewriter.getUnknownLoc(),
                                        shimDmaAllocOp.getSymName());
  }
  toBeErased.push_back(waitOp);
  return success();
}

/// Insert the control code operations into the NPU instruction function.
LogicalResult AIEDeviceBuilder::controlCodeToAIE(
    AMDAIE::ControlCodeOp controlCodeOp,
    xilinx::AIEX::RuntimeSequenceOp funcOp) {
  LLVM_DEBUG(llvm::dbgs() << "Convert [AMDAIE::ControlCodeOp]\n");
  Block *funcBlock = &funcOp.getBody().front();
  rewriter.setInsertionPointToEnd(funcBlock);
  auto insertIt = funcBlock->begin();
  auto controlCodeBegin = controlCodeOp.getBody()->begin();
  auto controlCodeEnd = controlCodeOp.getBody()->getTerminator()->getIterator();
  funcBlock->getOperations().splice(insertIt,
                                    controlCodeOp.getBody()->getOperations(),
                                    controlCodeBegin, controlCodeEnd);

  // Keep track of operations to be erased instead of erasing them directly as
  // there are bidirectional dependencies between operations. For example,
  // `amdaie.npu.dma_cpy_nd` potentially needs information from a sunsequent
  // `amdaie.npu.dma_wait` operation user and vice versa.
  // TODO(jornt): This is caused by differences between the `AMDAIE` dialect and
  // the `AIE` dialect and can be streamlined later by adjusting (both)
  // dialects.
  SmallVector<Operation *> toBeErased;
  WalkResult res =
      funcOp->walk<WalkOrder::PostOrder, ReverseIterator>([&](Operation *op) {
        if (TypeSwitch<Operation *, LogicalResult>(op)
                .Case<AMDAIE::NpuCircularDmaCpyNdOp>([&](auto dmaOp) {
                  // TODO(jornt): This is temporarily handled already by
                  // combining with `ConnectionOp` to create `aie.objectfifo`
                  // until we get rid of those.
                  eraseOp(dmaOp);
                  return success();
                })
                .Case<AMDAIE::NpuDmaCpyNdOp>([&](auto dmaOp) {
                  return npuDmaCpyNdOpToAIE(dmaOp, toBeErased);
                })
                .Case<AMDAIE::NpuDmaWaitOp>([&](auto waitOp) {
                  return npuDmaWaitToAIE(waitOp, toBeErased);
                })
                .Case<AMDAIE::EndOp>([&](auto endOp) {
                  eraseOp(endOp);
                  return success();
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
  if (res.wasInterrupted()) return failure();
  for (Operation *op : toBeErased) eraseOp(op);
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
  rewriter.setInsertionPointToEnd(deviceBlock);
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
  rewriter.setInsertionPointToEnd(deviceBlock);
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
    std::optional<size_t> maybeSize = maybeNpuDmaUserOp->getSourceStaticSize();
    if (!maybeSize) {
      return maybeNpuDmaUserOp->emitOpError()
             << "could not compute a static access size for source";
    }
    std::optional<size_t> maybeOffset =
        maybeNpuDmaUserOp->getSourceStaticBaseOffset();
    if (!maybeOffset) {
      return maybeNpuDmaUserOp->emitOpError()
             << "could not compute a static base offset for source";
    }
    AIE::BDDimLayoutArrayAttr dims = convertSizeStrideToBDDimLayoutArrayAttr(
        maybeNpuDmaUserOp->getSourceMixedSizes(),
        maybeNpuDmaUserOp->getSourceMixedStrides());
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
      rewriter.moveOpBefore(memOp, deviceBlock, deviceBlock->end());
      createDMA(memOp, AIE::DMAChannelDir::MM2S, channel.getValue(), dims,
                acqNum, acqNum, maybeSize.value(), maybeOffset.value(), buffers,
                lockPair, packetId);
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
    std::optional<size_t> maybeSize = maybeNpuDmaUserOp->getTargetStaticSize();
    if (!maybeSize) {
      return maybeNpuDmaUserOp->emitOpError()
             << "could not compute a static access size for source";
    }
    std::optional<size_t> maybeOffset =
        maybeNpuDmaUserOp->getTargetStaticBaseOffset();
    if (!maybeOffset) {
      return maybeNpuDmaUserOp->emitOpError()
             << "could not compute a static base offset for source";
    }
    AIE::BDDimLayoutArrayAttr dims = convertSizeStrideToBDDimLayoutArrayAttr(
        maybeNpuDmaUserOp->getTargetMixedSizes(),
        maybeNpuDmaUserOp->getTargetMixedStrides());
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
      rewriter.moveOpBefore(memOp, deviceBlock, deviceBlock->end());
      createDMA(memOp, AIE::DMAChannelDir::S2MM, channel.getValue(), dims,
                acqNum, acqNum, maybeSize.value(), maybeOffset.value(), buffers,
                lockPair, packetId);
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
  rewriter.setInsertionPointToEnd(deviceBlock);
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
  rewriter.setInsertionPointToEnd(deviceBlock);
  SmallVector<Operation *> flowOps =
      createFlowOps(flowOp, producerChannels, consumerChannels);
  return success();
}

LogicalResult AIEDeviceBuilder::lockToAIE(AMDAIE::LockOp lockOp,
                                          Block *deviceBlock, int &lockIndex) {
  LLVM_DEBUG(llvm::dbgs() << "Convert [AMDAIE::LockOp]\n");
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(deviceBlock);
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
    rewriter.setInsertionPointToEnd(deviceBlock);
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

LogicalResult AIEDeviceBuilder::workgroupToAIE(
    AMDAIE::WorkgroupOp workgroupOp, xilinx::AIE::DeviceOp deviceOp,
    xilinx::AIEX::RuntimeSequenceOp npuFuncOp) {
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
          if (failed(controlCodeToAIE(controlCodeOp, npuFuncOp))) {
            controlCodeOp.emitError("could not convert to AIEDialect ops");
            return WalkResult::interrupt();
          }
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
          rewriter.setInsertionPointToEnd(deviceBlock);
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
  rewriter.mergeBlocks(deviceCoreBlock, deviceBlock);
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

  // Retrieve the AMDAIEDevice from the executable target attribute.
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(moduleOp);
  std::optional<AMDAIEDevice> device = getConfigAMDAIEDevice(targetAttr);
  if (!device)
    return moduleOp.emitOpError()
           << "No AMDAIEDevice found in the target attribute configuration";
  xilinx::AIE::AIEDevice aieDevice = static_cast<xilinx::AIE::AIEDevice>(
      static_cast<uint32_t>(device.value()));

  auto funcRes = moduleOp.walk([&](func::FuncOp funcOp) {
    if (funcOp.isPrivate()) {
      return WalkResult::advance();
    }

    // Create aie.device.
    rewriter.setInsertionPoint(moduleBlock, moduleBlock->begin());
    auto deviceOp = rewriter.create<xilinx::AIE::DeviceOp>(
        rewriter.getUnknownLoc(),
        xilinx::AIE::AIEDeviceAttr::get(rewriter.getContext(), aieDevice));
    Block *deviceBlock = &deviceOp.getRegion().emplaceBlock();

    // The amdaie.controlcode operation has no operands, but the
    // aiex.runtime_sequence that it lowers to, does. Create the signature
    // of the aiex.runtime_sequence operation that replaces the
    // amdaie.controlcode. The HAL interface bindings are used to
    // order the function parameters correctly.
    SmallVector<IREE::HAL::InterfaceBindingSubspanOp> subspanOps;
    funcOp->walk([&](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
      subspanOps.push_back(subspanOp);
    });
    llvm::sort(subspanOps, [](IREE::HAL::InterfaceBindingSubspanOp a,
                              IREE::HAL::InterfaceBindingSubspanOp b) {
      return a.getBinding().getZExtValue() < b.getBinding().getZExtValue();
    });
    rewriter.setInsertionPoint(deviceBlock, deviceBlock->begin());

    // Create aiex.runtime_sequence inside aie.device
    auto npuFuncOp = rewriter.create<xilinx::AIEX::RuntimeSequenceOp>(
        rewriter.getUnknownLoc(), rewriter.getStringAttr(funcOp.getSymName()));
    Region &body = npuFuncOp.getBody();
    body.emplaceBlock();

    for (auto &&a : llvm::enumerate(subspanOps)) {
      body.addArgument(a.value().getType(), a.value().getLoc());
      bindingsMapper.map(a.value(), body.getArgument(a.index()));
    }

    // Walk the AIE regions ops and convert ops into pure AIEDialect ops.
    // IRMapping mapper;
    rewriter.setInsertionPointToStart(deviceBlock);
    WalkResult res = funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<func::FuncOp, func::ReturnOp>(op)) {
        return WalkResult::advance();
      } else if (auto workgroupOp = dyn_cast<AMDAIE::WorkgroupOp>(op)) {
        if (failed(workgroupToAIE(workgroupOp, deviceOp, npuFuncOp))) {
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

    // Move NPU instruction function to the end of the device block.
    rewriter.moveOpBefore(npuFuncOp, deviceBlock, deviceBlock->end());
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
    AIEDeviceBuilder builder(moduleOp.getContext());
    if (failed(builder.lowerToAIE(moduleOp))) return signalPassFailure();
  }
};

std::unique_ptr<Pass> createAMDAIELowerToAIEPass() {
  return std::make_unique<AMDAIELowerToAIEPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
