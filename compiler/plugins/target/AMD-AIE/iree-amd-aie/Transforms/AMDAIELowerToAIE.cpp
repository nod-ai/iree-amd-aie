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

#include <memory>
#include <numeric>

#include "aie/AIEDialect.h"
#include "aie/AIEXDialect.h"
#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
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

namespace {

/// Utility to remap the provided operation's operands.
void remapOperands(Operation *op, IRMapping &mapper) {
  for (int i = 0; i < op->getNumOperands(); ++i) {
    Value operand = op->getOperand(i);
    if (mapper.contains(operand)) {
      op->setOperand(i, mapper.lookup(operand));
    }
  }
}

/// It is dangerous to erase ops with `rewriter` without erasing them from
/// `mapper` too, as addresses of Operations/Values can be reused, resulting in
/// unexpected key-value pairs in `mapper`. Use this utility if `mapper` might
/// be used after `op` is erased.
void eraseOp(IRRewriter &rewriter, IRMapping &mapper, Operation *op) {
  for (Value result : op->getResults()) {
    mapper.erase(result);
  }
  mapper.erase(op);
  op->dropAllUses();
  rewriter.eraseOp(op);
}

//===----------------------------------------------------------------------===//
// Convert amdaie.core operation to aie.core
//===----------------------------------------------------------------------===//

/// Utility to convert vectors of `size` and `stride` into an
/// `AIE::BDDimLayoutArrayAttr`.
AIE::BDDimLayoutArrayAttr convertSizeStrideToBDDimLayoutArrayAttr(
    IRRewriter &rewriter, const SmallVector<OpFoldResult> &sizes,
    const SmallVector<OpFoldResult> &strides) {
  assert(sizes.size() == strides.size() &&
         "expected stride and size vectors of same size");
  SmallVector<AIE::BDDimLayoutAttr, 4> bdDimLayoutAttr;
  // If the access pattern (strides/sizes) have a single dimension, make it
  // implicit with an empty `BDDimLayoutAttr` as this is what the AIE dialect
  // expects.
  if (strides.size() == 1) {
    std::optional<int64_t> stride = getConstantIntValue(strides[0]);
    if (stride && stride.value() == 1) {
      return AIE::BDDimLayoutArrayAttr::get(rewriter.getContext(),
                                            ArrayRef(bdDimLayoutAttr));
    }
  }
  bdDimLayoutAttr.reserve(sizes.size());
  for (auto [size, stride] : llvm::zip(sizes, strides)) {
    bdDimLayoutAttr.push_back(AIE::BDDimLayoutAttr::get(
        rewriter.getContext(), getConstantIntValue(size).value(),
        getConstantIntValue(stride).value()));
  }
  return AIE::BDDimLayoutArrayAttr::get(rewriter.getContext(),
                                        ArrayRef(bdDimLayoutAttr));
}

/// Utility to create an `aie.objectfifo` operation from
/// `amdaie.circular_dma_cpy_nd`.
FailureOr<AIE::ObjectFifoCreateOp> createObjectFifo(
    IRRewriter &rewriter, AMDAIE::ConnectionOp connectionOp, IRMapping &mapper,
    AMDAIE::NpuCircularDmaCpyNdOp dmaOp, Value srcTile, ValueRange dstTiles,
    StringAttr &symName) {
  OpBuilder::InsertionGuard guard(rewriter);
  auto sourceType =
      cast<AMDAIE::LogicalObjectFifoType>(connectionOp.getSource().getType());
  auto targetType =
      cast<AMDAIE::LogicalObjectFifoType>(connectionOp.getTarget().getType());
  uint8_t sourceMemSpace = sourceType.getMemorySpaceAsUInt();
  uint8_t targetMemSpace = targetType.getMemorySpaceAsUInt();
  unsigned depth;
  unsigned sourceDepth = sourceType.getDepth();
  unsigned targetDepth = targetType.getDepth();
  if (sourceMemSpace == 0 && targetMemSpace == 0) {
    return connectionOp.emitOpError()
           << "both source and target on main memory not supported";
  } else if (sourceMemSpace == 0) {
    depth = targetDepth;
  } else if (targetMemSpace == 0) {
    depth = sourceDepth;
  } else {
    if (sourceDepth != targetDepth)
      return connectionOp.emitOpError()
             << "unsupported sourceDepth != targetDepth";
    depth = sourceDepth;
  }

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
             << "found non-`amdaie.channel` source channel";
    }
    consumerChannels.push_back(channelOp);
  }

  // Convert source and target sizes and strides to `BDDimLayoutArrayAttr`s,
  // which the `aie.objectfifo` works with.
  AIE::BDDimLayoutArrayAttr sourceDims =
      convertSizeStrideToBDDimLayoutArrayAttr(
          rewriter, dmaOp.getSourceMixedSizes(), dmaOp.getSourceMixedStrides());

  AIE::BDDimLayoutArrayAttr layoutAttr =
      convertSizeStrideToBDDimLayoutArrayAttr(
          rewriter, dmaOp.getTargetMixedSizes(), dmaOp.getTargetMixedStrides());
  // The aie.objectfifo expects a `BDDimLayoutArrayAttr` for each consumer. A
  // single one for all consumers will error out.
  SmallVector<AIE::BDDimLayoutArrayAttr> targetDimsVec(dstTiles.size(),
                                                       layoutAttr);

  AIE::BDDimLayoutArrayArrayAttr targetDims =
      AIE::BDDimLayoutArrayArrayAttr::get(rewriter.getContext(),
                                          ArrayRef(targetDimsVec));

  // For now, set data type based on source and target memory space. Use
  // L2/MemTile type if either source or target is located on L2. Otherwise, use
  // the most local type.
  // TODO(jornt): Not very clear and clean, but this is to mimic how AIE
  // objectfifos are set up and it is probably better to adjust AIE objectfifos
  // directly to make this more clean.
  // TODO(jornt): I think objectfifos should support source type != dest type.
  MemRefType srcType = cast<LogicalObjectFifoType>(connectionOp.getSourceType())
                           .getElementType();
  MemRefType dstType = cast<LogicalObjectFifoType>(connectionOp.getTargetType())
                           .getElementType();
  ArrayRef<int64_t> sourceShape = srcType.getShape();
  ArrayRef<int64_t> targetShape = dstType.getShape();
  int64_t sourceSize = std::accumulate(sourceShape.begin(), sourceShape.end(),
                                       1, std::multiplies<>());
  int64_t targetSize = std::accumulate(targetShape.begin(), targetShape.end(),
                                       1, std::multiplies<>());
  MemRefType memrefType =
      sourceSize < targetSize
          ? MemRefType::get({sourceSize}, srcType.getElementType(),
                            MemRefLayoutAttrInterface{},
                            srcType.getMemorySpace())
          : MemRefType::get({targetSize}, dstType.getElementType(),
                            MemRefLayoutAttrInterface{},
                            dstType.getMemorySpace());
  AIE::AIEObjectFifoType dtype = AIE::AIEObjectFifoType::get(memrefType);
  auto fifo = rewriter.create<AIE::ObjectFifoCreateOp>(
      rewriter.getUnknownLoc(), symName, srcTile, dstTiles,
      rewriter.getIntegerAttr(rewriter.getI32Type(), depth), dtype, sourceDims,
      targetDims);

  // Insert flow ops
  rewriter.setInsertionPoint(fifo);
  for (AMDAIE::ChannelOp producerChannel : producerChannels) {
    for (AMDAIE::ChannelOp consumerChannel : consumerChannels) {
      Value aieProducerTile = mapper.lookup(producerChannel.getTile());
      Value aieConsumerTile = mapper.lookup(consumerChannel.getTile());
      rewriter.create<AIE::FlowOp>(
          rewriter.getUnknownLoc(), aieProducerTile, AIE::WireBundle::DMA,
          producerChannel.getValue(), aieConsumerTile, AIE::WireBundle::DMA,
          consumerChannel.getValue(), FlatSymbolRefAttr::get(fifo->getContext(), fifo.getName()));
    }
  }

  return fifo;
}

/// Convert `amdaie.logicalobjectfifo.access` to
/// `aie.objectfifo.subview.access`, and refactor the memory space for
/// `memref.reinterpret_cast` ops.
LogicalResult accessOpToAIE(IRRewriter &rewriter,
                            AMDAIE::LogicalObjectFifoAccessOp accessOp,
                            IRMapping &mapper,
                            SmallVector<Operation *> &toBeErased) {
  LLVM_DEBUG(llvm::dbgs() << "Convert [AMDAIE::LogicalObjectFifoAccessOp]\n");
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(accessOp);
  if (!mapper.contains(accessOp.getInput())) {
    return accessOp.emitError()
           << "this access operation's input has not been mapped";
  }
  auto subviewOp = dyn_cast_if_present<AIE::ObjectFifoSubviewAccessOp>(
      mapper.lookup(accessOp.getInput()).getDefiningOp());
  if (!subviewOp) {
    return accessOp.emitError()
           << "access doesn't operate on an input that has been mapped to an "
              "`aie.objectfifo.acquire` + subview operation";
  }

  SmallVector<memref::ReinterpretCastOp> oldReinterpretOps;
  for (Operation *user : accessOp->getUsers()) {
    if (isa<memref::ReinterpretCastOp>(user)) {
      oldReinterpretOps.push_back(cast<memref::ReinterpretCastOp>(user));
    }
  }
  if (oldReinterpretOps.empty()) {
    return accessOp.emitError() << "reinterpret-cast op has not been generated";
  }
  assert(oldReinterpretOps.size() == 1 &&
         "expected a single reinterpret-cast op");
  auto oldReinterpretOp = oldReinterpretOps[0];

  auto type = cast<MemRefType>(oldReinterpretOp.getResult().getType());
  MemRefType newType = MemRefType::Builder(type);
  ArrayRef<int64_t> sizes = newType.getShape();
  auto [strides, baseOffset] = getStridesAndOffset(newType);
  auto reinterpretOp = rewriter.create<memref::ReinterpretCastOp>(
      rewriter.getUnknownLoc(), newType, subviewOp.getOutput(), baseOffset,
      sizes, strides);

  mapper.map(oldReinterpretOp.getOperation(), reinterpretOp.getOperation());
  mapper.map(oldReinterpretOp.getResult(), reinterpretOp.getResult());
  toBeErased.push_back(accessOp);
  toBeErased.push_back(oldReinterpretOp);
  return success();
}

/// Convert `amdaie.logicalobjectfifo.acquire` to `aie.objectfifo.acquire`.
/// Also insert `aie.objectfifo.subview.access` operations to access the
/// underlying memref and bridge the gap to AIE.
LogicalResult acquireOpToAIE(IRRewriter &rewriter,
                             AMDAIE::LogicalObjectFifoAcquire acquireOp,
                             IRMapping &mapper,
                             SmallVector<Operation *> &toBeErased) {
  LLVM_DEBUG(llvm::dbgs() << "Convert [AMDAIE::LogicalObjectFifoAcquire]\n");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(acquireOp);
  auto connectionOp = dyn_cast_if_present<AMDAIE::ConnectionOp>(
      acquireOp.getDma().getDefiningOp());
  if (!connectionOp) {
    return connectionOp.emitError()
           << "acquire doesn't operate on a `amdaie.connection`";
  }

  auto objFifo = dyn_cast<AIE::ObjectFifoCreateOp>(
      mapper.lookup(connectionOp.getOperation()));
  if (!objFifo) {
    return acquireOp.emitError()
           << "input isn't mapped to an `aie.objectifo` operation";
  }

  auto acquireOpType = dyn_cast<LogicalObjectFifoType>(acquireOp.getType());
  assert(acquireOpType &&
         "Expected LogicalObjectFifoAcquire to have type "
         "LogicalObjectFifoType");
  MemRefType elementType = acquireOpType.getElementType();

  auto subviewType = AIE::AIEObjectFifoSubviewType::get(elementType);
  AIE::ObjectFifoPort port =
      acquireOp.getPort() == LogicalObjectFifoPort::Produce
          ? AIE::ObjectFifoPort::Produce
          : AIE::ObjectFifoPort::Consume;
  auto objFifoAquireOp = rewriter.create<AIE::ObjectFifoAcquireOp>(
      rewriter.getUnknownLoc(), subviewType, port, objFifo.getName(), 1);

  auto subviewOp = rewriter.create<AIE::ObjectFifoSubviewAccessOp>(
      rewriter.getUnknownLoc(), elementType, objFifoAquireOp.getSubview(),
      /* index = */ rewriter.getIntegerAttr(rewriter.getI32Type(), 0));

  // Map acquire op to new acquire + subview op.
  mapper.map(acquireOp.getOperation(), subviewOp.getOperation());
  mapper.map(acquireOp.getResult(), subviewOp.getOutput());
  toBeErased.push_back(acquireOp);
  return success();
}

LogicalResult coreMemrefExtractStridedMetadataToAIE(
    IRRewriter &rewriter,
    memref::ExtractStridedMetadataOp extractStridedMetadataOp,
    IRMapping &mapper, SmallVector<Operation *> &toBeErased) {
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

LogicalResult coreFuncCallOpToAIE(IRRewriter &rewriter, func::CallOp oldCallOp,
                                  IRMapping &mapper,
                                  SmallVector<Operation *> &toBeErased) {
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
    mapper.map(fnDecl.getOperation(), newFnDecl.getOperation());
    fnDecl = newFnDecl;
  }
  // Fetch the new function declaration and create the new func.call op.
  auto newFnDecl = cast<func::FuncOp>(mapper.lookupOrDefault(fnDecl));
  rewriter.create<func::CallOp>(oldCallOp->getLoc(), newFnDecl, newArgs);
  toBeErased.push_back(oldCallOp);
  return success();
}

LogicalResult coreReleaseOpToAIE(IRRewriter &rewriter,
                                 AMDAIE::LogicalObjectFifoRelease releaseOp,
                                 IRMapping &mapper,
                                 SmallVector<Operation *> &toBeErased) {
  LLVM_DEBUG(llvm::dbgs() << "Convert [AMDAIE::LogicalObjectFifoRelease]\n");
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(releaseOp);
  Operation *dmaOp = releaseOp.getDma().getDefiningOp();
  auto objFifo = dyn_cast<AIE::ObjectFifoCreateOp>(mapper.lookup(dmaOp));
  if (!objFifo) {
    return releaseOp.emitError()
           << "input isn't mapped to an `aie.objectifo` operation";
  }
  AIE::ObjectFifoPort port =
      releaseOp.getPort() == LogicalObjectFifoPort::Produce
          ? AIE::ObjectFifoPort::Produce
          : AIE::ObjectFifoPort::Consume;
  std::optional<unsigned> maybeSize = releaseOp.getSize();
  unsigned size = maybeSize ? maybeSize.value() : 1;
  rewriter.replaceOpWithNewOp<AIE::ObjectFifoReleaseOp>(
      releaseOp, port, objFifo.getName(), size);
  return success();
}

/// Convert `amdaie.core` into `aie.core`.
LogicalResult coreToAIE(IRRewriter &rewriter, AMDAIE::CoreOp coreOp,
                        IRMapping &mapper, AIE::DeviceOp deviceOp,
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
            .Case<AMDAIE::LogicalObjectFifoAccessOp>([&](auto accessOp) {
              return accessOpToAIE(rewriter, accessOp, mapper, toBeErased);
            })
            .Case<AMDAIE::LogicalObjectFifoAcquire>([&](auto acquireOp) {
              return acquireOpToAIE(rewriter, acquireOp, mapper, toBeErased);
            })
            .Case<AMDAIE::LogicalObjectFifoRelease>([&](auto releaseOp) {
              return coreReleaseOpToAIE(rewriter, releaseOp, mapper,
                                        toBeErased);
            })
            .Case<memref::ExtractStridedMetadataOp>(
                [&](auto extractStridedMetadataOp) {
                  return coreMemrefExtractStridedMetadataToAIE(
                      rewriter, extractStridedMetadataOp, mapper, toBeErased);
                })
            .Case<func::CallOp>([&](auto oldCallOp) {
              return coreFuncCallOpToAIE(rewriter, oldCallOp, mapper,
                                         toBeErased);
            })
            .Default([&](Operation *op) {
              remapOperands(op, mapper);
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
  for (Operation *op : toBeErased) eraseOp(rewriter, mapper, op);

  mapper.map(coreOp.getResult(), aieCoreOp.getResult());
  mapper.map(coreOp.getOperation(), aieCoreOp.getOperation());
  return success();
}

}  // namespace

//===----------------------------------------------------------------------===//
// Convert amdaie.circular_dma_cpy_nd operation to aie.objectfifo
//===----------------------------------------------------------------------===//

/// Convert the `amdaie.connection` operation into bidirectional object
/// fifos.
LogicalResult flowToAIE(IRRewriter &rewriter, AMDAIE::ConnectionOp connectionOp,
                        IRMapping &mapper, Block *deviceBlock, int &dmaId) {
  LLVM_DEBUG(llvm::dbgs() << "Convert [AMDAIE::CircularDmaCpyNdOp]\n");
  rewriter.setInsertionPointToEnd(deviceBlock);
  if (!connectionOp.getSource())
    return connectionOp.emitOpError() << "expected a source";
  auto sourceLogicalObjFifo =
      dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
          connectionOp.getSource().getDefiningOp());
  if (!sourceLogicalObjFifo)
    return connectionOp.emitOpError() << "expected a logical objectFifo source";
  SmallVector<Value> newSourceTiles =
      llvm::map_to_vector(sourceLogicalObjFifo.getTiles(),
                          [&](Value tile) { return mapper.lookup(tile); });
  if (newSourceTiles.size() != 1) {
    return connectionOp.emitError()
           << "Can't create an `aie.objectfifo` from this flow operation as "
              "`ObjectFifoCreateOp` only handles a single source tile for now, "
              "but got: ";
  }
  Value newSourceTile = newSourceTiles[0];

  if (!connectionOp.getTarget())
    return connectionOp.emitOpError() << "expected a source";
  auto targetLogicalObjFifo =
      dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
          connectionOp.getTarget().getDefiningOp());
  if (!targetLogicalObjFifo)
    return connectionOp.emitOpError() << "expected a logical objectFifo source";
  SmallVector<Value> newTargetTiles =
      llvm::map_to_vector(targetLogicalObjFifo.getTiles(),
                          [&](Value tile) { return mapper.lookup(tile); });

  FailureOr<AMDAIE::NpuCircularDmaCpyNdOp> npuDmaUserOp =
      connectionOp.getNpuCircularDmaCpyNdUser();
  if (failed(npuDmaUserOp)) return failure();

  auto symName = "obj" + std::to_string(dmaId++);
  StringAttr symAttr = rewriter.getStringAttr(symName);
  FailureOr<AIE::ObjectFifoCreateOp> objFifo =
      createObjectFifo(rewriter, connectionOp, mapper, npuDmaUserOp.value(),
                       newSourceTile, newTargetTiles, symAttr);
  if (failed(objFifo)) return failure();
  mapper.map(connectionOp.getOperation(), objFifo.value().getOperation());
  return success();
}

//===----------------------------------------------------------------------===//
// Convert amdaie.controlcode operation to NPU instruction func
//===----------------------------------------------------------------------===//

/// Convert the `amdaie.npu.dma_cpy_nd` operation to `aiex.npu.dma_memcpy_nd`.
LogicalResult npuDmaCpyNdOpToAIE(IRRewriter &rewriter,
                                 AMDAIE::NpuDmaCpyNdOp dmaOp,
                                 SmallVector<Operation *> &toBeErased,
                                 IRMapping &mapper, IRMapping &bindingsMapper) {
  AMDAIE::ConnectionOp connectionOp = dmaOp.getConnectionOp();

  SmallVector<Value> offsets, sizes, strides;
  ArrayRef<int64_t> staticOffsets, staticSizes, staticStrides;
  AMDAIE::BdIdOp bdIdOp;
  LogicalObjectFifoFromMemrefOp logicalObjFifo;

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
  }

  else if (dmaOp.getTarget()) {
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
  }

  else {
    return dmaOp.emitOpError()
           << "has neither source not target memory space as L3.";
  }

  Value memref = bindingsMapper.lookup(logicalObjFifo.getMemref());

  auto objFifo = dyn_cast<AIE::ObjectFifoCreateOp>(
      mapper.lookup(connectionOp.getOperation()));

  uint32_t bdId = bdIdOp.getValue();

  if (!objFifo) {
    return dmaOp.emitError()
           << "input isn't mapped to an `aie.objectifo` operation";
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

  bool issueToken = dmaOp.hasDmaWaitOpUser();

  rewriter.setInsertionPoint(dmaOp);
  rewriter.create<AIEX::NpuDmaMemcpyNdOp>(
      dmaOp.getLoc(), SmallVector<Type, 1>{}, 0, 0, memref, offsets, sizes,
      strides, staticOffsets, staticSizes, staticStrides, nullptr,
      objFifo.getName(), bdId, issueToken);

  toBeErased.push_back(dmaOp);
  return success();
}

/// Convert the `amdaie.npu.dma_wait` operation to `aiex.npu.dma_wait`.
LogicalResult npuDmaWaitToAIE(IRRewriter &rewriter, AMDAIE::NpuDmaWaitOp waitOp,
                              SmallVector<Operation *> &toBeErased,
                              IRMapping &mapper, IRMapping &bindingsMapper) {
  rewriter.setInsertionPoint(waitOp);
  AMDAIE::ConnectionOp connectionOp = waitOp.getDmaOp().getConnectionOp();
  auto objFifo = dyn_cast<xilinx::AIE::ObjectFifoCreateOp>(
      mapper.lookup(connectionOp.getOperation()));
  if (!objFifo) {
    return waitOp.emitError()
           << "input isn't mapped to an `aie.objectifo` operation";
  }
  rewriter.create<AIEX::NpuDmaWaitOp>(rewriter.getUnknownLoc(),
                                      objFifo.getName());
  toBeErased.push_back(waitOp);
  return success();
}

/// Insert the control code operations into the NPU instruction function.
LogicalResult controlCodeToAie(IRRewriter &rewriter,
                               AMDAIE::ControlCodeOp controlCodeOp,
                               xilinx::AIEX::RuntimeSequenceOp funcOp,
                               IRMapping &mapper, IRMapping &bindingsMapper) {
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
                  eraseOp(rewriter, mapper, dmaOp);
                  return success();
                })
                .Case<AMDAIE::NpuDmaCpyNdOp>([&](auto dmaOp) {
                  return npuDmaCpyNdOpToAIE(rewriter, dmaOp, toBeErased, mapper,
                                            bindingsMapper);
                })
                .Case<AMDAIE::NpuDmaWaitOp>([&](auto waitOp) {
                  return npuDmaWaitToAIE(rewriter, waitOp, toBeErased, mapper,
                                         bindingsMapper);
                })
                .Case<AMDAIE::EndOp>([&](auto endOp) {
                  eraseOp(rewriter, mapper, endOp);
                  return success();
                })
                .Default([&](Operation *op) {
                  remapOperands(op, mapper);
                  return success();
                })
                .failed()) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  if (res.wasInterrupted()) return failure();
  for (Operation *op : toBeErased) eraseOp(rewriter, mapper, op);
  return success();
}

//===----------------------------------------------------------------------===//
// Convert amdaie.logicalobjectfifo.link operation to `aie.objectfifo.link`
//===----------------------------------------------------------------------===//

LogicalResult linkToAIE(IRRewriter &rewriter,
                        AMDAIE::LogicalObjectFifoLink linkOp, IRMapping &mapper,
                        Block *deviceBlock) {
  LLVM_DEBUG(llvm::dbgs() << "Convert [AMDAIE::LogicalObjectFifoLink]\n");
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(deviceBlock);
  SmallVector<Attribute> inSyms;
  for (auto in : linkOp.getIns()) {
    auto objFifo = dyn_cast<xilinx::AIE::ObjectFifoCreateOp>(
        mapper.lookup(in.getDefiningOp()));
    if (!objFifo) {
      return linkOp.emitError()
             << "input isn't mapped to an `aie.objectifo` operation";
    }
    inSyms.push_back(
        SymbolRefAttr::get(rewriter.getContext(), objFifo.getSymName()));
  }
  SmallVector<Attribute> outSyms;
  for (auto out : linkOp.getOuts()) {
    auto objFifo = dyn_cast<xilinx::AIE::ObjectFifoCreateOp>(
        mapper.lookup(out.getDefiningOp()));
    if (!objFifo) {
      return linkOp.emitError()
             << "output isn't mapped to an `aie.objectifo` operation";
    }
    outSyms.push_back(
        SymbolRefAttr::get(rewriter.getContext(), objFifo.getSymName()));
  }
  rewriter.create<AIE::ObjectFifoLinkOp>(
      rewriter.getUnknownLoc(), rewriter.getArrayAttr(inSyms),
      rewriter.getArrayAttr(outSyms), rewriter.getArrayAttr({}),
      rewriter.getArrayAttr({}));
  return success();
}

//===----------------------------------------------------------------------===//
// Convert amdaie.tile operation to aie.tile
//===----------------------------------------------------------------------===//

LogicalResult tileToAIE(IRRewriter &rewriter, AMDAIE::TileOp tileOp,
                        IRMapping &mapper, Block *deviceBlock) {
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

LogicalResult workgroupToAIE(IRRewriter &rewriter,
                             AMDAIE::WorkgroupOp workgroupOp,
                             xilinx::AIE::DeviceOp deviceOp,
                             xilinx::AIEX::RuntimeSequenceOp npuFuncOp,
                             IRMapping &mapper, IRMapping &bindingsMapper) {
  OpBuilder::InsertionGuard guard(rewriter);
  Block *deviceBlock = &deviceOp.getRegion().front();
  Block *deviceCoreBlock = rewriter.createBlock(&deviceOp.getRegion());
  rewriter.setInsertionPoint(deviceBlock, deviceBlock->begin());

  // Walk all operations in the AIE region and convert to AIE ops
  int dmaId = 0;
  WalkResult res = workgroupOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
    return TypeSwitch<Operation *, WalkResult>(op)
        .Case<AMDAIE::BdIdOp>([&](auto bdIdOp) {
          // BD ID ops are purely used for retrieving information in other ops
          // so don't convert to AIE dialect.
          return WalkResult::advance();
        })
        .Case<AMDAIE::CircularDmaCpyNdOp>([&](auto dmaOp) {
          dmaOp.emitOpError()
              << "`amdaie.circular_dma_cpy_nd` unsupported in lowering to AIE";
          return WalkResult::interrupt();
        })
        .Case<AMDAIE::ConnectionOp>([&](auto dmaOp) {
          if (failed(flowToAIE(rewriter, dmaOp, mapper, deviceBlock, dmaId))) {
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        })
        .Case<AMDAIE::ControlCodeOp>([&](auto controlCodeOp) {
          if (failed(controlCodeToAie(rewriter, controlCodeOp, npuFuncOp,
                                      mapper, bindingsMapper))) {
            controlCodeOp.emitError("could not convert to AIEDialect ops");
            return WalkResult::interrupt();
          }
          return WalkResult::skip();
        })
        .Case<AMDAIE::CoreOp>([&](auto coreOp) {
          if (failed(coreToAIE(rewriter, coreOp, mapper, deviceOp,
                               deviceCoreBlock))) {
            coreOp.emitError("could not convert to AIEDialect ops");
            return WalkResult::interrupt();
          }
          return WalkResult::skip();
        })
        .Case<AMDAIE::LogicalObjectFifoLink>([&](auto linkOp) {
          if (failed(linkToAIE(rewriter, linkOp, mapper, deviceBlock))) {
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        })
        .Case<AMDAIE::TileOp>([&](auto tileOp) {
          if (failed(tileToAIE(rewriter, tileOp, mapper, deviceBlock))) {
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        })
        .Default([&](Operation *op) {
          rewriter.setInsertionPointToEnd(deviceBlock);
          if (!isa_and_present<AMDAIEDialect>(op->getDialect())) {
            rewriter.clone(*op, mapper);
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
LogicalResult lowerToAIE(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
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
    IRMapping bindingsMapper;
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
    IRMapping mapper;
    rewriter.setInsertionPointToStart(deviceBlock);
    WalkResult res = funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<func::FuncOp, func::ReturnOp>(op)) {
        return WalkResult::advance();
      } else if (auto workgroupOp = dyn_cast<AMDAIE::WorkgroupOp>(op)) {
        if (failed(workgroupToAIE(rewriter, workgroupOp, deviceOp, npuFuncOp,
                                  mapper, bindingsMapper))) {
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
    eraseOp(rewriter, mapper, funcOp);
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
    registry.insert<AMDAIEDialect, xilinx::AIE::AIEDialect,
                    xilinx::AIEX::AIEXDialect>();
  }

  void runOnOperation() override {
    // Main function call to convert all operations into AIE dialect
    // operations inside an AIE device.
    if (failed(lowerToAIE(getOperation()))) return signalPassFailure();
  }
};

std::unique_ptr<Pass> createAMDAIELowerToAIEPass() {
  return std::make_unique<AMDAIELowerToAIEPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
