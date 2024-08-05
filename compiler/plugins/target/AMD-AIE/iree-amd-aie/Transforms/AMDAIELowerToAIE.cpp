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

#include <numeric>

#include "aie/AIEDialect.h"
#include "aie/AIEXDialect.h"
#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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

//===----------------------------------------------------------------------===//
// Convert amdaie.core operation to aie.core
//===----------------------------------------------------------------------===//

namespace {

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
AIE::ObjectFifoCreateOp createObjectFifo(IRRewriter &rewriter,
                                         AMDAIE::CircularDmaCpyNdOp dmaOp,
                                         Value srcTile, ValueRange dstTiles,
                                         StringAttr &symName) {
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
  MemRefType srcType =
      cast<LogicalObjectFifoType>(dmaOp.getSourceType()).getElementType();
  MemRefType dstType =
      cast<LogicalObjectFifoType>(dmaOp.getTargetType()).getElementType();
  ArrayRef<int64_t> sourceShape = srcType.getShape();
  ArrayRef<int64_t> targetShape = dstType.getShape();
  int64_t sourceSize = std::accumulate(sourceShape.begin(), sourceShape.end(),
                                       1, std::multiplies<>());
  int64_t targetSize = std::accumulate(targetShape.begin(), targetShape.end(),
                                       1, std::multiplies<>());
  // TODO(jornt) for now, memory space 1 is used for objectfifos. Maybe refactor
  // `aie.objectfifo` in the future to support different memory spaces.
  MemRefType memrefType =
      sourceSize < targetSize
          ? MemRefType::get({sourceSize}, srcType.getElementType(),
                            MemRefLayoutAttrInterface{},
                            rewriter.getI64IntegerAttr(1))
          : MemRefType::get({targetSize}, dstType.getElementType(),
                            MemRefLayoutAttrInterface{},
                            rewriter.getI64IntegerAttr(1));
  auto depthInBytes = srcType.getElementTypeBitWidth() / 8;
  auto fifo = rewriter.create<AIE::ObjectFifoCreateOp>(
      rewriter.getUnknownLoc(), symName, srcTile, dstTiles,
      rewriter.getIntegerAttr(rewriter.getI32Type(), depthInBytes),
      llvm::cast<mlir::Type>(memrefType), sourceDims, targetDims);
  return fifo;
}

/// Convert `amdaie.logicalobjectfifo.access` to `aie.objectfifo.subview.access`
/// + `memref.reinterpret_cast` to bridge the gap between the objectFifo type
/// and the type assumed by computational operations.
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
  auto subviewOp = dyn_cast<AIE::ObjectFifoSubviewAccessOp>(
      mapper.lookup(accessOp.getInput()).getDefiningOp());
  if (!subviewOp) {
    return accessOp.emitError()
           << "access doesn't operate on an input that has been mapped to an "
              "`aie.objectfifo.acquire` + subview operation";
  }
  auto type = cast<MemRefType>(accessOp.getOutput().getType());
  // TODO(jornt): for now, memory space 1 is used for objectFifos. Refactor
  // `aie.objectfifo` to support different memory spaces to avoid hardcoding.
  MemRefType newType =
      MemRefType::Builder(type).setMemorySpace(rewriter.getI64IntegerAttr(1));
  llvm::ArrayRef<int64_t> sizes = newType.getShape();
  auto [strides, baseOffset] = getStridesAndOffset(newType);
  auto reinterpretOp = rewriter.create<memref::ReinterpretCastOp>(
      rewriter.getUnknownLoc(), newType, subviewOp.getOutput(), baseOffset,
      sizes, strides);
  mapper.map(accessOp.getOperation(), reinterpretOp.getOperation());
  mapper.map(accessOp.getResult(), reinterpretOp.getResult());
  toBeErased.push_back(accessOp);
  return success();
}

/// Convert `amdaie.logicalobjectfifo.acquire` to `aie.objectfifo.acquire`.
/// There are some additional operations being added as well to bridge the gap
/// to AIE:
///   - Insert `aie.objectfifo.subview.access` operations to access the
///   underlying memref
///   - Insert `memref.reinterpret_cast` to get to the original local shape as
///   the `aie.objectfifo` has a single type, while the DMA operations converted
///   into objectfifos can have a different source and target type.
LogicalResult acquireOpToAIE(IRRewriter &rewriter,
                             AMDAIE::LogicalObjectFifoAcquire acquireOp,
                             IRMapping &mapper,
                             SmallVector<Operation *> &toBeErased) {
  LLVM_DEBUG(llvm::dbgs() << "Convert [AMDAIE::LogicalObjectFifoAcquire]\n");
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(acquireOp);
  auto dmaOp =
      dyn_cast<AMDAIE::CircularDmaCpyNdOp>(acquireOp.getDma().getDefiningOp());
  if (!dmaOp) {
    return dmaOp.emitError()
           << "acquire doesn't operate on a `amdaie.circular_dma_cpy_nd`";
  }

  auto objFifo =
      dyn_cast<AIE::ObjectFifoCreateOp>(mapper.lookup(dmaOp.getOperation()));
  if (!objFifo) {
    return acquireOp.emitError()
           << "input isn't mapped to an `aie.objectifo` operation";
  }
  MemRefType elementType = MemRefType::Builder(objFifo.getElemType())
                               .setMemorySpace(rewriter.getI64IntegerAttr(1));
  auto subviewType = elementType;
  AIE::ObjectFifoPort port =
      acquireOp.getPort() == LogicalObjectFifoPort::Produce
          ? AIE::ObjectFifoPort::Produce
          : AIE::ObjectFifoPort::Consume;
  auto objFifoAquireOp = rewriter.create<AIE::ObjectFifoAcquireOp>(
      rewriter.getUnknownLoc(), subviewType, port, objFifo.getName(), 1);
  auto subviewOp = rewriter.create<AIE::ObjectFifoSubviewAccessOp>(
      rewriter.getUnknownLoc(), elementType, objFifoAquireOp.getSubview(),
      rewriter.getIntegerAttr(rewriter.getI32Type(), 0));
  // Map acquire op to new acquire + subview op.
  mapper.map(acquireOp.getOperation(), subviewOp.getOperation());
  mapper.map(acquireOp.getResult(), subviewOp.getOutput());
  toBeErased.push_back(acquireOp);
  return success();
}

LogicalResult coreLinalgOpToAIE(IRRewriter &rewriter, linalg::LinalgOp linalgOp,
                                IRMapping &mapper,
                                SmallVector<Operation *> &toBeErased) {
  LLVM_DEBUG(llvm::dbgs() << "Convert [linalg.LinalgOp]\n");
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(linalgOp);
  rewriter.clone(*(linalgOp.getOperation()), mapper);
  rewriter.eraseOp(linalgOp);
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
  auto fnDecl = dyn_cast_or_null<func::FuncOp>(
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
  auto aieCoreBlock = rewriter.createBlock(&aieCoreRegion);
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
  auto walkResult = aieCoreOp.walk([&](Operation *op) {
    rewriter.setInsertionPoint(op);
    if (TypeSwitch<Operation *, LogicalResult>(op)
            .Case<AMDAIE::LogicalObjectFifoAccessOp>([&](auto accessOp) {
              return accessOpToAIE(rewriter, accessOp, mapper, toBeErased);
            })
            .Case<AMDAIE::LogicalObjectFifoAcquire>([&](auto acquireOp) {
              return acquireOpToAIE(rewriter, acquireOp, mapper, toBeErased);
            })
            .Case<AMDAIE::LogicalObjectFifoConsume>([&](auto consumeOp) {
              // TODO(jornt): get rid of LogicalObjectFifoConsume before this
              rewriter.eraseOp(consumeOp);
              return success();
            })
            .Case<AMDAIE::LogicalObjectFifoProduce>([&](auto produceOp) {
              // TODO(jornt): get rid of LogicalObjectFifoProduce before this
              rewriter.eraseOp(produceOp);
              return success();
            })
            .Case<AMDAIE::LogicalObjectFifoRelease>([&](auto releaseOp) {
              return coreReleaseOpToAIE(rewriter, releaseOp, mapper,
                                        toBeErased);
            })
            .Case<linalg::LinalgOp>([&](auto linalgOp) {
              return coreLinalgOpToAIE(rewriter, linalgOp, mapper, toBeErased);
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
  for (auto *op : toBeErased) {
    op->dropAllUses();
    rewriter.eraseOp(op);
  }

  mapper.map(coreOp.getResult(), aieCoreOp.getResult());
  mapper.map(coreOp.getOperation(), aieCoreOp.getOperation());
  return success();
}

}  // namespace

//===----------------------------------------------------------------------===//
// Convert amdaie.circular_dma_cpy_nd operation to aie.objectfifo
//===----------------------------------------------------------------------===//

/// Convert the `amdaie.circular_dma_cpy_nd` operation into bidirectional object
/// fifos.
LogicalResult circularDmaToAIE(IRRewriter &rewriter,
                               AMDAIE::CircularDmaCpyNdOp dmaOp,
                               IRMapping &mapper, Block *deviceBlock,
                               int &dmaId) {
  LLVM_DEBUG(llvm::dbgs() << "Convert [AMDAIE::CircularDmaCpyNdOp]\n");
  rewriter.setInsertionPointToEnd(deviceBlock);
  SmallVector<Value> newSourceTiles =
      llvm::map_to_vector(dmaOp.getSourceObjectFifo().getTiles(),
                          [&](Value tile) { return mapper.lookup(tile); });
  if (newSourceTiles.size() != 1) {
    return dmaOp.emitError()
           << "Can't create an `aie.objectfifo` from this DMA operation as "
              "`ObjectFifoCreateOp` only handles a single source tile for now.";
  }
  Value newSourceTile = newSourceTiles[0];
  SmallVector<Value> newTargetTiles =
      llvm::map_to_vector(dmaOp.getTargetObjectFifo().getTiles(),
                          [&](Value tile) { return mapper.lookup(tile); });

  auto symName = "obj" + std::to_string(dmaId++);
  auto symAttr = rewriter.getStringAttr(symName);
  AIE::ObjectFifoCreateOp objFifo =
      createObjectFifo(rewriter, dmaOp, newSourceTile, newTargetTiles, symAttr);
  mapper.map(dmaOp.getOperation(), objFifo.getOperation());
  return success();
}

//===----------------------------------------------------------------------===//
// Convert amdaie.controlcode operation to NPU instruction func
//===----------------------------------------------------------------------===//

namespace {

/// Utility to get the static offsets, sizes and strides for
/// `AIEX::NpuDmaMemcpyNdOp` with explicit addressing.
LogicalResult getStaticDimsForExplicitAddressing(
    Operation *op, const SmallVector<OpFoldResult> &offsets,
    const SmallVector<OpFoldResult> &sizes,
    const SmallVector<OpFoldResult> &strides,
    SmallVectorImpl<int64_t> &staticOffsets,
    SmallVectorImpl<int64_t> &staticSizes,
    SmallVectorImpl<int64_t> &staticStrides) {
  if (offsets.size() > staticOffsets.size()) {
    return op->emitError() << "size of `offsets` should be smaller or equal to "
                              "size of `staticOffsets`";
  }
  if (sizes.size() > staticSizes.size()) {
    return op->emitError() << "size of `sizes` should be smaller or equal to "
                              "size of `staticSizes`";
  }
  if (strides.size() > staticStrides.size()) {
    return op->emitError() << "size of `strides` should be smaller or equal to "
                              "size of `staticStrides`";
  }
  if (getConstantIntValue(strides[strides.size() - 1]).value() != 1) {
    return op->emitError() << "invalid last stride, should be 1";
  }
  for (int i = 0; i < offsets.size(); ++i)
    staticOffsets[staticOffsets.size() - offsets.size() + i] =
        getConstantIntValue(offsets[i]).value();
  for (int i = 0; i < sizes.size(); ++i)
    staticSizes[staticSizes.size() - sizes.size() + i] =
        getConstantIntValue(sizes[i]).value();
  for (int i = 0; i < strides.size(); ++i)
    staticStrides[staticStrides.size() - strides.size() + i] =
        getConstantIntValue(strides[i]).value();
  return success();
}

/// Utility to move 'repeat dimension' with stride 0 and size > 1 to outermost
/// dimension as only that one can support a stride with value 0 in AIE2(+)
/// hardware. But first check that such a dimension is actually the first 'real
/// dimension' in the access pattern.
LogicalResult canonicalizeNpuStridedPatternForAIE(
    SmallVectorImpl<int64_t> &offsets, SmallVectorImpl<int64_t> &sizes,
    SmallVectorImpl<int64_t> &strides) {
  bool foundNonUnitDim{false};
  for (size_t i = 0; i < offsets.size(); i++) {
    if (strides[i] == 0 && sizes[i] == 1) {
      continue;
    } else if (strides[i] == 0) {
      assert(sizes[i] > 0 && "size should be positive");
      if (foundNonUnitDim) return failure();
      foundNonUnitDim = true;
    } else {
      foundNonUnitDim = true;
    }
  }
  // Either dim 0 is a 'repeat dimension' or if the repeat is on a different
  // dimension, it guaranteed to be preceded by unit dimensions based on the
  // former check.
  for (size_t i = 1; i < offsets.size(); i++) {
    if (strides[i] == 0 && sizes[i] > 1) {
      strides[0] = 0;
      sizes[0] = sizes[i];
      sizes[i] = 1;
    }
  }
  return success();
}

/// Convert the `amdaie.npu.dma_cpy_nd` operation to `aiex.npu.dma_memcpy_nd`.
LogicalResult npuDmaCpyNdOpToAIE(IRRewriter &rewriter,
                                 AMDAIE::NpuDmaCpyNdOp dmaOp,
                                 SmallVector<Operation *> &toBeErased,
                                 IRMapping &mapper, IRMapping &bindingsMapper) {
  rewriter.setInsertionPoint(dmaOp);
  // Convert bidirectional `amdaie.npu.dma_cpy_nd` op into two halves.
  if (dmaOp.getSourceMemorySpaceAsUInt() == 0) {
    if (!dmaOp.hasSourceAddressing()) {
      return dmaOp.emitOpError()
             << "expected source addressing for DMA with source on L3";
    }
    AMDAIE::BdIdOp bdIdOp = dmaOp.getSourceBdIdOp();
    if (!bdIdOp)
      return dmaOp.emitOpError() << "expected to have a source BD ID op";

    // DmaOp either has explicit source addressing OR the defining op of its
    // source has its source on L3.
    SmallVector<Value> empty;
    SmallVector<int64_t, 4> staticOffsets(4, 0);
    SmallVector<int64_t, 4> staticSizes(4, 1);
    SmallVector<int64_t, 4> staticStrides(4, 0);
    if (failed(getStaticDimsForExplicitAddressing(
            dmaOp, dmaOp.getSourceMixedOffsets(), dmaOp.getSourceMixedSizes(),
            dmaOp.getSourceMixedStrides(), staticOffsets, staticSizes,
            staticStrides))) {
      return failure();
    }
    if (failed(canonicalizeNpuStridedPatternForAIE(staticOffsets, staticSizes,
                                                   staticStrides))) {
      return dmaOp.emitError() << "could not canonicalize for AIE";
    }

    AMDAIE::CircularDmaCpyNdOp dmaCpyNd = dmaOp.getDmaCpyNdOp();
    Value memref =
        bindingsMapper.lookup(dmaCpyNd.getSourceObjectFifo().getMemref());
    auto objFifo = dyn_cast<xilinx::AIE::ObjectFifoCreateOp>(
        mapper.lookup(dmaCpyNd.getOperation()));
    if (!objFifo) {
      return dmaOp.emitError()
             << "input isn't mapped to an `aie.objectifo` operation";
    }
    bool issueToken = dmaOp.hasDmaWaitOpUser();
    rewriter.create<AIEX::NpuDmaMemcpyNdOp>(
        rewriter.getUnknownLoc(), SmallVector<Type, 1>{}, 0, 0, memref, empty,
        empty, empty, staticOffsets, staticSizes, staticStrides,
        objFifo.getName(), bdIdOp.getValue(), issueToken);
  }
  if (dmaOp.getTargetMemorySpaceAsUInt() == 0) {
    if (!dmaOp.hasTargetAddressing()) {
      return dmaOp.emitOpError()
             << "expected target addressing for DMA with target on L3";
    }
    AMDAIE::BdIdOp bdIdOp = dmaOp.getTargetBdIdOp();
    if (!bdIdOp)
      return dmaOp.emitOpError() << "expected to have a target BD ID op";

    // DmaOp either has explicit target addressing OR the defining op of its
    // source has its target on L3.
    SmallVector<Value> empty;
    SmallVector<int64_t, 4> staticOffsets(4, 0);
    SmallVector<int64_t, 4> staticSizes(4, 1);
    SmallVector<int64_t, 4> staticStrides(4, 0);
    if (failed(getStaticDimsForExplicitAddressing(
            dmaOp, dmaOp.getTargetMixedOffsets(), dmaOp.getTargetMixedSizes(),
            dmaOp.getTargetMixedStrides(), staticOffsets, staticSizes,
            staticStrides))) {
      return failure();
    }
    if (failed(canonicalizeNpuStridedPatternForAIE(staticOffsets, staticSizes,
                                                   staticStrides))) {
      return dmaOp.emitError() << "could not canonicalize for AIE";
    }

    AMDAIE::CircularDmaCpyNdOp dmaCpyNd = dmaOp.getDmaCpyNdOp();
    Value memref =
        bindingsMapper.lookup(dmaCpyNd.getTargetObjectFifo().getMemref());
    auto objFifo = dyn_cast<xilinx::AIE::ObjectFifoCreateOp>(
        mapper.lookup(dmaCpyNd.getOperation()));
    if (!objFifo) {
      return dmaOp.emitError()
             << "input isn't mapped to an `aie.objectifo` operation";
    }
    bool issueToken = dmaOp.hasDmaWaitOpUser();
    rewriter.create<AIEX::NpuDmaMemcpyNdOp>(
        rewriter.getUnknownLoc(), SmallVector<Type, 1>{}, 0, 0, memref, empty,
        empty, empty, staticOffsets, staticSizes, staticStrides,
        objFifo.getName(), bdIdOp.getValue(), issueToken);
  }
  toBeErased.push_back(dmaOp);
  return success();
}

/// Convert the `amdaie.npu.dma_wait` operation to `aiex.npu.dma_wait`.
LogicalResult npuDmaWaitToAIE(IRRewriter &rewriter, AMDAIE::NpuDmaWaitOp waitOp,
                              SmallVector<Operation *> &toBeErased,
                              IRMapping &mapper, IRMapping &bindingsMapper) {
  rewriter.setInsertionPoint(waitOp);
  AMDAIE::CircularDmaCpyNdOp dmaCpyNd = waitOp.getDmaOp().getDmaCpyNdOp();
  auto objFifo = dyn_cast<xilinx::AIE::ObjectFifoCreateOp>(
      mapper.lookup(dmaCpyNd.getOperation()));
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
                               AMDAIE::ControlCodeOp &controlCodeOp,
                               xilinx::AIEX::RuntimeSequenceOp &funcOp,
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
                .Case<AMDAIE::NpuDmaCpyNdOp>([&](auto dmaOp) {
                  return npuDmaCpyNdOpToAIE(rewriter, dmaOp, toBeErased, mapper,
                                            bindingsMapper);
                })
                .Case<AMDAIE::NpuDmaWaitOp>([&](auto waitOp) {
                  return npuDmaWaitToAIE(rewriter, waitOp, toBeErased, mapper,
                                         bindingsMapper);
                })
                .Case<AMDAIE::EndOp>([&](auto endOp) {
                  rewriter.eraseOp(endOp);
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
  for (auto *op : toBeErased) {
    op->dropAllUses();
    rewriter.eraseOp(op);
  }
  return success();
}

}  // namespace

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
  rewriter.create<AIE::ObjectFifoLinkOp>(rewriter.getUnknownLoc(),
                                         rewriter.getArrayAttr(inSyms),
                                         rewriter.getArrayAttr(outSyms));
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
                             xilinx::AIEX::RuntimeSequenceOp ipuFuncOp,
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
          if (failed(circularDmaToAIE(rewriter, dmaOp, mapper, deviceBlock,
                                      dmaId))) {
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        })
        .Case<AMDAIE::ControlCodeOp>([&](auto controlCodeOp) {
          if (failed(controlCodeToAie(rewriter, controlCodeOp, ipuFuncOp,
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
          if (!isa_and_nonnull<AMDAIEDialect>(op->getDialect())) {
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
    // Insert AIE DeviceOp
    rewriter.setInsertionPoint(moduleBlock, moduleBlock->begin());
    auto deviceOp = rewriter.create<xilinx::AIE::DeviceOp>(
        rewriter.getUnknownLoc(),
        xilinx::AIE::AIEDeviceAttr::get(rewriter.getContext(), aieDevice));
    deviceOp.getRegion().emplaceBlock();
    Block *deviceBlock = &deviceOp.getRegion().front();

    // Create the signature of the NPU instruction sequence function. The HAL
    // interface bindings are used to order the function parameters correctly.
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
    auto ipuFuncOp = rewriter.create<xilinx::AIEX::RuntimeSequenceOp>(
        rewriter.getUnknownLoc(), rewriter.getStringAttr(funcOp.getSymName()));
    ipuFuncOp.getBody().push_back(new Block);
    for (int i = 0, e = subspanOps.size(); i < e; i++) {
      auto a = subspanOps[i].getResult();
      ipuFuncOp.getBody().addArgument(a.getType(), a.getLoc());
      bindingsMapper.map(a, ipuFuncOp.getBody().getArgument(i));
    }

    // Walk the AIE regions ops and convert ops into pure AIEDialect ops.
    IRMapping mapper;
    rewriter.setInsertionPointToStart(deviceBlock);
    WalkResult res = funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<func::FuncOp, func::ReturnOp>(op)) {
        return WalkResult::advance();
      } else if (auto workgroupOp = dyn_cast<AMDAIE::WorkgroupOp>(op)) {
        if (failed(workgroupToAIE(rewriter, workgroupOp, deviceOp, ipuFuncOp,
                                  mapper, bindingsMapper))) {
          return WalkResult::interrupt();
        }
        return WalkResult::skip();
      } else {
        if (!isa_and_nonnull<AMDAIEDialect>(op->getDialect())) {
          rewriter.clone(*op, mapper);
        }
      }
      return WalkResult::advance();
    });
    if (res.wasInterrupted()) return WalkResult::interrupt();

    // Move NPU instruction function to the end of the device block.
    rewriter.moveOpBefore(ipuFuncOp, deviceBlock, deviceBlock->end());
    // After walking the FuncOp, it has been converted into a DeviceOp and can
    // safely be erased.
    rewriter.eraseOp(funcOp);
    return WalkResult::advance();
  });
  if (funcRes.wasInterrupted()) return failure();
  return success();
}

/// Utility to erase all HAL bindings and dependent operations.
LogicalResult eraseHALBindings(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
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

/// Utility to move dependencies outside an operation into that operation. This
/// is for example needed for `aie.core` operations as MLIR-AIE expects all
/// dependencies, like constants, inside those core operations.
template <typename OpTy>
class MoveAllDependenciesIntoOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy parentOp,
                                PatternRewriter &rewriter) const override {
    bool addedDependency = false;
    parentOp->walk([&](Operation *op) {
      // Skip operations of type 'OpTy'.
      if (isa<OpTy>(op)) {
        return WalkResult::advance();
      }
      // Check all operands and whether their defining operations are located
      // outside the parentOp.
      for (Value operand : op->getOperands()) {
        if (!operand || !operand.getDefiningOp()) {
          continue;
        }
        Operation *dependencyOp = operand.getDefiningOp();
        if (isa_and_nonnull<xilinx::AIE::AIEDialect, xilinx::AIEX::AIEXDialect>(
                op->getDialect())) {
          // Skip AIE dialect operations.
          continue;
        } else if (!dependencyOp->getParentOfType<OpTy>()) {
          // Clone the dependency operation into the parent operation's block
          // and replace all uses.
          rewriter.setInsertionPointToStart(&parentOp->getRegion(0).front());
          Operation *newOp = rewriter.clone(*dependencyOp);
          dependencyOp->replaceUsesWithIf(newOp, [&](OpOperand &use) {
            return use.getOwner()->getParentOfType<OpTy>() == parentOp;
          });
          addedDependency = true;
        }
      }
      return WalkResult::advance();
    });
    return success(addedDependency);
  }
};

class AMDAIELowerToAIEPass
    : public impl::AMDAIELowerToAIEBase<AMDAIELowerToAIEPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect, xilinx::AIE::AIEDialect,
                    xilinx::AIEX::AIEXDialect>();
  }

  AMDAIELowerToAIEPass() = default;
  AMDAIELowerToAIEPass(const AMDAIELowerToAIEPass &pass){};
  void runOnOperation() override;
};

void AMDAIELowerToAIEPass::runOnOperation() {
  // Main function call to convert all operations into AIE dialect operations
  // inside an AIE device.
  if (failed(lowerToAIE(getOperation()))) {
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs() << "Module after lowerToAIE: " << getOperation());

  // Clean up the HAL bindings and it's uses as they are not needed anymore.
  if (failed(eraseHALBindings(getOperation()))) {
    return signalPassFailure();
  }

  // Move all dependencies, like for example constants, that are residing
  // outside core operations into those core operations. This is required by
  // the AIE dialect.
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<MoveAllDependenciesIntoOp<xilinx::AIE::CoreOp>>(context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }

  // All Ukernel related function declarations will be within aie.device, so
  // delete the ones outside from the SymbolTable.
  SymbolTable symbolTable(getOperation());
  getOperation()->walk([&](func::FuncOp funcOp) {
    if (funcOp.isPrivate() && !funcOp->getParentOfType<AIE::DeviceOp>()) {
      symbolTable.erase(funcOp);
    }
  });
}

}  // namespace

std::unique_ptr<Pass> createAMDAIELowerToAIEPass() {
  return std::make_unique<AMDAIELowerToAIEPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
