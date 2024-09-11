// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "iree-amd-aie/IR/AMDAIEOps.cpp.inc"

namespace mlir::iree_compiler::AMDAIE {

void AMDAIEDialect::initializeAMDAIEOps() {
  addOperations<
#define GET_OP_LIST
#include "iree-amd-aie/IR/AMDAIEOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// AMDAIE_BdIdOp
//===----------------------------------------------------------------------===//

void BdIdOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "bd_id");
}

//===----------------------------------------------------------------------===//
// AMDAIE_BufferOp
//===----------------------------------------------------------------------===//

void BufferOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "buffer");
}

//===----------------------------------------------------------------------===//
// AMDAIE_ChannelOp
//===----------------------------------------------------------------------===//

void ChannelOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "channel");
}

TileOp ChannelOp::getTileOp() {
  auto res = dyn_cast_if_present<TileOp>(getTile().getDefiningOp());
  assert(res && "`amdaie.channel` expects an `amdaie.tile` as tile operand");
  return res;
}

//===----------------------------------------------------------------------===//
// AMDAIE_ControlCodeOp
//===----------------------------------------------------------------------===//

LogicalResult ControlCodeOp::verify() {
  // Verify that this ControlCodeOp contains a EndOp terminator if one
  // exists.
  if (failed(OpTrait::SingleBlockImplicitTerminator<EndOp>::Impl<
             ControlCodeOp>::verifyRegionTrait(*this))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// AMDAIE_CoreOp
//===----------------------------------------------------------------------===//

void CoreOp::build(OpBuilder &b, OperationState &result, AMDAIE::TileOp tileOp,
                   ValueRange inputDmas, ValueRange outputDmas) {
  build(b, result, b.getIndexType(), tileOp, inputDmas, outputDmas, nullptr);
}

/// Hardcoded row_offset == 2 -> AIE core rows start from 2
/// TODO(jornt): avoid hardcoding here. Add a device model/identifier to loop up
/// core offset. This will be handled in a follow-up.
void CoreOp::build(OpBuilder &b, OperationState &result, Value coreCol,
                   Value coreRow, ValueRange inputDmas, ValueRange outputDmas) {
  auto rowOffset = b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), 2);
  auto row =
      b.createOrFold<arith::AddIOp>(b.getUnknownLoc(), rowOffset, coreRow);
  auto tileOp = b.create<AMDAIE::TileOp>(b.getUnknownLoc(), coreCol, row);
  build(b, result, tileOp, inputDmas, outputDmas, nullptr);
}

void CoreOp::build(OpBuilder &b, OperationState &result, Value coreCol,
                   Value coreRow) {
  build(b, result, coreCol, coreRow, {}, {});
}

LogicalResult CoreOp::verify() {
  // Verify that this CoreOp contains a EndOp terminator if one
  // exists.
  if (failed(OpTrait::SingleBlockImplicitTerminator<EndOp>::Impl<
             CoreOp>::verifyRegionTrait(*this))) {
    return failure();
  }
  return success();
}

TileOp CoreOp::getTileOp() {
  auto res = dyn_cast_if_present<TileOp>(getTile().getDefiningOp());
  assert(res && "`amdaie.core` expects an `amdaie.tile` as tile operand");
  return res;
}

//===----------------------------------------------------------------------===//
// AMDAIE_DmaCpyNdBaseOp
//===----------------------------------------------------------------------===//

namespace {
// Simplified from upstream MLIR's foldDynamicIndexList:
LogicalResult foldMixed(SmallVectorImpl<OpFoldResult> &ofrs) {
  bool valuesChanged = false;
  for (OpFoldResult &ofr : ofrs) {
    if (ofr.is<Attribute>()) continue;
    Attribute attr;
    if (matchPattern(ofr.get<Value>(), m_Constant(&attr))) {
      ofr = attr;
      valuesChanged = true;
    }
  }
  return success(valuesChanged);
}

template <typename OpType, typename ReplacementBuilder>
// Based on upstream MLIR's
// OpWithOffsetSizesAndStridesConstantArgumentFolder
class DoublyStridedFolder final : public OpRewritePattern<OpType> {
 public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    SmallVector<OpFoldResult> tgtMixedOffsets(op.getTargetMixedOffsets());
    SmallVector<OpFoldResult> tgtMixedSizes(op.getTargetMixedSizes());
    SmallVector<OpFoldResult> tgtMixedStrides(op.getTargetMixedStrides());
    SmallVector<OpFoldResult> srcMixedOffsets(op.getSourceMixedOffsets());
    SmallVector<OpFoldResult> srcMixedSizes(op.getSourceMixedSizes());
    SmallVector<OpFoldResult> srcMixedStrides(op.getSourceMixedStrides());

    // No constant operands were folded, just return;
    if (failed(foldMixed(tgtMixedOffsets)) &&
        failed(foldMixed(tgtMixedSizes)) &&
        failed(foldMixed(tgtMixedStrides)) &&
        failed(foldMixed(srcMixedOffsets)) &&
        failed(foldMixed(srcMixedSizes)) && failed(foldMixed(srcMixedStrides)))
      return failure();

    ReplacementBuilder::replace(op, rewriter, tgtMixedOffsets, tgtMixedSizes,
                                tgtMixedStrides, srcMixedOffsets, srcMixedSizes,
                                srcMixedStrides);

    return success();
  }
};

template <typename T>
struct DmaCpyNdBaseOpReplacementBuilder {
  static void replace(T dmaOp, PatternRewriter &rewriter,
                      ArrayRef<OpFoldResult> tgtMixedOffsets,
                      ArrayRef<OpFoldResult> tgtMixedSizes,
                      ArrayRef<OpFoldResult> tgtMixedStrides,
                      ArrayRef<OpFoldResult> srcMixedOffsets,
                      ArrayRef<OpFoldResult> srcMixedSizes,
                      ArrayRef<OpFoldResult> srcMixedStrides) {
    rewriter.replaceOpWithNewOp<T>(dmaOp, dmaOp.getTarget(), tgtMixedOffsets,
                                   tgtMixedSizes, tgtMixedStrides,
                                   dmaOp.getSource(), srcMixedOffsets,
                                   srcMixedSizes, srcMixedStrides);
  }
};
}  // namespace

// Build a DmaCpyNdOp with mixed static and dynamic entries.
void DmaCpyNdOp::build(OpBuilder &b, OperationState &result, Value target,
                       ArrayRef<OpFoldResult> targetOffsets,
                       ArrayRef<OpFoldResult> targetSizes,
                       ArrayRef<OpFoldResult> targetStrides, Value source,
                       ArrayRef<OpFoldResult> sourceOffsets,
                       ArrayRef<OpFoldResult> sourceSizes,
                       ArrayRef<OpFoldResult> sourceStrides) {
  SmallVector<int64_t> staticTargetOffsets, staticTargetSizes,
      staticTargetStrides;
  SmallVector<int64_t> staticSourceOffsets, staticSourceSizes,
      staticSourceStrides;
  SmallVector<Value> dynamicTargetOffsets, dynamicTargetSizes,
      dynamicTargetStrides;
  SmallVector<Value> dynamicSourceOffsets, dynamicSourceSizes,
      dynamicSourceStrides;
  dispatchIndexOpFoldResults(targetOffsets, dynamicTargetOffsets,
                             staticTargetOffsets);
  dispatchIndexOpFoldResults(targetSizes, dynamicTargetSizes,
                             staticTargetSizes);
  dispatchIndexOpFoldResults(targetStrides, dynamicTargetStrides,
                             staticTargetStrides);
  dispatchIndexOpFoldResults(sourceOffsets, dynamicSourceOffsets,
                             staticSourceOffsets);
  dispatchIndexOpFoldResults(sourceSizes, dynamicSourceSizes,
                             staticSourceSizes);
  dispatchIndexOpFoldResults(sourceStrides, dynamicSourceStrides,
                             staticSourceStrides);
  build(b, result, b.getIndexType(), target, dynamicTargetOffsets,
        dynamicTargetSizes, dynamicTargetStrides, staticTargetOffsets,
        staticTargetSizes, staticTargetStrides, source, dynamicSourceOffsets,
        dynamicSourceSizes, dynamicSourceStrides, staticSourceOffsets,
        staticSourceSizes, staticSourceStrides);
}

// Build a DmaCpyNdOp with static entries.
void DmaCpyNdOp::build(OpBuilder &b, OperationState &result, Value target,
                       ArrayRef<int64_t> targetOffsets,
                       ArrayRef<int64_t> targetSizes,
                       ArrayRef<int64_t> targetStrides, Value source,
                       ArrayRef<int64_t> sourceOffsets,
                       ArrayRef<int64_t> sourceSizes,
                       ArrayRef<int64_t> sourceStrides) {
  SmallVector<OpFoldResult> targetOffsetValues = llvm::to_vector<4>(
      llvm::map_range(targetOffsets, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> targetSizeValues = llvm::to_vector<4>(
      llvm::map_range(targetSizes, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> targetStrideValues = llvm::to_vector<4>(
      llvm::map_range(targetStrides, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> sourceOffsetValues = llvm::to_vector<4>(
      llvm::map_range(sourceOffsets, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> sourceSizeValues = llvm::to_vector<4>(
      llvm::map_range(sourceSizes, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> sourceStrideValues = llvm::to_vector<4>(
      llvm::map_range(sourceStrides, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  build(b, result, target, targetOffsetValues, targetSizeValues,
        targetStrideValues, source, sourceOffsetValues, sourceSizeValues,
        sourceStrideValues);
}

// Build a DmaCpyNdOp with dynamic entries.
void DmaCpyNdOp::build(OpBuilder &b, OperationState &result, Value target,
                       ValueRange targetOffsets, ValueRange targetSizes,
                       ValueRange targetStrides, Value source,
                       ValueRange sourceOffsets, ValueRange sourceSizes,
                       ValueRange sourceStrides) {
  SmallVector<OpFoldResult> targetOffsetValues =
      llvm::to_vector<4>(llvm::map_range(
          targetOffsets, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> targetSizeValues = llvm::to_vector<4>(
      llvm::map_range(targetSizes, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> targetStrideValues =
      llvm::to_vector<4>(llvm::map_range(
          targetStrides, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> sourceOffsetValues =
      llvm::to_vector<4>(llvm::map_range(
          sourceOffsets, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> sourceSizeValues = llvm::to_vector<4>(
      llvm::map_range(sourceSizes, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> sourceStrideValues =
      llvm::to_vector<4>(llvm::map_range(
          sourceStrides, [](Value v) -> OpFoldResult { return v; }));
  build(b, result, target, targetOffsetValues, targetSizeValues,
        targetStrideValues, source, sourceOffsetValues, sourceSizeValues,
        sourceStrideValues);
}

DoublyStridedOpInterface DmaCpyNdOp::createDoublyStridedOp(
    RewriterBase &rewriter, SmallVector<OpFoldResult> &newTargetOffsets,
    SmallVector<OpFoldResult> &newTargetSizes,
    SmallVector<OpFoldResult> &newTargetStrides,
    SmallVector<OpFoldResult> &newSourceOffsets,
    SmallVector<OpFoldResult> &newSourceSizes,
    SmallVector<OpFoldResult> &newSourceStrides) {
  Location loc = (*this)->getLoc();
  auto newOp = rewriter.create<AMDAIE::DmaCpyNdOp>(
      loc, getTarget(), newTargetOffsets, newTargetSizes, newTargetStrides,
      getSource(), newSourceOffsets, newSourceSizes, newSourceStrides);
  return cast<DoublyStridedOpInterface>(newOp.getOperation());
}

LogicalObjectFifoFromMemrefOp DmaCpyNdOp::getSourceObjectFifo() {
  return dyn_cast_if_present<LogicalObjectFifoFromMemrefOp>(
      getSource().getDefiningOp());
};

LogicalObjectFifoFromMemrefOp DmaCpyNdOp::getTargetObjectFifo() {
  return dyn_cast_if_present<LogicalObjectFifoFromMemrefOp>(
      getTarget().getDefiningOp());
};

void DmaCpyNdOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<DoublyStridedFolder<
      DmaCpyNdOp, DmaCpyNdBaseOpReplacementBuilder<DmaCpyNdOp>>>(context);
}

// Build a CircularDmaCpyNdOp with mixed static and dynamic entries.
void CircularDmaCpyNdOp::build(
    OpBuilder &b, OperationState &result, Value target,
    ArrayRef<OpFoldResult> targetOffsets, ArrayRef<OpFoldResult> targetSizes,
    ArrayRef<OpFoldResult> targetStrides, Value source,
    ArrayRef<OpFoldResult> sourceOffsets, ArrayRef<OpFoldResult> sourceSizes,
    ArrayRef<OpFoldResult> sourceStrides) {
  SmallVector<int64_t> staticTargetOffsets, staticTargetSizes,
      staticTargetStrides;
  SmallVector<int64_t> staticSourceOffsets, staticSourceSizes,
      staticSourceStrides;
  SmallVector<Value> dynamicTargetOffsets, dynamicTargetSizes,
      dynamicTargetStrides;
  SmallVector<Value> dynamicSourceOffsets, dynamicSourceSizes,
      dynamicSourceStrides;
  dispatchIndexOpFoldResults(targetOffsets, dynamicTargetOffsets,
                             staticTargetOffsets);
  dispatchIndexOpFoldResults(targetSizes, dynamicTargetSizes,
                             staticTargetSizes);
  dispatchIndexOpFoldResults(targetStrides, dynamicTargetStrides,
                             staticTargetStrides);
  dispatchIndexOpFoldResults(sourceOffsets, dynamicSourceOffsets,
                             staticSourceOffsets);
  dispatchIndexOpFoldResults(sourceSizes, dynamicSourceSizes,
                             staticSourceSizes);
  dispatchIndexOpFoldResults(sourceStrides, dynamicSourceStrides,
                             staticSourceStrides);
  build(b, result, b.getIndexType(), target, dynamicTargetOffsets,
        dynamicTargetSizes, dynamicTargetStrides, staticTargetOffsets,
        staticTargetSizes, staticTargetStrides, source, dynamicSourceOffsets,
        dynamicSourceSizes, dynamicSourceStrides, staticSourceOffsets,
        staticSourceSizes, staticSourceStrides);
}

// Build a CircularDmaCpyNdOp with static entries.
void CircularDmaCpyNdOp::build(OpBuilder &b, OperationState &result,
                               Value target, ArrayRef<int64_t> targetOffsets,
                               ArrayRef<int64_t> targetSizes,
                               ArrayRef<int64_t> targetStrides, Value source,
                               ArrayRef<int64_t> sourceOffsets,
                               ArrayRef<int64_t> sourceSizes,
                               ArrayRef<int64_t> sourceStrides) {
  SmallVector<OpFoldResult> targetOffsetValues = llvm::to_vector<4>(
      llvm::map_range(targetOffsets, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> targetSizeValues = llvm::to_vector<4>(
      llvm::map_range(targetSizes, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> targetStrideValues = llvm::to_vector<4>(
      llvm::map_range(targetStrides, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> sourceOffsetValues = llvm::to_vector<4>(
      llvm::map_range(sourceOffsets, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> sourceSizeValues = llvm::to_vector<4>(
      llvm::map_range(sourceSizes, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> sourceStrideValues = llvm::to_vector<4>(
      llvm::map_range(sourceStrides, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  build(b, result, target, targetOffsetValues, targetSizeValues,
        targetStrideValues, source, sourceOffsetValues, sourceSizeValues,
        sourceStrideValues);
}

// Build a CircularDmaCpyNdOp with dynamic entries.
void CircularDmaCpyNdOp::build(OpBuilder &b, OperationState &result,
                               Value target, ValueRange targetOffsets,
                               ValueRange targetSizes, ValueRange targetStrides,
                               Value source, ValueRange sourceOffsets,
                               ValueRange sourceSizes,
                               ValueRange sourceStrides) {
  SmallVector<OpFoldResult> targetOffsetValues =
      llvm::to_vector<4>(llvm::map_range(
          targetOffsets, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> targetSizeValues = llvm::to_vector<4>(
      llvm::map_range(targetSizes, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> targetStrideValues =
      llvm::to_vector<4>(llvm::map_range(
          targetStrides, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> sourceOffsetValues =
      llvm::to_vector<4>(llvm::map_range(
          sourceOffsets, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> sourceSizeValues = llvm::to_vector<4>(
      llvm::map_range(sourceSizes, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> sourceStrideValues =
      llvm::to_vector<4>(llvm::map_range(
          sourceStrides, [](Value v) -> OpFoldResult { return v; }));
  build(b, result, target, targetOffsetValues, targetSizeValues,
        targetStrideValues, source, sourceOffsetValues, sourceSizeValues,
        sourceStrideValues);
}

DoublyStridedOpInterface CircularDmaCpyNdOp::createDoublyStridedOp(
    RewriterBase &rewriter, SmallVector<OpFoldResult> &newTargetOffsets,
    SmallVector<OpFoldResult> &newTargetSizes,
    SmallVector<OpFoldResult> &newTargetStrides,
    SmallVector<OpFoldResult> &newSourceOffsets,
    SmallVector<OpFoldResult> &newSourceSizes,
    SmallVector<OpFoldResult> &newSourceStrides) {
  Location loc = (*this)->getLoc();
  auto newOp = rewriter.create<AMDAIE::CircularDmaCpyNdOp>(
      loc, getTarget(), newTargetOffsets, newTargetSizes, newTargetStrides,
      getSource(), newSourceOffsets, newSourceSizes, newSourceStrides);
  return cast<DoublyStridedOpInterface>(newOp.getOperation());
}

LogicalObjectFifoFromMemrefOp CircularDmaCpyNdOp::getSourceObjectFifo() {
  return dyn_cast_if_present<LogicalObjectFifoFromMemrefOp>(
      getSource().getDefiningOp());
};

LogicalObjectFifoFromMemrefOp CircularDmaCpyNdOp::getTargetObjectFifo() {
  return dyn_cast_if_present<LogicalObjectFifoFromMemrefOp>(
      getTarget().getDefiningOp());
};

void CircularDmaCpyNdOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                     MLIRContext *context) {
  results.add<DoublyStridedFolder<
      CircularDmaCpyNdOp,
      DmaCpyNdBaseOpReplacementBuilder<CircularDmaCpyNdOp>>>(context);
}

//===----------------------------------------------------------------------===//
// AMDAIE_ConnectionOp
//===----------------------------------------------------------------------===//

void ConnectionOp::build(mlir::OpBuilder &b, mlir::OperationState &result,
                         Value target, Value source) {
  build(b, result, target, {}, source, {});
}

FailureOr<AMDAIE::NpuCircularDmaCpyNdOp>
ConnectionOp::getNpuCircularDmaCpyNdUser() {
  SmallVector<AMDAIE::NpuCircularDmaCpyNdOp, 1> npuDmaUsers;
  for (Operation *userOp : getOperation()->getUsers()) {
    if (auto userNpuDmaOp = dyn_cast<AMDAIE::NpuCircularDmaCpyNdOp>(userOp))
      npuDmaUsers.push_back(userNpuDmaOp);
  }
  if (npuDmaUsers.size() != 1) {
    return emitOpError() << "only a single `amdaie.npu.circular_dma_cpy_nd` "
                            "user supported currently, but got: "
                         << npuDmaUsers.size();
  }
  return npuDmaUsers[0];
}

//===----------------------------------------------------------------------===//
// AMDAIE_LockOp
//===----------------------------------------------------------------------===//

void LockOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "lock");
}

//===----------------------------------------------------------------------===//
// AMDAIE_LogicalObjectFifoAccessOp
//===----------------------------------------------------------------------===//

void LogicalObjectFifoAccessOp::build(OpBuilder &b,
                                      mlir::OperationState &result, Value input,
                                      MemoryAccess accessType) {
  auto type = llvm::cast<LogicalObjectFifoType>(input.getType());
  build(b, result, type.getElementType(), input, accessType);
}

LogicalObjectFifoFromMemrefOp
LogicalObjectFifoAccessOp::getLogicalObjectFifo() {
  return dyn_cast_if_present<LogicalObjectFifoFromMemrefOp>(
      getInput().getDefiningOp());
};

//===----------------------------------------------------------------------===//
// AMDAIE_LogicalObjectFifoAcquire
//===----------------------------------------------------------------------===//

void LogicalObjectFifoAcquire::build(OpBuilder &b, mlir::OperationState &result,
                                     mlir::TypeRange resultTypes, Value dma,
                                     LogicalObjectFifoPort port) {
  build(b, result, resultTypes, dma, port, b.getI32IntegerAttr(1));
}

//===----------------------------------------------------------------------===//
// AMDAIE_LogicalObjectFifoFromBuffersOp
//===----------------------------------------------------------------------===//

SmallVector<AMDAIE::BufferOp> LogicalObjectFifoFromBuffersOp::getBuffersOnTile(
    TileOp tileOp) {
  SmallVector<AMDAIE::BufferOp> buffers;
  for (Value res : getBuffers()) {
    if (auto bufferOp =
            dyn_cast_if_present<AMDAIE::BufferOp>(res.getDefiningOp());
        bufferOp.getTile() == tileOp.getResult()) {
      buffers.push_back(bufferOp);
    }
  }
  return buffers;
}

SmallVector<AMDAIE::LockOp>
LogicalObjectFifoFromBuffersOp::getConsumerLocksOnTile(AMDAIE::TileOp tileOp) {
  SmallVector<AMDAIE::LockOp> locks;
  for (Value lockRes : getConsumerLocks()) {
    if (auto lockOp =
            dyn_cast_if_present<AMDAIE::LockOp>(lockRes.getDefiningOp());
        lockOp.getTile() == tileOp.getResult()) {
      locks.push_back(lockOp);
    }
  }
  return locks;
}

SmallVector<AMDAIE::LockOp>
LogicalObjectFifoFromBuffersOp::getProducerLocksOnTile(AMDAIE::TileOp tileOp) {
  SmallVector<AMDAIE::LockOp> locks;
  for (Value lockRes : getProducerLocks()) {
    if (auto lockOp =
            dyn_cast_if_present<AMDAIE::LockOp>(lockRes.getDefiningOp());
        lockOp.getTile() == tileOp.getResult()) {
      locks.push_back(lockOp);
    }
  }
  return locks;
}

SmallVector<Value> LogicalObjectFifoFromBuffersOp::getTiles() {
  llvm::SmallVector<mlir::Value> tiles;
  for (Value result : getBuffers()) {
    Value tile = cast<AMDAIE::BufferOp>(result.getDefiningOp()).getTile();
    tiles.push_back(tile);
  }
  return tiles;
}

LogicalResult LogicalObjectFifoFromBuffersOp::verify() {
  if (llvm::any_of(getBuffers(), [](Value result) {
        return !isa_and_present<AMDAIE::BufferOp>(result.getDefiningOp());
      })) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// AMDAIE_LogicalObjectFifoFromMemrefOp
//===----------------------------------------------------------------------===//

/// Build with an array of static tile locations.
void LogicalObjectFifoFromMemrefOp::build(
    OpBuilder &b, mlir::OperationState &result, Value memref,
    ArrayRef<std::pair<int64_t, int64_t>> tileLocations) {
  SmallVector<Value> tiles;
  tiles.reserve(tileLocations.size());
  for (auto [column, row] : tileLocations) {
    auto getCol = b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), column);
    auto getRow = b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), row);
    auto tileOp = b.create<AMDAIE::TileOp>(b.getUnknownLoc(), getCol, getRow);
    tiles.push_back(tileOp.getResult());
  }
  // For deterministic order.
  llvm::sort(tiles.begin(), tiles.end(),
             TileOp::tileValueColumnAndRowComparator);
  auto type = LogicalObjectFifoType::get(cast<MemRefType>(memref.getType()));
  build(b, result, type, memref, tiles);
}

LogicalResult LogicalObjectFifoFromMemrefOp::canonicalize(
    LogicalObjectFifoFromMemrefOp logicalObjectFifo,
    PatternRewriter &rewriter) {
  // We only canonicalize the tiles for now, so return if empty
  if (logicalObjectFifo.getTiles().empty()) {
    return success();
  }

  SmallVector<Value> tiles = logicalObjectFifo.getTiles();
  if (llvm::is_sorted(tiles, TileOp::tileValueColumnAndRowComparator)) {
    // Still erase duplicates.
    tiles.erase(std::unique(tiles.begin(), tiles.end()), tiles.end());
    return success();
  }

  // If tiles are not sorted, sort them, erase duplicates and replace the
  // logical objectfifo.
  llvm::sort(tiles.begin(), tiles.end(),
             TileOp::tileValueColumnAndRowComparator);
  tiles.erase(std::unique(tiles.begin(), tiles.end()), tiles.end());

  rewriter.replaceOpWithNewOp<AMDAIE::LogicalObjectFifoFromMemrefOp>(
      logicalObjectFifo,
      llvm::cast<LogicalObjectFifoType>(
          logicalObjectFifo.getOutput().getType()),
      logicalObjectFifo.getMemref(), tiles);
  return success();
}

LogicalResult LogicalObjectFifoFromMemrefOp::verify() {
  // Check whether the tile arguments are all of type AMDAIE::TileOp
  if (llvm::all_of(getTiles(), [](Value result) {
        return isa_and_present<TileOp>(result.getDefiningOp());
      })) {
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// AMDAIE_LogicalObjectFifoRelease
//===----------------------------------------------------------------------===//

void LogicalObjectFifoRelease::build(OpBuilder &b, mlir::OperationState &result,
                                     Value dma, LogicalObjectFifoPort port) {
  build(b, result, dma, port, b.getI32IntegerAttr(1));
}

//===----------------------------------------------------------------------===//
// AMDAIE_NpuDmaCpyNdOp
//===----------------------------------------------------------------------===//

// Build a NpuDmaCpyNdOp with mixed static and dynamic entries and target and
// source BD IDs.
void NpuDmaCpyNdOp::build(
    OpBuilder &b, OperationState &result, Value dma, Value target,
    ArrayRef<OpFoldResult> targetOffsets, ArrayRef<OpFoldResult> targetSizes,
    ArrayRef<OpFoldResult> targetStrides, Value targetBdId, Value source,
    ArrayRef<OpFoldResult> sourceOffsets, ArrayRef<OpFoldResult> sourceSizes,
    ArrayRef<OpFoldResult> sourceStrides, Value sourceBdId) {
  SmallVector<int64_t> staticTargetOffsets, staticTargetSizes,
      staticTargetStrides;
  SmallVector<int64_t> staticSourceOffsets, staticSourceSizes,
      staticSourceStrides;
  SmallVector<Value> dynamicTargetOffsets, dynamicTargetSizes,
      dynamicTargetStrides;
  SmallVector<Value> dynamicSourceOffsets, dynamicSourceSizes,
      dynamicSourceStrides;
  dispatchIndexOpFoldResults(targetOffsets, dynamicTargetOffsets,
                             staticTargetOffsets);
  dispatchIndexOpFoldResults(targetSizes, dynamicTargetSizes,
                             staticTargetSizes);
  dispatchIndexOpFoldResults(targetStrides, dynamicTargetStrides,
                             staticTargetStrides);
  dispatchIndexOpFoldResults(sourceOffsets, dynamicSourceOffsets,
                             staticSourceOffsets);
  dispatchIndexOpFoldResults(sourceSizes, dynamicSourceSizes,
                             staticSourceSizes);
  dispatchIndexOpFoldResults(sourceStrides, dynamicSourceStrides,
                             staticSourceStrides);
  build(b, result, b.getIndexType(), dma, target, dynamicTargetOffsets,
        dynamicTargetSizes, dynamicTargetStrides, staticTargetOffsets,
        staticTargetSizes, staticTargetStrides, targetBdId, source,
        dynamicSourceOffsets, dynamicSourceSizes, dynamicSourceStrides,
        staticSourceOffsets, staticSourceSizes, staticSourceStrides,
        sourceBdId);
}

// Build a NpuDmaCpyNdOp with static entries.
void NpuDmaCpyNdOp::build(
    OpBuilder &b, OperationState &result, Value dma, Value target,
    ArrayRef<int64_t> targetOffsets, ArrayRef<int64_t> targetSizes,
    ArrayRef<int64_t> targetStrides, mlir::Value targetBdId, Value source,
    ArrayRef<int64_t> sourceOffsets, ArrayRef<int64_t> sourceSizes,
    ArrayRef<int64_t> sourceStrides, mlir::Value sourceBdId) {
  SmallVector<OpFoldResult> targetOffsetValues = llvm::to_vector<4>(
      llvm::map_range(targetOffsets, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> targetSizeValues = llvm::to_vector<4>(
      llvm::map_range(targetSizes, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> targetStrideValues = llvm::to_vector<4>(
      llvm::map_range(targetStrides, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> sourceOffsetValues = llvm::to_vector<4>(
      llvm::map_range(sourceOffsets, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> sourceSizeValues = llvm::to_vector<4>(
      llvm::map_range(sourceSizes, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> sourceStrideValues = llvm::to_vector<4>(
      llvm::map_range(sourceStrides, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  build(b, result, dma, target, targetOffsetValues, targetSizeValues,
        targetStrideValues, targetBdId, source, sourceOffsetValues,
        sourceSizeValues, sourceStrideValues, sourceBdId);
}

// Build a NpuDmaCpyNdOp with dynamic entries.
void NpuDmaCpyNdOp::build(OpBuilder &b, OperationState &result, Value dma,
                          Value target, ValueRange targetOffsets,
                          ValueRange targetSizes, ValueRange targetStrides,
                          mlir::Value targetBdId, Value source,
                          ValueRange sourceOffsets, ValueRange sourceSizes,
                          ValueRange sourceStrides, mlir::Value sourceBdId) {
  SmallVector<OpFoldResult> targetOffsetValues =
      llvm::to_vector<4>(llvm::map_range(
          targetOffsets, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> targetSizeValues = llvm::to_vector<4>(
      llvm::map_range(targetSizes, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> targetStrideValues =
      llvm::to_vector<4>(llvm::map_range(
          targetStrides, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> sourceOffsetValues =
      llvm::to_vector<4>(llvm::map_range(
          sourceOffsets, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> sourceSizeValues = llvm::to_vector<4>(
      llvm::map_range(sourceSizes, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> sourceStrideValues =
      llvm::to_vector<4>(llvm::map_range(
          sourceStrides, [](Value v) -> OpFoldResult { return v; }));
  build(b, result, dma, target, targetOffsetValues, targetSizeValues,
        targetStrideValues, targetBdId, source, sourceOffsetValues,
        sourceSizeValues, sourceStrideValues, sourceBdId);
}

void NpuDmaCpyNdOp::print(OpAsmPrinter &p) {
  Operation *op = getOperation();
  p << " " << getConnection() << "(";
  if (getTarget()) p << getTarget();
  printDynamicIndexList(p, op, getTargetOffsets(), getTargetStaticOffsets());
  p << " ";
  printDynamicIndexList(p, op, getTargetSizes(), getTargetStaticSizes());
  p << " ";
  printDynamicIndexList(p, op, getTargetStrides(), getTargetStaticStrides());
  if (getTargetBdId()) p << " bd_id = " << getTargetBdId();
  p << ", ";
  if (getSource()) p << getSource();
  printDynamicIndexList(p, op, getSourceOffsets(), getSourceStaticOffsets());
  p << " ";
  printDynamicIndexList(p, op, getSourceSizes(), getSourceStaticSizes());
  p << " ";
  printDynamicIndexList(p, op, getSourceStrides(), getSourceStaticStrides());
  if (getSourceBdId()) p << " bd_id = " << getSourceBdId();
  p << ")";
  SmallVector<StringRef, 7> elidedAttrs;
  elidedAttrs.push_back("operandSegmentSizes");
  elidedAttrs.push_back("target_static_offsets");
  elidedAttrs.push_back("target_static_sizes");
  elidedAttrs.push_back("target_static_strides");
  elidedAttrs.push_back("source_static_offsets");
  elidedAttrs.push_back("source_static_sizes");
  elidedAttrs.push_back("source_static_strides");
  p.printOptionalAttrDictWithKeyword(op->getAttrs(), elidedAttrs);
  if (getTarget() || getSource()) p << " :";
  if (getTarget()) p << " target_type = " << getTarget().getType();
  if (getSource()) p << " source_type = " << getSource().getType();
}

ParseResult NpuDmaCpyNdOp::parse(OpAsmParser &parser, OperationState &result) {
  OpBuilder b(parser.getContext());
  auto indexType = b.getIndexType();

  SMLoc targetOperandsLoc, sourceOperandsLoc;
  OpAsmParser::UnresolvedOperand dma;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> targetOperands, sourceOperands,
      targetBdIdOperands, sourceBdIdOperands;
  DenseI64ArrayAttr targetStaticOffsets, targetStaticSizes, targetStaticStrides;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> targetDynamicOffsets,
      targetDynamicSizes, targetDynamicStrides;
  DenseI64ArrayAttr sourceStaticOffsets, sourceStaticSizes, sourceStaticStrides;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> sourceDynamicOffsets,
      sourceDynamicSizes, sourceDynamicStrides;
  SmallVector<Type, 1> targetTypes;
  SmallVector<Type, 1> sourceTypes;

  if (failed(parser.parseOperand(dma)) || failed(parser.parseLParen()))
    return failure();

  OpAsmParser::UnresolvedOperand target;
  if (parser.parseOptionalOperand(target).has_value()) {
    targetOperands.push_back(target);
  }
  if (failed(parseDynamicIndexList(parser, targetDynamicOffsets,
                                   targetStaticOffsets))) {
    return failure();
  }
  result.getOrAddProperties<NpuDmaCpyNdOp::Properties>().target_static_offsets =
      targetStaticOffsets;
  if (failed(parseDynamicIndexList(parser, targetDynamicSizes,
                                   targetStaticSizes))) {
    return failure();
  }
  result.getOrAddProperties<NpuDmaCpyNdOp::Properties>().target_static_sizes =
      targetStaticSizes;
  if (failed(parseDynamicIndexList(parser, targetDynamicStrides,
                                   targetStaticStrides))) {
    return failure();
  }
  result.getOrAddProperties<NpuDmaCpyNdOp::Properties>().target_static_strides =
      targetStaticStrides;

  if (succeeded(parser.parseOptionalKeyword("bd_id"))) {
    if (failed(parser.parseEqual())) return failure();
    OpAsmParser::UnresolvedOperand bdId;
    if (failed(parser.parseOperand(bdId))) return failure();
    targetBdIdOperands.push_back(bdId);
  }

  if (failed(parser.parseComma())) return failure();

  OpAsmParser::UnresolvedOperand source;
  if (parser.parseOptionalOperand(source).has_value()) {
    sourceOperands.push_back(source);
  }
  if (failed(parseDynamicIndexList(parser, sourceDynamicOffsets,
                                   sourceStaticOffsets))) {
    return failure();
  }
  result.getOrAddProperties<NpuDmaCpyNdOp::Properties>().source_static_offsets =
      sourceStaticOffsets;
  if (failed(parseDynamicIndexList(parser, sourceDynamicSizes,
                                   sourceStaticSizes))) {
    return failure();
  }
  result.getOrAddProperties<NpuDmaCpyNdOp::Properties>().source_static_sizes =
      sourceStaticSizes;
  if (failed(parseDynamicIndexList(parser, sourceDynamicStrides,
                                   sourceStaticStrides))) {
    return failure();
  }
  result.getOrAddProperties<NpuDmaCpyNdOp::Properties>().source_static_strides =
      sourceStaticStrides;

  if (succeeded(parser.parseOptionalKeyword("bd_id"))) {
    if (failed(parser.parseEqual())) return failure();
    OpAsmParser::UnresolvedOperand bdId;
    if (failed(parser.parseOperand(bdId))) return failure();
    sourceBdIdOperands.push_back(bdId);
  }

  if (failed(parser.parseRParen())) return failure();
  {
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalAttrDict(result.attributes)) return failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
          return parser.emitError(loc)
                 << "'" << result.name.getStringRef() << "' op ";
        }))) {
      return failure();
    }
  }

  if (succeeded(parser.parseOptionalColon())) {
    if (succeeded(parser.parseOptionalKeyword("target_type"))) {
      if (parser.parseEqual()) return failure();
      Type targetType;
      if (failed(parser.parseType(targetType))) return failure();
      targetTypes.push_back(targetType);
    }
    if (succeeded(parser.parseOptionalKeyword("source_type"))) {
      if (parser.parseEqual()) return failure();
      Type sourceType;
      if (failed(parser.parseType(sourceType))) return failure();
      sourceTypes.push_back(sourceType);
    }
  }

  llvm::copy(
      ArrayRef<int32_t>({1, static_cast<int32_t>(targetOperands.size()),
                         static_cast<int32_t>(targetDynamicOffsets.size()),
                         static_cast<int32_t>(targetDynamicSizes.size()),
                         static_cast<int32_t>(targetDynamicStrides.size()),
                         static_cast<int32_t>(targetBdIdOperands.size()),
                         static_cast<int32_t>(sourceOperands.size()),
                         static_cast<int32_t>(sourceDynamicOffsets.size()),
                         static_cast<int32_t>(sourceDynamicSizes.size()),
                         static_cast<int32_t>(sourceDynamicStrides.size()),
                         static_cast<int32_t>(sourceBdIdOperands.size())}),
      result.getOrAddProperties<NpuDmaCpyNdOp::Properties>()
          .operandSegmentSizes.begin());

  if (failed(parser.resolveOperand(dma, indexType, result.operands)))
    return failure();
  if (failed(parser.resolveOperands(targetOperands, targetTypes,
                                    targetOperandsLoc, result.operands))) {
    return failure();
  }
  if (failed(parser.resolveOperands(targetDynamicOffsets, indexType,
                                    result.operands))) {
    return failure();
  }
  if (failed(parser.resolveOperands(targetDynamicSizes, indexType,
                                    result.operands))) {
    return failure();
  }
  if (failed(parser.resolveOperands(targetDynamicStrides, indexType,
                                    result.operands))) {
    return failure();
  }
  if (failed(parser.resolveOperands(targetBdIdOperands, indexType,
                                    result.operands))) {
    return failure();
  }
  if (failed(parser.resolveOperands(sourceOperands, sourceTypes,
                                    sourceOperandsLoc, result.operands))) {
    return failure();
  }
  if (failed(parser.resolveOperands(sourceDynamicOffsets, indexType,
                                    result.operands))) {
    return failure();
  }
  if (failed(parser.resolveOperands(sourceDynamicSizes, indexType,
                                    result.operands))) {
    return failure();
  }
  if (failed(parser.resolveOperands(sourceDynamicStrides, indexType,
                                    result.operands))) {
    return failure();
  }
  if (failed(parser.resolveOperands(sourceBdIdOperands, indexType,
                                    result.operands))) {
    return failure();
  }

  result.addTypes(indexType);
  return success();
}

DoublyStridedOpInterface NpuDmaCpyNdOp::createDoublyStridedOp(
    ::mlir::RewriterBase &rewriter,
    ::llvm::SmallVector<OpFoldResult> &newTargetOffsets,
    ::llvm::SmallVector<OpFoldResult> &newTargetSizes,
    ::llvm::SmallVector<OpFoldResult> &newTargetStrides,
    ::llvm::SmallVector<OpFoldResult> &newSourceOffsets,
    ::llvm::SmallVector<OpFoldResult> &newSourceSizes,
    ::llvm::SmallVector<OpFoldResult> &newSourceStrides) {
  Location loc = (*this)->getLoc();
  auto newOp = rewriter.create<AMDAIE::NpuDmaCpyNdOp>(
      loc, getConnection(), getTarget(), newTargetOffsets, newTargetSizes,
      newTargetStrides, getTargetBdId(), getSource(), newSourceOffsets,
      newSourceSizes, newSourceStrides, getSourceBdId());
  return cast<DoublyStridedOpInterface>(newOp.getOperation());
}

bool NpuDmaCpyNdOp::hasDmaWaitOpUser() {
  return llvm::any_of((*this)->getUsers(),
                      [](auto userOp) { return isa<NpuDmaWaitOp>(userOp); });
}

namespace {
struct NpuDmaCpyNdOpReplacementBuilder {
  static void replace(NpuDmaCpyNdOp dmaOp, PatternRewriter &rewriter,
                      ArrayRef<OpFoldResult> tgtMixedOffsets,
                      ArrayRef<OpFoldResult> tgtMixedSizes,
                      ArrayRef<OpFoldResult> tgtMixedStrides,
                      ArrayRef<OpFoldResult> srcMixedOffsets,
                      ArrayRef<OpFoldResult> srcMixedSizes,
                      ArrayRef<OpFoldResult> srcMixedStrides) {
    rewriter.replaceOpWithNewOp<NpuDmaCpyNdOp>(
        dmaOp, dmaOp.getConnection(), dmaOp.getTarget(), tgtMixedOffsets,
        tgtMixedSizes, tgtMixedStrides, dmaOp.getTargetBdId(),
        dmaOp.getSource(), srcMixedOffsets, srcMixedSizes, srcMixedStrides,
        dmaOp.getSourceBdId());
  }
};
}  // namespace

void NpuDmaCpyNdOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results
      .add<DoublyStridedFolder<NpuDmaCpyNdOp, NpuDmaCpyNdOpReplacementBuilder>>(
          context);
}

//===----------------------------------------------------------------------===//
// AMDAIE_NpuCircularDmaCpyNdOp
//===----------------------------------------------------------------------===//

// Build a NpuCircularDmaCpyNdOp with mixed static and dynamic entries.
void NpuCircularDmaCpyNdOp::build(
    OpBuilder &b, OperationState &result, Value connection,
    ArrayRef<OpFoldResult> targetOffsets, ArrayRef<OpFoldResult> targetSizes,
    ArrayRef<OpFoldResult> targetStrides, ArrayRef<OpFoldResult> sourceOffsets,
    ArrayRef<OpFoldResult> sourceSizes, ArrayRef<OpFoldResult> sourceStrides) {
  SmallVector<int64_t> staticTargetOffsets, staticTargetSizes,
      staticTargetStrides;
  SmallVector<int64_t> staticSourceOffsets, staticSourceSizes,
      staticSourceStrides;
  SmallVector<Value> dynamicTargetOffsets, dynamicTargetSizes,
      dynamicTargetStrides;
  SmallVector<Value> dynamicSourceOffsets, dynamicSourceSizes,
      dynamicSourceStrides;
  dispatchIndexOpFoldResults(targetOffsets, dynamicTargetOffsets,
                             staticTargetOffsets);
  dispatchIndexOpFoldResults(targetSizes, dynamicTargetSizes,
                             staticTargetSizes);
  dispatchIndexOpFoldResults(targetStrides, dynamicTargetStrides,
                             staticTargetStrides);
  dispatchIndexOpFoldResults(sourceOffsets, dynamicSourceOffsets,
                             staticSourceOffsets);
  dispatchIndexOpFoldResults(sourceSizes, dynamicSourceSizes,
                             staticSourceSizes);
  dispatchIndexOpFoldResults(sourceStrides, dynamicSourceStrides,
                             staticSourceStrides);
  build(b, result, b.getIndexType(), connection, dynamicTargetOffsets,
        dynamicTargetSizes, dynamicTargetStrides, staticTargetOffsets,
        staticTargetSizes, staticTargetStrides, dynamicSourceOffsets,
        dynamicSourceSizes, dynamicSourceStrides, staticSourceOffsets,
        staticSourceSizes, staticSourceStrides);
}

// Build a NpuCircularDmaCpyNdOp with static entries.
void NpuCircularDmaCpyNdOp::build(
    OpBuilder &b, OperationState &result, Value connection,
    ArrayRef<int64_t> targetOffsets, ArrayRef<int64_t> targetSizes,
    ArrayRef<int64_t> targetStrides, ArrayRef<int64_t> sourceOffsets,
    ArrayRef<int64_t> sourceSizes, ArrayRef<int64_t> sourceStrides) {
  SmallVector<OpFoldResult> targetOffsetValues = llvm::to_vector<4>(
      llvm::map_range(targetOffsets, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> targetSizeValues = llvm::to_vector<4>(
      llvm::map_range(targetSizes, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> targetStrideValues = llvm::to_vector<4>(
      llvm::map_range(targetStrides, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> sourceOffsetValues = llvm::to_vector<4>(
      llvm::map_range(sourceOffsets, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> sourceSizeValues = llvm::to_vector<4>(
      llvm::map_range(sourceSizes, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> sourceStrideValues = llvm::to_vector<4>(
      llvm::map_range(sourceStrides, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  build(b, result, connection, targetOffsetValues, targetSizeValues,
        targetStrideValues, sourceOffsetValues, sourceSizeValues,
        sourceStrideValues);
}

// Build a NpuDmaCpyNdOp with dynamic entries.
void NpuCircularDmaCpyNdOp::build(OpBuilder &b, OperationState &result,
                                  Value connection, ValueRange targetOffsets,
                                  ValueRange targetSizes,
                                  ValueRange targetStrides,
                                  ValueRange sourceOffsets,
                                  ValueRange sourceSizes,
                                  ValueRange sourceStrides) {
  SmallVector<OpFoldResult> targetOffsetValues =
      llvm::to_vector<4>(llvm::map_range(
          targetOffsets, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> targetSizeValues = llvm::to_vector<4>(
      llvm::map_range(targetSizes, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> targetStrideValues =
      llvm::to_vector<4>(llvm::map_range(
          targetStrides, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> sourceOffsetValues =
      llvm::to_vector<4>(llvm::map_range(
          sourceOffsets, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> sourceSizeValues = llvm::to_vector<4>(
      llvm::map_range(sourceSizes, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> sourceStrideValues =
      llvm::to_vector<4>(llvm::map_range(
          sourceStrides, [](Value v) -> OpFoldResult { return v; }));
  build(b, result, connection, targetOffsetValues, targetSizeValues,
        targetStrideValues, sourceOffsetValues, sourceSizeValues,
        sourceStrideValues);
}

DoublyStridedOpInterface NpuCircularDmaCpyNdOp::createDoublyStridedOp(
    ::mlir::RewriterBase &rewriter,
    ::llvm::SmallVector<OpFoldResult> &newTargetOffsets,
    ::llvm::SmallVector<OpFoldResult> &newTargetSizes,
    ::llvm::SmallVector<OpFoldResult> &newTargetStrides,
    ::llvm::SmallVector<OpFoldResult> &newSourceOffsets,
    ::llvm::SmallVector<OpFoldResult> &newSourceSizes,
    ::llvm::SmallVector<OpFoldResult> &newSourceStrides) {
  Location loc = (*this)->getLoc();
  auto newOp = rewriter.create<AMDAIE::NpuCircularDmaCpyNdOp>(
      loc, getConnection(),
      getValueOrCreateConstantIndexOp(rewriter, loc, newTargetOffsets),
      getValueOrCreateConstantIndexOp(rewriter, loc, newTargetSizes),
      getValueOrCreateConstantIndexOp(rewriter, loc, newTargetStrides),
      getValueOrCreateConstantIndexOp(rewriter, loc, newSourceOffsets),
      getValueOrCreateConstantIndexOp(rewriter, loc, newSourceSizes),
      getValueOrCreateConstantIndexOp(rewriter, loc, newSourceStrides));
  return cast<DoublyStridedOpInterface>(newOp.getOperation());
}

namespace {
struct NpuCircularDmaCpyNdOpReplacementBuilder {
  static void replace(NpuCircularDmaCpyNdOp dmaOp, PatternRewriter &rewriter,
                      ArrayRef<OpFoldResult> tgtMixedOffsets,
                      ArrayRef<OpFoldResult> tgtMixedSizes,
                      ArrayRef<OpFoldResult> tgtMixedStrides,
                      ArrayRef<OpFoldResult> srcMixedOffsets,
                      ArrayRef<OpFoldResult> srcMixedSizes,
                      ArrayRef<OpFoldResult> srcMixedStrides) {
    rewriter.replaceOpWithNewOp<NpuCircularDmaCpyNdOp>(
        dmaOp, dmaOp.getConnection(), tgtMixedOffsets, tgtMixedSizes,
        tgtMixedStrides, srcMixedOffsets, srcMixedSizes, srcMixedStrides);
  }
};
}  // namespace

void NpuCircularDmaCpyNdOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<DoublyStridedFolder<NpuCircularDmaCpyNdOp,
                                  NpuCircularDmaCpyNdOpReplacementBuilder>>(
      context);
}

//===----------------------------------------------------------------------===//
// AMDAIE_TileOp
//===----------------------------------------------------------------------===//

void TileOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "tile");
}

bool TileOp::hasStaticLocation() {
  return getConstantIntValue(getCol()) && getConstantIntValue(getRow());
}

bool TileOp::tileColumnComparator(AMDAIE::TileOp &a, AMDAIE::TileOp &b) {
  int64_t colA = getConstantIntValue(a.getCol()).value();
  int64_t colB = getConstantIntValue(b.getCol()).value();
  return colA < colB;
}

bool TileOp::tileValueColumnAndRowComparator(Value a, Value b) {
  TileOp tileA = cast<AMDAIE::TileOp>(a.getDefiningOp());
  TileOp tileB = cast<AMDAIE::TileOp>(b.getDefiningOp());
  int64_t colA = getConstantIntValue(tileA.getCol()).value();
  int64_t rowA = getConstantIntValue(tileA.getRow()).value();
  int64_t colB = getConstantIntValue(tileB.getCol()).value();
  int64_t rowB = getConstantIntValue(tileB.getRow()).value();
  if (colA == colB) return rowA < rowB;
  return colA < colB;
};

//===----------------------------------------------------------------------===//
// AMDAIE_WorkgroupOp
//===----------------------------------------------------------------------===//

// Make sure the WorkgroupOp region is well-formed with a ControlCodeOp
// terminator
void WorkgroupOp::ensureTerminator(Region &region, OpBuilder &builder,
                                   Location loc) {
  OpTrait::SingleBlockImplicitTerminator<ControlCodeOp>::Impl<
      WorkgroupOp>::ensureTerminator(region, builder, loc);
  auto terminator =
      llvm::dyn_cast<ControlCodeOp>(region.front().getTerminator());
  if (terminator.getRegion().empty()) {
    Block *newBlock = builder.createBlock(&terminator.getRegion());
    builder.setInsertionPointToEnd(newBlock);
    builder.create<AMDAIE::EndOp>(builder.getUnknownLoc());
  }
}

// Builder that ensures the WorkgroupOp is well-formed with a block and a
// ControlCodeOp terminator
void WorkgroupOp::build(OpBuilder &builder, OperationState &result) {
  Region *bodyRegion = result.addRegion();
  OpBuilder::InsertionGuard guard(builder);
  builder.createBlock(bodyRegion);
  Block &bodyBlock = bodyRegion->front();
  builder.setInsertionPointToStart(&bodyBlock);
  WorkgroupOp::ensureTerminator(*bodyRegion, builder, result.location);
}

LogicalResult WorkgroupOp::verify() {
  // Verify that this WorkgroupOp contains a ControlCodeOp terminator if one
  // exists.
  if (failed(OpTrait::SingleBlockImplicitTerminator<ControlCodeOp>::Impl<
             WorkgroupOp>::verifyRegionTrait(*this))) {
    return failure();
  }
  return success();
}
}  // namespace mlir::iree_compiler::AMDAIE
