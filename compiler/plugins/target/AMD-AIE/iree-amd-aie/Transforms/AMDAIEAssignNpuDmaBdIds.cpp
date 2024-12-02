// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/aie_runtime/Utils/ChannelBdIdGenerator.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"

#define DEBUG_TYPE "iree-amdaie-assign-npu-dma-bd-ids"

namespace mlir::iree_compiler::AMDAIE {

namespace {

// Utility to retrieve a TileOp from a vector of tile values, while doing
// appropriate verifications.
FailureOr<AMDAIE::TileOp> getGeneratorTileOp(
    AMDAIE::NpuDmaCpyNdOp &npuDmaOp,
    DenseMap<Value, ChannelBdIdGenerator> &shimTileToGeneratorMap) {
  SmallVector<Value> tiles;
  if (npuDmaOp.getSource()) {
    auto logicalObjFifo =
        dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
            npuDmaOp.getSource().getDefiningOp());
    if (!logicalObjFifo)
      return npuDmaOp.emitOpError() << "expected a source logical objectFifo";
    tiles = logicalObjFifo.getTiles();
  }
  if (npuDmaOp.getTarget()) {
    auto logicalObjFifo =
        dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromMemrefOp>(
            npuDmaOp.getTarget().getDefiningOp());
    if (!logicalObjFifo)
      return npuDmaOp.emitOpError()
             << "expected a target `amdaie.logicalobjectfifo.from_memref`";
    tiles = logicalObjFifo.getTiles();
  }
  if (tiles.size() != 1) {
    if (tiles.empty()) {
      return npuDmaOp.emitOpError() << "no tiles found";
    } else {
      return npuDmaOp.emitOpError()
             << "operating on multiple tiles is not supported";
    }
  }
  Value tile = tiles[0];
  if (!shimTileToGeneratorMap.contains(tile))
    return npuDmaOp.emitOpError()
           << "no channel BD ID generator found for tile: " << tile;

  auto tileOp = dyn_cast_if_present<AMDAIE::TileOp>(tile.getDefiningOp());
  if (!tileOp) return npuDmaOp.emitOpError() << "no tile op found";
  return tileOp;
};

// Check if the DMA operation is in the innermost loop of controlcode.
bool isInMostInnerLoop(AMDAIE::NpuDmaCpyNdOp op) {
  auto parentLoop = op->getParentOfType<scf::ForOp>();
  if (!parentLoop) return false;

  bool hasNestedLoop = false;
  parentLoop.walk([&](scf::ForOp nestedLoop) {
    if (nestedLoop != parentLoop) hasNestedLoop = true;
  });
  return !hasNestedLoop;
}

// Count the number of BD IDs needed per loop iteration,
// so that we know where to start the BD ID for the next iteration.
uint32_t countBdIdPerLoopIteration(
    scf::ForOp loop, AMDAIE::TileOp tileOp,
    DenseMap<Value, ChannelBdIdGenerator> &shimTileToGeneratorMap) {
  uint32_t count = 0;
  loop.walk([&](AMDAIE::NpuDmaCpyNdOp dmaOp) {
    if (dmaOp.getSource() || dmaOp.getTarget()) {
      FailureOr<AMDAIE::TileOp> tile =
          getGeneratorTileOp(dmaOp, shimTileToGeneratorMap);
      if (succeeded(tile) && *tile == tileOp) count++;
    }
  });
  return count;
}

FailureOr<AMDAIE::BdIdOp> getBdIdOp(
    IRRewriter &rewriter, AMDAIE::NpuDmaCpyNdOp &npuDmaOp,
    DenseMap<Value, ChannelBdIdGenerator> &shimTileToGeneratorMap,
    DenseMap<TileOp, uint32_t> &tileToBdIdOffsetMap,
    DenseMap<TileOp, uint32_t> &tileToBdIdSizeMap, uint32_t channel) {
  FailureOr<AMDAIE::TileOp> tileOp =
      getGeneratorTileOp(npuDmaOp, shimTileToGeneratorMap);
  if (failed(tileOp)) return failure();

  ChannelBdIdGenerator &generator = shimTileToGeneratorMap[tileOp->getResult()];
  std::optional<uint32_t> bdId =
      generator.getAndAssignBdId(channel, BdIdAssignmentMode::Incremental);
  if (!bdId) return failure();
  AMDAIE::BdIdOp bdIdOp;
  rewriter.setInsertionPoint(npuDmaOp);
  if (isInMostInnerLoop(npuDmaOp)) {
    // If the DMA is in the innermost loop, assign a BD ID using the
    // semi-affine expression:
    // `(iv * step + bdId - offset) % size + offset`,
    // `step` represents the number of BD IDs needed per loop iteration,
    // `bdId` is the BD ID assigned by the generator,
    // `offset` is the BD ID assigned to the first DMA in the loop,
    // `size` is the number of BD IDs available.
    if (!tileToBdIdOffsetMap.contains(*tileOp))
      tileToBdIdOffsetMap[*tileOp] = bdId.value();
    // plus one because one BD ID is just assigned in this function.
    if (!tileToBdIdSizeMap.contains(*tileOp))
      tileToBdIdSizeMap[*tileOp] = generator.getAvailableBdIdNum(channel) + 1;
    auto loop = npuDmaOp->getParentOfType<scf::ForOp>();
    uint32_t bdIdCount =
        countBdIdPerLoopIteration(loop, *tileOp, shimTileToGeneratorMap);
    Value iv = loop.getInductionVar();
    auto step = rewriter.create<arith::ConstantOp>(
        rewriter.getUnknownLoc(), rewriter.getIndexAttr(bdIdCount));
    auto diff = rewriter.create<arith::ConstantOp>(
        rewriter.getUnknownLoc(),
        rewriter.getIndexAttr(bdId.value() -
                              tileToBdIdOffsetMap[*tileOp]));  // bdId - offset
    auto offset = rewriter.create<arith::ConstantOp>(
        rewriter.getUnknownLoc(),
        rewriter.getIndexAttr(tileToBdIdOffsetMap[*tileOp]));
    auto size = rewriter.create<arith::ConstantOp>(
        rewriter.getUnknownLoc(),
        rewriter.getIndexAttr(tileToBdIdSizeMap[*tileOp]));
    auto mul = rewriter.create<arith::MulIOp>(rewriter.getUnknownLoc(), iv,
                                              step);  // iv * step
    auto add1 =
        rewriter.create<arith::AddIOp>(rewriter.getUnknownLoc(), mul,
                                       diff);  // iv * step + bdId - offset
    auto mod = rewriter.create<arith::RemUIOp>(
        rewriter.getUnknownLoc(), add1,
        size);  // (iv * step + bdId - offset) % size
    auto add2 = rewriter.create<arith::AddIOp>(
        rewriter.getUnknownLoc(), mod,
        offset);  // (iv * step + bdId - offset) % size + offset
    bdIdOp = rewriter.create<AMDAIE::BdIdOp>(rewriter.getUnknownLoc(), *tileOp,
                                             add2.getResult());
  } else {
    // If the DMA is not in the innermost loop, assign a constant BD ID
    auto constant = rewriter.create<arith::ConstantOp>(
        rewriter.getUnknownLoc(), rewriter.getIndexAttr(bdId.value()));
    bdIdOp = rewriter.create<AMDAIE::BdIdOp>(rewriter.getUnknownLoc(), *tileOp,
                                             constant.getResult());
  }
  return bdIdOp;
};

FailureOr<uint32_t> retriveBdId(arith::AddIOp add2) {
  uint32_t offset = getConstantIndexOrAssert(add2.getOperand(1));
  if (auto mod = dyn_cast<arith::RemUIOp>(add2.getOperand(0).getDefiningOp())) {
    if (auto add1 =
            dyn_cast<arith::AddIOp>(mod.getOperand(0).getDefiningOp())) {
      uint32_t diff = getConstantIndexOrAssert(add1.getOperand(1));
      uint32_t bdId = offset + diff;
      return bdId;
    }
  }
  return failure();
};

/// Assign BD ids to NPU dma operations using the BD generator.
LogicalResult assignNpuDmaBdIds(AMDAIE::WorkgroupOp workgroupOp) {
  IRRewriter rewriter(workgroupOp->getContext());

  // Get the device model.
  std::optional<AMDAIEDevice> device = getConfigAMDAIEDevice(workgroupOp);
  if (!device)
    return workgroupOp->emitOpError()
           << "could not find an AMDAIEDevice attribute";
  AMDAIEDeviceModel deviceModel = AMDAIE::getDeviceModel(device.value());

  // Create a BD ID generator for every shim tile.
  DenseMap<Value, ChannelBdIdGenerator> shimTileToGeneratorMap;
  workgroupOp->walk([&](AMDAIE::TileOp tileOp) {
    std::optional<int64_t> col = getConstantIntValue(tileOp.getCol());
    std::optional<int64_t> row = getConstantIntValue(tileOp.getRow());
    if (col && row && deviceModel.isShimNOCTile(col.value(), row.value())) {
      ChannelBdIdGenerator generator(
          deviceModel.getChannelToValidBdIds(AMDAIETileType::SHIMNOC));
      shimTileToGeneratorMap[tileOp.getResult()] = std::move(generator);
    }
  });

  // TODO(jornt): Temporarily use channel 0 for all DMAs. This should
  // return correct results for Shim channels, however, for generality
  // towards other DMAs and future hardware generations, channel
  // assignment should happen before BD assignemnt. This requires more
  // refactoring.
  const uint32_t channel = 0;

  DenseMap<AMDAIE::TileOp, uint32_t> tileToBdIdOffsetMap;
  DenseMap<AMDAIE::TileOp, uint32_t> tileToBdIdSizeMap;
  // Walk `amdaie.npu_dma_cpy_nd` and  `amdaie.dma_wait` operations and assign
  // and release BD IDs when encountering the respective operations using the
  // tile BD ID generators initialized earlier.
  AMDAIE::ControlCodeOp controlCodeOp = workgroupOp.getControlCode();
  WalkResult res = controlCodeOp->walk([&](Operation *op) {
    if (auto npuDmaOp = dyn_cast<AMDAIE::NpuDmaCpyNdOp>(op)) {
      if (npuDmaOp.getSource()) {
        FailureOr<AMDAIE::BdIdOp> bdIdOp =
            getBdIdOp(rewriter, npuDmaOp, shimTileToGeneratorMap,
                      tileToBdIdOffsetMap, tileToBdIdSizeMap, channel);
        if (failed(bdIdOp)) return WalkResult::interrupt();
        rewriter.setInsertionPoint(npuDmaOp);
        npuDmaOp = rewriter.replaceOpWithNewOp<AMDAIE::NpuDmaCpyNdOp>(
            npuDmaOp, npuDmaOp.getResultTypes(), npuDmaOp.getConnection(),
            npuDmaOp.getTarget(), npuDmaOp.getTargetMixedOffsets(),
            npuDmaOp.getTargetMixedSizes(), npuDmaOp.getTargetMixedStrides(),
            npuDmaOp.getTargetBdId(), npuDmaOp.getSource(),
            npuDmaOp.getSourceMixedOffsets(), npuDmaOp.getSourceMixedSizes(),
            npuDmaOp.getSourceMixedStrides(), *bdIdOp);
      }
      if (npuDmaOp.getTarget()) {
        FailureOr<AMDAIE::BdIdOp> bdIdOp =
            getBdIdOp(rewriter, npuDmaOp, shimTileToGeneratorMap,
                      tileToBdIdOffsetMap, tileToBdIdSizeMap, channel);
        if (failed(bdIdOp)) return WalkResult::interrupt();
        rewriter.setInsertionPoint(npuDmaOp);
        (void)rewriter.replaceOpWithNewOp<AMDAIE::NpuDmaCpyNdOp>(
            npuDmaOp, npuDmaOp.getResultTypes(), npuDmaOp.getConnection(),
            npuDmaOp.getTarget(), npuDmaOp.getTargetMixedOffsets(),
            npuDmaOp.getTargetMixedSizes(), npuDmaOp.getTargetMixedStrides(),
            *bdIdOp, npuDmaOp.getSource(), npuDmaOp.getSourceMixedOffsets(),
            npuDmaOp.getSourceMixedSizes(), npuDmaOp.getSourceMixedStrides(),
            npuDmaOp.getSourceBdId());
      }
      return WalkResult::advance();
    } else if (auto npuWaitOp = dyn_cast<AMDAIE::NpuDmaWaitOp>(op)) {
      // Release BD ID used by input DMA op.
      for (AMDAIE::NpuDmaCpyNdOp npuDmaOp : npuWaitOp.getDmaOps()) {
        AMDAIE::BdIdOp bdIdOp;
        if (npuDmaOp.getSourceBdId()) {
          bdIdOp = dyn_cast_if_present<AMDAIE::BdIdOp>(
              npuDmaOp.getSourceBdId().getDefiningOp());
        } else if (npuDmaOp.getTargetBdId()) {
          bdIdOp = dyn_cast_if_present<AMDAIE::BdIdOp>(
              npuDmaOp.getTargetBdId().getDefiningOp());
        } else {
          return WalkResult::advance();
        }
        if (!bdIdOp) return WalkResult::advance();
        auto tileOp = dyn_cast_if_present<AMDAIE::TileOp>(
            bdIdOp.getTile().getDefiningOp());
        if (!tileOp) {
          bdIdOp.emitOpError()
              << "doesn't operate on a `amdaie.tile` operation";
          return WalkResult::interrupt();
        }
        if (!shimTileToGeneratorMap.contains(tileOp.getResult())) {
          bdIdOp.emitOpError()
              << "no BD ID generator found for this BD ID op's tile";
          return WalkResult::interrupt();
        }
        ChannelBdIdGenerator &generator =
            shimTileToGeneratorMap[tileOp.getResult()];
        Value value = bdIdOp.getValue();
        if (auto addOp = value.getDefiningOp<arith::AddIOp>()) {
          // If the BD ID is a semi-affine expression, retrieve the BD ID for
          // the first iteration.
          FailureOr<uint32_t> bdId = retriveBdId(addOp);
          if (failed(bdId)) return WalkResult::interrupt();
          generator.releaseBdId(*bdId);
        } else {
          // Else, must be a constant BD ID.
          uint32_t bdId = getConstantIndexOrAssert(value);
          generator.releaseBdId(bdId);
        }
      }
      return WalkResult::advance();
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return failure();
  return success();
}

class AMDAIEAssignNpuDmaBdIdsPass
    : public impl::AMDAIEAssignNpuDmaBdIdsBase<AMDAIEAssignNpuDmaBdIdsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEAssignNpuDmaBdIdsPass() = default;
  AMDAIEAssignNpuDmaBdIdsPass(const AMDAIEAssignNpuDmaBdIdsPass &pass){};
  void runOnOperation() override;
};

void AMDAIEAssignNpuDmaBdIdsPass::runOnOperation() {
  Operation *parentOp = getOperation();

  WalkResult res = parentOp->walk([&](AMDAIE::WorkgroupOp workgroupOp) {
    if (failed(assignNpuDmaBdIds(workgroupOp))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEAssignNpuDmaBdIdsPass() {
  return std::make_unique<AMDAIEAssignNpuDmaBdIdsPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
