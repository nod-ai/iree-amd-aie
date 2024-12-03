// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEAttrs.h"
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
template <CopyOpOperateOn OperateOn>
FailureOr<AMDAIE::TileOp> getGeneratorTileOp(
    AMDAIE::NpuDmaCpyNdOp &npuDmaOp,
    DenseMap<Value, ChannelBdIdGenerator> &shimTileToGeneratorMap) {
  SmallVector<Value> tiles;
  if constexpr (OperateOn == CopyOpOperateOn::Source) {
    auto logicalObjFifo =
        dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
            npuDmaOp.getSource().getDefiningOp());
    if (!logicalObjFifo)
      return npuDmaOp.emitOpError() << "expected a source logical objectFifo";
    tiles = logicalObjFifo.getTiles();

  } else if constexpr (OperateOn == CopyOpOperateOn::Target) {
    auto logicalObjFifo =
        dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromMemrefOp>(
            npuDmaOp.getTarget().getDefiningOp());
    if (!logicalObjFifo)
      return npuDmaOp.emitOpError()
             << "expected a target `amdaie.logicalobjectfifo.from_memref`";
    tiles = logicalObjFifo.getTiles();
  } else {
    return npuDmaOp.emitOpError()
           << "Function can only operate on Source or Target";
  }
  if (tiles.size() != 1) {
    return npuDmaOp.emitOpError()
           << "expected to operate on a singe tile, but found: "
           << tiles.size();
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
uint32_t getNumRequiredBdIds(
    scf::ForOp loop, AMDAIE::TileOp tileOp,
    DenseMap<Value, ChannelBdIdGenerator> &shimTileToGeneratorMap) {
  uint32_t count = 0;
  loop.walk([&](AMDAIE::NpuDmaCpyNdOp dmaOp) {
    if (dmaOp.getSource()) {
      FailureOr<AMDAIE::TileOp> tile =
          getGeneratorTileOp<CopyOpOperateOn::Source>(dmaOp,
                                                      shimTileToGeneratorMap);
      if (succeeded(tile) && *tile == tileOp) count++;
    }
    if (dmaOp.getTarget()) {
      FailureOr<AMDAIE::TileOp> tile =
          getGeneratorTileOp<CopyOpOperateOn::Target>(dmaOp,
                                                      shimTileToGeneratorMap);
      if (succeeded(tile) && *tile == tileOp) count++;
    }
  });
  return count;
}

template <CopyOpOperateOn OperateOn>
FailureOr<AMDAIE::BdIdOp> getBdIdOp(
    IRRewriter &rewriter, AMDAIE::NpuDmaCpyNdOp &npuDmaOp,
    DenseMap<Value, ChannelBdIdGenerator> &shimTileToGeneratorMap,
    DenseMap<AMDAIE::TileOp, uint32_t> &tileToBdIdSizeMap,
    DenseMap<AMDAIE::BdIdOp, SmallVector<uint32_t>> &bdIdOpToBdIdsMap,
    uint32_t channel) {
  FailureOr<AMDAIE::TileOp> tileOp =
      getGeneratorTileOp<OperateOn>(npuDmaOp, shimTileToGeneratorMap);
  if (failed(tileOp)) return failure();

  ChannelBdIdGenerator &generator = shimTileToGeneratorMap[tileOp->getResult()];
  AMDAIE::BdIdOp bdIdOp;
  rewriter.setInsertionPoint(npuDmaOp);
  if (isInMostInnerLoop(npuDmaOp)) {
    // If the DMA is in the innermost loop, using the semi-affine expression:
    // `iv % size + offset`,
    // `iv` is the loop induction variable,
    // `size` is the number of BD IDs assigned to each DMA op,
    // `offset` is the BD ID assigned to current DMA op at first iteration.
    scf::ForOp loop = npuDmaOp->getParentOfType<scf::ForOp>();
    Value iv = loop.getInductionVar();

    if (!tileToBdIdSizeMap.contains(*tileOp)) {
      uint32_t numRequired =
          getNumRequiredBdIds(loop, *tileOp, shimTileToGeneratorMap);
      uint32_t numAvailable = generator.getNumAvailableBdIds(channel);
      // In case of numRequired > numAvailable, BD ID size is set to 1, since
      // reusing will happen.
      tileToBdIdSizeMap[*tileOp] = std::max(numAvailable / numRequired, 1u);
    }
    uint32_t size = tileToBdIdSizeMap[*tileOp];

    // Assigning BD IDs for all iterations in the loop.
    SmallVector<uint32_t> bdIds;
    for (uint32_t i = 0; i < size; i++) {
      std::optional<uint32_t> bdId =
          generator.getAndAssignBdId(channel, BdIdAssignmentMode::Incremental);
      if (!bdId) return failure();
      bdIds.push_back(bdId.value());
    }
    // Get the BD ID for the first iteration as the offset.
    uint32_t offset = bdIds.front();

    // Create the semi-affine expression.
    AffineExpr ivExpr;
    bindDims(loop.getContext(), ivExpr);
    auto affineApply = rewriter.create<affine::AffineApplyOp>(
        loop.getLoc(), ivExpr % size + offset,
        ValueRange{
            iv,
        });
    bdIdOp = rewriter.create<AMDAIE::BdIdOp>(rewriter.getUnknownLoc(), *tileOp,
                                             affineApply.getResult());
    bdIdOpToBdIdsMap[bdIdOp] = bdIds;
  } else {
    // If the DMA is not in the innermost loop, assign a constant BD ID.
    std::optional<uint32_t> bdId =
        generator.getAndAssignBdId(channel, BdIdAssignmentMode::Incremental);
    if (!bdId) return failure();
    auto constant = rewriter.create<arith::ConstantOp>(
        rewriter.getUnknownLoc(), rewriter.getIndexAttr(bdId.value()));
    bdIdOp = rewriter.create<AMDAIE::BdIdOp>(rewriter.getUnknownLoc(), *tileOp,
                                             constant.getResult());
  }
  return bdIdOp;
};

LogicalResult releaseBdId(
    AMDAIE::BdIdOp bdIdOp,
    DenseMap<Value, ChannelBdIdGenerator> &shimTileToGeneratorMap,
    DenseMap<AMDAIE::BdIdOp, SmallVector<uint32_t>> &bdIdOpToBdIdsMap) {
  auto tileOp =
      dyn_cast_if_present<AMDAIE::TileOp>(bdIdOp.getTile().getDefiningOp());
  if (!tileOp)
    return bdIdOp.emitOpError()
           << "doesn't operate on a `amdaie.tile` operation";

  if (!shimTileToGeneratorMap.contains(tileOp.getResult()))
    return bdIdOp.emitOpError()
           << "no BD ID generator found for this BD ID op's tile";

  ChannelBdIdGenerator &generator = shimTileToGeneratorMap[tileOp.getResult()];
  Value value = bdIdOp.getValue();
  if (auto op = value.getDefiningOp<affine::AffineApplyOp>()) {
    // If the BD ID is a semi-affine expression.
    if (bdIdOpToBdIdsMap.contains(bdIdOp)) {
      for (uint32_t bdId : bdIdOpToBdIdsMap[bdIdOp]) {
        generator.releaseBdId(bdId);
      }
    } else {
      return bdIdOp.emitOpError() << "no BD IDs found for this expression";
    }
  } else {
    // Else, must be a constant BD ID.
    uint32_t bdId = getConstantIndexOrAssert(value);
    generator.releaseBdId(bdId);
  }
  return success();
}

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

  DenseMap<AMDAIE::TileOp, uint32_t> tileToBdIdSizeMap;
  DenseMap<AMDAIE::BdIdOp, SmallVector<uint32_t>> bdIdOpToBdIdsMap;
  // Walk `amdaie.npu_dma_cpy_nd` and  `amdaie.dma_wait` operations and assign
  // and release BD IDs when encountering the respective operations using the
  // tile BD ID generators initialized earlier.
  AMDAIE::ControlCodeOp controlCodeOp = workgroupOp.getControlCode();
  WalkResult res = controlCodeOp->walk([&](Operation *op) {
    if (auto npuDmaOp = dyn_cast<AMDAIE::NpuDmaCpyNdOp>(op)) {
      if (npuDmaOp.getSource()) {
        FailureOr<AMDAIE::BdIdOp> bdIdOp = getBdIdOp<CopyOpOperateOn::Source>(
            rewriter, npuDmaOp, shimTileToGeneratorMap, tileToBdIdSizeMap,
            bdIdOpToBdIdsMap, channel);
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
        FailureOr<AMDAIE::BdIdOp> bdIdOp = getBdIdOp<CopyOpOperateOn::Target>(
            rewriter, npuDmaOp, shimTileToGeneratorMap, tileToBdIdSizeMap,
            bdIdOpToBdIdsMap, channel);
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
          if (!bdIdOp) return WalkResult::advance();
          if (failed(releaseBdId(bdIdOp, shimTileToGeneratorMap,
                                 bdIdOpToBdIdsMap)))
            return WalkResult::interrupt();
        }

        if (npuDmaOp.getTargetBdId()) {
          bdIdOp = dyn_cast_if_present<AMDAIE::BdIdOp>(
              npuDmaOp.getTargetBdId().getDefiningOp());
          if (!bdIdOp) return WalkResult::advance();
          if (failed(releaseBdId(bdIdOp, shimTileToGeneratorMap,
                                 bdIdOpToBdIdsMap)))
            return WalkResult::interrupt();
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
    registry.insert<AMDAIEDialect, affine::AffineDialect>();
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
