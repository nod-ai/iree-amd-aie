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

std::optional<uint32_t> getNumberIterations(scf::ForOp loop) {
  std::optional<uint32_t> lowerBound =
      getConstantIntValue(loop.getLowerBound());
  std::optional<uint32_t> upperBound =
      getConstantIntValue(loop.getUpperBound());
  std::optional<uint32_t> step = getConstantIntValue(loop.getStep());

  if (!lowerBound || !upperBound || !step) {
    return std::nullopt;
  } else {
    return 1 + (upperBound.value() - lowerBound.value()) / step.value();
  }
}

// `bdIdCount` represents the upper bound on the number of BD IDs
// needed between the current DMA copy and its corresponding DMA wait.
//
// Example:
// %0 = dma_copy {bd_id = 0}   // Current DMA copy
// scf.for %arg0 = %c0 to %c1 step %c2 {
//   %1 = dma_copy {bd_id = %arg0 + 1}  // DMA copy inside a sub-loop
//   dma_wait(%1)                       // Wait for the sub-loop DMA copy
// }
// dma_wait(%0)   // Current DMA wait
//
// In this example:
// - The current DMA copy (%0) requires 1 BD ID.
// - The sub-loop executes 2 iterations, each requiring 1 BD ID.
// - Therefore, the upper bound for `bdIdCount` is:
//  3 = 1 (current) + 2 * 1(sub-loop).
//
// The lower bound will be 2 instead, if the DMA copy inside the sub-loop uses
// constant BD ID.
//
// To ensure the inner loop has access to more BD IDs, we compute
// the upper bound and use it to allocate BD IDs effectively.
void getNumRequiredBdIds(
    scf::ForOp loop, AMDAIE::NpuDmaCpyNdOp currDmaOp, AMDAIE::TileOp tileOp,
    DenseMap<Value, ChannelBdIdGenerator> &shimTileToGeneratorMap,
    uint32_t &bdIdCount) {
  bool startCounting =
      (currDmaOp == nullptr);  // Start immediately if no currDmaOp

  for (auto &op : loop.getOps()) {
    if (auto npuDmaOp = dyn_cast<AMDAIE::NpuDmaCpyNdOp>(op)) {
      // Skip until currDmaOp is found
      if (npuDmaOp == currDmaOp) startCounting = true;
      if (!startCounting) continue;
      if (npuDmaOp.getSource()) {
        FailureOr<AMDAIE::TileOp> tile =
            getGeneratorTileOp<CopyOpOperateOn::Source>(npuDmaOp,
                                                        shimTileToGeneratorMap);
        if (succeeded(tile) && *tile == tileOp) bdIdCount++;
      }
      if (npuDmaOp.getTarget()) {
        FailureOr<AMDAIE::TileOp> tile =
            getGeneratorTileOp<CopyOpOperateOn::Target>(npuDmaOp,
                                                        shimTileToGeneratorMap);
        if (succeeded(tile) && *tile == tileOp) bdIdCount++;
      }
    } else if (auto npuWaitOp = dyn_cast<AMDAIE::NpuDmaWaitOp>(op)) {
      for (AMDAIE::NpuDmaCpyNdOp npuDmaOp : npuWaitOp.getDmaOps()) {
        // Reached the DMA wait operation, stop counting.
        if (npuDmaOp == currDmaOp) return;
      }
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      uint32_t subLoopBdIdCount = 0;
      getNumRequiredBdIds(forOp, nullptr, tileOp, shimTileToGeneratorMap,
                          subLoopBdIdCount);
      std::optional<uint32_t> subIterations = getNumberIterations(forOp);
      if (subIterations) {
        bdIdCount += subLoopBdIdCount * subIterations.value();
      } else {
        bdIdCount += subLoopBdIdCount;
      }
    }
  }
}

template <CopyOpOperateOn OperateOn>
FailureOr<AMDAIE::BdIdOp> getBdIdOp(
    IRRewriter &rewriter, AMDAIE::NpuDmaCpyNdOp &npuDmaOp,
    DenseMap<Value, ChannelBdIdGenerator> &shimTileToGeneratorMap,
    DenseMap<AMDAIE::BdIdOp, SmallVector<uint32_t>> &bdIdOpToBdIdsMap,
    uint32_t channel) {
  FailureOr<AMDAIE::TileOp> tileOp =
      getGeneratorTileOp<OperateOn>(npuDmaOp, shimTileToGeneratorMap);
  if (failed(tileOp)) return failure();

  ChannelBdIdGenerator &generator = shimTileToGeneratorMap[tileOp->getResult()];
  rewriter.setInsertionPoint(npuDmaOp);
  if (scf::ForOp loop = npuDmaOp->getParentOfType<scf::ForOp>();
      loop && getNumberIterations(loop)) {
    // If the DMA is in a loop, using the semi-affine expression:
    // `iv % size + offset`,
    // `iv` is the loop induction variable,
    // `size` is the number of BD IDs assigned to each DMA op,
    // `offset` is the BD ID assigned to current DMA op at first iteration.
    Value iv = loop.getInductionVar();

    // Get the number of BD IDs will be assigned to current DMA op.
    uint32_t numRequired = 0;
    getNumRequiredBdIds(loop, npuDmaOp, *tileOp, shimTileToGeneratorMap,
                        numRequired);
    uint32_t numAvailable = generator.getNumAvailableBdIds(channel);
    uint32_t size = std::max(numAvailable / numRequired, 1u);

    // Only create expression if more than 1 BD ID is needed,
    // otherwise, fall back to constant BD ID.
    if (size > 1) {
      // Assigning BD IDs for all iterations in the loop.
      SmallVector<uint32_t> bdIds;
      for (uint32_t i = 0; i < size; i++) {
        std::optional<uint32_t> bdId = generator.getAndAssignBdId(
            channel, BdIdAssignmentMode::Incremental);
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
      AMDAIE::BdIdOp bdIdOp = rewriter.create<AMDAIE::BdIdOp>(
          rewriter.getUnknownLoc(), *tileOp, affineApply.getResult());
      bdIdOpToBdIdsMap[bdIdOp] = bdIds;
      return bdIdOp;
    }
  }

  // Assign a constant BD ID.
  std::optional<uint32_t> bdId =
      generator.getAndAssignBdId(channel, BdIdAssignmentMode::Incremental);
  if (!bdId) return failure();
  auto constant = rewriter.create<arith::ConstantOp>(
      rewriter.getUnknownLoc(), rewriter.getIndexAttr(bdId.value()));
  AMDAIE::BdIdOp bdIdOp = rewriter.create<AMDAIE::BdIdOp>(
      rewriter.getUnknownLoc(), *tileOp, constant.getResult());
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

  DenseMap<AMDAIE::BdIdOp, SmallVector<uint32_t>> bdIdOpToBdIdsMap;
  // Walk `amdaie.npu_dma_cpy_nd` and  `amdaie.dma_wait` operations and assign
  // and release BD IDs when encountering the respective operations using the
  // tile BD ID generators initialized earlier.
  AMDAIE::ControlCodeOp controlCodeOp = workgroupOp.getControlCode();
  WalkResult res = controlCodeOp->walk([&](Operation *op) {
    if (auto npuDmaOp = dyn_cast<AMDAIE::NpuDmaCpyNdOp>(op)) {
      if (npuDmaOp.getSource()) {
        FailureOr<AMDAIE::BdIdOp> bdIdOp = getBdIdOp<CopyOpOperateOn::Source>(
            rewriter, npuDmaOp, shimTileToGeneratorMap, bdIdOpToBdIdsMap,
            channel);
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
            rewriter, npuDmaOp, shimTileToGeneratorMap, bdIdOpToBdIdsMap,
            channel);
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
