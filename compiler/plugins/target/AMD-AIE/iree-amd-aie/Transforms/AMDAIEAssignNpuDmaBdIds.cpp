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

#define DEBUG_TYPE "iree-amdaie-assign-npu-dma-bd-ids"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Assign BD ids to NPU dma operations using the BD generator
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

  // Utility to retrieve a TileOp from a vector of tile values, while doing
  // appropriate verifications.
  auto getGeneratorTileOp = [&](AMDAIE::NpuDmaCpyNdOp &npuDmaOp,
                                const SmallVector<Value> &tiles,
                                AMDAIE::TileOp &tileOp) -> LogicalResult {
    if (tiles.size() != 1) {
      return npuDmaOp.emitOpError()
             << "operating on multiple tiles is not supported";
    }
    Value tile = tiles[0];
    if (!shimTileToGeneratorMap.contains(tile)) {
      return npuDmaOp.emitOpError()
             << "no channel BD ID generator found for tile: " << tile;
    }
    tileOp = dyn_cast_if_present<AMDAIE::TileOp>(tile.getDefiningOp());
    if (!tileOp) return npuDmaOp.emitOpError() << "no tile op found";
    return success();
  };

  // Walk `amdaie.npu_dma_cpy_nd` and  `amdaie.dma_wait` operations and assign
  // and release BD IDs when encountering the respective operations using the
  // tile BD ID generators initialized earlier.
  AMDAIE::ControlCodeOp controlCodeOp = workgroupOp.getControlCode();
  WalkResult res = controlCodeOp->walk([&](Operation *op) {
    if (auto npuDmaOp = dyn_cast<AMDAIE::NpuDmaCpyNdOp>(op)) {
      if (npuDmaOp.getSource()) {
        auto logicalObjFifo =
            dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
                npuDmaOp.getSource().getDefiningOp());
        if (!logicalObjFifo) {
          npuDmaOp.emitOpError() << "expected a source logical objectFifo";
          return WalkResult::interrupt();
        }
        SmallVector<Value> tiles = logicalObjFifo.getTiles();
        AMDAIE::TileOp tileOp;
        if (failed(getGeneratorTileOp(npuDmaOp, tiles, tileOp)))
          return WalkResult::interrupt();
        ChannelBdIdGenerator &generator =
            shimTileToGeneratorMap[tileOp.getResult()];
        // TODO(jornt): Temporarily use channel 0 for all DMAs. This should
        // return correct results for Shim channels, however, for generality
        // towards other DMAs and future hardware generations, channel
        // assignment should happen before BD assignemnt. This requires more
        // refactoring.
        std::optional<uint32_t> bdId = generator.getAndAssignBdId(0);
        rewriter.setInsertionPointAfter(tileOp);
        auto bdIdOp = rewriter.create<AMDAIE::BdIdOp>(rewriter.getUnknownLoc(),
                                                      tileOp, bdId.value());
        rewriter.setInsertionPoint(npuDmaOp);
        npuDmaOp = rewriter.replaceOpWithNewOp<AMDAIE::NpuDmaCpyNdOp>(
            npuDmaOp, npuDmaOp.getDma(), npuDmaOp.getTarget(),
            npuDmaOp.getTargetMixedOffsets(), npuDmaOp.getTargetMixedSizes(),
            npuDmaOp.getTargetMixedStrides(), npuDmaOp.getTargetBdId(),
            npuDmaOp.getSource(), npuDmaOp.getSourceMixedOffsets(),
            npuDmaOp.getSourceMixedSizes(), npuDmaOp.getSourceMixedStrides(),
            bdIdOp);
      }
      if (npuDmaOp.getTarget()) {
        auto logicalObjFifo =
            dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromMemrefOp>(
                npuDmaOp.getTarget().getDefiningOp());
        if (!logicalObjFifo) {
          npuDmaOp.emitOpError()
              << "expected a target `amdaie.logicalobjectfifo.from_memref`";
          return WalkResult::interrupt();
        }
        SmallVector<Value> tiles = logicalObjFifo.getTiles();
        AMDAIE::TileOp tileOp;
        if (failed(getGeneratorTileOp(npuDmaOp, tiles, tileOp)))
          return WalkResult::interrupt();
        ChannelBdIdGenerator &generator =
            shimTileToGeneratorMap[tileOp.getResult()];
        // TODO(jornt): Temporarily use channel 0 for all DMAs. This should
        // return correct results for Shim channels, however, for generality
        // towards other DMAs and future hardware generations, channel
        // assignment should happen before BD assignemnt. This requires more
        // refactoring.
        std::optional<uint32_t> bdId = generator.getAndAssignBdId(0);
        rewriter.setInsertionPointAfter(tileOp);
        auto bdIdOp = rewriter.create<AMDAIE::BdIdOp>(rewriter.getUnknownLoc(),
                                                      tileOp, bdId.value());
        rewriter.setInsertionPoint(npuDmaOp);
        (void)rewriter.replaceOpWithNewOp<AMDAIE::NpuDmaCpyNdOp>(
            npuDmaOp, npuDmaOp.getDma(), npuDmaOp.getTarget(),
            npuDmaOp.getTargetMixedOffsets(), npuDmaOp.getTargetMixedSizes(),
            npuDmaOp.getTargetMixedStrides(), bdIdOp, npuDmaOp.getSource(),
            npuDmaOp.getSourceMixedOffsets(), npuDmaOp.getSourceMixedSizes(),
            npuDmaOp.getSourceMixedStrides(), npuDmaOp.getSourceBdId());
      }
      return WalkResult::advance();
    } else if (auto npuWaitOp = dyn_cast<AMDAIE::NpuDmaWaitOp>(op)) {
      // Release BD ID used by input DMA op.
      AMDAIE::NpuDmaCpyNdOp npuDmaOp = npuWaitOp.getDmaOp();
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
      auto tileOp =
          dyn_cast_if_present<AMDAIE::TileOp>(bdIdOp.getTile().getDefiningOp());
      if (!tileOp) {
        bdIdOp.emitOpError() << "doesn't operate on a `amdaie.tile` operation";
        return WalkResult::interrupt();
      }
      if (!shimTileToGeneratorMap.contains(tileOp.getResult())) {
        bdIdOp.emitOpError()
            << "no BD ID generator found for this BD ID op's tile";
        return WalkResult::interrupt();
      }
      ChannelBdIdGenerator &generator =
          shimTileToGeneratorMap[tileOp.getResult()];
      uint32_t value = bdIdOp.getValue();
      generator.releaseBdId(value);
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
