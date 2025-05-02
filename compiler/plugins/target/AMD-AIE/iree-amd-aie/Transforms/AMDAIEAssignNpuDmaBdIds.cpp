// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEAttrs.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/Utils/ChannelBdIdGenerator.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"

#define DEBUG_TYPE "iree-amdaie-assign-npu-dma-bd-ids"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Utility to retrieve a TileOp from a vector of tile values, while doing
/// appropriate verifications.
template <CopyOpOperateOn OperateOn>
FailureOr<AMDAIE::TileOp> getGeneratorTileOp(
    AMDAIE::NpuDmaCpyNdOp npuDmaOp,
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

/// Given a list of DMA Ops and the number of required BD IDs by the first DMA
/// op in the list, split the available BD IDs equally amongst all and assign it
/// to the DMA ops.
LogicalResult assignBdIdsToDMAOpsBatch(
    IRRewriter &rewriter, SmallVector<AMDAIE::NpuDmaCpyNdOp> &dmaOps,
    AMDAIE::TileOp tileOp,
    DenseMap<Value, ChannelBdIdGenerator> &shimTileToGeneratorMap,
    DenseMap<AMDAIE::BdIdOp, SmallVector<uint32_t>> &bdIdOpToBdIdsMap,
    DenseMap<AMDAIE::NpuDmaCpyNdOp, SmallVector<AMDAIE::BdIdOp>>
        &dmaOpToBdIdMap,
    uint32_t numRequiredBdIds) {
  llvm::outs() << "assignBdIdsToDMAOpsBatch BEGIN ======\n";
  llvm::outs() << "DB - 1\n";
  llvm::outs().flush();
  // Get the channel.
  FailureOr<AMDAIE::ChannelOp> maybeChannelOp;
  if (dmaOps[0].getSource()) {
    FailureOr<AMDAIE::TileOp> tile =
        getGeneratorTileOp<CopyOpOperateOn::Source>(dmaOps[0],
                                                    shimTileToGeneratorMap);
    if (succeeded(tile)) maybeChannelOp = dmaOps[0].getSourceChannelOp();
  }
  if (dmaOps[0].getTarget()) {
    FailureOr<AMDAIE::TileOp> tile =
        getGeneratorTileOp<CopyOpOperateOn::Target>(dmaOps[0],
                                                    shimTileToGeneratorMap);
    if (succeeded(tile)) maybeChannelOp = dmaOps[0].getTargetChannelOp();
  }
  llvm::outs() << "DB - 2\n";
  llvm::outs().flush();
  if (failed(maybeChannelOp)) return failure();
  uint32_t channel = maybeChannelOp.value().getValue();
  // Compute BD ID split amongst all DMA ops.
  ChannelBdIdGenerator &generator = shimTileToGeneratorMap[tileOp.getResult()];
  uint32_t numAvailable = generator.getNumAvailableBdIds(channel);
  uint32_t totalDmaOps = dmaOps.size();
  uint32_t size = std::max(numAvailable / numRequiredBdIds, 1u);
  scf::ForOp loop = nullptr;
  AffineExpr ivExpr = nullptr;
  Value iv = nullptr;
  // In case the parent of the DMA ops is a scf.for we need to keep track of the
  // loop induction variable in order to create a semi affine expression later
  // for distributing BD IDs for each iteration.
  if (loop = dmaOps[0]->getParentOfType<scf::ForOp>();
      loop && getNumberIterations(loop)) {
    iv = loop.getInductionVar();
    bindDims(loop.getContext(), ivExpr);
  } else {
    // In case the DMA ops are not surrounded by scf.for, we will assign only
    // one BD ID.
    size = 1;
  }
  llvm::outs() << "DB - 3\n";
  llvm::outs().flush();
  // In case there are not enough BD ids available, return failure.
  if (size * totalDmaOps > numAvailable) return failure();

  llvm::outs() << "DB - 4\n";
  llvm::outs().flush();
  // Traverse each DMA op found in step 1, assign BD IDs and keep a track of the
  // first BD ID op assigned.
  rewriter.setInsertionPoint(dmaOps[0]);
  llvm::outs() << "DMAOps batch size = " << dmaOps.size() << "\n";
  llvm::outs().flush();
  for (AMDAIE::NpuDmaCpyNdOp dmaOp : dmaOps) {
    uint32_t bdIdMapIndex = 0;
    if (dmaOp.getSource()) {
      FailureOr<AMDAIE::TileOp> tile =
          getGeneratorTileOp<CopyOpOperateOn::Source>(dmaOp,
                                                      shimTileToGeneratorMap);
      if (succeeded(tile) && *tile == tileOp) {
        bdIdMapIndex = 0;
      }
    }
    if (dmaOp.getTarget()) {
      FailureOr<AMDAIE::TileOp> tile =
          getGeneratorTileOp<CopyOpOperateOn::Target>(dmaOp,
                                                      shimTileToGeneratorMap);
      if (succeeded(tile) && *tile == tileOp) {
        bdIdMapIndex = 1;
      }
    }
    // Only create expression if more than 1 BD ID is needed and if,
    // otherwise, fall back to constant BD ID.
    if (size > 1) {
      // Assigning BD IDs for all iterations in the loop.
      SmallVector<uint32_t> bdIds;
      llvm::outs() << "DB - 1.0\n";
      llvm::outs().flush();
      for (uint32_t i = 0; i < size; i++) {
        std::optional<uint32_t> bdId = generator.getAndAssignBdId(
            channel, BdIdAssignmentMode::Incremental);
        if (!bdId) return failure();
        bdIds.push_back(bdId.value());
      }
      llvm::outs() << "DB - 1.1\n";
      llvm::outs().flush();
      // Get the BD ID for the first iteration as the offset.
      uint32_t offset = bdIds.front();

      // Create the semi-affine expression.
      auto affineApply = rewriter.create<affine::AffineApplyOp>(
          loop.getLoc(), ivExpr % size + offset,
          ValueRange{
              iv,
          });
      AMDAIE::BdIdOp bdIdOp = rewriter.create<AMDAIE::BdIdOp>(
          rewriter.getUnknownLoc(), tileOp, affineApply.getResult());
      bdIdOpToBdIdsMap[bdIdOp] = bdIds;
      if (!dmaOpToBdIdMap.contains(dmaOp)) {
        SmallVector<AMDAIE::BdIdOp> bdIdOps = {nullptr, nullptr};
        dmaOpToBdIdMap[dmaOp] = bdIdOps;
      }

      dmaOpToBdIdMap[dmaOp][bdIdMapIndex] = bdIdOp;
    } else {
      // Assign a constant BD ID.
      std::optional<uint32_t> bdId =
          generator.getAndAssignBdId(channel, BdIdAssignmentMode::Incremental);
      llvm::outs() << "DB - 2.0\n";
      llvm::outs().flush();
      if (!bdId) return dmaOp.emitOpError() << "no BD ID available";
      llvm::outs() << "DB - 2.1\n";
      llvm::outs().flush();
      auto constant = rewriter.create<arith::ConstantOp>(
          rewriter.getUnknownLoc(), rewriter.getIndexAttr(bdId.value()));
      AMDAIE::BdIdOp bdIdOp = rewriter.create<AMDAIE::BdIdOp>(
          rewriter.getUnknownLoc(), tileOp, constant.getResult());
      if (!dmaOpToBdIdMap.contains(dmaOp)) {
        SmallVector<AMDAIE::BdIdOp> bdIdOps = {nullptr, nullptr};
        dmaOpToBdIdMap[dmaOp] = bdIdOps;
      }
      dmaOpToBdIdMap[dmaOp][bdIdMapIndex] = bdIdOp;
    }
  }
  llvm::outs() << "END =================\n";
  llvm::outs().flush();
  return success();
}

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

static LogicalResult releaseAllBdIds(
    SmallVector<AMDAIE::NpuDmaCpyNdOp> dmaOps,
    DenseMap<Value, ChannelBdIdGenerator> &shimTileToGeneratorMap,
    DenseMap<AMDAIE::BdIdOp, SmallVector<uint32_t>> &bdIdOpToBdIdsMap) {
  // Release BD ID used by input DMA op.
  for (AMDAIE::NpuDmaCpyNdOp npuDmaOp : dmaOps) {
    AMDAIE::BdIdOp bdIdOp;
    if (npuDmaOp.getSourceBdId()) {
      bdIdOp = dyn_cast_if_present<AMDAIE::BdIdOp>(
          npuDmaOp.getSourceBdId().getDefiningOp());
      if (!bdIdOp) continue;
      if (failed(releaseBdId(bdIdOp, shimTileToGeneratorMap, bdIdOpToBdIdsMap)))
        return failure();
    }

    if (npuDmaOp.getTargetBdId()) {
      bdIdOp = dyn_cast_if_present<AMDAIE::BdIdOp>(
          npuDmaOp.getTargetBdId().getDefiningOp());
      if (!bdIdOp) continue;
      if (failed(releaseBdId(bdIdOp, shimTileToGeneratorMap, bdIdOpToBdIdsMap)))
        return failure();
    }
  }
  return success();
}

typedef struct ControlCodeGraphStruct {
  llvm::MapVector<
      AMDAIE::TileOp,
      llvm::MapVector<Operation *,
                      SmallVector<SmallVector<AMDAIE::NpuDmaCpyNdOp>>>>
      tileParentOpDmaBatchMap;
  llvm::MapVector<Operation *, SmallVector<Operation *>>
      parentOpToImmediateInnerParentOps;
  DenseMap<AMDAIE::NpuDmaCpyNdOp, AMDAIE::NpuDmaWaitOp> dmaOpToWaitOp;
} ControlCodeGraph;

static void formControlCodeGraph(
    Operation *parentOp, ControlCodeGraph &controlCodeGraph,
    DenseMap<Value, ChannelBdIdGenerator> &shimTileToGeneratorMap) {
  DenseMap<AMDAIE::TileOp, AMDAIE::NpuDmaCpyNdOp> tileToFirstDmaOpMap;
  SmallVector<Operation *> immediateInnerParentOps;
  auto updatetileParentOpDmaBatchMap = [&](AMDAIE::TileOp tile,
                                           Operation *parentOp,
                                           AMDAIE::NpuDmaCpyNdOp dmaOp) {
    if (!controlCodeGraph.tileParentOpDmaBatchMap.contains(tile)) {
      llvm::MapVector<Operation *,
                      SmallVector<SmallVector<AMDAIE::NpuDmaCpyNdOp>>>
          parentOpToBatchMap;
      parentOpToBatchMap[parentOp] = {};
      controlCodeGraph.tileParentOpDmaBatchMap[tile] = parentOpToBatchMap;
    }
    if (!tileToFirstDmaOpMap.contains(tile) || !tileToFirstDmaOpMap[tile]) {
      controlCodeGraph.tileParentOpDmaBatchMap[tile][parentOp].push_back(
          {dmaOp});
      tileToFirstDmaOpMap[tile] = dmaOp;
    } else {
      int32_t totalBatchSoFar =
          controlCodeGraph.tileParentOpDmaBatchMap[tile][parentOp].size();
      assert((totalBatchSoFar >= 1) &&
             "expected at least on DMAOp in the batch");
      controlCodeGraph
          .tileParentOpDmaBatchMap[tile][parentOp][totalBatchSoFar - 1]
          .push_back(dmaOp);
    }
  };
  AMDAIE::NpuDmaCpyNdOp currDmaOp = nullptr;
  // Traverse the parent operation's immediate child ops and form the
  // tileParentOpDmaBatchMap.
  for (Operation &op : parentOp->getRegion(0).getOps()) {
    if (isa<scf::ForOp, scf::ForallOp>(op))
      immediateInnerParentOps.push_back(&op);
    else if (auto dmaOp = dyn_cast<AMDAIE::NpuDmaCpyNdOp>(op)) {
      if (dmaOp.getSource()) {
        FailureOr<AMDAIE::TileOp> tile =
            getGeneratorTileOp<CopyOpOperateOn::Source>(dmaOp,
                                                        shimTileToGeneratorMap);
        if (succeeded(tile))
          updatetileParentOpDmaBatchMap(*tile, parentOp, dmaOp);
      }
      if (dmaOp.getTarget()) {
        FailureOr<AMDAIE::TileOp> tile =
            getGeneratorTileOp<CopyOpOperateOn::Target>(dmaOp,
                                                        shimTileToGeneratorMap);
        if (succeeded(tile))
          updatetileParentOpDmaBatchMap(*tile, parentOp, dmaOp);
      }
      if (!currDmaOp) currDmaOp = dmaOp;
    } else if (auto npuWaitOp = dyn_cast<AMDAIE::NpuDmaWaitOp>(op)) {
      for (AMDAIE::NpuDmaCpyNdOp npuDmaOp : npuWaitOp.getDmaOps()) {
        // Reached the DMA wait operation, reset tracking of current DMA op for
        // the tile.
        if (npuDmaOp == currDmaOp) {
          controlCodeGraph.dmaOpToWaitOp[npuDmaOp] = npuWaitOp;
          currDmaOp = nullptr;
          if (npuDmaOp.getSource()) {
            FailureOr<AMDAIE::TileOp> tile =
                getGeneratorTileOp<CopyOpOperateOn::Source>(
                    npuDmaOp, shimTileToGeneratorMap);
            if (succeeded(tile)) {
              tileToFirstDmaOpMap[*tile] = nullptr;
            }
          }
          if (npuDmaOp.getTarget()) {
            FailureOr<AMDAIE::TileOp> tile =
                getGeneratorTileOp<CopyOpOperateOn::Target>(
                    npuDmaOp, shimTileToGeneratorMap);
            if (succeeded(tile)) {
              tileToFirstDmaOpMap[*tile] = nullptr;
            }
          }
        }
      }
    }
  }
  controlCodeGraph.parentOpToImmediateInnerParentOps[parentOp] =
      immediateInnerParentOps;
  // Traverse the immediate inner child operations of the parent op which can in
  // turn have DMA ops.
  for (Operation *op : immediateInnerParentOps)
    formControlCodeGraph(op, controlCodeGraph, shimTileToGeneratorMap);
}

static uint32_t traverseInnerParentOpChain(AMDAIE::TileOp tile,
                                           Operation *parentOp,
                                           ControlCodeGraph &controlCodeGraph) {
  uint32_t requiredBdIds = 0;
  if (isa<scf::ForallOp>(parentOp)) return 0;
  if (!controlCodeGraph.tileParentOpDmaBatchMap[tile].contains(parentOp)) {
    requiredBdIds = 0;
  } else {
    SmallVector<SmallVector<AMDAIE::NpuDmaCpyNdOp>> batches =
        controlCodeGraph.tileParentOpDmaBatchMap[tile][parentOp];
    uint32_t totalDmaOpsInParentOpOperatingOnTile = 0;
    for (SmallVector<AMDAIE::NpuDmaCpyNdOp> batch : batches) {
      totalDmaOpsInParentOpOperatingOnTile += batch.size();
    }
    requiredBdIds = totalDmaOpsInParentOpOperatingOnTile;
  }
  for (Operation *immediateInnerParentOp :
       controlCodeGraph.parentOpToImmediateInnerParentOps[parentOp]) {
    uint32_t requiredBdIdsByInnerParentOp = traverseInnerParentOpChain(
        tile, immediateInnerParentOp, controlCodeGraph);
    std::optional<uint32_t> loopIterations =
        getNumberIterations(cast<scf::ForOp>(immediateInnerParentOp));
    if (loopIterations) {
      requiredBdIds += requiredBdIdsByInnerParentOp * loopIterations.value();
    } else {
      requiredBdIds += requiredBdIdsByInnerParentOp;
    }
  }
  return requiredBdIds;
}

/// Computes the number of BD IDs required between the current
/// DMA copy operation and its corresponding DMA wait operation; returns the DMA
/// ops it traverses within same block and on same tile. If a sub-loop is
/// encountered, it takes into account any DMA ops being used on the same tile.
/// This approach ensures that the inner loop has access to a greater
/// number of BD IDs, which is favorable for enabling efficient BD chaining in
/// subsequent passes.
///
/// Example:
/// %0 = dma_copy {bd_id = 0}   // Current DMA copy
/// scf.for %arg0 = %c0 to %c1 step %c2 {
///   %1 = dma_copy {bd_id = %arg0 + 1}  // DMA copy inside a sub-loop
///   dma_wait(%1)                       // Wait for the sub-loop DMA copy
/// }
/// dma_wait(%0)   // Current DMA wait
///
/// In this example:
/// - The current DMA copy (%0) requires 1 BD ID.
/// - The sub-loop executes 2 iterations, each requiring 1 BD ID.
/// - Therefore, the required number of BD IDs is:
///  3 = 1 (current) + 2 * 1(sub-loop).
static uint32_t traverseParentOpChainToGetBdIds(
    AMDAIE::TileOp tile, Operation *parentOp,
    ControlCodeGraph &controlCodeGraph, AMDAIE::NpuDmaCpyNdOp npuDmaOp) {
  uint32_t requiredBdIds = 0;
  AMDAIE::NpuDmaWaitOp npuWaitOp = controlCodeGraph.dmaOpToWaitOp[npuDmaOp];
  for (Operation *immediateInnerParentOp :
       controlCodeGraph.parentOpToImmediateInnerParentOps[parentOp]) {
    if (!immediateInnerParentOp->isBeforeInBlock(npuWaitOp)) break;
    if (auto forOp = dyn_cast<scf::ForOp>(immediateInnerParentOp)) {
      uint32_t requiredBdIdsByInnerParentOp = traverseInnerParentOpChain(
          tile, immediateInnerParentOp, controlCodeGraph);
      std::optional<uint32_t> loopIterations = getNumberIterations(forOp);
      if (loopIterations) {
        requiredBdIds += requiredBdIdsByInnerParentOp * loopIterations.value();
      } else {
        requiredBdIds += requiredBdIdsByInnerParentOp;
      }
    }
  }
  return requiredBdIds;
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

  DenseMap<AMDAIE::BdIdOp, SmallVector<uint32_t>> bdIdOpToBdIdsMap;
  // Walk `amdaie.npu_dma_cpy_nd` and  `amdaie.dma_wait` operations and assign
  // and release BD IDs when encountering the respective operations using the
  // tile BD ID generators initialized earlier.
  AMDAIE::ControlCodeOp controlCodeOp = workgroupOp.getControlCode();
  // Since a DMA op can have source and target, therefore we can have two BD IDs
  // for any DMA op. Hence we maintain a map from DMA op to a vector of BD IDs.
  DenseMap<AMDAIE::NpuDmaCpyNdOp, SmallVector<AMDAIE::BdIdOp>> dmaOpToBdIdMap;
  ControlCodeGraph controlCodeGraph;
  formControlCodeGraph(controlCodeOp, controlCodeGraph, shimTileToGeneratorMap);
  // For each (tile, parentOp) pair assign BD Ids to DMA Ops batch by updating
  // `dmaOpToBdIdMap`.
  for (auto tileParentOpDmaBatch : controlCodeGraph.tileParentOpDmaBatchMap) {
    AMDAIE::TileOp tile = tileParentOpDmaBatch.first;
    llvm::outs() << "Tile = " << tile << "\n";
    llvm::outs().flush();
    for (auto parentOpDmaBatch : tileParentOpDmaBatch.second) {
      Operation *parentOp = parentOpDmaBatch.first;
      llvm::outs() << "ParentOp = " << (*parentOp) << "\n";
      llvm::outs().flush();
      for (SmallVector<AMDAIE::NpuDmaCpyNdOp> batch : parentOpDmaBatch.second) {
        uint32_t requiredBdIdsByInnerOpChain = traverseParentOpChainToGetBdIds(
            tile, parentOp, controlCodeGraph, batch[0]);
        uint32_t totalBdIdsRequiredPerIteration =
            requiredBdIdsByInnerOpChain + batch.size();
        if (failed(assignBdIdsToDMAOpsBatch(
                rewriter, batch, tile, shimTileToGeneratorMap, bdIdOpToBdIdsMap,
                dmaOpToBdIdMap, totalBdIdsRequiredPerIteration)))
          return failure();
        if (failed(releaseAllBdIds(batch, shimTileToGeneratorMap,
                                   bdIdOpToBdIdsMap)))
          return failure();
      }
    }
  }
  // At this step we have all the information to traverse and perform the
  // replacements of the DMA Ops.
  WalkResult res = controlCodeOp->walk([&](Operation *op) {
    if (auto npuDmaOp = dyn_cast<AMDAIE::NpuDmaCpyNdOp>(op)) {
      if (npuDmaOp.getSource()) {
        assert(dmaOpToBdIdMap.contains(npuDmaOp) && "No BD ID mapping found");
        assert((dmaOpToBdIdMap[npuDmaOp][/*sourceBdIdIndex=*/0] != nullptr) &&
               "No source BD ID mapping found");
        AMDAIE::BdIdOp bdIdOp = dmaOpToBdIdMap[npuDmaOp][/*sourceBdIdIndex=*/0];
        rewriter.setInsertionPoint(npuDmaOp);
        AMDAIE::NpuDmaCpyNdOp oldNpuDmaOp = npuDmaOp;
        npuDmaOp = rewriter.replaceOpWithNewOp<AMDAIE::NpuDmaCpyNdOp>(
            npuDmaOp, npuDmaOp.getResultTypes(), npuDmaOp.getConnection(),
            npuDmaOp.getTarget(), npuDmaOp.getTargetMixedOffsets(),
            npuDmaOp.getTargetMixedSizes(), npuDmaOp.getTargetMixedStrides(),
            npuDmaOp.getTargetBdId(), npuDmaOp.getSource(),
            npuDmaOp.getSourceMixedOffsets(), npuDmaOp.getSourceMixedSizes(),
            npuDmaOp.getSourceMixedStrides(), bdIdOp);
        // Since we have created a new NPU DMA op by assigning BD ID to the
        // source, we need to copy the BD ID assignment to this new NPU DMA op
        // in order to preserve the information about target BD ID candidate.
        dmaOpToBdIdMap[npuDmaOp] = dmaOpToBdIdMap[oldNpuDmaOp];
      }
      if (npuDmaOp.getTarget()) {
        assert(dmaOpToBdIdMap.contains(npuDmaOp) && "No BD ID mapping found");
        assert((dmaOpToBdIdMap[npuDmaOp][/*targetBdIdIndex=*/1] != nullptr) &&
               "No target BD ID mapping found");
        AMDAIE::BdIdOp bdIdOp = dmaOpToBdIdMap[npuDmaOp][/*targetBdIdIndex=*/1];
        rewriter.setInsertionPoint(npuDmaOp);
        (void)rewriter.replaceOpWithNewOp<AMDAIE::NpuDmaCpyNdOp>(
            npuDmaOp, npuDmaOp.getResultTypes(), npuDmaOp.getConnection(),
            npuDmaOp.getTarget(), npuDmaOp.getTargetMixedOffsets(),
            npuDmaOp.getTargetMixedSizes(), npuDmaOp.getTargetMixedStrides(),
            bdIdOp, npuDmaOp.getSource(), npuDmaOp.getSourceMixedOffsets(),
            npuDmaOp.getSourceMixedSizes(), npuDmaOp.getSourceMixedStrides(),
            npuDmaOp.getSourceBdId());
      }
      return WalkResult::advance();
    } else if (auto npuWaitOp = dyn_cast<AMDAIE::NpuDmaWaitOp>(op)) {
      // // Release BD ID used by input DMA op.
      // for (AMDAIE::NpuDmaCpyNdOp npuDmaOp : npuWaitOp.getDmaOps()) {
      //   AMDAIE::BdIdOp bdIdOp;
      //   if (npuDmaOp.getSourceBdId()) {
      //     bdIdOp = dyn_cast_if_present<AMDAIE::BdIdOp>(
      //         npuDmaOp.getSourceBdId().getDefiningOp());
      //     if (!bdIdOp) return WalkResult::advance();
      //     if (failed(releaseBdId(bdIdOp, shimTileToGeneratorMap,
      //                            bdIdOpToBdIdsMap)))
      //       return WalkResult::interrupt();
      //   }

      //   if (npuDmaOp.getTargetBdId()) {
      //     bdIdOp = dyn_cast_if_present<AMDAIE::BdIdOp>(
      //         npuDmaOp.getTargetBdId().getDefiningOp());
      //     if (!bdIdOp) return WalkResult::advance();
      //     if (failed(releaseBdId(bdIdOp, shimTileToGeneratorMap,
      //                            bdIdOpToBdIdsMap)))
      //       return WalkResult::interrupt();
      //   }
      // }
      // return WalkResult::advance();
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
