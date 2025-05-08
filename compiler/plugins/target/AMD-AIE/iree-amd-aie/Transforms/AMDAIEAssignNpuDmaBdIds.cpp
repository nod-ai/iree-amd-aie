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

/// A DmaBatch contains the following :-
///   1. DmaOps within the current batch.
///   2. Required Bd Ids by the current batch.
///   3. Nested DmaBatch chain.
///   4. Pointer to the sibling DmaBatch.
typedef struct DmaBatchStruct {
  SmallVector<AMDAIE::NpuDmaCpyNdOp> currentDmaOps = {};
  int32_t requiredBdIds = 0;
  SmallVector<DmaBatchStruct *> immediateInnerBatches = {};
  DmaBatchStruct *nextDmaBatch = nullptr;
} DmaBatch;

using TileDmaBatchGraph = llvm::MapVector<AMDAIE::TileOp, DmaBatch *>;

/// Create a new TileDmaBatchGraph by traversing over each tiles in a workgroup.
static TileDmaBatchGraph initTileDmaBatchGraph(
    AMDAIE::WorkgroupOp workgroupOp) {
  TileDmaBatchGraph tileDmaBatchGraph;
  workgroupOp.walk([&](AMDAIE::TileOp tile) {
    if (tileDmaBatchGraph.contains(tile)) return WalkResult::skip();
    tileDmaBatchGraph[tile] = new DmaBatch();
    return WalkResult::advance();
  });
  return tileDmaBatchGraph;
}

/// A DmaBatch with no DmaOps and no nested DmaBatches is an empty DmaBatch.
static bool isEmptyDmaBatch(DmaBatch *dmaBatch) {
  return (dmaBatch->currentDmaOps.empty() &&
          dmaBatch->immediateInnerBatches.empty());
}

static void updateInnerBatchesOfCurrentTileDmaBatchGraph(
    DenseMap<AMDAIE::TileOp, DmaBatch *> &currentTileBatch,
    TileDmaBatchGraph &innerTileDmaBatchGraph) {
  for (auto [tile, dmaBatch] : innerTileDmaBatchGraph) {
    if (isEmptyDmaBatch(dmaBatch)) continue;
    currentTileBatch[tile]->immediateInnerBatches.push_back(dmaBatch);
  }
}

static TileDmaBatchGraph formTileDmaBatchGraph(
    AMDAIE::WorkgroupOp &workgroupOp, Operation *parentOp,
    DenseMap<Value, ChannelBdIdGenerator> &shimTileToGeneratorMap) {
  TileDmaBatchGraph tileDmaBatchGraph = initTileDmaBatchGraph(workgroupOp);

  DenseMap<AMDAIE::TileOp, DmaBatch *> currentTileBatch;
  for (auto [tile, dmaBatch] : tileDmaBatchGraph) {
    currentTileBatch[tile] = dmaBatch;
  }
  auto addDmaToBatch = [&](AMDAIE::TileOp tile, AMDAIE::NpuDmaCpyNdOp dmaOp) {
    assert(tileDmaBatchGraph.contains(tile) && "Tile op not found");
    currentTileBatch[tile]->currentDmaOps.push_back(dmaOp);
  };

  auto updateCurrentTileBatch = [&](AMDAIE::TileOp tile) {
    currentTileBatch[tile]->nextDmaBatch = new DmaBatch();
    currentTileBatch[tile] = currentTileBatch[tile]->nextDmaBatch;
  };

  AMDAIE::NpuDmaCpyNdOp currDmaOp = nullptr;
  // Traverse the parent operation's immediate child ops and form the
  // tileParentOpDmaBatchMap.
  for (Operation &op : parentOp->getRegion(0).getOps()) {
    if (isa<scf::ForOp, scf::ForallOp>(op)) {
      TileDmaBatchGraph innerTileDmaBatchGraph =
          formTileDmaBatchGraph(workgroupOp, &op, shimTileToGeneratorMap);
      updateInnerBatchesOfCurrentTileDmaBatchGraph(currentTileBatch,
                                                   innerTileDmaBatchGraph);
    } else if (auto dmaOp = dyn_cast<AMDAIE::NpuDmaCpyNdOp>(op)) {
      if (dmaOp.getSource()) {
        FailureOr<AMDAIE::TileOp> tile =
            getGeneratorTileOp<CopyOpOperateOn::Source>(dmaOp,
                                                        shimTileToGeneratorMap);
        if (succeeded(tile)) addDmaToBatch(*tile, dmaOp);
      }
      if (dmaOp.getTarget()) {
        FailureOr<AMDAIE::TileOp> tile =
            getGeneratorTileOp<CopyOpOperateOn::Target>(dmaOp,
                                                        shimTileToGeneratorMap);
        if (succeeded(tile)) addDmaToBatch(*tile, dmaOp);
      }
      if (!currDmaOp) currDmaOp = dmaOp;
    } else if (auto npuWaitOp = dyn_cast<AMDAIE::NpuDmaWaitOp>(op)) {
      for (AMDAIE::NpuDmaCpyNdOp npuDmaOp : npuWaitOp.getDmaOps()) {
        // Reached the DMA wait operation, reset tracking of current DMA op for
        // the tile.
        if (npuDmaOp == currDmaOp) {
          currDmaOp = nullptr;
          if (npuDmaOp.getSource()) {
            FailureOr<AMDAIE::TileOp> tile =
                getGeneratorTileOp<CopyOpOperateOn::Source>(
                    npuDmaOp, shimTileToGeneratorMap);
            if (succeeded(tile)) updateCurrentTileBatch(*tile);
          }
          if (npuDmaOp.getTarget()) {
            FailureOr<AMDAIE::TileOp> tile =
                getGeneratorTileOp<CopyOpOperateOn::Target>(
                    npuDmaOp, shimTileToGeneratorMap);
            if (succeeded(tile)) updateCurrentTileBatch(*tile);
          }
        }
      }
    }
  }
  return tileDmaBatchGraph;
}

/// Given a non-empty DmaBatch find the immediate surrounding parent op.
static Operation *findParentOpOfBatch(DmaBatch *dmaBatch) {
  if (!dmaBatch->currentDmaOps.empty())
    return dmaBatch->currentDmaOps[0]->getParentOp();
  assert(!dmaBatch->immediateInnerBatches.empty() &&
         "DmaBatch found with no DmaOps and no immediate inner batches");
  Operation *parentOp = findParentOpOfBatch(dmaBatch->immediateInnerBatches[0]);
  assert(parentOp && "Found inner DmaBatch with no parent");
  return parentOp->getParentOp();
}

/// Given a DmaBatch `outerDmaBatch`, traverse each nested DmaBatch in it and
/// infer the required Bd Ids for them. If the nsted DmaBatch is surrounded with
/// a scf.for, we account for that while sending the required Bd Ids to the
/// caller of this API.
static int32_t getRequiredBdIdsForInnerBatches(DmaBatch *outerDmaBatch) {
  if (outerDmaBatch->immediateInnerBatches.empty()) return 0;
  int32_t requiredBdIds = 0;
  for (DmaBatch *dmaBatch : outerDmaBatch->immediateInnerBatches) {
    DmaBatch *currDmaBatch = dmaBatch;
    do {
      if (isEmptyDmaBatch(currDmaBatch)) break;
      int32_t totalDmaOpsInCurrentBatch = currDmaBatch->currentDmaOps.size();
      int32_t requiredBdIdsForInnerBatches =
          getRequiredBdIdsForInnerBatches(currDmaBatch);
      currDmaBatch->requiredBdIds =
          requiredBdIdsForInnerBatches + totalDmaOpsInCurrentBatch;
      requiredBdIds += totalDmaOpsInCurrentBatch;
      Operation *parentOp = findParentOpOfBatch(currDmaBatch);
      if (auto forOp = dyn_cast_if_present<scf::ForOp>(parentOp)) {
        std::optional<uint32_t> numIterations = getNumberIterations(forOp);
        if (numIterations) {
          requiredBdIds += requiredBdIdsForInnerBatches * numIterations.value();
        } else {
          requiredBdIds += requiredBdIdsForInnerBatches;
        }
      }
      currDmaBatch = currDmaBatch->nextDmaBatch;
    } while (currDmaBatch != nullptr);
  }
  return requiredBdIds;
}

/// Traverse each DmaBatch in a given (Tile -> DmaBatch) and infer Bd Ids
/// required for it.
static void inferBdIdsRequiredInBatches(TileDmaBatchGraph &tileDmaBatchGraph) {
  for (auto [tile, dmaBatch] : tileDmaBatchGraph) {
    DmaBatch *currDmaBatch = dmaBatch;
    do {
      if (isEmptyDmaBatch(currDmaBatch)) break;
      int32_t totalDmaOpsInCurrentBatch = currDmaBatch->currentDmaOps.size();
      int32_t requiredBdIdsForInnerBatches =
          getRequiredBdIdsForInnerBatches(currDmaBatch);
      currDmaBatch->requiredBdIds =
          requiredBdIdsForInnerBatches + totalDmaOpsInCurrentBatch;
      currDmaBatch = currDmaBatch->nextDmaBatch;
    } while (currDmaBatch != nullptr);
  }
}

/// A struct that maintains the channel and source/target information for a
/// given DmaOp and TileOp.
typedef struct DmaTileDataStruct {
  uint32_t channel;
  uint32_t bdIdMapIndex;
} DmaTileData;

/// Given a DmaOp and a TileOp - extract whether the source (or target) operates
/// on the tile and also extract the corresponding channel.
static FailureOr<DmaTileData> getDmaTileData(
    AMDAIE::NpuDmaCpyNdOp npuDmaOp, AMDAIE::TileOp tileOp,
    DenseMap<Value, ChannelBdIdGenerator> &shimTileToGeneratorMap) {
  FailureOr<AMDAIE::ChannelOp> maybeChannelOp;
  uint32_t bdIdMapIndex;
  if (npuDmaOp.getSource()) {
    FailureOr<AMDAIE::TileOp> tile =
        getGeneratorTileOp<CopyOpOperateOn::Source>(npuDmaOp,
                                                    shimTileToGeneratorMap);
    if (succeeded(tile) && *tile == tileOp) {
      maybeChannelOp = npuDmaOp.getSourceChannelOp();
      bdIdMapIndex = 0;
    }
  }
  if (npuDmaOp.getTarget()) {
    FailureOr<AMDAIE::TileOp> tile =
        getGeneratorTileOp<CopyOpOperateOn::Target>(npuDmaOp,
                                                    shimTileToGeneratorMap);
    if (succeeded(tile) && *tile == tileOp) {
      maybeChannelOp = npuDmaOp.getTargetChannelOp();
      bdIdMapIndex = 1;
    }
  }

  if (failed(maybeChannelOp)) return failure();

  DmaTileData dmaTileData = {maybeChannelOp.value().getValue(), bdIdMapIndex};
  return dmaTileData;
}

/// Assign required Bd Ids to the DmaOps of the current DmaBatch. This
/// assignment is tracked by maintaining `dmaOpToBdIdMap`, which essentially
/// maps a DmaOp to its source/target Bd Ids. Also, the API splits the available
/// BD IDs equally amongst all DmaOps in the DmaBatch when assigning
static LogicalResult assignRequiredBdIdsInCurrentBatch(
    IRRewriter &rewriter, AMDAIE::TileOp tileOp, DmaBatch *dmaBatch,
    DenseMap<Value, ChannelBdIdGenerator> &shimTileToGeneratorMap,
    DenseMap<AMDAIE::BdIdOp, SmallVector<uint32_t>> &bdIdOpToBdIdsMap,
    DenseMap<AMDAIE::NpuDmaCpyNdOp, SmallVector<AMDAIE::BdIdOp>>
        &dmaOpToBdIdMap) {
  // Get the channel.
  if (dmaBatch->currentDmaOps.empty()) return success();
  FailureOr<DmaTileData> maybeDmaTileData = getDmaTileData(
      dmaBatch->currentDmaOps[0], tileOp, shimTileToGeneratorMap);
  if (failed(maybeDmaTileData)) return failure();
  DmaTileData dmaTileData = *maybeDmaTileData;
  ChannelBdIdGenerator &generator = shimTileToGeneratorMap[tileOp.getResult()];
  uint32_t numAvailableBdIds =
      generator.getNumAvailableBdIds(dmaTileData.channel);
  uint32_t size = std::max(numAvailableBdIds / dmaBatch->requiredBdIds, 1u);
  scf::ForOp loop = nullptr;
  AffineExpr ivExpr = nullptr;
  Value iv = nullptr;
  // In case the parent of the DMA ops is a scf.for we need to keep track of
  // the loop induction variable in order to create a semi affine expression
  // later for distributing BD IDs for each iteration.
  if (loop = dmaBatch->currentDmaOps[0]->getParentOfType<scf::ForOp>();
      loop && getNumberIterations(loop)) {
    iv = loop.getInductionVar();
    bindDims(loop.getContext(), ivExpr);
  } else {
    // In case the DMA ops are not surrounded by scf.for, we will assign
    // only one BD ID.
    size = 1;
  }

  rewriter.setInsertionPoint(dmaBatch->currentDmaOps[0]);
  for (AMDAIE::NpuDmaCpyNdOp dmaOp : dmaBatch->currentDmaOps) {
    maybeDmaTileData = getDmaTileData(dmaOp, tileOp, shimTileToGeneratorMap);
    if (failed(maybeDmaTileData)) return failure();
    dmaTileData = *maybeDmaTileData;
    // Only create expression if more than 1 BD ID is needed and if,
    // otherwise, fall back to constant BD ID.
    if (size > 1) {
      // Assigning BD IDs for all iterations in the loop.
      SmallVector<uint32_t> bdIds;
      for (uint32_t i = 0; i < size; i++) {
        std::optional<uint32_t> bdId = generator.getAndAssignBdId(
            dmaTileData.channel, BdIdAssignmentMode::Incremental);
        if (!bdId) return failure();
        bdIds.push_back(bdId.value());
      }
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

      dmaOpToBdIdMap[dmaOp][dmaTileData.bdIdMapIndex] = bdIdOp;
    } else {
      // Assign a constant BD ID.
      std::optional<uint32_t> bdId = generator.getAndAssignBdId(
          dmaTileData.channel, BdIdAssignmentMode::Incremental);
      if (!bdId) return dmaOp.emitOpError() << "no BD ID available";
      auto constant = rewriter.create<arith::ConstantOp>(
          rewriter.getUnknownLoc(), rewriter.getIndexAttr(bdId.value()));
      AMDAIE::BdIdOp bdIdOp = rewriter.create<AMDAIE::BdIdOp>(
          rewriter.getUnknownLoc(), tileOp, constant.getResult());
      if (!dmaOpToBdIdMap.contains(dmaOp)) {
        SmallVector<AMDAIE::BdIdOp> bdIdOps = {nullptr, nullptr};
        dmaOpToBdIdMap[dmaOp] = bdIdOps;
      }
      dmaOpToBdIdMap[dmaOp][dmaTileData.bdIdMapIndex] = bdIdOp;
    }
  }
  return success();
}

/// Release a given BdId op.
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

/// DmaOps are assigned Bd Ids prior to invoking this function and a map
/// `dmaOpToBdIdMap` is maintained that maps a DmaOp to its source/target Bd
/// Ids. For each DmaOp in the list `dmaOps`, this API will check the
/// `dmaOpToBdIdMap` and release Bd Ids if assigned.
static LogicalResult releaseAssignedBdIdsInCurrentBatch(
    SmallVector<AMDAIE::NpuDmaCpyNdOp> dmaOps,
    DenseMap<Value, ChannelBdIdGenerator> &shimTileToGeneratorMap,
    DenseMap<AMDAIE::BdIdOp, SmallVector<uint32_t>> &bdIdOpToBdIdsMap,
    DenseMap<AMDAIE::NpuDmaCpyNdOp, SmallVector<AMDAIE::BdIdOp>>
        &dmaOpToBdIdMap) {
  // Release BD ID used by input DMA op.
  for (AMDAIE::NpuDmaCpyNdOp npuDmaOp : dmaOps) {
    if (dmaOpToBdIdMap[npuDmaOp][0]) {
      if (failed(releaseBdId(dmaOpToBdIdMap[npuDmaOp][0],
                             shimTileToGeneratorMap, bdIdOpToBdIdsMap)))
        return failure();
    }

    if (dmaOpToBdIdMap[npuDmaOp][1]) {
      if (failed(releaseBdId(dmaOpToBdIdMap[npuDmaOp][1],
                             shimTileToGeneratorMap, bdIdOpToBdIdsMap)))
        return failure();
    }
  }
  return success();
}

/// Declaration of an API which will work on assigning Bd Ids to the inner
/// DmaBatch of the current DmaBatch `parentDmaBatch`. And maintain a mapping of
/// DmaOp -> source/target Bd Id which would be used later during new DmaOp
/// creation/replacement.
static LogicalResult assignRequiredBdIdsInInnerBatch(
    IRRewriter &rewriter, AMDAIE::TileOp tileOp, DmaBatch *parentDmaBatch,
    DenseMap<Value, ChannelBdIdGenerator> &shimTileToGeneratorMap,
    DenseMap<AMDAIE::BdIdOp, SmallVector<uint32_t>> &bdIdOpToBdIdsMap,
    DenseMap<AMDAIE::NpuDmaCpyNdOp, SmallVector<AMDAIE::BdIdOp>>
        &dmaOpToBdIdMap);

/// Since a DmaBatch contains the following :-
///   1. DmaOps within the current batch.
///   2. Nested DmaBatch chain.
///   3. Pointer to the sibling DmaBatch.
/// This API processes a given DmaBatch by :-
///   a. Assigning Bd Ids to 1.
///   b. Assigning Bd Ids to 2.
///   c. Releasing Bd Ids assigned to 1.
///   d. Moving on to 3 and repeating steps a-to-d for it.
static LogicalResult processDmaBatch(
    IRRewriter &rewriter, AMDAIE::TileOp tileOp, DmaBatch *currDmaBatch,
    DenseMap<Value, ChannelBdIdGenerator> &shimTileToGeneratorMap,
    DenseMap<AMDAIE::BdIdOp, SmallVector<uint32_t>> &bdIdOpToBdIdsMap,
    DenseMap<AMDAIE::NpuDmaCpyNdOp, SmallVector<AMDAIE::BdIdOp>>
        &dmaOpToBdIdMap) {
  do {
    if (isEmptyDmaBatch(currDmaBatch)) break;
    if (failed(assignRequiredBdIdsInCurrentBatch(
            rewriter, tileOp, currDmaBatch, shimTileToGeneratorMap,
            bdIdOpToBdIdsMap, dmaOpToBdIdMap)))
      return failure();

    if (failed(assignRequiredBdIdsInInnerBatch(
            rewriter, tileOp, currDmaBatch, shimTileToGeneratorMap,
            bdIdOpToBdIdsMap, dmaOpToBdIdMap)))
      return failure();

    if (failed(releaseAssignedBdIdsInCurrentBatch(
            currDmaBatch->currentDmaOps, shimTileToGeneratorMap,
            bdIdOpToBdIdsMap, dmaOpToBdIdMap)))
      return failure();

    currDmaBatch = currDmaBatch->nextDmaBatch;
  } while (currDmaBatch != nullptr);
  return success();
}

static LogicalResult assignRequiredBdIdsInInnerBatch(
    IRRewriter &rewriter, AMDAIE::TileOp tileOp, DmaBatch *parentDmaBatch,
    DenseMap<Value, ChannelBdIdGenerator> &shimTileToGeneratorMap,
    DenseMap<AMDAIE::BdIdOp, SmallVector<uint32_t>> &bdIdOpToBdIdsMap,
    DenseMap<AMDAIE::NpuDmaCpyNdOp, SmallVector<AMDAIE::BdIdOp>>
        &dmaOpToBdIdMap) {
  for (DmaBatch *dmaBatch : parentDmaBatch->immediateInnerBatches) {
    if (failed(processDmaBatch(rewriter, tileOp, dmaBatch,
                               shimTileToGeneratorMap, bdIdOpToBdIdsMap,
                               dmaOpToBdIdMap)))
      return failure();
  }
  return success();
}

/// This API will work on assigning Bd Ids to each DmaBatch belonging to a
/// particular Tile. And maintain a mapping of DmaOp -> source/target Bd Id
/// which would be used later during new DmaOp creation/replacement.
static LogicalResult assignRequiredBdIdsInBatch(
    IRRewriter &rewriter, TileDmaBatchGraph &tileDmaBatchGraph,
    DenseMap<Value, ChannelBdIdGenerator> &shimTileToGeneratorMap,
    DenseMap<AMDAIE::BdIdOp, SmallVector<uint32_t>> &bdIdOpToBdIdsMap,
    DenseMap<AMDAIE::NpuDmaCpyNdOp, SmallVector<AMDAIE::BdIdOp>>
        &dmaOpToBdIdMap) {
  for (auto [tileOp, dmaBatch] : tileDmaBatchGraph) {
    if (failed(processDmaBatch(rewriter, tileOp, dmaBatch,
                               shimTileToGeneratorMap, bdIdOpToBdIdsMap,
                               dmaOpToBdIdMap)))
      return failure();
  }
  return success();
}

/// Traverse each DmaOp inside ControlCode and replace it with new new DmaOp
/// that has Bd Ids assigned using `dmaOpToBdIdMap`.
static LogicalResult createNewDmaOpsAndReplaceOldDmaOps(
    IRRewriter &rewriter, AMDAIE::ControlCodeOp controlCodeOp,
    DenseMap<AMDAIE::NpuDmaCpyNdOp, SmallVector<AMDAIE::BdIdOp>>
        &dmaOpToBdIdMap) {
  WalkResult res = controlCodeOp->walk([&](AMDAIE::NpuDmaCpyNdOp npuDmaOp) {
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
  });
  if (res.wasInterrupted()) return failure();
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

  DenseMap<AMDAIE::BdIdOp, SmallVector<uint32_t>> bdIdOpToBdIdsMap;
  // Walk `amdaie.npu_dma_cpy_nd` and  `amdaie.dma_wait` operations and assign
  // and release BD IDs when encountering the respective operations using the
  // tile BD ID generators initialized earlier.
  AMDAIE::ControlCodeOp controlCodeOp = workgroupOp.getControlCode();
  // Since a DMA op can have source and target, therefore we can have two BD IDs
  // for any DMA op. Hence we maintain a map from DMA op to a vector of BD IDs.
  DenseMap<AMDAIE::NpuDmaCpyNdOp, SmallVector<AMDAIE::BdIdOp>> dmaOpToBdIdMap;
  TileDmaBatchGraph tileDmaBatchGraph =
      formTileDmaBatchGraph(workgroupOp, controlCodeOp, shimTileToGeneratorMap);
  inferBdIdsRequiredInBatches(tileDmaBatchGraph);
  if (failed(assignRequiredBdIdsInBatch(rewriter, tileDmaBatchGraph,
                                        shimTileToGeneratorMap,
                                        bdIdOpToBdIdsMap, dmaOpToBdIdMap)))
    return failure();
  if (failed(createNewDmaOpsAndReplaceOldDmaOps(rewriter, controlCodeOp,
                                                dmaOpToBdIdMap)))
    return failure();
  // At this step we have all the information to traverse and perform the
  // replacements of the DMA Ops.

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
