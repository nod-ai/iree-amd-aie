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
///   4. A pointer to the parent op if the parent of is a scf.for.
struct DmaBatch {
  SmallVector<AMDAIE::NpuDmaCpyNdOp> currentDmaOps = {};
  int32_t requiredBdIds = 0;
  SmallVector<std::shared_ptr<DmaBatch>> immediateInnerBatches = {};
  std::shared_ptr<DmaBatch> nextDmaBatch = nullptr;
  scf::ForOp forOpParent = nullptr;

  /// A DmaBatch with no DmaOps and no nested DmaBatches is an empty DmaBatch.
  bool isEmpty() const {
    return (currentDmaOps.empty() && immediateInnerBatches.empty());
  }

  /// Given a DmaBatch `outerDmaBatch`, traverse each nested DmaBatch in it and
  /// infer the required Bd Ids for them. If the nested DmaBatch is surrounded
  /// with a scf.for, we account for that while sending the required Bd Ids to
  /// the caller of this API.
  int32_t getRequiredBdIdsForInnerBatches() const {
    if (immediateInnerBatches.empty()) return 0;
    int32_t requiredBdIds = 0;
    for (const std::shared_ptr<DmaBatch> &dmaBatch : immediateInnerBatches) {
      std::shared_ptr<DmaBatch> currDmaBatch = dmaBatch;
      do {
        if (currDmaBatch->isEmpty()) break;
        int32_t totalDmaOpsInCurrentBatch = currDmaBatch->currentDmaOps.size();
        int32_t requiredBdIdsForInnerBatches =
            currDmaBatch->getRequiredBdIdsForInnerBatches();
        currDmaBatch->requiredBdIds =
            requiredBdIdsForInnerBatches + totalDmaOpsInCurrentBatch;
        requiredBdIds += totalDmaOpsInCurrentBatch;
        if (currDmaBatch->forOpParent) {
          std::optional<uint32_t> numIterations =
              getNumberIterations(currDmaBatch->forOpParent);
          if (numIterations) {
            requiredBdIds +=
                requiredBdIdsForInnerBatches * numIterations.value();
          } else {
            requiredBdIds += requiredBdIdsForInnerBatches;
          }
        }
        currDmaBatch = currDmaBatch->nextDmaBatch;
      } while (currDmaBatch != nullptr);
    }
    return requiredBdIds;
  }
};

/// `TileDmaBatchGraph` is a struct that maintains a mapping from a tile to its
/// corresponding `DmaBatch`. It contains utilities that help form/update the
/// `DmaBatch` graph for a particular tile.
class TileDmaBatchGraph {
  /// The main tile->DmaBatch graph.
  llvm::MapVector<AMDAIE::TileOp, std::shared_ptr<DmaBatch>> tileDmaBatchGraph;
  /// Helps keep track of the current DmaBatch of a tile that's being processed.
  /// It holds the reference to the last DmaBatch for every tile on the main
  /// graph.
  DenseMap<AMDAIE::TileOp, std::shared_ptr<DmaBatch>> currentTileBatch;

 public:
  /// Create a new TileDmaBatchGraph by traversing over each tiles in a
  /// workgroup.
  TileDmaBatchGraph(AMDAIE::WorkgroupOp workgroupOp, Operation *parentOp) {
    workgroupOp.walk([&](AMDAIE::TileOp tile) {
      if (tileDmaBatchGraph.contains(tile)) return WalkResult::skip();
      tileDmaBatchGraph[tile] = std::make_shared<DmaBatch>();
      currentTileBatch[tile] = tileDmaBatchGraph[tile];
      currentTileBatch[tile]->forOpParent = dyn_cast<scf::ForOp>(parentOp);
      return WalkResult::advance();
    });
  }

  llvm::MapVector<AMDAIE::TileOp, std::shared_ptr<DmaBatch>> &getGraph() {
    return tileDmaBatchGraph;
  }

  void addDmaToBatch(AMDAIE::TileOp tile, AMDAIE::NpuDmaCpyNdOp dmaOp) {
    assert(tileDmaBatchGraph.contains(tile) && "Tile op not found");
    currentTileBatch[tile]->currentDmaOps.push_back(dmaOp);
  };

  void updateCurrentTileBatch(AMDAIE::TileOp tile) {
    currentTileBatch[tile]->nextDmaBatch = std::make_shared<DmaBatch>();
    currentTileBatch[tile]->nextDmaBatch->forOpParent =
        currentTileBatch[tile]->forOpParent;
    currentTileBatch[tile] = currentTileBatch[tile]->nextDmaBatch;
  };

  /// Given another `TileDmaBatchGraph` instance, we consider that as the nested
  /// graph to be added to the current graph instance. We do this by updating
  /// the `immediateInnerBatches` of the corresponding tile's DmaBatch.
  void updateInnerBatchesOfCurrentTileDmaBatchGraph(
      TileDmaBatchGraph &innerTileDmaBatchGraph) {
    for (auto &[tile, dmaBatch] : innerTileDmaBatchGraph.getGraph()) {
      if (dmaBatch->isEmpty()) continue;
      currentTileBatch[tile]->immediateInnerBatches.push_back(
          std::move(dmaBatch));
    }
  };

  /// Traverse each DmaBatch in a given (Tile -> DmaBatch) and infer Bd Ids
  /// required for it.
  void inferBdIdsRequiredInBatches() {
    for (auto &[tile, dmaBatch] : getGraph()) {
      std::shared_ptr<DmaBatch> currDmaBatch = dmaBatch;
      do {
        if (currDmaBatch->isEmpty()) break;
        int32_t totalDmaOpsInCurrentBatch = currDmaBatch->currentDmaOps.size();
        int32_t requiredBdIdsForInnerBatches =
            currDmaBatch->getRequiredBdIdsForInnerBatches();
        currDmaBatch->requiredBdIds =
            requiredBdIdsForInnerBatches + totalDmaOpsInCurrentBatch;
        currDmaBatch = currDmaBatch->nextDmaBatch;
      } while (currDmaBatch != nullptr);
    }
  }
};

/// Create a BD ID generator for every shim tile.
static void createShimTileToGeneratorMap(
    AMDAIE::WorkgroupOp workgroupOp, AMDAIEDeviceModel &deviceModel,
    DenseMap<Value, ChannelBdIdGenerator> &shimTileToGeneratorMap) {
  workgroupOp->walk([&](AMDAIE::TileOp tileOp) {
    std::optional<int64_t> col = getConstantIntValue(tileOp.getCol());
    std::optional<int64_t> row = getConstantIntValue(tileOp.getRow());
    if (col && row && deviceModel.isShimNOCTile(col.value(), row.value())) {
      ChannelBdIdGenerator generator(
          deviceModel.getChannelToValidBdIds(AMDAIETileType::SHIMNOC));
      shimTileToGeneratorMap[tileOp.getResult()] = std::move(generator);
    }
  });
}

/// A struct that maintains the channel and source/target information for a
/// given DmaOp and TileOp.
struct DmaTileData {
  uint32_t channel = -1;
  uint32_t bdIdMapIndex = -1;
};

class BdIdAssignmentUtil {
  // A mapping from a shim tile to its BD ID generator.
  DenseMap<Value, ChannelBdIdGenerator> shimTileToGeneratorMap;
  // A mapping from a BD ID Op to the BD IDs it covers.
  DenseMap<AMDAIE::BdIdOp, SmallVector<uint32_t>> bdIdOpToBdIdsMap;
  // A mapping from DMAOp to its corresponding source/target BD IDs.
  DenseMap<AMDAIE::NpuDmaCpyNdOp, SmallVector<AMDAIE::BdIdOp, 2>>
      dmaOpToBdIdMap;

  /// Given a DmaOp and a TileOp - extract whether the source (or target)
  /// operates on the tile and also extract the corresponding channel.
  FailureOr<DmaTileData> getDmaTileData(AMDAIE::NpuDmaCpyNdOp npuDmaOp,
                                        AMDAIE::TileOp tileOp) {
    FailureOr<AMDAIE::ChannelOp> maybeChannelOp;
    uint32_t bdIdMapIndex = -1;
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
        if (bdIdMapIndex == -1) {
          maybeChannelOp = npuDmaOp.getTargetChannelOp();
          bdIdMapIndex = 1;
        } else {
          return failure();
        }
      }
    }

    if (failed(maybeChannelOp)) return failure();

    DmaTileData dmaTileData = {maybeChannelOp.value().getValue(), bdIdMapIndex};
    return dmaTileData;
  }

  /// Since a DmaBatch contains the following :-
  ///   1. DmaOps within the current batch.
  ///   2. Nested DmaBatch chain.
  ///   3. Pointer to the sibling DmaBatch.
  /// This API processes a given DmaBatch by :-
  ///   a. Assigning Bd Ids to 1.
  ///   b. Assigning Bd Ids to 2.
  ///   c. Releasing Bd Ids assigned to 1.
  ///   d. Moving on to 3 and repeating steps a-to-d for it.
  LogicalResult processDmaBatch(IRRewriter &rewriter, AMDAIE::TileOp tileOp,
                                std::shared_ptr<DmaBatch> currDmaBatch) {
    OpBuilder::InsertionGuard guard(rewriter);
    do {
      if (currDmaBatch->isEmpty()) break;
      if (failed(assignRequiredBdIdsInCurrentBatch(rewriter, tileOp,
                                                   currDmaBatch))) {
        return failure();
      }

      if (failed(assignRequiredBdIdsInInnerBatch(rewriter, tileOp,
                                                 currDmaBatch))) {
        return failure();
      }

      if (failed(releaseAssignedBdIdsInDmaOps(currDmaBatch->currentDmaOps))) {
        return failure();
      }

      currDmaBatch = currDmaBatch->nextDmaBatch;
    } while (currDmaBatch != nullptr);
    return success();
  }

  /// Assign Bd Ids to the inner DmaBatch of the current DmaBatch
  /// `parentDmaBatch`. And maintain a mapping of DmaOp -> source/target Bd Id
  /// which would be used later during new DmaOp creation/replacement.
  LogicalResult assignRequiredBdIdsInInnerBatch(
      IRRewriter &rewriter, AMDAIE::TileOp tileOp,
      std::shared_ptr<DmaBatch> parentDmaBatch) {
    for (std::shared_ptr<DmaBatch> &dmaBatch :
         parentDmaBatch->immediateInnerBatches) {
      if (failed(processDmaBatch(rewriter, tileOp, dmaBatch))) return failure();
    }
    return success();
  }

  /// Assign required Bd Ids to the DmaOps of the current DmaBatch. This
  /// assignment is tracked by maintaining `dmaOpToBdIdMap`, which essentially
  /// maps a DmaOp to its source/target Bd Ids. Also, the API splits the
  /// available BD IDs equally amongst all DmaOps in the DmaBatch when assigning
  LogicalResult assignRequiredBdIdsInCurrentBatch(
      IRRewriter &rewriter, AMDAIE::TileOp tileOp,
      std::shared_ptr<DmaBatch> dmaBatch) {
    // Get the channel.
    if (dmaBatch->currentDmaOps.empty()) return success();
    FailureOr<DmaTileData> maybeDmaTileData =
        getDmaTileData(dmaBatch->currentDmaOps[0], tileOp);
    if (failed(maybeDmaTileData)) return failure();
    DmaTileData dmaTileData = *maybeDmaTileData;
    ChannelBdIdGenerator &generator =
        shimTileToGeneratorMap[tileOp.getResult()];
    uint32_t numAvailableBdIds =
        generator.getNumAvailableBdIds(dmaTileData.channel);
    uint32_t size = std::max(numAvailableBdIds / dmaBatch->requiredBdIds, 1u);
    scf::ForOp loop = nullptr;
    AffineExpr ivExpr = nullptr;
    Value iv = nullptr;
    // In case the parent of the DMA ops is a scf.for we need to keep track of
    // the loop induction variable in order to create a semi affine expression
    // later for distributing BD IDs for each iteration.
    if (loop = dmaBatch->forOpParent; loop && getNumberIterations(loop)) {
      iv = loop.getInductionVar();
      bindDims(loop.getContext(), ivExpr);
    } else {
      // In case the DMA ops are not surrounded by scf.for, we will assign
      // only one BD ID.
      size = 1;
    }

    rewriter.setInsertionPoint(dmaBatch->currentDmaOps[0]);
    for (AMDAIE::NpuDmaCpyNdOp dmaOp : dmaBatch->currentDmaOps) {
      // If DmaTileData for the current DMA Op has not been inferred, do so.
      if (dmaTileData.bdIdMapIndex == -1) {
        maybeDmaTileData = getDmaTileData(dmaOp, tileOp);
        if (failed(maybeDmaTileData)) return failure();
        dmaTileData = *maybeDmaTileData;
      }
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
          SmallVector<AMDAIE::BdIdOp, 2> bdIdOps = {nullptr, nullptr};
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
          SmallVector<AMDAIE::BdIdOp, 2> bdIdOps = {nullptr, nullptr};
          dmaOpToBdIdMap[dmaOp] = bdIdOps;
        }
        dmaOpToBdIdMap[dmaOp][dmaTileData.bdIdMapIndex] = bdIdOp;
      }
      // Reset to fetch next DmaOps' DmaTileData.
      dmaTileData.bdIdMapIndex = -1;
    }
    return success();
  }

  /// Release a given BdId op.
  LogicalResult releaseBdId(AMDAIE::BdIdOp bdIdOp) {
    auto tileOp =
        dyn_cast_if_present<AMDAIE::TileOp>(bdIdOp.getTile().getDefiningOp());
    if (!tileOp) {
      return bdIdOp.emitOpError()
             << "doesn't operate on a `amdaie.tile` operation";
    }

    if (!shimTileToGeneratorMap.contains(tileOp.getResult())) {
      return bdIdOp.emitOpError()
             << "no BD ID generator found for this BD ID op's tile";
    }

    ChannelBdIdGenerator &generator =
        shimTileToGeneratorMap[tileOp.getResult()];
    Value value = bdIdOp.getValue();
    if (auto op = value.getDefiningOp<affine::AffineApplyOp>()) {
      // If the BD ID is a semi-affine expression.
      if (bdIdOpToBdIdsMap.contains(bdIdOp)) {
        for (uint32_t bdId : bdIdOpToBdIdsMap[bdIdOp])
          generator.releaseBdId(bdId);
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
  LogicalResult releaseAssignedBdIdsInDmaOps(
      SmallVectorImpl<AMDAIE::NpuDmaCpyNdOp> &dmaOps) {
    // Release BD ID used by input DMA op.
    for (AMDAIE::NpuDmaCpyNdOp npuDmaOp : dmaOps) {
      if (AMDAIE::BdIdOp bdIdOp = dmaOpToBdIdMap[npuDmaOp][0]; bdIdOp) {
        if (failed(releaseBdId(bdIdOp))) return failure();
      }
      if (AMDAIE::BdIdOp bdIdOp = dmaOpToBdIdMap[npuDmaOp][1]; bdIdOp) {
        if (failed(releaseBdId(bdIdOp))) return failure();
      }
    }
    return success();
  }

 public:
  BdIdAssignmentUtil(
      DenseMap<Value, ChannelBdIdGenerator> &shimTileToGeneratorMap) {
    this->shimTileToGeneratorMap = shimTileToGeneratorMap;
  }

  /// Assign Bd Ids to each DmaBatch belonging to a particular Tile in the
  /// TileDmaBatch. And maintain a mapping of DmaOp -> source/target Bd Id which
  /// would be used later during new DmaOp creation/replacement.
  LogicalResult assignRequiredBdIdsInBatch(
      IRRewriter &rewriter, TileDmaBatchGraph &tileDmaBatchGraph) {
    for (auto &[tileOp, dmaBatch] : tileDmaBatchGraph.getGraph()) {
      if (failed(processDmaBatch(rewriter, tileOp, dmaBatch))) return failure();
    }
    return success();
  }

  /// Traverse each DmaOp inside ControlCode and replace it with new new DmaOp
  /// that has Bd Ids assigned using `dmaOpToBdIdMap`.
  LogicalResult replaceDmaOps(IRRewriter &rewriter,
                              AMDAIE::ControlCodeOp controlCodeOp) {
    WalkResult res = controlCodeOp->walk([&](AMDAIE::NpuDmaCpyNdOp npuDmaOp) {
      assert(dmaOpToBdIdMap.contains(npuDmaOp) && "No BD ID mapping found");
      Value sourceBdId = nullptr;
      Value targetBdId = nullptr;
      if (AMDAIE::BdIdOp bdIdOp =
              dmaOpToBdIdMap[npuDmaOp][/*sourceBdIdIndex=*/0];
          bdIdOp) {
        sourceBdId = bdIdOp.getResult();
      }
      if (AMDAIE::BdIdOp bdIdOp =
              dmaOpToBdIdMap[npuDmaOp][/*targetBdIdIndex=*/1];
          bdIdOp) {
        targetBdId = bdIdOp.getResult();
      }
      rewriter.setInsertionPoint(npuDmaOp);
      rewriter.replaceOpWithNewOp<AMDAIE::NpuDmaCpyNdOp>(
          npuDmaOp, npuDmaOp.getResultTypes(), npuDmaOp.getConnection(),
          npuDmaOp.getTarget(), npuDmaOp.getTargetMixedOffsets(),
          npuDmaOp.getTargetMixedSizes(), npuDmaOp.getTargetMixedStrides(),
          targetBdId, npuDmaOp.getSource(), npuDmaOp.getSourceMixedOffsets(),
          npuDmaOp.getSourceMixedSizes(), npuDmaOp.getSourceMixedStrides(),
          sourceBdId);
      return WalkResult::advance();
    });
    if (res.wasInterrupted()) return failure();
    return success();
  }
};

/// The function `createTileDmaBatchGraph` returns a `TileDmaBatchGraph` to the
/// caller.
/// Each invocation of the following function helps create a graph of DmaBatch
/// as follows :-
/// 1. Creates a new TileDmaBatchGraph.
/// 2. For each op in the parent op:
///     a. If the op is a DMA Op, we get the tiles it operates on and add it to
///        the current tileDmaBatchGraph formed in step 1 (and updates
///        `currDmaOp` for tracking the current DmaBatch).
///     b. If the op is a DMA Wait Op, that means the current DmaBatch can end
///        and we therefore start a new DmaBatch.
///     c. If the op is scf.for/forall, we recursively invoke the same function
///        with this op as the parent op. This basically means we are getting
///        into a nested structure. And since this API returns a
///        `TileDmaBatchGraph`, the returned structure is essentially the nested
///        `TileDmaBatchGraph`. Hence we need to update the current
///        `TileDmaBatchGraph` formed in step 1 to have this returned structure
///        as its nested/subgraph.
static TileDmaBatchGraph createTileDmaBatchGraph(
    AMDAIE::WorkgroupOp workgroupOp, Operation *parentOp,
    DenseMap<Value, ChannelBdIdGenerator> &shimTileToGeneratorMap) {
  TileDmaBatchGraph tileDmaBatchGraph =
      TileDmaBatchGraph(workgroupOp, parentOp);

  AMDAIE::NpuDmaCpyNdOp currDmaOp = nullptr;
  // Traverse the parent operation's immediate child ops and form the
  // tileParentOpDmaBatchMap.
  for (Operation &op : parentOp->getRegion(0).getOps()) {
    TypeSwitch<Operation *, void>(&op)
        .Case<scf::ForOp, scf::ForallOp>([&](auto forOp) {
          TileDmaBatchGraph innerTileDmaBatchGraph =
              createTileDmaBatchGraph(workgroupOp, &op, shimTileToGeneratorMap);
          tileDmaBatchGraph.updateInnerBatchesOfCurrentTileDmaBatchGraph(
              innerTileDmaBatchGraph);
        })
        .Case<AMDAIE::NpuDmaCpyNdOp>([&](auto dmaOp) {
          if (dmaOp.getSource()) {
            FailureOr<AMDAIE::TileOp> tile =
                getGeneratorTileOp<CopyOpOperateOn::Source>(
                    dmaOp, shimTileToGeneratorMap);
            if (succeeded(tile)) tileDmaBatchGraph.addDmaToBatch(*tile, dmaOp);
          }
          if (dmaOp.getTarget()) {
            FailureOr<AMDAIE::TileOp> tile =
                getGeneratorTileOp<CopyOpOperateOn::Target>(
                    dmaOp, shimTileToGeneratorMap);
            if (succeeded(tile)) tileDmaBatchGraph.addDmaToBatch(*tile, dmaOp);
          }
          if (!currDmaOp) currDmaOp = dmaOp;
        })
        .Case<AMDAIE::NpuDmaWaitOp>([&](auto npuWaitOp) {
          for (AMDAIE::NpuDmaCpyNdOp npuDmaOp : npuWaitOp.getDmaOps()) {
            // Reached the DMA wait operation, reset tracking of current DMA op
            // for the tile.
            if (npuDmaOp == currDmaOp) {
              currDmaOp = nullptr;
              if (npuDmaOp.getSource()) {
                FailureOr<AMDAIE::TileOp> tile =
                    getGeneratorTileOp<CopyOpOperateOn::Source>(
                        npuDmaOp, shimTileToGeneratorMap);
                if (succeeded(tile))
                  tileDmaBatchGraph.updateCurrentTileBatch(*tile);
              }
              if (npuDmaOp.getTarget()) {
                FailureOr<AMDAIE::TileOp> tile =
                    getGeneratorTileOp<CopyOpOperateOn::Target>(
                        npuDmaOp, shimTileToGeneratorMap);
                if (succeeded(tile))
                  tileDmaBatchGraph.updateCurrentTileBatch(*tile);
              }
            }
          }
        });
  }
  return tileDmaBatchGraph;
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

  DenseMap<Value, ChannelBdIdGenerator> shimTileToGeneratorMap;
  createShimTileToGeneratorMap(workgroupOp, deviceModel,
                               shimTileToGeneratorMap);
  BdIdAssignmentUtil bdIdAssignmentUtil(shimTileToGeneratorMap);

  // Walk `amdaie.npu_dma_cpy_nd` and  `amdaie.dma_wait` operations and assign
  // and release BD IDs when encountering the respective operations using the
  // tile BD ID generators initialized earlier.
  AMDAIE::ControlCodeOp controlCodeOp = workgroupOp.getControlCode();
  // Since a DMA op can have source and target, therefore we can have two BD IDs
  // for any DMA op. Hence we maintain a map from DMA op to a vector of BD IDs.
  DenseMap<AMDAIE::NpuDmaCpyNdOp, SmallVector<AMDAIE::BdIdOp>> dmaOpToBdIdMap;
  TileDmaBatchGraph tileDmaBatchGraph = createTileDmaBatchGraph(
      workgroupOp, controlCodeOp, shimTileToGeneratorMap);
  tileDmaBatchGraph.inferBdIdsRequiredInBatches();
  if (failed(bdIdAssignmentUtil.assignRequiredBdIdsInBatch(
          rewriter, tileDmaBatchGraph))) {
    return failure();
  }
  if (failed(bdIdAssignmentUtil.replaceDmaOps(rewriter, controlCodeOp)))
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
