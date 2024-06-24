// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-assign-tile-locations"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Comparator for a pair of `amdaie.dma_cpy_nd` on the first tile operation's
/// column index.
bool dmaColComparator(
    std::pair<AMDAIE::DmaCpyNdOp, SmallVector<AMDAIE::TileOp>> &a,
    std::pair<AMDAIE::DmaCpyNdOp, SmallVector<AMDAIE::TileOp>> &b) {
  return TileOp::tileColumnComparator(a.second[0], b.second[0]);
};

/// Return the tiles of the sources respectively targets of the users of this
/// logical objectfifo, depending on whether the OperateOn template parameter is
/// set to `OperateOn::Source` respectively `OperateOn::Target`.
template <CopyOpOperateOn OperateOn>
LogicalResult getUserTiles(
    AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo,
    SmallVectorImpl<AMDAIE::TileOp> &tiles) {
  llvm::SmallSetVector<AMDAIE::TileOp, 16> tileSet;
  for (Operation *user : logicalObjectFifo->getUsers()) {
    if (auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(user)) {
      ValueRange tileIndices;
      if constexpr (OperateOn == CopyOpOperateOn::Source) {
        if (dmaOp.getTargetObjectFifo() != logicalObjectFifo) continue;
        tileIndices = dmaOp.getSourceObjectFifo().getTiles();
      } else if constexpr (OperateOn == CopyOpOperateOn::Target) {
        if (dmaOp.getSourceObjectFifo() != logicalObjectFifo) continue;
        tileIndices = dmaOp.getTargetObjectFifo().getTiles();
      }

      // Only fill in tiles when all sources have tiles.
      if (tileIndices.empty()) return failure();
      for (Value index : tileIndices)
        tileSet.insert(dyn_cast<AMDAIE::TileOp>(index.getDefiningOp()));
    }
  }
  tiles = tileSet.takeVector();
  return success();
}

/// Utility to recursively find users of the provided logical objectFifo inside
/// `amdaie.core` operations and return the tile coordinates.
LogicalResult findUsersInCoreAndAddTiles(
    Operation *op, AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo,
    llvm::SmallSetVector<std::pair<int64_t, int64_t>, 16> &tiles) {
  for (Operation *userOp : op->getUsers()) {
    if (auto coreOp = userOp->getParentOfType<AMDAIE::CoreOp>()) {
      AMDAIE::TileOp tileOp = coreOp.getTileOp();
      std::optional<int64_t> column = getConstantIntValue(tileOp.getCol());
      std::optional<int64_t> row = getConstantIntValue(tileOp.getRow());
      if (!column || !row) {
        return coreOp.emitOpError() << "has non-constant tile location";
      }
      tiles.insert(std::make_pair(column.value(), row.value()));
    }
    if (auto subviewOp = dyn_cast<memref::SubViewOp>(userOp)) {
      return findUsersInCoreAndAddTiles(subviewOp, logicalObjectFifo, tiles);
    } else if (auto userLogicalObjectFifo =
                   dyn_cast<AMDAIE::LogicalObjectFifoFromMemrefOp>(userOp)) {
      return findUsersInCoreAndAddTiles(userLogicalObjectFifo,
                                        logicalObjectFifo, tiles);
    }
  }
  return success();
}

/// Assign tiles to the logical objectfifos with local memory space (L1).
/// The tiles are derived from the usage of the logical objectfifos within
/// core operations, which are already assigned a tile location.
LogicalResult assignLocalAieTiles(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());

  WalkResult res = moduleOp->walk(
      [&](AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo) {
        Attribute memSpace = logicalObjectFifo.getMemorySpace();
        if (!memSpace || dyn_cast<IntegerAttr>(memSpace).getInt() != 2)
          return WalkResult::advance();

        llvm::SmallSetVector<std::pair<int64_t, int64_t>, 16> tileLocations;
        if (failed(findUsersInCoreAndAddTiles(
                logicalObjectFifo, logicalObjectFifo, tileLocations))) {
          return WalkResult::interrupt();
        }
        // Handle subviews.
        for (Operation *userOp :
             logicalObjectFifo.getMemref().getDefiningOp()->getUsers()) {
          if (auto subviewOp = dyn_cast<memref::SubViewOp>(userOp)) {
            if (failed(findUsersInCoreAndAddTiles(subviewOp, logicalObjectFifo,
                                                  tileLocations))) {
              return WalkResult::interrupt();
            }
          }
        }

        SmallVector<Value> tiles;
        tiles.reserve(tileLocations.size());
        rewriter.setInsertionPoint(logicalObjectFifo);
        for (auto [column, row] : tileLocations) {
          auto colIndex = rewriter.create<arith::ConstantIndexOp>(
              rewriter.getUnknownLoc(), column);
          auto rowIndex = rewriter.create<arith::ConstantIndexOp>(
              rewriter.getUnknownLoc(), row);
          auto tileOp = rewriter.create<AMDAIE::TileOp>(
              rewriter.getUnknownLoc(), colIndex, rowIndex);
          tiles.push_back(tileOp.getResult());
        }
        // Sort for deterministic output IR.
        llvm::sort(tiles.begin(), tiles.end(),
                   AMDAIE::TileOp::tileValueColumnAndRowComparator);
        rewriter.replaceOpWithNewOp<AMDAIE::LogicalObjectFifoFromMemrefOp>(
            logicalObjectFifo,
            cast<LogicalObjectFifoType>(
                logicalObjectFifo.getOutput().getType()),
            logicalObjectFifo.getMemref(), tiles);
        return WalkResult::advance();
      });
  if (res.wasInterrupted()) return failure();
  return success();
}

/// Assign a set of potential physical AIE tiles to logical objectFifos. This
/// rewrite takes an iterative approach by matching logical objectfifos and only
/// assigning tiles when linked through dma ops with other logical objectfifos
/// which already have tiles assigned. If the linked logical objectfifos don't
/// have tiles assigned yet, we will return a failure and give the linked
/// logical objectfifos a chance to assign tiles before returning to this one.
///
/// TODO(jornt): There are decisions being made in this pass on which tiles to
/// assign to a logical objectfifo. This logic is very simple for now and tries
/// to use the tiles in the same columns as targets and sources. At some point,
/// we probably need some AIE device model to guide the assignement here for
/// performance and to avoid hardware resource issues later on.
class FillAieTiles
    : public OpRewritePattern<AMDAIE::LogicalObjectFifoFromMemrefOp> {
  using OpRewritePattern<
      AMDAIE::LogicalObjectFifoFromMemrefOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo,
      PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "FillAieTiles: " << logicalObjectFifo << "\n");
    if (!logicalObjectFifo.getTiles().empty()) {
      return failure();
    }

    Attribute memSpace = logicalObjectFifo.getMemorySpace();
    // Skip logical objectfifos within local memory as they should already be
    // assigned.
    if (memSpace && dyn_cast<IntegerAttr>(memSpace).getInt() == 2) {
      if (logicalObjectFifo.getTiles().empty()) {
        logicalObjectFifo.emitOpError()
            << "found logical objectfifo on local memory space with no tiles "
               "assigned.";
      }
      return failure();
    }
    // HandLe both L3/shim and L2/Memtiles.
    // Skip logical objectfifos within non-global and non-shared memory.
    if (memSpace && dyn_cast<IntegerAttr>(memSpace).getInt() != 1) {
      return logicalObjectFifo.emitOpError()
             << "found logical objectfifo with unknown memory space";
    }

    SmallVector<AMDAIE::TileOp, 16> targetTiles;
    SmallVector<AMDAIE::TileOp, 16> sourceTiles;
    LogicalResult dstRes =
        getUserTiles<CopyOpOperateOn::Target>(logicalObjectFifo, targetTiles);
    LogicalResult srcRes =
        getUserTiles<CopyOpOperateOn::Source>(logicalObjectFifo, sourceTiles);

    // If no source and target tiles found, skip.
    if (failed(dstRes) && failed(srcRes)) {
      return failure();
    }

    // TODO(jornt): avoid row hardcoding. Will need to update the mlir-aie
    // target model for this.
    int64_t rowInt = memSpace ? 1 : 0;
    llvm::SmallSetVector<std::pair<int64_t, int64_t>, 16> tileLocations;
    auto createTileLocations =
        [&](SmallVector<AMDAIE::TileOp, 16> &tiles) -> LogicalResult {
      // TODO(jornt): For now, for deterministic behaviour, sort on column
      // index and use first one. This needs to be generalized to assign
      // tiles based on a resource model.
      std::sort(tiles.begin(), tiles.end(),
                AMDAIE::TileOp::tileColumnComparator);
      // Erase duplicates.
      tiles.erase(std::unique(tiles.begin(), tiles.end()), tiles.end());
      for (AMDAIE::TileOp tile : tiles) {
        std::optional<int64_t> column = getConstantIntValue(tile.getCol());
        if (!column) return tile.emitOpError() << "found non-constant column";
        tileLocations.insert(std::make_pair(column.value(), rowInt));
      }
      return success();
    };

    if (!targetTiles.empty() && !sourceTiles.empty()) {
      return logicalObjectFifo.emitOpError()
             << "found logical objectfifo with both source and target tiles, "
                "which is not supported yet";
    } else if (!targetTiles.empty()) {
      // Create tile locations for this logical objectfifo based on target
      // tiles.
      if (failed(createTileLocations(targetTiles))) {
        return failure();
      }
    } else if (!sourceTiles.empty()) {
      // Create tile locations for this logical objectfifo based on source
      // tiles.
      if (failed(createTileLocations(sourceTiles))) {
        return failure();
      }
    } else {
      // Don't assign this logicalObjectFifo to a physical tile (yet!). Wait
      // for other logical objectfifos to be assigned first.
      return failure();
    }

    // If no tile results, skip, and maybe in a next iteration another tile will
    // be found.
    if (tileLocations.empty()) {
      return failure();
    }

    rewriter.setInsertionPoint(logicalObjectFifo);
    rewriter.replaceOpWithNewOp<AMDAIE::LogicalObjectFifoFromMemrefOp>(
        logicalObjectFifo, logicalObjectFifo.getMemref(),
        tileLocations.takeVector());
    return success();
  }
};

/// Return the user DMA operations and corresponding assigned tiles in the
/// specified direction (source or target).
template <CopyOpOperateOn OperateOn>
SmallVector<std::pair<AMDAIE::DmaCpyNdOp, SmallVector<AMDAIE::TileOp>>>
getUserDmasAndTiles(AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo) {
  SmallVector<std::pair<AMDAIE::DmaCpyNdOp, SmallVector<AMDAIE::TileOp>>>
      dmaOps;
  for (Operation *user : logicalObjectFifo->getUsers()) {
    if (auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(user)) {
      ValueRange tileIndices;
      if constexpr (OperateOn == CopyOpOperateOn::Source) {
        if (dmaOp.getTargetObjectFifo() != logicalObjectFifo) continue;
        tileIndices = dmaOp.getSourceObjectFifo().getTiles();
      } else if constexpr (OperateOn == CopyOpOperateOn::Target) {
        if (dmaOp.getSourceObjectFifo() != logicalObjectFifo) continue;
        tileIndices = dmaOp.getTargetObjectFifo().getTiles();
      }
      SmallVector<AMDAIE::TileOp> tiles;
      for (Value index : tileIndices)
        tiles.push_back(dyn_cast<AMDAIE::TileOp>(index.getDefiningOp()));
      dmaOps.push_back(std::make_pair(dmaOp, tiles));
    }
  }
  return dmaOps;
}

/// Assign specific tile locations to objectFifos, starting from the set of
/// potential tile locations filled in earlier.
LogicalResult assignAieTilesAndDistributeLogicalObjectFifos(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());

  moduleOp->walk([&](AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo) {
    Attribute memSpace = logicalObjectFifo.getMemorySpace();
    if (memSpace && dyn_cast<IntegerAttr>(memSpace).getInt() != 1)
      return WalkResult::advance();

    SmallVector<AMDAIE::TileOp> tiles = llvm::map_to_vector(
        logicalObjectFifo.getTiles(),
        [](Value tile) { return dyn_cast<TileOp>(tile.getDefiningOp()); });
    llvm::sort(tiles.begin(), tiles.end(),
               AMDAIE::TileOp::tileColumnComparator);
    SmallVector<std::pair<AMDAIE::DmaCpyNdOp, SmallVector<AMDAIE::TileOp>>>
        sourceDmaOps =
            getUserDmasAndTiles<CopyOpOperateOn::Source>(logicalObjectFifo);
    SmallVector<std::pair<AMDAIE::DmaCpyNdOp, SmallVector<AMDAIE::TileOp>>>
        targetDmaOps =
            getUserDmasAndTiles<CopyOpOperateOn::Target>(logicalObjectFifo);

    // Assign tiles for following cases:
    // 1) No source DMA operations (e.g. L3 -> L2): distribute onto multiple
    // tiles to potentially use multiple shim DMAs for reading from global
    // memory in different columns.
    // 2) No target DMA operations (e.g. L2 -> L3):
    // distribute onto multiple tiles to potentially use multiple shim DMAs for
    // writing to global memory in different columns.
    // 3) Default: assign first tile from the sorted sequence of potential
    // tiles.
    if (sourceDmaOps.empty() && targetDmaOps.size() == tiles.size()) {
      llvm::sort(targetDmaOps.begin(), targetDmaOps.end(), dmaColComparator);
      for (auto &&[tile, dmaOpElem] : llvm::zip(tiles, targetDmaOps)) {
        rewriter.setInsertionPoint(logicalObjectFifo);
        SmallVector<Value> tileResults = {cast<Value>(tile.getResult())};
        auto newLogicalObjectFifo =
            rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
                rewriter.getUnknownLoc(),
                cast<LogicalObjectFifoType>(
                    logicalObjectFifo.getOutput().getType()),
                logicalObjectFifo.getMemref(), tileResults);
        dmaOpElem.first->replaceUsesOfWith(logicalObjectFifo.getResult(),
                                           newLogicalObjectFifo.getResult());
      }
    } else if (targetDmaOps.empty() && sourceDmaOps.size() == tiles.size()) {
      llvm::sort(sourceDmaOps.begin(), sourceDmaOps.end(), dmaColComparator);
      for (auto &&[tile, dmaOpElem] : llvm::zip(tiles, sourceDmaOps)) {
        rewriter.setInsertionPoint(logicalObjectFifo);
        SmallVector<Value> tileResults = {cast<Value>(tile.getResult())};
        auto newLogicalObjectFifo =
            rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
                rewriter.getUnknownLoc(),
                cast<LogicalObjectFifoType>(
                    logicalObjectFifo.getOutput().getType()),
                logicalObjectFifo.getMemref(), tileResults);
        dmaOpElem.first->replaceUsesOfWith(logicalObjectFifo.getResult(),
                                           newLogicalObjectFifo.getResult());
      }
    } else {
      // For now, use first tile in sorted list. This will need to become more
      // complex in the future to account for potential hardware limitations and
      // constraints.
      SmallVector<Value> tileResults = {cast<Value>(tiles[0].getResult())};
      rewriter.setInsertionPoint(logicalObjectFifo);
      rewriter.replaceOpWithNewOp<AMDAIE::LogicalObjectFifoFromMemrefOp>(
          logicalObjectFifo,
          cast<LogicalObjectFifoType>(logicalObjectFifo.getOutput().getType()),
          logicalObjectFifo.getMemref(), tileResults);
    }
    return WalkResult::advance();
  });
  return success();
}

class AMDAIEAssignTileLocationsPass
    : public impl::AMDAIEAssignTileLocationsBase<
          AMDAIEAssignTileLocationsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEAssignTileLocationsPass() = default;
  AMDAIEAssignTileLocationsPass(const AMDAIEAssignTileLocationsPass &pass){};
  void runOnOperation() override;
};

void AMDAIEAssignTileLocationsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp moduleOp = getOperation();

  // Assign tile locations to logical objectfifos on local (L1) memory.
  if (failed(assignLocalAieTiles(moduleOp))) {
    moduleOp.emitOpError() << "local tile assignment failed";
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs() << "Module after assignLocalAieTiles: \n"
                          << moduleOp << "\n");

  // Assign a set of potential tile locations to the remaining logical
  // objectFifos.
  RewritePatternSet assignAieTilePatters(context);
  assignAieTilePatters.insert<FillAieTiles>(context);
  if (failed(applyPatternsAndFoldGreedily(moduleOp,
                                          std::move(assignAieTilePatters)))) {
    moduleOp.emitOpError()
        << "collection of tile candidates for logical objectFifos failed";
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs() << "Module after FillAieTiles: \n"
                          << moduleOp << "\n");

  // Assign specific tile locations to objectFifos, starting from the set of
  // potential tile locations filled in earlier.
  if (failed(assignAieTilesAndDistributeLogicalObjectFifos(moduleOp))) {
    moduleOp.emitOpError()
        << "tile assignment and logical objectFifo distribution failed";
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEAssignTileLocationsPass() {
  return std::make_unique<AMDAIEAssignTileLocationsPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
