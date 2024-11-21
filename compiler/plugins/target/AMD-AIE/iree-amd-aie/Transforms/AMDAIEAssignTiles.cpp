// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/aie_runtime/Utils/ChannelGenerator.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-assign-tiles"

namespace mlir::iree_compiler::AMDAIE {

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
      for (Value index : tileIndices) {
        tileSet.insert(
            dyn_cast_if_present<AMDAIE::TileOp>(index.getDefiningOp()));
      }
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
      if (!column || !row)
        return coreOp.emitOpError() << "has non-constant tile location";
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

/// Utility to clear non-local tile assignments.
LogicalResult clearNonLocalTiles(RewriterBase &rewriter, Operation *op) {
  op->walk([&](AMDAIE::LogicalObjectFifoFromMemrefOp objFifo) {
    if (objFifo.getMemorySpaceAsUInt() != 2) {
      rewriter.setInsertionPoint(objFifo);
      SmallVector<Value> tiles;
      rewriter.replaceOpWithNewOp<AMDAIE::LogicalObjectFifoFromMemrefOp>(
          objFifo, cast<LogicalObjectFifoType>(objFifo.getOutput().getType()),
          objFifo.getMemref(), tiles);
    }
  });
  return success();
}

/// Utility to duplicate global objFifos for each strided copy-like operation
/// user to allow global logical objectFifos to be assigned to different tile
/// locations.
LogicalResult duplicateGlobalObjFifos(RewriterBase &rewriter, Operation *op) {
  op->walk([&](AMDAIE::DoublyStridedCopyOpInterface copyOp) {
    auto source = dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromMemrefOp>(
        copyOp.getSource().getDefiningOp());
    auto target = dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromMemrefOp>(
        copyOp.getTarget().getDefiningOp());
    if (source && source.getMemorySpaceAsUInt() == 0) {
      rewriter.setInsertionPoint(copyOp);
      auto newSource = rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
          rewriter.getUnknownLoc(),
          cast<LogicalObjectFifoType>(source.getOutput().getType()),
          source.getMemref());
      rewriter.replaceUsesWithIf(
          source.getOutput(), newSource.getOutput(), [&](OpOperand &use) {
            return use.getOwner() == copyOp.getOperation();
          });
    }
    if (target && target.getMemorySpaceAsUInt() == 0) {
      rewriter.setInsertionPoint(copyOp);
      auto newTarget = rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
          rewriter.getUnknownLoc(),
          cast<LogicalObjectFifoType>(target.getOutput().getType()),
          target.getMemref());
      rewriter.replaceUsesWithIf(
          target.getOutput(), newTarget.getOutput(), [&](OpOperand &use) {
            return use.getOwner() == copyOp.getOperation();
          });
    }
  });
  return success();
}

/// Assign tiles to the logical objectfifos with local memory space (L1).
/// The tiles are derived from the usage of the logical objectfifos within
/// core operations, which are already assigned a tile location.
LogicalResult assignLocalTiles(RewriterBase &rewriter, Operation *op) {
  WalkResult res =
      op->walk([&](AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo) {
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

/// Assign a set of candidate physical AIE tiles to logical objectFifos. This
/// rewrite takes an iterative approach by matching logical objectfifos and only
/// assigning tiles when linked through dma ops with other logical objectfifos
/// which already have tiles assigned. If the linked logical objectfifos don't
/// have tiles assigned yet, we will return a failure and give the linked
/// logical objectfifos a chance to assign tiles before returning to this one.
class FillTiles
    : public OpRewritePattern<AMDAIE::LogicalObjectFifoFromMemrefOp> {
  using OpRewritePattern<
      AMDAIE::LogicalObjectFifoFromMemrefOp>::OpRewritePattern;

 public:
  FillTiles(MLIRContext *context, const AMDAIE::AMDAIEDeviceModel &deviceModel)
      : OpRewritePattern(context), deviceModel(deviceModel) {}

  LogicalResult matchAndRewrite(
      AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo,
      PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "FillTiles: " << logicalObjectFifo << "\n");
    if (!logicalObjectFifo.getTiles().empty()) {
      return rewriter.notifyMatchFailure(logicalObjectFifo,
                                         "Tiles are already assigned.");
    }
    uint8_t memSpace = logicalObjectFifo.getMemorySpaceAsUInt();
    if (memSpace != 0 && memSpace != 1) {
      return rewriter.notifyMatchFailure(
          logicalObjectFifo,
          "Skip logical objFifos that don't operate on L3 or L2");
    }

    SmallVector<AMDAIE::TileOp, 16> targetTiles;
    SmallVector<AMDAIE::TileOp, 16> sourceTiles;
    LogicalResult dstRes =
        getUserTiles<CopyOpOperateOn::Target>(logicalObjectFifo, targetTiles);
    LogicalResult srcRes =
        getUserTiles<CopyOpOperateOn::Source>(logicalObjectFifo, sourceTiles);
    if (failed(dstRes) && failed(srcRes)) {
      return rewriter.notifyMatchFailure(logicalObjectFifo,
                                         "No source or target tiles found");
    }

    SmallVector<uint32_t> memSpaceRows = deviceModel.getMemSpaceRows(memSpace);
    if (memSpaceRows.size() == 0) {
      return rewriter.notifyMatchFailure(
          logicalObjectFifo,
          "No rows found for the memory space of this logical objFifo");
    }
    if (memSpaceRows.size() > 1) {
      logicalObjectFifo.emitWarning()
          << "has a memory space with multiple available rows, the first one "
             "of which is chosen for tile assignment, but this might not lead "
             "to good usage of the available resources.";
    }
    uint32_t row = memSpaceRows[0];
    llvm::SmallSetVector<std::pair<int64_t, int64_t>, 16> tileLocations;
    auto createTileLocations =
        [&](SmallVector<AMDAIE::TileOp, 16> &tiles) -> LogicalResult {
      // For deterministic and canonical output, sort on column index and erase
      // duplicates.
      std::sort(tiles.begin(), tiles.end(),
                AMDAIE::TileOp::tileColumnComparator);
      tiles.erase(std::unique(tiles.begin(), tiles.end()), tiles.end());
      for (AMDAIE::TileOp tile : tiles) {
        std::optional<int64_t> column = getConstantIntValue(tile.getCol());
        if (!column) return tile.emitOpError() << "found non-constant column";
        tileLocations.insert(std::make_pair(column.value(), row));
      }
      return success();
    };

    if (!targetTiles.empty() && !sourceTiles.empty()) {
      return rewriter.notifyMatchFailure(
          logicalObjectFifo,
          "Found logical objectfifo with both source and target tiles, which "
          "is not supported yet");
    } else if (!targetTiles.empty()) {
      // Create tile locations for this logical objectfifo based on the
      // consumers' tiles.
      if (failed(createTileLocations(targetTiles))) {
        return rewriter.notifyMatchFailure(
            logicalObjectFifo,
            "Could not find tile locations based on the consumers' tiles.");
      }
    } else if (!sourceTiles.empty()) {
      // Create tile locations for this logical objectfifo based on producers'
      // tiles.
      if (failed(createTileLocations(sourceTiles))) {
        return rewriter.notifyMatchFailure(
            logicalObjectFifo,
            "Could not find tile locations based on the producers' tiles.");
      }
    } else {
      return rewriter.notifyMatchFailure(
          logicalObjectFifo,
          "Don't assign this logicalObjectFifo to a physical tile (yet!). Wait "
          "for other logical objectfifos to be assigned first.");
    }

    if (tileLocations.empty()) {
      return rewriter.notifyMatchFailure(
          logicalObjectFifo,
          "No tile locations found for this logical objFifo. Maybe in a next "
          "iteration, with more information, a tile location can be found.");
    }
    rewriter.setInsertionPoint(logicalObjectFifo);
    rewriter.replaceOpWithNewOp<AMDAIE::LogicalObjectFifoFromMemrefOp>(
        logicalObjectFifo, logicalObjectFifo.getMemref(),
        tileLocations.takeVector());
    return success();
  }

 private:
  // The device model used to retrieve device specific information.
  const AMDAIEDeviceModel &deviceModel;
};

/// Assign tile locations to objectFifos. Start by searching for a set of
/// candidate tile locations and then assign tiles based on a simple usage-based
/// model that prioritizes tiles that have the least usage.
LogicalResult assignNonLocalTiles(RewriterBase &rewriter, Operation *op,
                                  const AMDAIEDeviceModel &deviceModel) {
  MLIRContext *context = rewriter.getContext();
  if (failed(clearNonLocalTiles(rewriter, op)))
    return op->emitOpError() << "failed to clear non-local tile assignemts";

  // Find and fill the tile candidates.
  RewritePatternSet fillTilePatterns(context);
  fillTilePatterns.insert<FillTiles>(context, deviceModel);
  if (failed(applyPatternsAndFoldGreedily(op, std::move(fillTilePatterns)))) {
    return op->emitOpError()
           << "collection of tile candidates for logical objectFifos failed";
  }
  if (failed(verify(op, true))) {
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "After fillTiles: \n" << *op << "\n");

  // Keep track of the buffer usage on tiles to try distributing buffers equally
  // over available tiles.
  DenseMap<TileLoc, size_t> tileLocToUsage;
  auto tileLocAndUsageCmp = [&](AMDAIE::TileOp a, AMDAIE::TileOp b) -> bool {
    int64_t colA = getConstantIndexOrAssert(a.getCol());
    int64_t rowA = getConstantIndexOrAssert(a.getRow());
    int64_t colB = getConstantIndexOrAssert(b.getCol());
    int64_t rowB = getConstantIndexOrAssert(b.getRow());
    size_t usageA = tileLocToUsage[TileLoc(colA, rowA)];
    size_t usageB = tileLocToUsage[TileLoc(colB, rowB)];
    if (usageA < usageB) return true;
    if (usageA > usageB) return false;
    if (colA < colB) return true;
    if (colA > colB) return false;
    if (rowA < rowB) return true;
    if (rowA > rowB) return false;
    assert(false && "same tiles should never be compared");
  };

  // After filling tile candidates, find and assign a specific one.
  DenseMap<MemRefType, int64_t> logicalObjFifoToTileId;
  WalkResult res =
      op->walk([&](AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo) {
        uint8_t memSpace = logicalObjectFifo.getMemorySpaceAsUInt();
        if (memSpace != 0 && memSpace != 1) return WalkResult::advance();
        if (logicalObjectFifo.getTiles().size() == 0) {
          logicalObjectFifo.emitOpError()
              << "should have at least one tile candidate";
          return WalkResult::interrupt();
        }

        SmallVector<AMDAIE::TileOp> tiles =
            llvm::map_to_vector(logicalObjectFifo.getTiles(), [](Value tile) {
              return dyn_cast_if_present<TileOp>(tile.getDefiningOp());
            });
        AMDAIE::TileOp assignedTileOp =
            *std::min_element(tiles.begin(), tiles.end(), tileLocAndUsageCmp);

        // Increase usage of the chosen tile.
        int64_t col = getConstantIndexOrAssert(assignedTileOp.getCol());
        int64_t row = getConstantIndexOrAssert(assignedTileOp.getRow());
        tileLocToUsage[TileLoc(col, row)] += 1;

        rewriter.setInsertionPoint(logicalObjectFifo);
        SmallVector<Value> tileResults = {
            cast<Value>(assignedTileOp.getResult())};
        rewriter.replaceOpWithNewOp<AMDAIE::LogicalObjectFifoFromMemrefOp>(
            logicalObjectFifo,
            cast<LogicalObjectFifoType>(
                logicalObjectFifo.getOutput().getType()),
            logicalObjectFifo.getMemref(), tileResults);
        return WalkResult::advance();
      });
  if (res.wasInterrupted()) return failure();
  return success();
}

namespace {

class AMDAIEAssignTilesPass
    : public impl::AMDAIEAssignTilesBase<AMDAIEAssignTilesPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  void runOnOperation() override;
};

void AMDAIEAssignTilesPass::runOnOperation() {
  Operation *parentOp = getOperation();
  IRRewriter rewriter(&getContext());
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(parentOp);
  std::optional<AMDAIEDevice> maybeDevice = getConfigAMDAIEDevice(targetAttr);
  if (!maybeDevice) {
    parentOp->emitOpError()
        << "has no AMDAIEDevice in the target attribute configuration. This "
           "device-specific information is required to determine when loops "
           "can be subsumed into DMA operations, and must be attached to a "
           "containing ModuleOp.";
    return signalPassFailure();
  }
  AMDAIEDeviceModel deviceModel = getDeviceModel(maybeDevice.value());

  // Assign tile locations to logical objectfifos on local (L1) memory.
  if (failed(assignLocalTiles(rewriter, parentOp))) {
    parentOp->emitOpError() << "local tile assignment failed";
    return signalPassFailure();
  }
  if (failed(verify(parentOp, true))) {
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs() << "After assignLocalTiles: \n" << *parentOp << "\n");

  // Duplicate global objFifos for each strided copy-like operation user to
  // allow global logical objectFifos to be assigned to different tile
  // locations.
  if (failed(duplicateGlobalObjFifos(rewriter, parentOp))) {
    parentOp->emitOpError() << "failed duplicating global object fifos";
    return signalPassFailure();
  }
  if (failed(verify(parentOp, true))) {
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs() << "After duplicateGlobalObjFifos: \n"
                          << *parentOp << "\n");

  // Assign tile locations to logical objectfifos on non-local (not L1) memory.
  if (failed(assignNonLocalTiles(rewriter, parentOp, deviceModel))) {
    parentOp->emitOpError() << "local tile assignment failed";
    return signalPassFailure();
  }
  if (failed(verify(parentOp, true))) {
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs() << "After assignNonLocalTiles: \n"
                          << *parentOp << "\n");
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEAssignTilesPass() {
  return std::make_unique<AMDAIEAssignTilesPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
