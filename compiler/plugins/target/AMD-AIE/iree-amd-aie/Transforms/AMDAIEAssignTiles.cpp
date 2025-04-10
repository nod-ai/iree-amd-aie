// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEOpUtils.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-assign-tiles"

namespace mlir::iree_compiler::AMDAIE {

/// Return the tiles of the sources respectively targets of the users of this
/// logical objectfifo, depending on whether the OperateOn template parameter is
/// set to `OperateOn::Source` respectively `OperateOn::Target`.
template <CopyOpOperateOn OperateOn>
LogicalResult getUserTiles(AMDAIE::LogicalObjFifoOpInterface logicalObjectFifo,
                           SmallVectorImpl<AMDAIE::TileOp> &tiles) {
  llvm::SmallSetVector<AMDAIE::TileOp, 16> tileSet;
  for (Operation *user : logicalObjectFifo->getUsers()) {
    if (auto copyOp = dyn_cast<CopyOpInterface>(user)) {
      auto source = dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
          copyOp.getSource().getDefiningOp());
      auto target = dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
          copyOp.getTarget().getDefiningOp());
      if (!source || !target) continue;
      SmallVector<Value> tileIndices;
      if constexpr (OperateOn == CopyOpOperateOn::Source) {
        if (target != logicalObjectFifo) continue;
        tileIndices = source.getTiles();
      } else if constexpr (OperateOn == CopyOpOperateOn::Target) {
        if (source != logicalObjectFifo) continue;
        tileIndices = target.getTiles();
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
    Operation *op, AMDAIE::LogicalObjFifoOpInterface logicalObjectFifo,
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
                   dyn_cast<AMDAIE::LogicalObjFifoOpInterface>(userOp)) {
      return findUsersInCoreAndAddTiles(userLogicalObjectFifo,
                                        logicalObjectFifo, tiles);
    }
  }
  return success();
}

/// Utility to clear non-local tile assignments.
LogicalResult clearNonLocalTiles(RewriterBase &rewriter, Operation *op) {
  WalkResult res = op->walk([&](AMDAIE::LogicalObjFifoOpInterface objFifo) {
    if (objFifo.getMemorySpaceAsUInt() != 2) {
      rewriter.setInsertionPoint(objFifo);
      SmallVector<Value> tiles;
      if (failed(objFifo.replaceWithNewTiles(rewriter, tiles))) {
        objFifo.emitOpError() << "could not replace its tiles";
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return failure();
  return success();
}

/// Utility to duplicate global objectFifos (L3) for each strided copy-like
/// operation user to allow global logical objectFifos to be assigned to
/// different tile locations.
LogicalResult duplicateGlobalObjFifos(RewriterBase &rewriter, Operation *op) {
  op->walk([&](AMDAIE::DoublyStridedCopyOpInterface copyOp) {
    auto source = dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
        copyOp.getSource().getDefiningOp());
    auto target = dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
        copyOp.getTarget().getDefiningOp());
    auto createNewObjFifoAndReplaceUsesFrom =
        [&](AMDAIE::LogicalObjFifoOpInterface oldObjFifo) {
          rewriter.setInsertionPoint(copyOp);
          auto newObjFifo = cast<AMDAIE::LogicalObjFifoOpInterface>(
              rewriter.clone(*oldObjFifo.getOperation()));
          rewriter.replaceUsesWithIf(
              oldObjFifo->getResult(0), newObjFifo->getResult(0),
              [&](OpOperand &use) {
                return use.getOwner() == copyOp.getOperation();
              });
        };
    if (source && source.getMemorySpaceAsUInt() == 0) {
      createNewObjFifoAndReplaceUsesFrom(source);
    }
    if (target && target.getMemorySpaceAsUInt() == 0) {
      createNewObjFifoAndReplaceUsesFrom(target);
    }
  });
  return success();
}

/// Base class for tile allocators to support different tile assignment
/// strategies.
class TileAllocatorBase {
 public:
  TileAllocatorBase(RewriterBase &rewriter,
                    const AMDAIE::AMDAIEDeviceModel &deviceModel)
      : rewriter(rewriter), deviceModel(deviceModel) {}
  virtual ~TileAllocatorBase(){};
  /// Assign tiles to the provided vector of logical objectFifos on the same
  /// memory space. This method expects a set of valid tile candidates to be
  /// provided inside the operation.
  virtual LogicalResult assignTiles(
      SmallVector<AMDAIE::LogicalObjFifoOpInterface> &objFifos,
      uint8_t memSpace, function_ref<InFlightDiagnostic()> emitError) = 0;

 protected:
  RewriterBase &rewriter;
  const AMDAIE::AMDAIEDeviceModel &deviceModel;
};

/// A custom tile allocater that takes into consideration the usage and column
/// of users to determine tile locations.
class UsageAndColumnBasedTileAllocator final : public TileAllocatorBase {
 public:
  DenseMap<Operation *, DenseSet<Operation *>> uniqueL3L2Pair;

  UsageAndColumnBasedTileAllocator(
      RewriterBase &rewriter, const AMDAIE::AMDAIEDeviceModel &deviceModel,
      DenseMap<Operation *, DenseSet<Operation *>> uniqueL3L2Pair)
      : TileAllocatorBase(rewriter, deviceModel),
        uniqueL3L2Pair(uniqueL3L2Pair) {}

  LogicalResult assignTiles(
      SmallVector<AMDAIE::LogicalObjFifoOpInterface> &objFifos,
      uint8_t memSpace, function_ref<InFlightDiagnostic()> emitError) {
    assert(llvm::all_of(objFifos,
                        [&](AMDAIE::LogicalObjFifoOpInterface objFifo) {
                          return objFifo.getMemorySpaceAsUInt() == memSpace;
                        }) &&
           "All logical objectFifos should have ths same memory space");
    if (memSpace == 2) return assignLocalTiles(objFifos, memSpace, emitError);
    if (memSpace == 0 || memSpace == 1)
      return assignNonLocalTiles(objFifos, memSpace, emitError);
    return emitError() << "Unsupported memory space : "
                       << std::to_string(memSpace);
  }

 private:
  /// Assign tiles to the logical objectfifos with local memory space (L1).
  /// The tiles are derived from the usage of the logical objectfifos within
  /// core operations, which are already assigned a tile location.
  LogicalResult assignLocalTiles(
      SmallVector<AMDAIE::LogicalObjFifoOpInterface> &objFifos,
      uint8_t memSpace, function_ref<InFlightDiagnostic()> emitError) {
    assert(memSpace == 2 && "Local memory space should be `2`.");
    for (AMDAIE::LogicalObjFifoOpInterface objFifo : objFifos) {
      llvm::SmallSetVector<std::pair<int64_t, int64_t>, 16> tileLocations;
      if (failed(findUsersInCoreAndAddTiles(objFifo, objFifo, tileLocations))) {
        return failure();
      }
      // Handle subviews.
      if (auto fromMemrefOp = dyn_cast<AMDAIE::LogicalObjectFifoFromMemrefOp>(
              objFifo.getOperation())) {
        for (Operation *userOp :
             fromMemrefOp.getMemref().getDefiningOp()->getUsers()) {
          if (auto subviewOp = dyn_cast<memref::SubViewOp>(userOp)) {
            if (failed(findUsersInCoreAndAddTiles(subviewOp, objFifo,
                                                  tileLocations))) {
              return failure();
            }
          }
        }
      }

      SmallVector<Value> tiles;
      tiles.reserve(tileLocations.size());
      rewriter.setInsertionPoint(objFifo);
      for (auto [column, row] : tileLocations) {
        auto colIndex = rewriter.create<arith::ConstantIndexOp>(
            rewriter.getUnknownLoc(), column);
        auto rowIndex = rewriter.create<arith::ConstantIndexOp>(
            rewriter.getUnknownLoc(), row);
        auto tileOp = rewriter.create<AMDAIE::TileOp>(rewriter.getUnknownLoc(),
                                                      colIndex, rowIndex);
        tiles.push_back(tileOp.getResult());
      }
      // Sort for deterministic output IR.
      llvm::sort(tiles.begin(), tiles.end(),
                 AMDAIE::TileOp::tileValueColumnAndRowComparator);
      if (failed(objFifo.replaceWithNewTiles(rewriter, tiles))) {
        return objFifo.emitOpError() << "could not replace its tiles.";
      }
    }
    return success();
  }

  /// Assign tile locations to objectFifos. Start by searching for a set of
  /// candidate tile locations and then assign tiles based on a simple
  /// usage-based model that prioritizes tiles that have the least usage.
  LogicalResult assignNonLocalTiles(
      SmallVector<AMDAIE::LogicalObjFifoOpInterface> &objFifos,
      uint8_t memSpace, function_ref<InFlightDiagnostic()> emitError) {
    assert((memSpace == 0 || memSpace == 1) &&
           "The memory space of non-local objectFifos should be `0` or `1`");
    // Keep track of the buffer usage on tiles to try distributing buffers
    // evenly over available tile resources.
    DenseMap<TileLoc, size_t> tileLocToUsage;
    auto tileLocAndUsageCmp = [&](const TileLoc &a, const TileLoc &b) -> bool {
      size_t usageA = tileLocToUsage[a];
      size_t usageB = tileLocToUsage[b];
      if (usageA < usageB) return true;
      if (usageA > usageB) return false;
      if (a.col < b.col) return true;
      if (a.col > b.col) return false;
      if (a.row < b.row) return true;
      if (a.row > b.row) return false;
      assert(false && "same tiles should never be compared");
      return false;
    };

    SmallVector<uint32_t> memSpaceRows = deviceModel.getMemSpaceRows(memSpace);
    if (memSpaceRows.size() == 0) {
      return emitError()
             << "No rows found for the memory space of this logical objFifo";
    }
    uint32_t row = memSpaceRows[0];

    for (AMDAIE::LogicalObjFifoOpInterface objFifo : objFifos) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Assign tile for objFifo: " << objFifo << "\n");

      mlir::FunctionOpInterface funcOp =
          objFifo->getParentOfType<mlir::FunctionOpInterface>();
      if (!funcOp) {
        return objFifo.emitOpError()
               << "Could not find a function-like parent op.";
      }
      FailureOr<CoreRegionInfo> coreRegionInfo = getCoreRegionInfo(funcOp);
      if (failed(coreRegionInfo)) return failure();
      int startCol = coreRegionInfo.value().startCol;
      int numCols = coreRegionInfo.value().numCols;
      llvm::SmallSetVector<TileLoc, 16> tileLocations;
      for (int i = startCol; i < startCol + numCols; i++)
        tileLocations.insert(TileLoc(i, row));
      if (tileLocations.empty()) {
        return objFifo.emitOpError() << "No tile locations found for this "
                                        "logical objFifo. Maybe in a next "
                                        "iteration, with more information, a "
                                        "tile location can be found.";
      }
      SmallVector<TileLoc> tiles = tileLocations.takeVector();

      // Sort tiles on priority column + left to right;
      FailureOr<int64_t> maybePriorityCol = getPriorityColumn(objFifo);
      if (failed(maybePriorityCol)) return failure();
      int64_t priorityCol = maybePriorityCol.value();
      llvm::sort(tiles, [&](const TileLoc &a, const TileLoc &b) {
        if (a.col == priorityCol) return true;
        return a.col < b.col;
      });

      // Here we are limiting the number of tile options to buffer count to
      // avoid repeated accesses of the same buffer being assigned to
      // different tiles. Because if the repeated access of the same buffer
      // are assigned to different tiles, that unnecessarily ends up consuming
      // more DMA channels on those new tiles than needed, and as a result we
      // will end up exhausting the DMA channels. Currently the following fix
      // works for L3 buffers.
      auto fromMemrefOp = dyn_cast<AMDAIE::LogicalObjectFifoFromMemrefOp>(
          objFifo.getOperation());
      if (fromMemrefOp) {
        Operation *defOp = fromMemrefOp.getMemref().getDefiningOp();
        if (defOp && uniqueL3L2Pair.contains(defOp))
          tiles.truncate(
              std::min((size_t)uniqueL3L2Pair[defOp].size(), tiles.size()));
      }

      // Assing a tile location.
      TileLoc assignedTileLoc;
      const auto *tileIt = llvm::find_if(tiles, [&](const TileLoc &tileLoc) {
        return tileLoc.col == priorityCol;
      });
      if (tileIt != tiles.end()) {
        assignedTileLoc = *tileIt;
        LLVM_DEBUG(llvm::dbgs()
                   << "Assign to priority column: " << priorityCol << "\n");
      } else {
        LLVM_DEBUG(llvm::dbgs() << "Assign based on usage comparator\n");
        assignedTileLoc =
            *std::min_element(tiles.begin(), tiles.end(), tileLocAndUsageCmp);
      }

      // Increase usage of the chosen tile as a new logical objectFifo will be
      // assigned to it. This allows distributing the logical objectFifos
      // evenly across the available tile resources.
      LLVM_DEBUG(llvm::dbgs()
                 << "Assign to tile (col, row): (" << assignedTileLoc.col
                 << ", " << assignedTileLoc.row << ")\n");
      tileLocToUsage[assignedTileLoc] += 1;

      rewriter.setInsertionPoint(objFifo);
      auto getCol = rewriter.create<arith::ConstantIndexOp>(
          rewriter.getUnknownLoc(), assignedTileLoc.col);
      auto getRow = rewriter.create<arith::ConstantIndexOp>(
          rewriter.getUnknownLoc(), assignedTileLoc.row);
      auto assignedTileOp = rewriter.create<AMDAIE::TileOp>(
          rewriter.getUnknownLoc(), getCol, getRow);
      SmallVector<Value> tileResults = {
          cast<Value>(assignedTileOp.getResult())};
      if (failed(objFifo.replaceWithNewTiles(rewriter, tileResults))) {
        return objFifo.emitOpError() << "Could not replace its tiles.";
      }
    }
    return success();
  }

  /// Utility to return a priority column for the provided objectFifo if tiles
  /// of users are all within the same column. Returns `-1` if no priority
  /// column was found.
  FailureOr<int64_t> getPriorityColumn(
      AMDAIE::LogicalObjFifoOpInterface objFifo) const {
    int64_t priorityCol{-1};
    SmallVector<AMDAIE::TileOp> targetTiles;
    SmallVector<AMDAIE::TileOp> sourceTiles;
    LogicalResult dstRes =
        getUserTiles<CopyOpOperateOn::Target>(objFifo, targetTiles);
    LogicalResult srcRes =
        getUserTiles<CopyOpOperateOn::Source>(objFifo, sourceTiles);
    if (failed(dstRes) && failed(srcRes)) {
      return objFifo.emitOpError() << "No source or target tiles found";
    }
    SmallVector<int64_t> targetColsVec =
        llvm::map_to_vector(targetTiles, [](AMDAIE::TileOp tileOp) {
          std::optional<int64_t> column = getConstantIntValue(tileOp.getCol());
          return column.has_value() ? column.value() : -1;
        });
    DenseSet<int64_t> targetCols(targetColsVec.begin(), targetColsVec.end());
    SmallVector<int64_t> sourceColsVec =
        llvm::map_to_vector(sourceTiles, [](AMDAIE::TileOp tileOp) {
          std::optional<int64_t> column = getConstantIntValue(tileOp.getCol());
          return column.has_value() ? column.value() : -1;
        });
    DenseSet<int64_t> sourceCols(sourceColsVec.begin(), sourceColsVec.end());
    if (targetCols.size() == 1 && sourceCols.size() == 1) {
      int64_t targetCol = *targetCols.begin();
      int64_t sourceCol = *sourceCols.begin();
      if (targetCol != -1 && sourceCol != -1 && targetCol != sourceCol) {
        return objFifo.emitOpError()
               << "Found two different priority columns, column " << targetCol
               << " on the target side and column " << sourceCol
               << " on the source side.";
      } else {
        priorityCol = targetCol;
      }
    } else if (targetCols.size() == 1) {
      priorityCol = *targetCols.begin();
    } else if (sourceCols.size() == 1) {
      priorityCol = *sourceCols.begin();
    }
    return priorityCol;
  }
};

/// Assign tile locations to objectFifos based on available resources. Visit
/// objectFifos based on locality to the cores, i.e. first visit the objectFifos
/// on L1, then L2, etc.
LogicalResult assignTiles(
    RewriterBase &rewriter, Operation *op, const AMDAIEDeviceModel &deviceModel,
    DenseMap<Operation *, DenseSet<Operation *>> uniqueL3L2Pair) {
  if (failed(clearNonLocalTiles(rewriter, op)))
    return op->emitOpError() << "failed to clear non-local tile assignments";

  UsageAndColumnBasedTileAllocator tileAllocator(rewriter, deviceModel,
                                                 uniqueL3L2Pair);

  DenseMap<uint8_t, SmallVector<AMDAIE::LogicalObjFifoOpInterface>>
      memSpaceToObjFifos;
  op->walk([&](AMDAIE::LogicalObjFifoOpInterface logicalObjectFifo) {
    uint8_t memSpace = logicalObjectFifo.getMemorySpaceAsUInt();
    memSpaceToObjFifos[memSpace].push_back(logicalObjectFifo);
  });
  SmallVector<uint8_t> memSpaces = llvm::map_to_vector(
      memSpaceToObjFifos,
      [](const std::pair<uint8_t,
                         SmallVector<AMDAIE::LogicalObjFifoOpInterface>>
             &memSpaceAndObjFifos) { return memSpaceAndObjFifos.first; });
  llvm::sort(memSpaces, std::greater<uint8_t>());
  for (uint8_t memSpace : memSpaces) {
    if (failed(
            tileAllocator.assignTiles(memSpaceToObjFifos[memSpace], memSpace,
                                      [&]() { return op->emitOpError(); }))) {
      return failure();
    }
  }
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
           "device-specific information is required to looking up column and "
           "row related information, and must be attached to a containing "
           "ModuleOp.";
    return signalPassFailure();
  }
  AMDAIEDeviceModel deviceModel = getDeviceModel(maybeDevice.value());

  // Duplicate global objectFifos for each strided copy-like operation user to
  // allow global logical objectFifos to be assigned to different tile
  // locations.
  if (failed(duplicateGlobalObjFifos(rewriter, parentOp))) {
    parentOp->emitOpError() << "failed duplicating global object fifos";
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs() << "After duplicateGlobalObjFifos: \n"
                          << *parentOp << "\n");

  // Analyse the count of each L3 logicalObjectFifos appearing in L3<->L2
  // CopyOp. We would be maintaining this count in a map `l3BufferCount` and
  // using this later to pick one tile to assign to the L3 logicalObjectFifo.
  // The analysis below is maintaining all unique pairs of (L3 source, L2
  // target) and (L3 target, L2 source).
  DenseMap<Operation *, DenseSet<Operation *>> uniqueL3L2Pair;
  parentOp->walk([&](Operation *op) -> WalkResult {
    if (auto copyOp = dyn_cast<CopyOpInterface>(op)) {
      auto source = dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
          copyOp.getSource().getDefiningOp());
      auto target = dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
          copyOp.getTarget().getDefiningOp());
      if (!source || !target) {
        return WalkResult::interrupt();
      }
      auto sourceFromMemrefOp = dyn_cast<AMDAIE::LogicalObjectFifoFromMemrefOp>(
          source.getOperation());
      auto targetFromMemrefOp = dyn_cast<AMDAIE::LogicalObjectFifoFromMemrefOp>(
          target.getOperation());
      if (!sourceFromMemrefOp || !targetFromMemrefOp) {
        return WalkResult::interrupt();
      }
      Operation *l3DefOp = nullptr;
      Operation *l2DefOp = nullptr;
      if (source.getMemorySpaceAsUInt() == 0) {
        l3DefOp = sourceFromMemrefOp.getMemref().getDefiningOp();
        l2DefOp = targetFromMemrefOp.getMemref().getDefiningOp();
      } else if (target.getMemorySpaceAsUInt() == 0) {
        l3DefOp = targetFromMemrefOp.getMemref().getDefiningOp();
        l2DefOp = sourceFromMemrefOp.getMemref().getDefiningOp();
      } else {
        return WalkResult::advance();
      }
      uniqueL3L2Pair[l3DefOp].insert(l2DefOp);
      return WalkResult::advance();
    }
    return WalkResult::advance();
  });
  // Assign tile locations to logical objectFifos on non-local (not L1) memory.
  if (failed(assignTiles(rewriter, parentOp, deviceModel, uniqueL3L2Pair))) {
    parentOp->emitOpError() << "non-local tile assignment failed";
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
