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

#define DEBUG_TYPE "iree-amdaie-distribute-shared-memory"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Utility to use tuple coordinates as key of a `DenseMap`.
struct LocationMapInfo {
  static SmallVector<std::pair<int64_t, int64_t>> getEmptyKey() {
    return {std::make_pair(int64_t(-1), int64_t(-1))};
  }

  static SmallVector<std::pair<int64_t, int64_t>> getTombstoneKey() {
    return {std::make_pair(int64_t(-2), int64_t(-2))};
  }

  static unsigned getHashValue(
      const SmallVector<std::pair<int64_t, int64_t>> &v) {
    return static_cast<unsigned>(llvm::hash_combine_range(v.begin(), v.end()));
  }

  static bool isEqual(const SmallVector<std::pair<int64_t, int64_t>> &lhs,
                      const SmallVector<std::pair<int64_t, int64_t>> &rhs) {
    return lhs == rhs;
  }
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

/// Allocate different memories for logical objectFifos on the same shared
/// memory tile to ensure different buffers will be used for them.
LogicalResult distributeSharedMemory(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());

  // Map from local objectfifos found to the tiles where they are used
  DenseMap<SmallVector<std::pair<int64_t, int64_t>>, Value, LocationMapInfo>
      locationsToMemref;

  moduleOp->walk([&](AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo) {
    Attribute memSpace = logicalObjectFifo.getMemorySpace();
    if (!memSpace || dyn_cast<IntegerAttr>(memSpace).getInt() != 1)
      return WalkResult::advance();

    SmallVector<AMDAIE::TileOp> tiles =
        llvm::map_to_vector(logicalObjectFifo.getTiles(), [](Value tile) {
          return dyn_cast<AMDAIE::TileOp>(tile.getDefiningOp());
        });
    llvm::sort(tiles.begin(), tiles.end(),
               AMDAIE::TileOp::tileValueColumnAndRowComparator);

    SmallVector<AMDAIE::TileOp> targetTiles;
    (void)getUserTiles<CopyOpOperateOn::Target>(logicalObjectFifo, targetTiles);
    llvm::sort(targetTiles.begin(), targetTiles.end(),
               AMDAIE::TileOp::tileValueColumnAndRowComparator);
    tiles.insert(tiles.end(), std::make_move_iterator(targetTiles.begin()),
                 std::make_move_iterator(targetTiles.end()));

    SmallVector<AMDAIE::TileOp> sourceTiles;
    (void)getUserTiles<CopyOpOperateOn::Source>(logicalObjectFifo, sourceTiles);
    llvm::sort(sourceTiles.begin(), sourceTiles.end(),
               AMDAIE::TileOp::tileValueColumnAndRowComparator);
    tiles.insert(tiles.end(), std::make_move_iterator(sourceTiles.begin()),
                 std::make_move_iterator(sourceTiles.end()));
    LLVM_DEBUG(llvm::dbgs() << "Op: " << logicalObjectFifo
                            << ", number of tiles: " << tiles.size() << "\n");

    SmallVector<std::pair<int64_t, int64_t>> locations =
        llvm::map_to_vector(tiles, [](AMDAIE::TileOp tile) {
          return std::make_pair(
              (int64_t)getConstantIntValue(tile.getCol()).value(),
              (int64_t)getConstantIntValue(tile.getRow()).value());
        });
    if (!locationsToMemref.contains(locations)) {
      auto allocOp = dyn_cast<memref::AllocOp>(
          logicalObjectFifo.getMemref().getDefiningOp());
      rewriter.setInsertionPoint(allocOp);
      auto newAllocOp =
          dyn_cast<memref::AllocOp>(rewriter.clone(*allocOp.getOperation()));
      auto newDeallocOp = rewriter.create<memref::DeallocOp>(
          rewriter.getUnknownLoc(), newAllocOp);
      newDeallocOp->moveBefore(&newAllocOp->getBlock()->back());
      locationsToMemref[locations] = newAllocOp.getResult();
    }
    rewriter.setInsertionPoint(logicalObjectFifo);
    rewriter.replaceOpWithNewOp<AMDAIE::LogicalObjectFifoFromMemrefOp>(
        logicalObjectFifo,
        cast<LogicalObjectFifoType>(logicalObjectFifo.getOutput().getType()),
        locationsToMemref[locations], logicalObjectFifo.getTiles());
    return WalkResult::advance();
  });
  return success();
}

class AMDAIEDistributeSharedMemoryPass
    : public impl::AMDAIEDistributeSharedMemoryBase<
          AMDAIEDistributeSharedMemoryPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEDistributeSharedMemoryPass() = default;
  AMDAIEDistributeSharedMemoryPass(
      const AMDAIEDistributeSharedMemoryPass &pass){};
  void runOnOperation() override;
};

void AMDAIEDistributeSharedMemoryPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  // Allocate different memories for logical objectFifos on the same shared
  // memory tile to ensure different buffers will be used for them.
  if (failed(distributeSharedMemory(moduleOp))) {
    moduleOp.emitOpError() << "distribution of shared memory failed";
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEDistributeSharedMemoryPass() {
  return std::make_unique<AMDAIEDistributeSharedMemoryPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
