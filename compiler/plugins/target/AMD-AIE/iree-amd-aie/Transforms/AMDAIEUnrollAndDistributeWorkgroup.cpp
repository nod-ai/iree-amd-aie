// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-unroll-and-distribute-workgroup"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Convert scf.forall ops within a workgroup to scf.for ops
/// TODO(jornt): use upstream `forallToFor` function once merged.
LogicalResult workgroupForallToFor(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
  WalkResult res = moduleOp->walk([&](AMDAIE::WorkgroupOp workgroupOp) {
    WalkResult workgroupRes = workgroupOp->walk([&](scf::ForallOp forallOp) {
      if (failed(scf::forallToForLoop(rewriter, forallOp))) {
        workgroupOp.emitOpError()
            << "failed to transform scf.forall to scf.for";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (workgroupRes.wasInterrupted()) return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return failure();
  return success();
}

/// Hoist an affine apply op on a scf.for op's induction variable into that
/// scf.for block.
LogicalResult hoistAffineApplyDependingOnFor(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
  moduleOp->walk([&](affine::AffineApplyOp applyOp) {
    (void)hoistForAffineApplyOp(rewriter, applyOp);
  });
  return success();
}

/// Unroll the scf.for loops selected for parallel execution on the AIE
/// array. Try to hoist dma ops that don't depend on the loops' induction
/// variables to avoid duplicated copies.
///
/// This rewriter consists of a sequence of transformations:
///   1) First, try to promote the for loop if possible.
///   2) Try hoisting dma ops outside the scf.for operation.
///   3) Unroll and distribute the logical objectfifos remaining in the scf.for
///   loop.
class AMDAIEUnrollWorkgroupLoops : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  /// Hoist dma ops outside the scf.for operation if there are no dependencies
  /// on the scf.for loop, or on other dmas producing or consuming data from the
  /// one considered.
  ///
  /// NOTE: This method makes an assumption that the operations
  /// happen on the most local memory and therefore dmas moving data into more
  /// local memory can be hoisted before the scf.for loop. And on the other
  /// hand, dmas moving data away from local memory can be hoisted behind the
  /// scf.for. This assumption could be removed in the future.
  template <typename Iterator>
  LogicalResult hoistDmaOps(PatternRewriter &rewriter, scf::ForOp forOp) const {
    // Keep track of whether a hoist happened.
    bool hoistHappened{false};

    // Create utility function to check whether an operand depends on a scf.for
    // induction variable or a value within the scf.for's scope
    auto dependsOnLoop = [&](OpOperand &operand) -> bool {
      Operation *op = operand.get().getDefiningOp();
      if (!op) return operand.get() == forOp.getInductionVar();

      // Check for an induction var and whether the parent scf.for is the same
      // as the one we're using for the hoist
      auto parentForOp = op->getParentOfType<scf::ForOp>();
      return (operand.get() == forOp.getInductionVar()) ||
             (parentForOp == forOp);
    };

    // Logical objectfifo dependencies introduced in loop body walk.
    DenseSet<AMDAIE::LogicalObjectFifoFromMemrefOp> dependencies;

    // Utility to add logical objectfifos to the dependencies set.
    auto addDependencies = [&](AMDAIE::DmaCpyNdOp dmaOp) {
      if (std::is_same<Iterator, ForwardIterator>::value) {
        dependencies.insert(dmaOp.getTargetObjectFifo());
      } else if (std::is_same<Iterator, ReverseIterator>::value) {
        dependencies.insert(dmaOp.getSourceObjectFifo());
      }
    };

    // Walk all dma ops and try to hoist them if:
    //   1) There are no dependencies on the loop's induction variable.
    //   2) There are no dependencies on other dmas producing into a logical
    //   objectfifo which this dma is consuming.
    //   3) There are no dmas waiting for this dma to produce data.
    //
    // The last two conditions are partially checked through comparing source
    // and target memory spaces (Global < L2 < L1):
    //   1) In the forward sweep, only hoist dmas for which
    //   (source.memory < target.memory), i.e. moving data to AIE cores.
    //   2) In the backward sweep, only hoist dmas for which
    //   (source.memory > target.memory), i.e. moving data away from AIE cores.
    //
    // These last checks in theory limit the hoisting detection capability, but
    // should be valid.
    forOp.walk<WalkOrder::PostOrder, Iterator>([&](AMDAIE::DmaCpyNdOp dmaOp) {
      if (llvm::any_of(dmaOp->getOpOperands(), [&](OpOperand &operand) {
            return dependsOnLoop(operand);
          })) {
        addDependencies(dmaOp);
        return WalkResult::advance();
      }

      uint64_t sourceMemspace =
          dmaOp.getSourceObjectFifo().getMemorySpaceAsUInt();
      uint64_t targetMemspace =
          dmaOp.getTargetObjectFifo().getMemorySpaceAsUInt();
      if (std::is_same<Iterator, ForwardIterator>::value &&
          !dependencies.contains(dmaOp.getSourceObjectFifo()) &&
          sourceMemspace < targetMemspace) {
        rewriter.moveOpBefore(dmaOp, forOp);
        hoistHappened = true;
        return WalkResult::advance();
      } else if (std::is_same<Iterator, ReverseIterator>::value &&
                 !dependencies.contains(dmaOp.getTargetObjectFifo()) &&
                 sourceMemspace > targetMemspace) {
        rewriter.moveOpAfter(dmaOp, forOp);
        hoistHappened = true;
        return WalkResult::advance();
      }

      // If this dma op can't be hoisted due to dependencies, keep adding new
      // dependencies.
      addDependencies(dmaOp);
      return WalkResult::advance();
    });
    if (hoistHappened) return success();
    return failure();
  }

  /// Unroll the for loop, skipping the first iteration. While unrolling, create
  /// new clones of `LogicalObjectFifoFromMemrefOp` so they can be distributed
  /// onto multiple physical locations later.
  LogicalResult loopUnrollAndDistributeLogicalObjectFifos(
      RewriterBase &rewriter, scf::ForOp forOp) const {
    Block *loopBodyBlock = forOp.getBody();
    OpBuilder builder = OpBuilder::atBlockTerminator(loopBodyBlock);

    // Keep a pointer to the last non-terminator operation in the original block
    // so that we know what to clone (since we are doing this in-place).
    Block::iterator srcBlockEnd = std::prev(loopBodyBlock->end(), 2);

    // Update loop bounds
    int64_t lbInt = getConstantIntValue(forOp.getLowerBound()).value();
    int64_t ubInt = getConstantIntValue(forOp.getUpperBound()).value();
    int64_t stepInt = getConstantIntValue(forOp.getStep()).value();
    // TODO(avarma): Either :-
    //               1. Enforce loop normalisation before this pass.
    //                  OR
    //               2. Adapt this better during PR review.
    if (lbInt != 0) return failure();
    ubInt = std::ceil(ubInt / stepInt);
    stepInt = 1;
    if (stepInt == (ubInt - lbInt)) return failure();
    Value forOpIV = forOp.getInductionVar();
    forOp.setUpperBound(
        rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), 1));

    // Iterate through the loop and create body
    IRMapping operandMap;
    for (auto i = lbInt + stepInt; i < ubInt; i += stepInt) {
      Value ivUnroll =
          builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), i);
      if (!forOpIV.use_empty()) {
        operandMap.map(forOpIV, ivUnroll);
      }

      // Iterate through body and clone ops
      for (auto it = loopBodyBlock->begin(); it != std::next(srcBlockEnd);
           it++) {
        if (auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(*it)) {
          AMDAIE::LogicalObjectFifoFromMemrefOp source =
              dmaOp.getSourceObjectFifo();
          AMDAIE::LogicalObjectFifoFromMemrefOp target =
              dmaOp.getTargetObjectFifo();
          if (!operandMap.contains(source.getOutput())) {
            rewriter.setInsertionPoint(source);
            auto cloneOp = dyn_cast<AMDAIE::LogicalObjectFifoFromMemrefOp>(
                rewriter.clone(*dmaOp.getSource().getDefiningOp()));
            operandMap.map(source.getOutput(), cloneOp.getOutput());
          }
          if (!operandMap.contains(target.getOutput())) {
            rewriter.setInsertionPoint(target);
            auto cloneOp = dyn_cast<AMDAIE::LogicalObjectFifoFromMemrefOp>(
                rewriter.clone(*dmaOp.getTarget().getDefiningOp()));
            operandMap.map(target.getOutput(), cloneOp.getOutput());
          }
          builder.clone(*it, operandMap);
        } else {
          builder.clone(*it, operandMap);
        }
      }
    }
    return success();
  }

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Only unroll loops within a workgroup
    if (!forOp->getParentOfType<AMDAIE::WorkgroupOp>()) return failure();

    // Skip for ops with nested for ops. Wait until nested ones get resolved
    // first.
    auto nestedForOps = forOp.getOps<scf::ForOp>();
    if (!nestedForOps.empty()) return failure();

    // First, try to promote the for loop
    if (succeeded(forOp.promoteIfSingleIteration(rewriter))) {
      return success();
    }

    // Try hoisting dma ops outside the scf.for operation by sweeping once
    // forward and once backward to hoist to before, respectively after the
    // scf.for.
    (void)hoistDmaOps<ForwardIterator>(rewriter, forOp);
    (void)hoistDmaOps<ReverseIterator>(rewriter, forOp);

    // Unroll and distribute the logical objectfifos
    if (failed(loopUnrollAndDistributeLogicalObjectFifos(rewriter, forOp))) {
      return failure();
    }
    return success();
  }
};

/// Assign tiles to the logical objectfifos with local memory space (L1).
/// The tiles are derived from the usage of the logical objectfifos within
/// core operations, which are already assigned a tile location.
LogicalResult assignLocalAieTiles(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());

  // Map from local objectfifos found to the tiles where they are used
  DenseMap<AMDAIE::LogicalObjectFifoFromMemrefOp,
           llvm::SmallSetVector<Value, 16>>
      logicalObjectFifosToTiles;

  // Utility function insert a local objectfifo - tile pair into the local
  // objectfifo to tile map
  auto insertTile = [&](AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo,
                        Value tileResult) -> void {
    if (!logicalObjectFifosToTiles.contains(logicalObjectFifo)) {
      logicalObjectFifosToTiles[logicalObjectFifo] = {};
    }
    logicalObjectFifosToTiles[logicalObjectFifo].insert(tileResult);
  };

  // Walk DMA ops and find the ones which are used in cores to update
  // source/target logical objectfifos
  moduleOp->walk([&](AMDAIE::DmaCpyNdOp dmaOp) {
    for (Operation *userOp : dmaOp->getUsers()) {
      if (auto coreOp = userOp->getParentOfType<AMDAIE::CoreOp>()) {
        auto workgroupOp = coreOp->getParentOfType<AMDAIE::WorkgroupOp>();
        if (!workgroupOp) continue;

        Attribute sourceMemspace = dmaOp.getSourceObjectFifo().getMemorySpace();
        Attribute targetMemspace = dmaOp.getTargetObjectFifo().getMemorySpace();
        if (sourceMemspace &&
            dyn_cast<IntegerAttr>(sourceMemspace).getInt() == 2) {
          // Source on L1
          insertTile(dmaOp.getSourceObjectFifo(),
                     coreOp.getTileOp().getResult());
        } else if (targetMemspace &&
                   dyn_cast<IntegerAttr>(targetMemspace).getInt() == 2) {
          // Target on L1
          insertTile(dmaOp.getTargetObjectFifo(),
                     coreOp.getTileOp().getResult());
        }

        // Move tile to beginning of workgroup to make sure ssa values are not
        // dominated
        Block *workgroupBlock = workgroupOp.getBody();
        rewriter.moveOpBefore(coreOp.getTileOp(), workgroupBlock,
                              workgroupBlock->begin());
      }
    }
    return WalkResult::advance();
  });

  // Update the logical objectfifos with assigned tiles
  for (auto &&[logicalObjectFifo, tiles] : logicalObjectFifosToTiles) {
    rewriter.setInsertionPoint(logicalObjectFifo);
    rewriter.replaceOpWithNewOp<AMDAIE::LogicalObjectFifoFromMemrefOp>(
        logicalObjectFifo,
        cast<LogicalObjectFifoType>(logicalObjectFifo.getOutput().getType()),
        logicalObjectFifo.getMemref(), tiles.takeVector());
  }
  return success();
}

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

/// Assign logical objectfifos to physical AIE tiles. This rewrite takes an
/// iterative approach by matching logical objectfifos and only assigning tiles
/// when linked through dma ops with other logical objectfifos which already
/// have tiles assigned. If the linked logical objectfifos don't have tiles
/// assigned yet, we will return a failure and give the linked logical
/// objectfifos a chance to assign tiles before returning to this one.
///
/// TODO(jornt): There are decisions being made in this pass on which tile to
/// assign to a logical objectfifo. This logic is very simple for now and tries
/// to use the leftmost available column. At some point, we probably need some
/// AIE device model to guide the assignement here for performance and to avoid
/// resource issues down below.
class AssignAieTiles
    : public OpRewritePattern<AMDAIE::LogicalObjectFifoFromMemrefOp> {
  using OpRewritePattern<
      AMDAIE::LogicalObjectFifoFromMemrefOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo,
      PatternRewriter &rewriter) const override {
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

    SmallVector<Value> tileResults;
    if (!memSpace || dyn_cast<IntegerAttr>(memSpace).getInt() == 1) {
      // HandLe both L3/shim and L2/Memtiles. Try to use memtiles in the same
      // column as the AIE tiles where the data needs to go to.

      // TODO(jornt): avoid row hardcoding. Will need to update the mlir-aie
      // target model for this.
      int rowInt = memSpace ? 1 : 0;
      Value row = rewriter.create<arith::ConstantIndexOp>(
          rewriter.getUnknownLoc(), rowInt);

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

      auto colComparator = [](AMDAIE::TileOp &a, AMDAIE::TileOp &b) -> bool {
        int64_t colA = getConstantIntValue(a.getCol()).value();
        int64_t colB = getConstantIntValue(b.getCol()).value();
        return colA < colB;
      };
      if (!targetTiles.empty()) {
        // TODO(jornt): For now, for deterministic behaviour, sort on column
        // index and use first one. This needs to be generalized to assign tiles
        // based on a resource model.
        std::sort(targetTiles.begin(), targetTiles.end(), colComparator);
        Value col = targetTiles[0].getCol();
        tileResults.push_back(
            rewriter.create<AMDAIE::TileOp>(rewriter.getUnknownLoc(), col, row)
                .getResult());
      } else if (!sourceTiles.empty()) {
        // TODO(jornt): For now, for deterministic behaviour, sort on column
        // index and use first one. This needs to be generalized to assign tiles
        // based on a resource model.
        std::sort(sourceTiles.begin(), sourceTiles.end(), colComparator);
        Value col = sourceTiles[0].getCol();
        tileResults.push_back(
            rewriter.create<AMDAIE::TileOp>(rewriter.getUnknownLoc(), col, row)
                .getResult());
      } else {
        // Don't assign this logicalObjectFifo to a physical tile (yet!). Wait
        // for other logical objectfifos to be assigned first.
        return failure();
      }
    } else {
      return logicalObjectFifo.emitOpError()
             << "found logical objectfifo with unknown memory space";
    }
    // If no tile results, skip, and maybe in a next iteration another tile will
    // be found.
    if (tileResults.empty()) {
      return failure();
    }

    // Extend this logical objectfifo's tile set.
    SmallVector<Value> objFifoTiles = logicalObjectFifo.getTiles();
    DenseSet<Value> tileSet(objFifoTiles.begin(), objFifoTiles.end());

    // If the logical objectfifo already contains all the new tiles, skip.
    if (llvm::all_of(tileResults,
                     [&](Value val) { return tileSet.contains(val); })) {
      return failure();
    }

    // Concatenate existing with new tiles and replace the logicalObjectFifo
    std::move(objFifoTiles.begin(), objFifoTiles.end(),
              std::back_inserter(tileResults));
    rewriter.replaceOpWithNewOp<AMDAIE::LogicalObjectFifoFromMemrefOp>(
        logicalObjectFifo,
        cast<LogicalObjectFifoType>(logicalObjectFifo.getOutput().getType()),
        logicalObjectFifo.getMemref(), tileResults);
    return success();
  }
};

class AMDAIEUnrollAndDistributeWorkgroupPass
    : public impl::AMDAIEUnrollAndDistributeWorkgroupBase<
          AMDAIEUnrollAndDistributeWorkgroupPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEUnrollAndDistributeWorkgroupPass() = default;
  AMDAIEUnrollAndDistributeWorkgroupPass(
      const AMDAIEUnrollAndDistributeWorkgroupPass &pass){};
  void runOnOperation() override;
};

void AMDAIEUnrollAndDistributeWorkgroupPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp moduleOp = getOperation();
  // Convert scf.forall operations within a workgroup to nested scf.for
  // operations.
  if (failed(workgroupForallToFor(moduleOp))) {
    return signalPassFailure();
  }
  // Hoist the affine apply ops on scf.for induction variables to the
  // corresponding scf.for's body.
  if (failed(hoistAffineApplyDependingOnFor(moduleOp))) {
    return signalPassFailure();
  }
  // Unroll loops inside the workgroups and try hoisting dma operations if
  // possible.
  RewritePatternSet unrollWorkgroupPatterns(context);
  unrollWorkgroupPatterns.insert<AMDAIEUnrollWorkgroupLoops>(context);
  if (failed(applyPatternsAndFoldGreedily(
          moduleOp, std::move(unrollWorkgroupPatterns)))) {
    return signalPassFailure();
  }
  // Assign tile locations to logical objectfifos on local (L1) memory.
  if (failed(assignLocalAieTiles(moduleOp))) {
    return signalPassFailure();
  }
  // Assign tile locations to the remaining logical objectfifos.
  RewritePatternSet assignAieTilePatters(context);
  assignAieTilePatters.insert<AssignAieTiles>(context);
  if (failed(applyPatternsAndFoldGreedily(moduleOp,
                                          std::move(assignAieTilePatters)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEUnrollAndDistributeWorkgroupPass() {
  return std::make_unique<AMDAIEUnrollAndDistributeWorkgroupPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
