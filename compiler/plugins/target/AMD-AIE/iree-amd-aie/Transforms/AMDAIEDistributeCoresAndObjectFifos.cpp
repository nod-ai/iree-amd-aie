// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#define DEBUG_TYPE "iree-amdaie-distribute-cores-and-objectfifos"

namespace mlir::iree_compiler::AMDAIE {

// Used to annotate loops that should be unrolled.
static const llvm::StringLiteral kAMDAIELoopUnroll = "amdaie.unroll";

namespace {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// AMDAIEDistributeCoresAndObjectFifosPass
//===----------------------------------------------------------------------===//

/// Convert inner scf.forall ops chosen for parallel distribution to scf.for
/// ops.
LogicalResult localForallToFor(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
  WalkResult res = moduleOp->walk([&](scf::ForallOp forallOp) {
    auto maybeMapping = forallOp.getMapping();
    if (!maybeMapping) return WalkResult::advance();
    SmallVector<Attribute> mapping = llvm::to_vector(maybeMapping->getValue());
    if (mapping.empty()) return WalkResult::advance();

    // We index on thread mapping for core and dma unrolling and buffer
    // distribution.
    if (!isa<mlir::gpu::GPUThreadMappingAttr>(*mapping.begin()))
      return WalkResult::advance();

    SmallVector<Operation *> loopResults;
    if (failed(scf::forallToForLoop(rewriter, forallOp, &loopResults))) {
      forallOp.emitOpError() << "failed to transform scf.forall to scf.for";
      return WalkResult::interrupt();
    }
    // Set attribute to unroll this loop later in this pass.
    for (Operation *loopRes : loopResults) {
      scf::ForOp forOp = dyn_cast<scf::ForOp>(loopRes);
      if (!forOp) {
        forallOp.emitOpError() << "failed to retrieve generated scf.for from "
                                  "scf::forallToForLoop conversion";
        return WalkResult::interrupt();
      }
      forOp->setAttr(kAMDAIELoopUnroll,
                     mlir::BoolAttr::get(forOp->getContext(), true));
    }
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
class AMDAIEUnrollLocalLoops : public OpRewritePattern<scf::ForOp> {
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
    // Bail out early on if the scf.for is within amdaie.core op. This is
    // because such loops will not have DmaOps and are indeed formed due to
    // vectorizing loop nest, therefore we can bail out early.
    if (forOp->getParentOfType<AMDAIE::CoreOp>()) return failure();
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

      std::optional<uint8_t> sourceMemspace =
          dmaOp.getSourceMemorySpaceAsUInt();
      std::optional<uint8_t> targetMemspace =
          dmaOp.getTargetMemorySpaceAsUInt();
      if (!sourceMemspace || !targetMemspace) {
        dmaOp.emitOpError() << "expected a source and target memory space";
        return WalkResult::interrupt();
      }
      if (std::is_same<Iterator, ForwardIterator>::value &&
          !dependencies.contains(dmaOp.getSourceObjectFifo()) &&
          sourceMemspace.value() < targetMemspace.value()) {
        rewriter.moveOpBefore(dmaOp, forOp);
        hoistHappened = true;
        return WalkResult::advance();
      } else if (std::is_same<Iterator, ReverseIterator>::value &&
                 !dependencies.contains(dmaOp.getTargetObjectFifo()) &&
                 sourceMemspace.value() > targetMemspace.value()) {
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
    if (lbInt != 0 || stepInt != 1) return failure();
    if (stepInt == (ubInt - lbInt)) return failure();
    Value forOpIV = forOp.getInductionVar();
    forOp.setUpperBound(
        rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), 1));

    // Iterate through the loop and create body
    for (int64_t i = lbInt + stepInt; i < ubInt; i += stepInt) {
      IRMapping operandMap;
      Value ivUnroll =
          builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), i);
      if (!forOpIV.use_empty()) {
        operandMap.map(forOpIV, ivUnroll);
      }

      // Iterate through body and map internal logical objectfifos to new ones
      // and fill operand map.
      for (auto it = loopBodyBlock->begin(); it != std::next(srcBlockEnd);
           it++) {
        if (auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(*it)) {
          std::optional<uint8_t> sourceMemSpaceInt =
              dmaOp.getSourceMemorySpaceAsUInt();
          std::optional<uint8_t> targetMemSpaceInt =
              dmaOp.getTargetMemorySpaceAsUInt();
          if (!sourceMemSpaceInt || !targetMemSpaceInt) {
            return rewriter.notifyMatchFailure(
                dmaOp, "expected a source and target memory space");
          }
          if (targetMemSpaceInt.value() > sourceMemSpaceInt.value()) {
            AMDAIE::LogicalObjectFifoFromMemrefOp target =
                dmaOp.getTargetObjectFifo();
            rewriter.setInsertionPoint(target);
            auto cloneOp =
                dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromMemrefOp>(
                    rewriter.clone(*dmaOp.getTarget().getDefiningOp()));
            operandMap.map(target.getOutput(), cloneOp.getOutput());
          } else if (sourceMemSpaceInt.value() > targetMemSpaceInt.value()) {
            AMDAIE::LogicalObjectFifoFromMemrefOp source =
                dmaOp.getSourceObjectFifo();
            rewriter.setInsertionPoint(source);
            auto cloneOp =
                dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromMemrefOp>(
                    rewriter.clone(*dmaOp.getSource().getDefiningOp()));
            operandMap.map(source.getOutput(), cloneOp.getOutput());
          }
        }
      }

      // Iterate through body and clone ops
      for (auto it = loopBodyBlock->begin(); it != std::next(srcBlockEnd);
           it++) {
        builder.clone(*it, operandMap);
      }
    }
    return success();
  }

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Unroll loops only if annotated to be unrolled earlier in the pass.
    if (!forOp->hasAttr(kAMDAIELoopUnroll) ||
        !cast<BoolAttr>(forOp->getAttr(kAMDAIELoopUnroll)).getValue())
      return failure();

    // Skip for ops with nested for ops. Wait until nested ones get resolved
    // first.
    auto nestedForOps = forOp.getOps<scf::ForOp>();
    if (!nestedForOps.empty()) return failure();

    // First, try to promote the for loop
    if (succeeded(forOp.promoteIfSingleIteration(rewriter))) {
      return success();
    }

    // Hoist non-dma loop invariant operations (like constants, affine apply,
    // etc) out of the loop like operation to allow more DMA operations to be
    // hoisted.
    moveLoopInvariantCode(dyn_cast<LoopLikeOpInterface>(forOp.getOperation()));

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

/// Insert `amdaie.logicalobjectfifo.access` operations which retrieve the
/// memrefs from logical objectfifos and update the computational operations to
/// operate on these local memrefs.
LogicalResult insertLogicalObjectFifoAccess(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());

  SmallVector<AMDAIE::CoreOp> coreOps;
  moduleOp->walk([&](AMDAIE::CoreOp coreOp) { coreOps.push_back(coreOp); });

  for (AMDAIE::CoreOp coreOp : coreOps) {
    DenseMap<Value, std::tuple<AMDAIE::LogicalObjectFifoFromMemrefOp,
                               AMDAIE::MemoryAccess>>
        memrefToLogicalObjectFifo;

    SmallVector<AMDAIE::DmaCpyNdOp> inputDmaOps =
        llvm::map_to_vector(coreOp.getInputDmas(), [](Value inputDma) {
          return cast<AMDAIE::DmaCpyNdOp>(inputDma.getDefiningOp());
        });
    for (AMDAIE::DmaCpyNdOp inputDmaOp : inputDmaOps) {
      Value targetMemref = inputDmaOp.getTargetObjectFifo().getMemref();
      memrefToLogicalObjectFifo[targetMemref] = std::make_pair(
          inputDmaOp.getTargetObjectFifo(), AMDAIE::MemoryAccess::Read);
    }
    SmallVector<AMDAIE::DmaCpyNdOp> outputDmaOps =
        llvm::map_to_vector(coreOp.getOutputDmas(), [](Value outputDma) {
          return cast<AMDAIE::DmaCpyNdOp>(outputDma.getDefiningOp());
        });
    for (AMDAIE::DmaCpyNdOp outputDmaOp : outputDmaOps) {
      Value sourceMemref = outputDmaOp.getSourceObjectFifo().getMemref();
      memrefToLogicalObjectFifo[sourceMemref] = std::make_pair(
          outputDmaOp.getSourceObjectFifo(), AMDAIE::MemoryAccess::Write);
    }

    // We maintain a map from AllocOp to LogicalObjectFifoAccessOp in order to
    // avoid creating a new LogicalObjectFifoAccessOp for the same AllocOp being
    // used in a different op.
    DenseMap<Value, AMDAIE::LogicalObjectFifoAccessOp>
        memrefToLogicalObjectFifoAccess;

    WalkResult res = coreOp->walk([&](Operation *op) {
      // We want to insert amdaie.logicalobjectfifo.access ops right before
      // the first usage. But for vectorized ops this would mean they'd get
      // inserted within the vectorized scf.for ops. We therefore would want
      // to traverse to the outermost scf.for op in that case. Currently we
      // bubble up this traversal till that operation whose parent is not a
      // scf.for op. TODO: Generalize this later.
      Operation *opToInsertRewriterPoint = op;
      while (isa<scf::ForOp>(opToInsertRewriterPoint->getParentOp())) {
        opToInsertRewriterPoint = opToInsertRewriterPoint->getParentOp();
      }
      for (auto &&[idx, operand] : llvm::enumerate(op->getOpOperands())) {
        Operation *operandDefiningOp = operand.get().getDefiningOp();
        if (!dyn_cast_if_present<memref::AllocOp>(operandDefiningOp)) continue;
        if (memrefToLogicalObjectFifoAccess.contains(operand.get())) {
          op->setOperand(idx, memrefToLogicalObjectFifoAccess[operand.get()]);
        } else if (memrefToLogicalObjectFifo.contains(operand.get())) {
          rewriter.setInsertionPoint(opToInsertRewriterPoint);
          std::tuple<AMDAIE::LogicalObjectFifoFromMemrefOp,
                     AMDAIE::MemoryAccess>
              value = memrefToLogicalObjectFifo[operand.get()];
          auto accessOp = rewriter.create<AMDAIE::LogicalObjectFifoAccessOp>(
              rewriter.getUnknownLoc(), std::get<0>(value), std::get<1>(value));
          memrefToLogicalObjectFifoAccess[operand.get()] = accessOp;
          op->setOperand(idx, accessOp);
        } else if (auto type =
                       llvm::dyn_cast<MemRefType>(operand.get().getType())) {
          Value memref = operand.get();
          rewriter.setInsertionPoint(coreOp);

          auto logicalObjectFifo =
              rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
                  rewriter.getUnknownLoc(), LogicalObjectFifoType::get(type),
                  memref);

          rewriter.setInsertionPoint(opToInsertRewriterPoint);

          AMDAIE::LogicalObjectFifoAccessOp accessOp;
          if (memrefToLogicalObjectFifo.contains(memref)) {
            std::tuple<AMDAIE::LogicalObjectFifoFromMemrefOp,
                       AMDAIE::MemoryAccess>
                value = memrefToLogicalObjectFifo[memref];
            accessOp = rewriter.create<AMDAIE::LogicalObjectFifoAccessOp>(
                rewriter.getUnknownLoc(), std::get<0>(value),
                std::get<1>(value));
          } else {
            accessOp = rewriter.create<AMDAIE::LogicalObjectFifoAccessOp>(
                rewriter.getUnknownLoc(), logicalObjectFifo,
                AMDAIE::MemoryAccess::None);
          }
          memrefToLogicalObjectFifoAccess[memref] = accessOp;
          op->setOperand(idx, accessOp);
        }
      }
      return WalkResult::advance();
    });
    if (res.wasInterrupted()) return failure();
  }
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

/// Assign specific tile locations to objectFifos, starting from the set of
/// potential tile locations filled in earlier.
LogicalResult assignAieTilesAndDistributeLogicalObjectFifos(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());

  moduleOp->walk([&](AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo) {
    Attribute memSpace = logicalObjectFifo.getMemorySpace();
    if (memSpace && dyn_cast<IntegerAttr>(memSpace).getInt() != 1)
      return WalkResult::advance();

    SmallVector<AMDAIE::TileOp> tiles =
        llvm::map_to_vector(logicalObjectFifo.getTiles(), [](Value tile) {
          return dyn_cast_if_present<TileOp>(tile.getDefiningOp());
        });
    llvm::sort(tiles.begin(), tiles.end(),
               AMDAIE::TileOp::tileColumnComparator);

    // For now, use first tile in sorted list.
    // TODO(jornt): This will need to become more complex in the future to
    // account for potential hardware limitations and constraints.
    SmallVector<Value> tileResults = {cast<Value>(tiles[0].getResult())};
    rewriter.setInsertionPoint(logicalObjectFifo);
    rewriter.replaceOpWithNewOp<AMDAIE::LogicalObjectFifoFromMemrefOp>(
        logicalObjectFifo,
        cast<LogicalObjectFifoType>(logicalObjectFifo.getOutput().getType()),
        logicalObjectFifo.getMemref(), tileResults);
    return WalkResult::advance();
  });
  return success();
}

class AMDAIEDistributeCoresAndObjectFifosPass
    : public impl::AMDAIEDistributeCoresAndObjectFifosBase<
          AMDAIEDistributeCoresAndObjectFifosPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEDistributeCoresAndObjectFifosPass() = default;
  AMDAIEDistributeCoresAndObjectFifosPass(
      const AMDAIEDistributeCoresAndObjectFifosPass &pass){};
  void runOnOperation() override;
};

void AMDAIEDistributeCoresAndObjectFifosPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp moduleOp = getOperation();

  // Convert local scf.forall operations selected for parallel distribution to
  // nested scf.for operations.
  if (failed(localForallToFor(moduleOp))) {
    moduleOp.emitOpError()
        << "local `scf.forall` to `scf.for` conversion failed";
    return signalPassFailure();
  }

  // Hoist the affine apply ops on scf.for induction variables to the
  // corresponding scf.for's body.
  if (failed(hoistAffineApplyDependingOnFor(moduleOp))) {
    moduleOp.emitOpError() << "`affine.apply` hoisting failed";
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs() << "Module after localForallToFor: \n"
                          << moduleOp << "\n");

  if (failed(verify(moduleOp, true))) {
    return signalPassFailure();
  }

  // Unroll local parallel loops and try hoisting dma operations if
  // possible.
  RewritePatternSet unrollLocalLoopsPatterns(context);
  unrollLocalLoopsPatterns.insert<AMDAIEUnrollLocalLoops>(context);
  if (failed(applyPatternsAndFoldGreedily(
          moduleOp, std::move(unrollLocalLoopsPatterns)))) {
    moduleOp.emitOpError()
        << "loop unrolling of loops selected for parallel execution failed";
    return signalPassFailure();
  }

  if (failed(verify(moduleOp, true))) {
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs() << "Module after AMDAIEUnrollLocalLoops: \n"
                          << moduleOp << "\n");

  // Insert `amdaie.logicalobjectfifo.access` operations which retrieve the
  // memrefs from logical objectfifos and update the computational operations
  // to operate on these local memrefs. These access operations will be used
  // to assign local AIE tiles to local logical objectFifos later.
  if (failed(insertLogicalObjectFifoAccess(moduleOp))) {
    moduleOp.emitOpError()
        << "insertion of `amdaie.logicalobjectfifo.access` operations failed";
    return signalPassFailure();
  }

  if (failed(verify(moduleOp, true))) {
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs() << "Module after insertLogicalObjectFifoAccess: \n"
                          << moduleOp << "\n");

  // Assign tile locations to logical objectfifos on local (L1) memory.
  if (failed(assignLocalAieTiles(moduleOp))) {
    moduleOp.emitOpError() << "local tile assignment failed";
    return signalPassFailure();
  }

  if (failed(verify(moduleOp, true))) {
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

  if (failed(verify(moduleOp, true))) {
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

  if (failed(verify(moduleOp, true))) {
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs()
             << "Module after assignAieTilesAndDistributeLogicalObjectFifos: \n"
             << moduleOp << "\n");
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEDistributeCoresAndObjectFifosPass() {
  return std::make_unique<AMDAIEDistributeCoresAndObjectFifosPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
