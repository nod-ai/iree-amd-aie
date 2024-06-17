// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Common.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#define DEBUG_TYPE "iree-amdaie-unroll-cores-and-distribute-memory"

namespace mlir::iree_compiler::AMDAIE {

static const llvm::StringLiteral kAMDAIELoopUnroll = "amdaie.unroll";

namespace {

/// Utility to use tuple coordinates as key of a `DenseMap`.
struct LocationMapInfo {
  static SmallVector<std::tuple<int64_t, int64_t>> getEmptyKey() {
    return {std::make_tuple(int64_t(-1), int64_t(-1))};
  }

  static SmallVector<std::tuple<int64_t, int64_t>> getTombstoneKey() {
    return {std::make_tuple(int64_t(-2), int64_t(-2))};
  }

  static unsigned getHashValue(
      const SmallVector<std::tuple<int64_t, int64_t>> &v) {
    return static_cast<unsigned>(llvm::hash_combine_range(v.begin(), v.end()));
  }

  static bool isEqual(const SmallVector<std::tuple<int64_t, int64_t>> &lhs,
                      const SmallVector<std::tuple<int64_t, int64_t>> &rhs) {
    return lhs == rhs;
  }
};

LogicalResult distributeLocalMemoryHack(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());

  SmallVector<Operation *> toBeErased;

  // Map from memref result to a new memref to be used.
  DenseMap<memref::AllocOp, memref::AllocOp> memrefToNew;

  // Walk DMA ops and find the ones which are used in cores to update
  // source/target logical objectfifos
  moduleOp->walk([&](memref::AllocOp allocOp) {
    // TODO source/target
    // AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo =
    //     dmaOp.getSourceObjectFifo();
    // Attribute memSpace = logicalObjectFifo.getMemorySpace();
    Attribute memSpace =
        cast<MemRefType>(allocOp.getResult().getType()).getMemorySpace();
    if (!memSpace || dyn_cast<IntegerAttr>(memSpace).getInt() != 2)
      return WalkResult::advance();

    LLVM_DEBUG(llvm::dbgs() << "Alloc op: " << allocOp << "\n");

    SmallVector<AMDAIE::DmaCpyNdOp> dmaUsers;
    for (Operation *userOp : allocOp->getUsers()) {
      if (auto logicalObjectFifo =
              dyn_cast<AMDAIE::LogicalObjectFifoFromMemrefOp>(userOp)) {
        for (Operation *objFifoUserOp : logicalObjectFifo->getUsers()) {
          if (auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(objFifoUserOp);
              dmaOp.getSourceObjectFifo() == logicalObjectFifo) {
            dmaUsers.push_back(dmaOp);
          }
        }
      }
    }
    if (dmaUsers.empty()) return WalkResult::advance();
    LLVM_DEBUG(llvm::dbgs() << "DMA users: " << dmaUsers.size() << "\n");

    // auto allocOp =
    //     dyn_cast<memref::AllocOp>(logicalObjectFifo.getMemref().getDefiningOp());
    // if (!allocOp) return WalkResult::advance();

    for (Operation *userOp : allocOp->getUsers()) {
      auto subviewOp = dyn_cast<memref::SubViewOp>(userOp);
      if (!subviewOp) continue;

      if (!memrefToNew.contains(allocOp)) {
        LLVM_DEBUG(llvm::dbgs() << "Create new allocate\n");
        rewriter.setInsertionPoint(allocOp);
        auto memRefType = cast<MemRefType>(subviewOp.getResult().getType());
        MemRefType allocType = MemRefType::get(
            memRefType.getShape(), memRefType.getElementType(),
            MemRefLayoutAttrInterface{}, memRefType.getMemorySpace());
        auto newAllocOp = rewriter.create<memref::AllocOp>(
            rewriter.getUnknownLoc(), allocType);
        auto newDeallocOp = rewriter.create<memref::DeallocOp>(
            rewriter.getUnknownLoc(), newAllocOp);
        newDeallocOp->moveBefore(&newAllocOp->getBlock()->back());
        // memrefToNew.map(allocOp.getOperation(), newAllocOp.getOperation());
        // memrefToNew.map(allocOp, newAllocOp);
        memrefToNew[allocOp] = newAllocOp;
      }
      LLVM_DEBUG(llvm::dbgs() << "replaceAllUsesWith of subview\n");
      auto newAlloc = memrefToNew[allocOp];
      rewriter.replaceAllUsesWith(subviewOp, newAlloc);
      // subviewOp->dropAllUses();
      // rewriter.eraseOp(subviewOp);
      toBeErased.push_back(subviewOp);
    }

    LLVM_DEBUG(llvm::dbgs() << "BEFORE\n");
    if (memrefToNew.contains(allocOp)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Update allocate DMA users: " << dmaUsers.size() << "\n");
      auto newAlloc = memrefToNew[allocOp];
      auto type = cast<MemRefType>(newAlloc.getType());
      for (AMDAIE::DmaCpyNdOp dmaOp : dmaUsers) {
        SmallVector<Value> empty;
        rewriter.setInsertionPoint(dmaOp.getSourceObjectFifo());
        auto source = rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
            rewriter.getUnknownLoc(), LogicalObjectFifoType::get(type),
            newAlloc.getResult());
        rewriter.replaceOp(dmaOp.getSourceObjectFifo(), source);
        rewriter.setInsertionPoint(dmaOp);
        auto newDmaOp = rewriter.create<AMDAIE::DmaCpyNdOp>(
            dmaOp.getLoc(), dmaOp.getTarget(), dmaOp.getTargetOffsets(),
            dmaOp.getTargetSizes(), dmaOp.getTargetStrides(), source, empty,
            empty, empty);
        rewriter.replaceOp(dmaOp, newDmaOp);
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "Before deallocOp\n");

      memref::DeallocOp deallocOp;
      for (Operation *userOp : allocOp->getUsers()) {
        if (auto deallocUser = dyn_cast<memref::DeallocOp>(userOp)) {
          deallocOp = deallocUser;
        }
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "Before deallocOp if\n");
      if (deallocOp) {
        toBeErased.push_back(deallocOp);
      }
      toBeErased.push_back(allocOp);
    }
    LLVM_DEBUG(llvm::dbgs()
                 << "Before advance\n");
    return WalkResult::advance();
  });

  LLVM_DEBUG(llvm::dbgs() << "erase\n");
  for (auto *op : toBeErased) {
    op->dropAllUses();
    rewriter.eraseOp(op);
  }
  return success();
}

/// Convert scf.forall ops within a workgroup to scf.for ops
/// TODO(jornt): use upstream `forallToFor` function once merged.
LogicalResult workgroupForallToFor(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
  WalkResult res = moduleOp->walk([&](scf::ForallOp forallOp) {
    SmallVector<Attribute> mapping =
        llvm::to_vector(forallOp.getMapping()->getValue());
    // We index on thread mapping for core and dma unrolling and buffer
    // distribution.
    if (!isa<mlir::gpu::GPUThreadMappingAttr>(*mapping.begin()))
      return WalkResult::advance();

    SmallVector<Operation *> results;
    if (failed(scf::forallToForLoop(rewriter, forallOp, &results))) {
      forallOp.emitOpError() << "failed to transform scf.forall to scf.for";
      return WalkResult::interrupt();
    }
    // Set attribute to unroll this loop later in this pass.
    for (Operation *res : results) {
      scf::ForOp forOp = dyn_cast<scf::ForOp>(res);
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
    if (lbInt != 0 || stepInt != 1) return failure();
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
          // TODO(jornt): needs refactoring for case when DMAs are not found
          // inside workgroup.
          AMDAIE::LogicalObjectFifoFromMemrefOp source =
              dmaOp.getSourceObjectFifo();
          uint64_t sourceMemSpaceInt = source.getMemorySpaceAsUInt();
          AMDAIE::LogicalObjectFifoFromMemrefOp target =
              dmaOp.getTargetObjectFifo();
          uint64_t targetMemSpaceInt = target.getMemorySpaceAsUInt();
          if (targetMemSpaceInt > sourceMemSpaceInt) {
            rewriter.setInsertionPoint(target);
            auto cloneOp = dyn_cast<AMDAIE::LogicalObjectFifoFromMemrefOp>(
                rewriter.clone(*dmaOp.getTarget().getDefiningOp()));
            operandMap.map(target.getOutput(), cloneOp.getOutput());
          } else if (sourceMemSpaceInt > targetMemSpaceInt) {
            rewriter.setInsertionPoint(source);
            auto cloneOp = dyn_cast<AMDAIE::LogicalObjectFifoFromMemrefOp>(
                rewriter.clone(*dmaOp.getSource().getDefiningOp()));
            operandMap.map(source.getOutput(), cloneOp.getOutput());
          }
          // if (!operandMap.contains(source.getOutput())) {
          //   rewriter.setInsertionPoint(source);
          //   auto cloneOp = dyn_cast<AMDAIE::LogicalObjectFifoFromMemrefOp>(
          //       rewriter.clone(*dmaOp.getSource().getDefiningOp()));
          //   operandMap.map(source.getOutput(), cloneOp.getOutput());
          // }
          // if (!operandMap.contains(target.getOutput())) {
          //   rewriter.setInsertionPoint(target);
          //   auto cloneOp = dyn_cast<AMDAIE::LogicalObjectFifoFromMemrefOp>(
          //       rewriter.clone(*dmaOp.getTarget().getDefiningOp()));
          //   operandMap.map(target.getOutput(), cloneOp.getOutput());
          // }
          builder.clone(*it, operandMap);
        } else {
          builder.clone(*it, operandMap);
        }
        // builder.clone(*it, operandMap);
      }
    }
    return success();
  }

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Only unroll loops only if inidcated to be unrolled earlier in the pass.
    // if (!forOp->getParentOfType<AMDAIE::WorkgroupOp>()) return failure();
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
      for (Value index : tileIndices)
        tileSet.insert(dyn_cast<AMDAIE::TileOp>(index.getDefiningOp()));
    }
  }
  tiles = tileSet.takeVector();
  return success();
}

/// TODO
LogicalResult distributeLocalMemoryAccess(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());

  SmallVector<AMDAIE::CoreOp> coreOps;
  moduleOp->walk([&](AMDAIE::CoreOp coreOp) { coreOps.push_back(coreOp); });

  for (AMDAIE::CoreOp coreOp : coreOps) {
    DenseMap<Value, std::tuple<AMDAIE::LogicalObjectFifoFromMemrefOp,
                               AMDAIE::MemoryAccess>>
        memrefToLogicalObjectFifo;
    // First walk to collect consume/produce DMA accesses and map respective
    // memrefs to logical objectifos.
    coreOp->walk([&](Operation *op) {
      // TODO(jornt): can we avoid produce/consume?
      if (auto consumeOp = dyn_cast<AMDAIE::LogicalObjectFifoConsume>(op)) {
        Value targetMemref =
            consumeOp.getDmaCpyNdOp().getTargetObjectFifo().getMemref();
        memrefToLogicalObjectFifo[targetMemref] =
            std::make_tuple(consumeOp.getDmaCpyNdOp().getTargetObjectFifo(),
                            AMDAIE::MemoryAccess::Read);
      } else if (auto produceOp =
                     dyn_cast<AMDAIE::LogicalObjectFifoProduce>(op)) {
        Value sourceMemref =
            produceOp.getDmaCpyNdOp().getSourceObjectFifo().getMemref();
        memrefToLogicalObjectFifo[sourceMemref] =
            std::make_tuple(produceOp.getDmaCpyNdOp().getSourceObjectFifo(),
                            AMDAIE::MemoryAccess::Write);
      }
    });

    WalkResult res = coreOp->walk([&](Operation *op) {
      if (isa<linalg::LinalgOp>(op)) {
        Operation *currOp = op;
        while (currOp->getParentOp() != coreOp) {
          currOp = currOp->getParentOp();
        }
        auto linalgOp = cast<linalg::LinalgOp>(op);
        for (auto &&[idx, operand] :
             llvm::enumerate(linalgOp->getOpOperands())) {
          if (memrefToLogicalObjectFifo.contains(operand.get())) {
            rewriter.setInsertionPointToStart(coreOp.getBody());
            std::tuple<AMDAIE::LogicalObjectFifoFromMemrefOp,
                       AMDAIE::MemoryAccess>
                value = memrefToLogicalObjectFifo[operand.get()];
            auto accessOp = rewriter.create<AMDAIE::LogicalObjectFifoAccessOp>(
                rewriter.getUnknownLoc(), std::get<0>(value),
                std::get<1>(value));
            linalgOp->setOperand(idx, accessOp);
          } else if (auto type =
                         llvm::dyn_cast<MemRefType>(operand.get().getType())) {
            Value memref = operand.get();
            bool hasSubViewOp = false;
            if (auto subViewOp = memref.getDefiningOp<memref::SubViewOp>()) {
              memref = subViewOp.getViewSource();
              type = cast<MemRefType>(memref.getType());
              hasSubViewOp = true;
            }
            rewriter.setInsertionPoint(coreOp);
            auto logicalObjectFifo =
                rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
                    rewriter.getUnknownLoc(), LogicalObjectFifoType::get(type),
                    memref);
            rewriter.setInsertionPointToStart(coreOp.getBody());
            auto accessOp = rewriter.create<AMDAIE::LogicalObjectFifoAccessOp>(
                rewriter.getUnknownLoc(), logicalObjectFifo,
                AMDAIE::MemoryAccess::Any);
            if (hasSubViewOp) {
              linalgOp->getOperand(idx).getDefiningOp()->setOperand(0,
                                                                    accessOp);
            } else {
              linalgOp->setOperand(idx, accessOp);
            }
          }
        }
      }
      return WalkResult::advance();
    });
    if (res.wasInterrupted()) return failure();
  }
  return success();
}

/// TODO
LogicalResult distributeLocalMemory(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());

  // Map from memref result to a new memref to be used, using the assigned tiles
  // as a key to map to different memrefs on different tiles.
  DenseMap<Value, DenseMap<SmallVector<std::tuple<int64_t, int64_t>>, Value,
                           LocationMapInfo>>
      memrefToNew;

  // Walk DMA ops and find the ones which are used in cores to update
  // source/target logical objectfifos
  moduleOp->walk([&](AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo) {
    // Skip non-local memories.
    Attribute memSpace = logicalObjectFifo.getMemorySpace();
    if (!memSpace || dyn_cast<IntegerAttr>(memSpace).getInt() != 2)
      return WalkResult::advance();

    // Recognize local memory distribution opportunity as a `memref.subview` on
    // a `memref.alloc`.
    auto subviewOp = dyn_cast<memref::SubViewOp>(
        logicalObjectFifo.getMemref().getDefiningOp());
    if (!subviewOp) return WalkResult::advance();

    auto allocOp =
        dyn_cast<memref::AllocOp>(subviewOp.getViewSource().getDefiningOp());
    if (!allocOp) return WalkResult::advance();

    if (!memrefToNew.contains(allocOp.getResult())) {
      DenseMap<SmallVector<std::tuple<int64_t, int64_t>>, Value,
               LocationMapInfo>
          value;
      memrefToNew[allocOp.getResult()] = value;
    }
    SmallVector<std::tuple<int64_t, int64_t>> locations =
        llvm::map_to_vector(logicalObjectFifo.getTiles(), [](Value res) {
          auto tile = dyn_cast<AMDAIE::TileOp>(res.getDefiningOp());
          return std::make_tuple(
              (int64_t)getConstantIntValue(tile.getCol()).value(),
              (int64_t)getConstantIntValue(tile.getRow()).value());
        });
    if (!memrefToNew[allocOp.getResult()].contains(locations)) {
      // Create new allocate.
      LLVM_DEBUG(llvm::dbgs() << "Create new allocate\n");
      // TODO(jornt): add more checks on subview to make sure access pattern is
      // ok for new allocate.
      rewriter.setInsertionPoint(allocOp);
      auto memRefType = cast<MemRefType>(subviewOp.getResult().getType());
      MemRefType allocType = MemRefType::get(
          memRefType.getShape(), memRefType.getElementType(),
          MemRefLayoutAttrInterface{}, memRefType.getMemorySpace());
      auto newAllocOp =
          rewriter.create<memref::AllocOp>(rewriter.getUnknownLoc(), allocType);
      auto newDeallocOp = rewriter.create<memref::DeallocOp>(
          rewriter.getUnknownLoc(), newAllocOp);
      newDeallocOp->moveBefore(&newAllocOp->getBlock()->back());
      memrefToNew[allocOp.getResult()][locations] = newAllocOp.getResult();
    }
    LLVM_DEBUG(llvm::dbgs() << "Distribute logical objectfifo: "
                            << logicalObjectFifo << "\n");
    // Update the `amdaie.logicalobjectfifo.from_memref` to use a dedicated
    // allocate instead of the subview.
    rewriter.setInsertionPoint(logicalObjectFifo);
    rewriter.replaceOpWithNewOp<AMDAIE::LogicalObjectFifoFromMemrefOp>(
        logicalObjectFifo,
        cast<LogicalObjectFifoType>(logicalObjectFifo.getOutput().getType()),
        memrefToNew[allocOp.getResult()][locations],
        logicalObjectFifo.getTiles());

    // Value memref = logicalObjectFifo.getMemref();
    // llvm::SmallSetVector<Value, 16> tiles;
    // for (Operation *userOp : memref.getDefiningOp()->getUsers()) {
    //   if (auto coreOp = userOp->getParentOfType<AMDAIE::CoreOp>()) {
    //     AMDAIE::TileOp tileOp = coreOp.getTileOp();
    //     rewriter.setInsertionPoint(logicalObjectFifo);
    //     auto newTileOp =
    //         dyn_cast<AMDAIE::TileOp>(rewriter.clone(*tileOp.getOperation()));
    //     tiles.insert(newTileOp.getResult());
    //   }
    // }
    // rewriter.setInsertionPoint(logicalObjectFifo);
    // rewriter.replaceOpWithNewOp<AMDAIE::LogicalObjectFifoFromMemrefOp>(
    //     logicalObjectFifo,
    //     cast<LogicalObjectFifoType>(logicalObjectFifo.getOutput().getType()),
    //     logicalObjectFifo.getMemref(), tiles.takeVector());
    return WalkResult::advance();
  });
  return success();
}

void findUsersInCoreAndAddTiles(
    IRRewriter &rewriter, Operation *op,
    AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo,
    llvm::SmallSetVector<Value, 16> &tiles) {
  for (Operation *userOp : op->getUsers()) {
    if (auto coreOp = userOp->getParentOfType<AMDAIE::CoreOp>()) {
      AMDAIE::TileOp tileOp = coreOp.getTileOp();
      rewriter.setInsertionPoint(logicalObjectFifo);
      auto newTileOp =
          dyn_cast<AMDAIE::TileOp>(rewriter.clone(*tileOp.getOperation()));
      tiles.insert(newTileOp.getResult());
    }
    // if (auto subviewOp = dyn_cast<memref::SubViewOp>(userOp)) {
    //   findUsersInCoreAndAddTiles(rewriter, subviewOp, logicalObjectFifo,
    //   tiles);
    // } else if (auto userLogicalObjectFifo =
    //                dyn_cast<AMDAIE::LogicalObjectFifoFromMemrefOp>(userOp)) {
    //   findUsersInCoreAndAddTiles(rewriter, userLogicalObjectFifo,
    //                              logicalObjectFifo, tiles);
    // }
    findUsersInCoreAndAddTiles(rewriter, userOp, logicalObjectFifo, tiles);
  }
}

/// Assign tiles to the logical objectfifos with local memory space (L1).
/// The tiles are derived from the usage of the logical objectfifos within
/// core operations, which are already assigned a tile location.
LogicalResult assignLocalAieTiles(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());

  // Walk DMA ops and find the ones which are used in cores to update
  // source/target logical objectfifos
  moduleOp->walk([&](AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo) {
    Attribute memSpace = logicalObjectFifo.getMemorySpace();
    if (!memSpace || dyn_cast<IntegerAttr>(memSpace).getInt() != 2)
      return WalkResult::advance();

    llvm::SmallSetVector<Value, 16> tiles;
    findUsersInCoreAndAddTiles(rewriter, logicalObjectFifo, logicalObjectFifo,
                               tiles);
    // Handle subviews. Refactor.
    for (Operation *userOp :
         logicalObjectFifo.getMemref().getDefiningOp()->getUsers()) {
      if (auto subviewOp = dyn_cast<memref::SubViewOp>(userOp)) {
        findUsersInCoreAndAddTiles(rewriter, subviewOp, logicalObjectFifo,
                                   tiles);
      }
    }
    rewriter.setInsertionPoint(logicalObjectFifo);
    rewriter.replaceOpWithNewOp<AMDAIE::LogicalObjectFifoFromMemrefOp>(
        logicalObjectFifo,
        cast<LogicalObjectFifoType>(logicalObjectFifo.getOutput().getType()),
        logicalObjectFifo.getMemref(), tiles.takeVector());
    return WalkResult::advance();
  });

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

///
LogicalResult distributeSharedMemory(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());

  // Map from local objectfifos found to the tiles where they are used
  DenseMap<SmallVector<std::tuple<int64_t, int64_t>>, Value, LocationMapInfo>
      locationsToMemref;

  auto comparator = [](Value a, Value b) -> bool {
    TileOp tileA = dyn_cast<AMDAIE::TileOp>(a.getDefiningOp());
    TileOp tileB = dyn_cast<AMDAIE::TileOp>(b.getDefiningOp());
    int64_t colA = getConstantIntValue(tileA.getCol()).value();
    int64_t rowA = getConstantIntValue(tileA.getRow()).value();
    int64_t colB = getConstantIntValue(tileB.getCol()).value();
    int64_t rowB = getConstantIntValue(tileB.getRow()).value();
    if (colA == colB) return rowA < rowB;
    return colA < colB;
  };

  // TODO
  moduleOp->walk([&](AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo) {
    Attribute memSpace = logicalObjectFifo.getMemorySpace();
    if (!memSpace || dyn_cast<IntegerAttr>(memSpace).getInt() != 1)
      return WalkResult::advance();

    SmallVector<AMDAIE::TileOp> tiles =
        llvm::map_to_vector(logicalObjectFifo.getTiles(), [](Value tile) {
          return dyn_cast<AMDAIE::TileOp>(tile.getDefiningOp());
        });
    llvm::sort(tiles.begin(), tiles.end(), comparator);

    SmallVector<AMDAIE::TileOp> targetTiles;
    (void)getUserTiles<CopyOpOperateOn::Target>(logicalObjectFifo, targetTiles);
    llvm::sort(targetTiles.begin(), targetTiles.end(), comparator);
    tiles.insert(tiles.end(), std::make_move_iterator(targetTiles.begin()),
                 std::make_move_iterator(targetTiles.end()));

    SmallVector<AMDAIE::TileOp> sourceTiles;
    (void)getUserTiles<CopyOpOperateOn::Source>(logicalObjectFifo, sourceTiles);
    llvm::sort(sourceTiles.begin(), sourceTiles.end(), comparator);
    tiles.insert(tiles.end(), std::make_move_iterator(sourceTiles.begin()),
                 std::make_move_iterator(sourceTiles.end()));
    llvm::outs() << "Op: " << logicalObjectFifo << ", TILES: " << tiles.size()
                 << "\n";

    SmallVector<std::tuple<int64_t, int64_t>> locations =
        llvm::map_to_vector(tiles, [](AMDAIE::TileOp tile) {
          return std::make_tuple(
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
  if (failed(distributeLocalMemoryHack(moduleOp))) {
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs() << "Module after distributeLocalMemory: \n"
                          << moduleOp << "\n");

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
  LLVM_DEBUG(llvm::dbgs() << "Module after AMDAIEUnrollWorkgroupLoops: \n"
                          << moduleOp << "\n");
  if (failed(distributeLocalMemoryAccess(moduleOp))) {
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs() << "Module after distributeLocalMemoryAccess: \n"
                          << moduleOp << "\n");
  // Assign tile locations to logical objectfifos on local (L1) memory.
  if (failed(assignLocalAieTiles(moduleOp))) {
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs() << "Module after assignLocalAieTiles: \n"
                          << moduleOp << "\n");
  // Assign tile locations to the remaining logical objectfifos.
  RewritePatternSet assignAieTilePatters(context);
  assignAieTilePatters.insert<AssignAieTiles>(context);
  if (failed(applyPatternsAndFoldGreedily(moduleOp,
                                          std::move(assignAieTilePatters)))) {
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs() << "Module after AssignAieTiles: \n"
                          << moduleOp << "\n");
  //
  if (failed(distributeSharedMemory(moduleOp))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEDistributeCoresAndObjectFifosPass() {
  return std::make_unique<AMDAIEDistributeCoresAndObjectFifosPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
