// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#define DEBUG_TYPE "iree-amdaie-unroll-local-loops"

namespace mlir::iree_compiler::AMDAIE {

// Used to annotate loops that were unrolled.
// TODO(avarma): Shift this to AMDAIEUtils.h.
static const llvm::StringLiteral kAMDAIELoopUnroll = "amdaie.unroll";

namespace {

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
    for (auto i = lbInt + stepInt; i < ubInt; i += stepInt) {
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

class AMDAIEUnrollLocalLoopsPass
    : public impl::AMDAIEUnrollLocalLoopsBase<AMDAIEUnrollLocalLoopsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEUnrollLocalLoopsPass() = default;
  AMDAIEUnrollLocalLoopsPass(const AMDAIEUnrollLocalLoopsPass &pass){};
  void runOnOperation() override;
};

void AMDAIEUnrollLocalLoopsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp moduleOp = getOperation();

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
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEUnrollLocalLoopsPass() {
  return std::make_unique<AMDAIEUnrollLocalLoopsPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
