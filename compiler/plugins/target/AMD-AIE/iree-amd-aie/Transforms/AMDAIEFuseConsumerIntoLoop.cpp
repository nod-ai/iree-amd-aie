// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-fuse-consumer-into-loop"
namespace mlir::iree_compiler::AMDAIE {
namespace {

class AMDAIEFuseConsumerIntoLoopPass
    : public impl::AMDAIEFuseConsumerIntoLoopBase<
          AMDAIEFuseConsumerIntoLoopPass> {
 public:
  AMDAIEFuseConsumerIntoLoopPass() = default;
  AMDAIEFuseConsumerIntoLoopPass(const AMDAIEFuseConsumerIntoLoopPass &pass) {}
  AMDAIEFuseConsumerIntoLoopPass(
      const AMDAIEFuseConsumerIntoLoopOptions &options)
      : AMDAIEFuseConsumerIntoLoopBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }
  void runOnOperation() override;
};

void AMDAIEFuseConsumerIntoLoopPass::runOnOperation() {
  MLIRContext *context = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();
  IRRewriter rewriter(context);

  RewritePatternSet patterns(context);
  tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, context);
  tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, context);
  FrozenRewritePatternSet canonicalizationPatterns(std::move(patterns));

  // Step 1. The depth until which we would keep fusing consumer chain.
  // TODO(avarma): This should also be part of KernelDispatch logic.
  unsigned fuseDepth = 1;
  // Check if there is matmul-elementwise fusion opportunity. If so, overwrite
  // the `fuseDepth` to be 2.
  funcOp->walk<WalkOrder::PostOrder, ReverseIterator>([&](linalg::LinalgOp op) {
    if (isMatmulProducerOfElementwise(op)) {
      fuseDepth = 2;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  // Step 2. Find the first scf loop in postorder walk.
  Operation *scfLoopOp = nullptr;
  funcOp->walk<WalkOrder::PostOrder, ReverseIterator>(
      [&](LoopLikeOpInterface op) {
        if (isa<scf::ForOp, scf::ForallOp>(op)) {
          scfLoopOp = op;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  if (!scfLoopOp) {
    LLVM_DEBUG(llvm::dbgs()
               << "There is no scf.for/forall loop to fuse with\n");
    return;
  }

  // Step 3. Search the compute op and its consumer slices.
  Operation *computeOp;
  scfLoopOp->walk<WalkOrder::PostOrder, ReverseIterator>(
      [&](linalg::LinalgOp op) {
        computeOp = op;
        return WalkResult::interrupt();
      });
  if (!computeOp) {
    LLVM_DEBUG(llvm::dbgs() << "Could not find any compute op\n");
    return;
  }

  // Step 4. Greedily fuse the consumer ops for a specified fusion depth and
  // while fusion keeps occurring or until the maximum number of iterations is
  // exceeded.
  bool changed{true};
  int64_t iter{0};
  while (changed && iter < maxIterations) {
    changed = false;
    // Canonicalize before every iteration to enable more back-to-back fusion
    // opportunities.
    (void)applyPatternsGreedily(funcOp, canonicalizationPatterns);
    Operation *producerOp = computeOp;
    // TODO(jornt): Refactor fuseDepth to avoid hardcoding and fuse greedily
    // with any depth instead.
    for (unsigned depth = 1; depth <= fuseDepth; depth++) {
      do {
        ResultRange results = producerOp->getResults();
        SmallVector<Operation *> allUsers = std::accumulate(
            results.begin(), results.end(), SmallVector<Operation *>{},
            [](SmallVector<Operation *> init, OpResult res) {
              for (Operation *op : res.getUsers()) init.push_back(op);
              return init;
            });
        if (allUsers.size() != 1) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Expected only one user of the compute op\n");
          break;
        }

        Operation *candidateSliceOp = allUsers[0];
        if (!(isa<tensor::InsertSliceOp, tensor::ParallelInsertSliceOp>(
                candidateSliceOp))) {
          producerOp = producerOp->getParentOfType<LoopLikeOpInterface>();
          LLVM_DEBUG(
              llvm::dbgs()
              << "Going to reattempt fusion because didn't find "
                 "tensor.insert_slice/tensor.parallel_insert_slice as the "
                 "user of the compute op\n");
          continue;
        }
        std::optional<scf::SCFFuseConsumerOfSliceResult> fusedConsumer =
            scf::tileAndFuseConsumerOfSlice(rewriter, candidateSliceOp);
        if (!fusedConsumer) {
          producerOp = producerOp->getParentOfType<LoopLikeOpInterface>();
          LLVM_DEBUG(llvm::dbgs()
                     << "Failed to fuse any consumer op into the producer. "
                        "Reattempt with loop-like parent operation.\n");
          continue;
        }
        changed = true;
        fusedConsumer->origConsumerOperand->getOwner()->erase();
        Operation *fusedOp =
            fusedConsumer->tiledAndFusedConsumerOperand->getOwner();
        if (getAncestorInBlock(fusedOp, computeOp->getBlock()) != nullptr) {
          // The consumer is fused all the way into the producer's block, so
          // operate on this op from now on, but with reduced depth.
          computeOp = fusedOp;
          fuseDepth -= 1;
        }
        producerOp = fusedOp;
        break;
      } while (producerOp && producerOp->getParentOp() != funcOp);
    }
    iter++;
  }
  if (iter >= maxIterations) {
    funcOp.emitOpError() << "Maximum number of iterations reached, consumer "
                            "fusion is likely stuck in an infinite loop.";
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEFuseConsumerIntoLoopPass(
    AMDAIEFuseConsumerIntoLoopOptions options) {
  return std::make_unique<AMDAIEFuseConsumerIntoLoopPass>(options);
}
}  // namespace mlir::iree_compiler::AMDAIE
