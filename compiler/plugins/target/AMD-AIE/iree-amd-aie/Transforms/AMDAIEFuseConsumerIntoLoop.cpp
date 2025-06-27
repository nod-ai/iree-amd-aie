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

template <typename ComputeOp>
Operation *findPostOrderComputeOpInLoop(LoopLikeOpInterface loop) {
  Operation *computeOp = nullptr;
  loop->walk<WalkOrder::PostOrder, ReverseIterator>([&](ComputeOp op) {
    if (op != loop) {
      computeOp = op;
      return WalkResult::interrupt();
    }
    return WalkResult::interrupt();
  });
  return computeOp;
}

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
    if (isElementwiseWithMatmulProducer(op)) {
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
  LoopLikeOpInterface loop = cast<LoopLikeOpInterface>(scfLoopOp);

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

  // Step 4. Find outermost nested scf loop and maintain the loop nest count.
  int64_t loopNestDepth = 1;
  do {
    ResultRange results = loop->getResults();
    SmallVector<Operation *> allUsers = std::accumulate(
        results.begin(), results.end(), SmallVector<Operation *>{},
        [](SmallVector<Operation *> init, OpResult res) {
          for (Operation *op : res.getUsers()) init.push_back(op);
          return init;
        });
    if (allUsers.size() != 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Expected only one user of the compute/loop op.\n");
      return;
    }
    if (!(isa<tensor::InsertSliceOp, tensor::ParallelInsertSliceOp>(
            allUsers[0]))) {
      break;
    }
    computeOp = loop;
    loop = loop->getParentOfType<LoopLikeOpInterface>();
    loopNestDepth++;
  } while (loop);

  if (!loop) {
    LLVM_DEBUG(llvm::dbgs() << "Could not find the outermost scf loop from "
                               "where consumer fusion can begin.\n");
    return;
  }
  // Step 5. Greedily fuse the consumer ops for a specified fusion depth at each
  // level of the loopnest. While doing so we maintain `changeLocal` to track if
  // consumer fusion has taken place at a particular loopnest; and
  // `changeGlobal` to track if at least one consumer fusion has taken place at
  // any level of the loopnest.
  bool changedGlobal{false}, changedLocal{true};
  while (loopNestDepth != 0) {
    changedLocal = false;
    if (loopNestDepth > 1) {
      computeOp = findPostOrderComputeOpInLoop<LoopLikeOpInterface>(loop);
    } else {
      computeOp = findPostOrderComputeOpInLoop<linalg::LinalgOp>(loop);
    }
    assert(computeOp && "could not find either a scf loop or a linalg.generic");
    // TODO(jornt): Refactor fuseDepth to avoid hardcoding and fuse greedily
    // with any depth instead.
    for (unsigned depth = 1; depth <= fuseDepth; depth++) {
      do {
        ResultRange results = computeOp->getResults();
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
          break;
        }
        std::optional<scf::SCFFuseConsumerOfSliceResult> fusedConsumer =
            scf::tileAndFuseConsumerOfSlices(rewriter, candidateSliceOp,
                                             MutableArrayRef(loop));
        if (!fusedConsumer) {
          break;
        }
        changedLocal = true;
        changedGlobal = true;
        fusedConsumer->origConsumerOperands.front()->getOwner()->erase();
        computeOp =
            fusedConsumer->tiledAndFusedConsumerOperands.front()->getOwner();
        break;
      } while (computeOp && computeOp->getParentOp() != funcOp);
    }
    if (changedLocal == false) break;
    if (loopNestDepth > 1) {
      loop = cast<LoopLikeOpInterface>(
          findPostOrderComputeOpInLoop<LoopLikeOpInterface>(
              computeOp->getParentOfType<LoopLikeOpInterface>()));
    }
    loopNestDepth--;
    // Canonicalize before every iteration to enable more back-to-back fusion
    // opportunities.
    (void)applyPatternsGreedily(funcOp, canonicalizationPatterns);
  }
  if (!changedGlobal) {
    LLVM_DEBUG(llvm::dbgs()
               << "Failed to fuse any consumer op into the producer.");
    return;
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEFuseConsumerIntoLoopPass(
    AMDAIEFuseConsumerIntoLoopOptions options) {
  return std::make_unique<AMDAIEFuseConsumerIntoLoopPass>(options);
}
}  // namespace mlir::iree_compiler::AMDAIE
