// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"

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

  // The depth until which we would keep fusing consumer chain.
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

  // Based on the `fuseDepth`, we would greedily fuse the consumer ops.
  for (unsigned depth = 1; depth <= fuseDepth; depth++) {
    // Walk through the graph in post order and find the loop.
    Operation *scfLoopOp = nullptr;
    funcOp->walk<WalkOrder::PostOrder, ReverseIterator>(
        [&](LoopLikeOpInterface op) {
          if (isa<scf::ForOp>(op) && useSCFFor) {
            scfLoopOp = op;
            return WalkResult::interrupt();
          } else if (isa<scf::ForallOp>(op) && !useSCFFor) {
            scfLoopOp = op;
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });

    if (!scfLoopOp) {
      LLVM_DEBUG(llvm::dbgs()
                 << "There is no scf.for/forall loop to fuse with.\n");
      return;
    }

    // Search the compute op and its consumer slices.
    linalg::LinalgOp linalgOp;
    scfLoopOp->walk<WalkOrder::PostOrder, ReverseIterator>(
        [&](linalg::LinalgOp op) {
          linalgOp = op;
          return WalkResult::interrupt();
        });

    if (!linalgOp) {
      LLVM_DEBUG(llvm::dbgs() << "----- Could not find any compute op \n");
      return;
    }

    Value::user_range users = linalgOp->getResult(0).getUsers();
    if (!llvm::hasSingleElement(users)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "----- Expected only one user of the compute op : "
                 << linalgOp << "\n");
      return;
    }

    Operation *terminatorStoreOp = *(users.begin());
    if (!(isa<tensor::InsertSliceOp, tensor::ParallelInsertSliceOp>(
            terminatorStoreOp))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "----- Expected either tensor.insert_slice OR "
                    "tensor.parallel_insert_slice to be the only user of the "
                    "compute op : "
                 << linalgOp << "\n");
      return;
    }

    std::optional<scf::SCFFuseConsumerOfSliceResult> fusedConsumer =
        scf::tileAndFuseConsumerOfSlice(rewriter, terminatorStoreOp);
    if (!fusedConsumer) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "Failed to fuse any consumer op into the producer scf loop\n");
      return;
    }
    fusedConsumer->origConsumerOperand->getOwner()->erase();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEFuseConsumerIntoLoopPass(
    AMDAIEFuseConsumerIntoLoopOptions options) {
  return std::make_unique<AMDAIEFuseConsumerIntoLoopPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
