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

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }
  void runOnOperation() override;
};

void AMDAIEFuseConsumerIntoLoopPass::runOnOperation() {
  MLIRContext *context = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();
  IRRewriter rewriter(context);

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
        if (isa<scf::ForOp>(op)) {
          scfLoopOp = op;
          return WalkResult::interrupt();
        } else if (isa<scf::ForallOp>(op)) {
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
  linalg::LinalgOp linalgOp;
  scfLoopOp->walk<WalkOrder::PostOrder, ReverseIterator>(
      [&](linalg::LinalgOp op) {
        linalgOp = op;
        return WalkResult::interrupt();
      });
  if (!linalgOp) {
    LLVM_DEBUG(llvm::dbgs() << "Could not find any compute op\n");
    return;
  }

  Operation *computeOp = linalgOp;
  // Step 4. Based on the `fuseDepth`, we would greedily fuse the consumer ops.
  for (unsigned depth = 1; depth <= fuseDepth; depth++) {
    LLVM_DEBUG(llvm::dbgs() << "Compute op = " << (*computeOp) << "\n");
    do {
      Value::user_range users = computeOp->getResult(0).getUsers();
      if (!llvm::hasSingleElement(users)) {
        computeOp->emitOpError("Expected only one user of the compute op");
        return signalPassFailure();
      }

      Operation *terminatorStoreOp = *(users.begin());
      if (!(isa<tensor::InsertSliceOp, tensor::ParallelInsertSliceOp>(
              terminatorStoreOp))) {
        computeOp = computeOp->getParentOfType<LoopLikeOpInterface>();
        LLVM_DEBUG(llvm::dbgs()
                   << "Going to reattempt fusion because didn't find "
                      "tensor.insert_slice/tensor.parallel_insert_slice as the "
                      "user of the compute op\n");
        continue;
      }
      std::optional<scf::SCFFuseConsumerOfSliceResult> fusedConsumer =
          scf::tileAndFuseConsumerOfSlice(rewriter, terminatorStoreOp);
      if (!fusedConsumer) {
        terminatorStoreOp->emitOpError(
            "Failed to fuse any consumer op into the producer");
        return signalPassFailure();
      }
      fusedConsumer->origConsumerOperand->getOwner()->erase();
      computeOp = fusedConsumer->tiledAndFusedConsumerOperand->getOwner();
      break;
    } while (computeOp && computeOp->getParentOp() != funcOp);
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEFuseConsumerIntoLoopPass() {
  return std::make_unique<AMDAIEFuseConsumerIntoLoopPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
