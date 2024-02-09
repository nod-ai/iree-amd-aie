// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-fuse-pack-into-for"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIEFusePackIntoForLoopPass
    : public impl::AMDAIEFusePackIntoForLoopBase<
          AMDAIEFusePackIntoForLoopPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  AMDAIEFusePackIntoForLoopPass() = default;
  AMDAIEFusePackIntoForLoopPass(const AMDAIEFusePackIntoForLoopPass &pass){};
  void runOnOperation() override;
};

void AMDAIEFusePackIntoForLoopPass::runOnOperation() {
  MLIRContext *context = &getContext();
  func::FuncOp funcOp = getOperation();
  IRRewriter rewriter(context);

  // Walk through the graph in post order and find the for loop.
  scf::ForOp forOp;
  funcOp->walk<WalkOrder::PostOrder, ReverseIterator>(
      [&](LoopLikeOpInterface op) {
        if (isa<scf::ForOp>(op)) {
          forOp = cast<scf::ForOp>(op);
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  if (!forOp) {
    LLVM_DEBUG(llvm::dbgs() << "There is no for loop to fuse with.\n");
    return;
  }

  // Search the compute op and its producer slices within the For loop.
  BlockArgument bbArg = forOp.getRegionIterArg(0);
  SmallVector<tensor::ExtractSliceOp> sliceOps;
  for (auto user : bbArg.getUsers()) {
    if (auto genericOp = dyn_cast<linalg::GenericOp>(user)) {
      for (auto [index, operand] : llvm::enumerate(genericOp.getOperands())) {
        if (!isa<BlockArgument>(operand)) {
          auto defOp = operand.getDefiningOp();
          if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(defOp)) {
            sliceOps.push_back(sliceOp);
          }
        }
      }
    }
  }

  if (sliceOps.empty()) {
    funcOp->emitOpError("There is no extract tensor slice.");
    return signalPassFailure();
  }

  // Materialize each slice of the producer in place.
  LoopLikeOpInterface loops = cast<LoopLikeOpInterface>(forOp.getOperation());
  for (auto sliceOp : sliceOps) {
    auto defOp = sliceOp.getOperand(0).getDefiningOp();
    if (!isa<tensor::PackOp>(defOp)) {
      LLVM_DEBUG(llvm::dbgs() << "The producer is not a pack op.\n");
      continue;
    }
    std::optional<scf::SCFFuseProducerOfSliceResult> fusedProducer =
        scf::tileAndFuseProducerOfSlice(rewriter, sliceOp,
                                        MutableArrayRef(&loops, 1));
    if (!fusedProducer) {
      funcOp->emitOpError("Failed to fuse pack ops into for loop.");
      return signalPassFailure();
    }
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEFusePackIntoForLoopPass() {
  return std::make_unique<AMDAIEFusePackIntoForLoopPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
