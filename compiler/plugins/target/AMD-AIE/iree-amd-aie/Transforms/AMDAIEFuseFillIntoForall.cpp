// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-fuse-fill-into-forall"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIEFuseFillIntoForallPass
    : public impl::AMDAIEFuseFillIntoForallBase<AMDAIEFuseFillIntoForallPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  AMDAIEFuseFillIntoForallPass() = default;
  AMDAIEFuseFillIntoForallPass(const AMDAIEFuseFillIntoForallPass &pass){};
  void runOnOperation() override;
};

void AMDAIEFuseFillIntoForallPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  // Find a unique FillOp with a single output, or return.
  SmallVector<linalg::FillOp> fillOps;
  getOperation()->walk(
      [&](linalg::FillOp fillOp) { fillOps.push_back(fillOp); });
  if (fillOps.size() != 1) {
    LLVM_DEBUG(llvm::dbgs() << "Expected exactly 1 fill op, but found "
                            << fillOps.size() << ".\n");
    return;
  }

  linalg::FillOp fillOp = fillOps[0];
  if (fillOp.getResults().size() != 1) {
    LLVM_DEBUG(llvm::dbgs() << "Expected fill op to have exactly 1 result, but "
                            << "found " << fillOp.getResults().size() << ".\n");

    return;
  };

  // Confirm that there is a unique user that is a forall, and match
  // the block argument that is used by the fill op, or return.
  if (!fillOp->hasOneUse()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Expected exactly 1 use of fill op, but found 0 or 2+.");
    return;
  }
  OpOperand &fillUse = *fillOp->getUses().begin();
  auto forallOp = dyn_cast<scf::ForallOp>(fillUse.getOwner());
  if (!forallOp) {
    LLVM_DEBUG(llvm::dbgs() << "Expected fill op to be used by a forall op, "
                            << "but unique user is "
                            << fillUse.getOwner()->getName() << ".\n");
    return;
  }
  BlockArgument bbArg = forallOp.getTiedBlockArgument(&fillUse);

  // Find 0 or 1 ExtractSliceOps that use the fill result, or return.
  tensor::ExtractSliceOp extractSliceOp;
  for (Operation *user : bbArg.getUsers()) {
    if (auto nxt = dyn_cast<tensor::ExtractSliceOp>(user)) {
      if (extractSliceOp) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Expected at most 1 extract_slice op, but found 2+.\n");
        return;
      }
      extractSliceOp = nxt;
    }
  }

  if (extractSliceOp) {
    LoopLikeOpInterface loops =
        cast<LoopLikeOpInterface>(forallOp.getOperation());

    // Materialize the slice of the producer in place.
    std::optional<scf::SCFFuseProducerOfSliceResult> fusedProducer =
        scf::tileAndFuseProducerOfSlice(rewriter, extractSliceOp,
                                        MutableArrayRef(&loops, 1));
    if (!fusedProducer) {
      fillOp->emitOpError("could not be fused into forall");
      return signalPassFailure();
    }
  } else {
    // In the case where there are no extract_slice ops, we manually create the
    // fill at the beginning of the forall body. This situation might arise
    // if the extract_slice has been folded, for example if the forall is
    // over a grid if size 1.
    rewriter.setInsertionPointToStart(forallOp.getBody());
    auto fusedFill =
        rewriter.create<linalg::FillOp>(fillOp.getLoc(), fillOp.value(), bbArg);
    rewriter.replaceUsesWithIf(
        bbArg, fusedFill.getResult(0), [&](OpOperand &operand) {
          Operation *owner = operand.getOwner();
          if (owner == fusedFill || isa<tensor::ParallelInsertSliceOp>(owner)) {
            return false;
          }
          return true;
        });

    // Do not use the result of the old fill.
    rewriter.replaceAllUsesWith(fillOp.getResults()[0], fillOp.getOutputs()[0]);
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEFuseFillIntoForallPass() {
  return std::make_unique<AMDAIEFuseFillIntoForallPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
