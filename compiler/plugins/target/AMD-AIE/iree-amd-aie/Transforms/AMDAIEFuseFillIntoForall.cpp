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
  mlir::FunctionOpInterface funcOp = getOperation();
  IRRewriter rewriter(context);

  // Find a unique FillOp with a single output, or return.
  SmallVector<linalg::FillOp> fillOps;
  funcOp->walk([&](linalg::FillOp fillOp) { fillOps.push_back(fillOp); });
  if (fillOps.size() != 1) return;
  linalg::FillOp fillOp = fillOps[0];
  if (fillOp.getResults().size() != 1) return;

  // Confirm that there is a unique user that is a forall, and match
  // the block argument that is used by the fill op, or return.
  ResultRange::use_range fillUses = fillOp->getUses();
  if (std::distance(fillUses.begin(), fillUses.end()) != 1) return;
  OpOperand &fillUse = *fillUses.begin();
  auto forallOp = dyn_cast<scf::ForallOp>(fillUse.getOwner());
  if (!forallOp) return;
  BlockArgument bbArg = forallOp.getTiedBlockArgument(&fillUse);

  // Find 0 or 1 ExtractSliceOps that use the fill result, or return.
  tensor::ExtractSliceOp extractSliceOp;
  for (Operation *user : bbArg.getUsers()) {
    if (auto nxt = dyn_cast<tensor::ExtractSliceOp>(user)) {
      if (extractSliceOp) return;
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
      funcOp->emitOpError("Failed to fuse fill op into forall loop.");
      return signalPassFailure();
    }
    return;
  }

  // In the case where there are no extract_slice ops, we manually create the
  // fill at the beginning of the forall body.
  assert(!extractSliceOp);
  rewriter.setInsertionPointToStart(forallOp.getBody());
  Value scalar = fillOp.value();
  Location loc = fillOp.getLoc();
  auto fusedFill = rewriter.create<linalg::FillOp>(loc, scalar, bbArg);
  rewriter.replaceUsesWithIf(
      bbArg, fusedFill.getResult(0), [&](OpOperand &operand) {
        Operation *owner = operand.getOwner();
        if (owner == fusedFill) {
          return false;
        } else if (isa<tensor::ParallelInsertSliceOp>(owner)) {
          return false;
        } else {
          return true;
        }
      });

  // Do not use the result of the old fill.
  rewriter.replaceAllUsesWith(fillOp.getResults()[0], fillOp.getOutputs()[0]);
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEFuseFillIntoForallPass() {
  return std::make_unique<AMDAIEFuseFillIntoForallPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
