// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
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
  func::FuncOp funcOp = getOperation();
  IRRewriter rewriter(context);

  // Find the producer op, in this case is linalg.fill.
  TilingInterface tileableProducer;
  funcOp->walk([&](TilingInterface op) {
    if (isa<linalg::FillOp>(op)) {
      tileableProducer = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (!tileableProducer) {
    LLVM_DEBUG(llvm::dbgs() << "There is no producer op to be fused.\n");
    return;
  }

  // Search the first use by a scf::ForallOp user.
  scf::ForallOp forallOp;
  auto itProducerUses =
      llvm::find_if(tileableProducer->getUses(), [&](OpOperand &use) {
        forallOp = dyn_cast<scf::ForallOp>(use.getOwner());
        return forallOp;
      });
  if (!forallOp) {
    LLVM_DEBUG(llvm::dbgs() << "There is no forall Op.\n");
    return;
  }

  // Search the producer slices accessed within the Forall op.
  OpOperand *pUse = &(*itProducerUses);
  BlockArgument bbArg = forallOp.getTiedBlockArgument(pUse);

  auto itBBArgUsers = llvm::find_if(bbArg.getUsers(), [&](Operation *user) {
    auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
    return sliceOp;
  });
  if (itBBArgUsers == bbArg.getUsers().end()) {
    funcOp->emitOpError("There is no extract tensor slice.");
    return signalPassFailure();
  }
  auto sliceOpToTile = cast<tensor::ExtractSliceOp>(*itBBArgUsers);

  LoopLikeOpInterface loops =
      cast<LoopLikeOpInterface>(forallOp.getOperation());

  // Materialize the slice of the producer in place.
  std::optional<scf::SCFFuseProducerOfSliceResult> fusedProducer =
      scf::tileAndFuseProducerOfSlice(rewriter, sliceOpToTile,
                                      MutableArrayRef(&loops, 1));
  if (!fusedProducer) {
    funcOp->emitOpError("Failed to fuse fill op into forall loop.");
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEFuseFillIntoForallPass() {
  return std::make_unique<AMDAIEFuseFillIntoForallPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
