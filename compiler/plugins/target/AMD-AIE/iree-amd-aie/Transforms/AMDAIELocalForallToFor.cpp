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
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-local-forall-to-for"

namespace mlir::iree_compiler::AMDAIE {

// Used to annotate loops that should be unrolled.
static const llvm::StringLiteral kAMDAIELoopUnroll = "amdaie.unroll";

namespace {

/// Convert inner scf.forall ops chosen for parallel distribution to scf.for
/// ops.
LogicalResult localForallToFor(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
  WalkResult res = moduleOp->walk([&](scf::ForallOp forallOp) {
    SmallVector<Attribute> mapping =
        llvm::to_vector(forallOp.getMapping()->getValue());
    // We index on thread mapping for core and dma unrolling and buffer
    // distribution.
    if (!isa<mlir::gpu::GPUThreadMappingAttr>(*mapping.begin()))
      return WalkResult::advance();

    SmallVector<Operation *> loopResults;
    if (failed(scf::forallToForLoop(rewriter, forallOp, &loopResults))) {
      forallOp.emitOpError() << "failed to transform scf.forall to scf.for";
      return WalkResult::interrupt();
    }
    // Set attribute to unroll this loop later in this pass.
    for (Operation *loopRes : loopResults) {
      scf::ForOp forOp = dyn_cast<scf::ForOp>(loopRes);
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

class AMDAIELocalForallToForPass
    : public impl::AMDAIELocalForallToForBase<AMDAIELocalForallToForPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIELocalForallToForPass() = default;
  AMDAIELocalForallToForPass(const AMDAIELocalForallToForPass &pass){};
  void runOnOperation() override;
};

void AMDAIELocalForallToForPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  // Convert local scf.forall operations selected for parallel distribution to
  // nested scf.for operations.
  if (failed(localForallToFor(moduleOp))) {
    moduleOp.emitOpError()
        << "local `scf.forall` to `scf.for` conversion failed";
    return signalPassFailure();
  }
  // Hoist the affine apply ops on scf.for induction variables to the
  // corresponding scf.for's body.
  if (failed(hoistAffineApplyDependingOnFor(moduleOp))) {
    moduleOp.emitOpError() << "`affine.apply` hoisting failed";
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIELocalForallToForPass() {
  return std::make_unique<AMDAIELocalForallToForPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
