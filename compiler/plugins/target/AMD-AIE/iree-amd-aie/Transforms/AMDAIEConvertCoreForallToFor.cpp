// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file the conversion of `scf.forall` into `scf.for` with `aie.core`
// operations. The resulting `scf.for` loops are coalesced as well into a single
// loop. This is needed for now for further lowering as `scf.forall` isn't
// supported in the further lowering of `aie.core`.
//
//===----------------------------------------------------------------------===//

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"

#define DEBUG_TYPE "iree-amdaie-convert-core-forall-to-for"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Converts `scf.forall` into nested `scf.for` and then coalesce the `scf.for`
/// loops.
LogicalResult coreForallToFor(RewriterBase &rewriter, AMDAIE::CoreOp coreOp) {
  WalkResult res = coreOp->walk([&](scf::ForallOp forallOp) {
    SmallVector<Operation *> forOpResults;
    if (failed(scf::forallToForLoop(rewriter, forallOp, &forOpResults))) {
      coreOp.emitOpError() << "failed to transform scf.forall to scf.for";
      return WalkResult::interrupt();
    }

    SmallVector<scf::ForOp> forOps = llvm::map_to_vector(
        forOpResults,
        [](Operation *loop) { return dyn_cast<scf::ForOp>(loop); });
    if (failed(coalesceLoops(rewriter, forOps))) {
      coreOp.emitOpError() << "failed to coalesce for loops";
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return failure();
  return success();
}

class AMDAIEConvertCoreForallToForPass
    : public impl::AMDAIEConvertCoreForallToForBase<
          AMDAIEConvertCoreForallToForPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, affine::AffineDialect>();
  }

  AMDAIEConvertCoreForallToForPass() = default;
  AMDAIEConvertCoreForallToForPass(
      const AMDAIEConvertCoreForallToForPass &pass) {};
  void runOnOperation() override;
};

void AMDAIEConvertCoreForallToForPass::runOnOperation() {
  Operation *parentOp = getOperation();
  IRRewriter rewriter(parentOp->getContext());

  parentOp->walk([&](AMDAIE::CoreOp coreOp) {
    if (failed(coreForallToFor(rewriter, coreOp))) {
      return signalPassFailure();
    }
  });
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEConvertCoreForallToForPass() {
  return std::make_unique<AMDAIEConvertCoreForallToForPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
