// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-peel-for-loop"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIEPeelForLoopPass
    : public impl::AMDAIEPeelForLoopBase<AMDAIEPeelForLoopPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  AMDAIEPeelForLoopPass() = default;
  AMDAIEPeelForLoopPass(const AMDAIEPeelForLoopPass &pass){};
  void runOnOperation() override;
};

void AMDAIEPeelForLoopPass::runOnOperation() {
  MLIRContext *context = &getContext();
  func::FuncOp funcOp = getOperation();
  IRRewriter rewriter(context);

  funcOp->walk([&](scf::ForOp forOp) {
    auto lbInt = getConstantIntValue(forOp.getLowerBound());
    auto ubInt = getConstantIntValue(forOp.getUpperBound());
    auto stepInt = getConstantIntValue(forOp.getStep());

    // Peeling is not needed if there is one or less iteration.
    if (lbInt && ubInt && stepInt &&
        ceil(float(*ubInt - *lbInt) / *stepInt) <= 1)
      return;

    scf::ForOp result;
    LogicalResult status =
        scf::peelForLoopFirstIteration(rewriter, forOp, result);
    if (failed(status)) {
      return signalPassFailure();
    }
  });
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEPeelForLoopPass() {
  return std::make_unique<AMDAIEPeelForLoopPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
