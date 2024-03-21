// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-peel-for-loop"

namespace mlir::iree_compiler::AMDAIE {

namespace {

LogicalResult peelForLoopLastIteration(RewriterBase &b, scf::ForOp forOp,
                                       scf::ForOp &lastIteration) {
  RewriterBase::InsertionGuard guard(b);
  auto lbInt = getConstantIntValue(forOp.getLowerBound());
  auto ubInt = getConstantIntValue(forOp.getUpperBound());
  auto stepInt = getConstantIntValue(forOp.getStep());

  // Check again when the first iteration is already peeled off.
  // Peeling is not needed if there is one or less iteration.
  if (lbInt && ubInt && stepInt && ceil(float(*ubInt - *lbInt) / *stepInt) <= 1)
    return failure();

  AffineExpr lbSymbol, ubSymbol, stepSymbol;
  bindSymbols(b.getContext(), lbSymbol, ubSymbol, stepSymbol);

  // New upper bound: %ub - (%ub - %lb) mod %step
  auto modMap =
      AffineMap::get(0, 3, {ubSymbol - ((ubSymbol - lbSymbol) % stepSymbol)});
  b.setInsertionPoint(forOp);
  auto loc = forOp.getLoc();
  Value splitBound = b.createOrFold<affine::AffineApplyOp>(
      loc, modMap,
      ValueRange{forOp.getLowerBound(), forOp.getUpperBound(),
                 forOp.getStep()});

  // Create ForOp for partial iteration.
  b.setInsertionPointAfter(forOp);
  lastIteration = cast<scf::ForOp>(b.clone(*forOp.getOperation()));
  lastIteration.getLowerBoundMutable().assign(splitBound);
  b.replaceAllUsesWith(forOp.getResults(), lastIteration->getResults());
  lastIteration.getInitArgsMutable().assign(forOp->getResults());

  // Set new upper loop bound.
  b.modifyOpInPlace(forOp,
                    [&]() { forOp.getUpperBoundMutable().assign(splitBound); });

  return success();
}

class AMDAIEPeelForLoopPass
    : public impl::AMDAIEPeelForLoopBase<AMDAIEPeelForLoopPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  AMDAIEPeelForLoopPass() = default;
  AMDAIEPeelForLoopPass(const AMDAIEPeelForLoopPass &pass){};
  AMDAIEPeelForLoopPass(const AMDAIEPeelForLoopOptions &options)
      : AMDAIEPeelForLoopBase(options) {}
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
    switch (peelLoopType) {
      case PeelLoopType::First:
        if (failed(scf::peelForLoopFirstIteration(rewriter, forOp, result))) {
          forOp->emitOpError("failed to peel the first iteration.");
          return signalPassFailure();
        }
        break;
      case PeelLoopType::Last:
        if (failed(peelForLoopLastIteration(rewriter, forOp, result))) {
          forOp->emitOpError("failed to peel the last iteration.");
          return signalPassFailure();
        }
        break;
      case PeelLoopType::FirstLast:
        if (failed(scf::peelForLoopFirstIteration(rewriter, forOp, result))) {
          forOp->emitOpError("failed to peel the first iteration.");
          return signalPassFailure();
        }
        if (failed(peelForLoopLastIteration(rewriter, forOp, result))) {
          LLVM_DEBUG(llvm::dbgs() << "Skip peeling for the last iteration.\n");
        }
        break;
      default:
        forOp->emitOpError("failed to specify the type of loop peeling.");
        return signalPassFailure();
    }
  });
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEPeelForLoopPass(
    AMDAIEPeelForLoopOptions options) {
  return std::make_unique<AMDAIEPeelForLoopPass>(options);
}
}  // namespace mlir::iree_compiler::AMDAIE
