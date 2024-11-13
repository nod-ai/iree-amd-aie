// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "iree-amdaie-unpeel"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIEUnpeelPass : public impl::AMDAIEUnpeelBase<AMDAIEUnpeelPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }
  AMDAIEUnpeelPass() = default;
  AMDAIEUnpeelPass(const AMDAIEUnpeelPass &pass){};
  void runOnOperation() override;
};

// replace the peeled code which looks something like
//
// linalg.generic ...
// scf.for i = 1 to 7 {
//     linalq.generic ...
// }
// linalg.generic ...
//
// with something like
// scf.for i = 0 to 8 {
//    linalg.generic ...
// }
void doMerge(IRRewriter &rewriter, mlir::FunctionOpInterface operation) {
  // If true, don't merge the epilogue, just the prologue into main.
  constexpr bool mergeJustPrologue = true;
  (void)mergeJustPrologue;

  // Find 3 linalg.generics with a reduction dimension:
  SmallVector<linalg::GenericOp> generics;
  operation->walk([&](linalg::GenericOp genericOp) {
    bool hasRed{false};
    for (auto t : genericOp.getIteratorTypesArray()) {
      if (linalg::isReductionIterator(t)) {
        hasRed = true;
      }
    }
    if (hasRed) {
      generics.push_back(genericOp);
    }
  });
  if (generics.size() != 3) return;

  // If we have 3, we assume they're the prologue, main, epilogue.
  // Replace all uses of prologue and (maybe) epilogue with the 'out' input
  SmallVector<linalg::GenericOp> toMerge{generics[0]};
  if (!mergeJustPrologue) toMerge.push_back(generics[2]);
  for (auto g : toMerge) {
    auto output = g.getResult(0);
    auto input = g.getOperands().back();
    rewriter.replaceAllUsesWith(output, input);
    rewriter.eraseOp(g);
  }

  // Reset the scf.for loop bounds
  auto containingScf = generics[1]->getParentOfType<scf::ForOp>();
  assert(containingScf);
  rewriter.setInsertionPoint(containingScf);
  auto zeroLbIndex =
      rewriter.create<arith::ConstantIndexOp>(generics[1].getLoc(), 0);
  containingScf.setLowerBound(zeroLbIndex);

  if (!mergeJustPrologue) {
    auto oneIndex =
        rewriter.create<arith::ConstantIndexOp>(generics[1].getLoc(), 1);
    auto currentUb = containingScf.getUpperBound();
    auto newUb = rewriter.create<arith::AddIOp>(generics[1].getLoc(), currentUb,
                                                oneIndex);
    containingScf.setUpperBound(newUb);
  }
}

void AMDAIEUnpeelPass::runOnOperation() {
  Operation *parentOp = getOperation();
  IRRewriter rewriter(parentOp);
  parentOp->walk([&](func::FuncOp funcOp) { doMerge(rewriter, funcOp); });
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEUnpeelPass() {
  return std::make_unique<AMDAIEUnpeelPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
