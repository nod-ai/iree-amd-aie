// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#define DEBUG_TYPE "iree-amdaie-hoist-logical-objectfifo"

namespace mlir::iree_compiler::AMDAIE {

/// Hoist logical objectFifo operations until one of the operands is located
/// within the same scope.
LogicalResult hoistLogicalObjFifoOp(RewriterBase &rewriter,
                                    AMDAIE::LogicalObjectFifoFromMemrefOp op) {
  Operation *ancestorOp = op;
  while (ancestorOp) {
    Operation *newAncestorOp = ancestorOp->getParentOp();
    if (llvm::any_of(op->getOperands(), [&](Value operand) {
          return operand.getDefiningOp() &&
                 newAncestorOp->isProperAncestor(operand.getDefiningOp());
        })) {
      break;
    }
    if (isa<AMDAIE::WorkgroupOp, AMDAIE::ControlCodeOp, func::FuncOp>(
            newAncestorOp)) {
      break;
    }
    ancestorOp = newAncestorOp;
  }
  if (ancestorOp && ancestorOp != op) rewriter.moveOpBefore(op, ancestorOp);
  return failure();
}

namespace {
struct AMDAIEHoistLogicalObjFifoPass
    : public impl::AMDAIEHoistLogicalObjFifoBase<
          AMDAIEHoistLogicalObjFifoPass> {
  void runOnOperation() override;
};

void AMDAIEHoistLogicalObjFifoPass::runOnOperation() {
  Operation *parentOp = getOperation();
  IRRewriter rewriter(parentOp->getContext());

  SmallVector<AMDAIE::LogicalObjectFifoFromMemrefOp> logicalObjFifos;
  parentOp->walk([&](AMDAIE::LogicalObjectFifoFromMemrefOp op) {
    (void)hoistLogicalObjFifoOp(rewriter, op);
  });
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEHoistLogicalObjFifoPass() {
  return std::make_unique<AMDAIEHoistLogicalObjFifoPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
