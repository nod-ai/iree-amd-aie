// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-insert-copy-ops"

namespace mlir::iree_compiler::AMDAIE {

namespace {

FailureOr<Value> promoteValue(IRRewriter &rewriter, Location loc, Value v) {
  auto tensorType = dyn_cast<RankedTensorType>(v.getType());
  if (!tensorType) {
    llvm::errs() << "expected a ranked tensor type\n";
    return failure();
  }
  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(rewriter, loc, v);
  Value empty = rewriter.create<tensor::EmptyOp>(loc, mixedSizes,
                                                 tensorType.getElementType());
  auto copy = rewriter.create<linalg::CopyOp>(loc, v, empty);
  return copy.getResult(0);
}

LogicalResult promoteResults(IRRewriter &rewriter, Operation *op) {
  OpBuilder::InsertionGuard g(rewriter);
  for (auto [i, operand] : llvm::enumerate(op->getResults())) {
    rewriter.setInsertionPointAfter(op);
    FailureOr<Value> maybeReplacement =
        promoteValue(rewriter, op->getLoc(), operand);
    if (failed(maybeReplacement))
      return op->emitError() << "failed to promote result " << i;
    rewriter.replaceUsesWithIf(operand, *maybeReplacement, [&](OpOperand &use) {
      // Only replace uses not inside the newly created copy op.
      return use.getOwner() != maybeReplacement->getDefiningOp();
    });
  }
  return success();
}

LogicalResult promoteInputs(IRRewriter &rewriter, Operation *op) {
  OpBuilder::InsertionGuard g(rewriter);
  auto dstStyleOp = dyn_cast<DestinationStyleOpInterface>(op);
  if (!dstStyleOp) return failure();
  for (auto [i, operand] : llvm::enumerate(dstStyleOp.getDpsInputs())) {
    rewriter.setInsertionPoint(op);
    FailureOr<Value> maybeReplacement =
        promoteValue(rewriter, dstStyleOp.getLoc(), operand);
    if (failed(maybeReplacement))
      return dstStyleOp.emitError() << "failed to promote input " << i;
    op->setOperand(i, *maybeReplacement);
  }
  return success();
}

class AMDAIEInsertCopyOpsPass
    : public impl::AMDAIEInsertCopyOpsBase<AMDAIEInsertCopyOpsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, AMDAIEDialect>();
  }

  AMDAIEInsertCopyOpsPass() = default;
  AMDAIEInsertCopyOpsPass(const AMDAIEInsertCopyOpsPass &pass){};
  void runOnOperation() override;
};

void AMDAIEInsertCopyOpsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);
  mlir::FunctionOpInterface funcOp = getOperation();
  SmallVector<Operation *> targetOps;
  funcOp->walk<WalkOrder::PostOrder, ReverseIterator>([&](Operation *op) {
    if (isa<linalg::SoftmaxOp>(op)) targetOps.push_back(op);
  });
  for (Operation *targetOp : targetOps) {
    if (failed(promoteInputs(rewriter, targetOp))) return signalPassFailure();
    if (failed(promoteResults(rewriter, targetOp))) return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEInsertCopyOpsPass() {
  return std::make_unique<AMDAIEInsertCopyOpsPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
