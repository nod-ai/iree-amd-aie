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

// Promote a value by allocating a new buffer or using the provided output
// buffer, then copying the value into it.
FailureOr<Value> promoteValue(IRRewriter &rewriter, Location loc, Value v,
                              Value copyDest = Value()) {
  auto tensorType = dyn_cast<RankedTensorType>(v.getType());
  if (!tensorType) {
    llvm::errs() << "expected a ranked tensor type\n";
    return failure();
  }

  // If no output buffer is provided, allocate a new buffer.
  if (!copyDest) {
    SmallVector<Value> dynamicSizes;
    for (auto [idx, size] : llvm::enumerate(tensorType.getShape())) {
      if (ShapedType::isDynamic(size)) {
        dynamicSizes.push_back(rewriter.create<tensor::DimOp>(loc, v, idx));
      }
    }
    auto alloc = rewriter.create<bufferization::AllocTensorOp>(loc, tensorType,
                                                               dynamicSizes);
    copyDest = alloc.getResult();
  }

  auto copy = rewriter.create<linalg::CopyOp>(loc, v, copyDest);
  return copy.getResult(0);
}

LogicalResult promoteResults(IRRewriter &rewriter, Operation *op) {
  OpBuilder::InsertionGuard g(rewriter);

  // Only a single output is supported.
  if (op->getNumResults() != 1)
    return op->emitError("expected a single output");

  // Only ranked tensor is supported.
  Value result = op->getResults()[0];
  auto resultType = dyn_cast<RankedTensorType>(result.getType());
  if (!resultType) return op->emitError("expected ranked tensor result");

  // Handle case where the target op is inside scf.forall and we want to use
  // block arguments as copy destinations.
  Value outputBuffer;
  if (auto forallOp = op->getParentOfType<scf::ForallOp>()) {
    if (forallOp.getNumResults() != 1)
      return forallOp->emitError("expected a single output");

    Block &block = forallOp.getRegion().front();
    unsigned numIvs = forallOp.getInductionVars().size();
    Value blockArg = block.getArgument(numIvs);
    auto blockArgType = dyn_cast<RankedTensorType>(blockArg.getType());
    if (!blockArgType)
      return forallOp->emitError("expected ranked tensor block argument");

    // Create tensor.extract_slice on block arg.
    rewriter.setInsertionPoint(op);
    SmallVector<OpFoldResult> offsets, sizes, strides;
    for (Value iv : forallOp.getInductionVars()) offsets.push_back(iv);
    // Pad offset with zeros if iv size is smaller than the rank.
    for (unsigned i = numIvs; i < resultType.getRank(); ++i)
      offsets.push_back(rewriter.getIndexAttr(0));
    for (int64_t d : resultType.getShape())
      sizes.push_back(rewriter.getIndexAttr(d));
    strides.assign(resultType.getRank(), rewriter.getIndexAttr(1));

    outputBuffer = rewriter.create<tensor::ExtractSliceOp>(
        op->getLoc(), blockArg, offsets, sizes, strides);
  }

  rewriter.setInsertionPointAfter(op);
  FailureOr<Value> maybeReplacement =
      promoteValue(rewriter, op->getLoc(), result, outputBuffer);
  if (failed(maybeReplacement))
    return op->emitError() << "failed to promote result";

  rewriter.replaceUsesWithIf(result, *maybeReplacement, [&](OpOperand &use) {
    // Only replace uses not inside the newly created copy op.
    return use.getOwner() != maybeReplacement->getDefiningOp();
  });
  return success();
}

LogicalResult promoteInputs(IRRewriter &rewriter, Operation *op) {
  OpBuilder::InsertionGuard g(rewriter);
  auto dstStyleOp = dyn_cast<DestinationStyleOpInterface>(op);
  if (!dstStyleOp) return failure();

  Location loc = dstStyleOp.getLoc();
  unsigned numDpsInputs = dstStyleOp.getNumDpsInputs();

  // Promote the input operands.
  for (auto [i, operand] : llvm::enumerate(dstStyleOp.getDpsInputs())) {
    rewriter.setInsertionPoint(op);
    FailureOr<Value> maybeReplacement = promoteValue(rewriter, loc, operand);
    if (failed(maybeReplacement))
      return dstStyleOp.emitError() << "failed to promote input " << i;
    op->setOperand(i, *maybeReplacement);
  }

  // Promote the init operands.
  for (auto [i, operand] : llvm::enumerate(dstStyleOp.getDpsInits())) {
    rewriter.setInsertionPoint(op);
    FailureOr<Value> maybeReplacement = promoteValue(rewriter, loc, operand);
    if (failed(maybeReplacement))
      return dstStyleOp.emitError() << "failed to promote init " << i;
    op->setOperand(numDpsInputs + i, *maybeReplacement);
  }

  return success();
}

class AMDAIEInsertCopyOpsPass
    : public impl::AMDAIEInsertCopyOpsBase<AMDAIEInsertCopyOpsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, linalg::LinalgDialect,
                    AMDAIEDialect>();
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
    if (isa<linalg::SoftmaxOp>(op) || isa<linalg::GenericOp>(op))
      targetOps.push_back(op);
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
