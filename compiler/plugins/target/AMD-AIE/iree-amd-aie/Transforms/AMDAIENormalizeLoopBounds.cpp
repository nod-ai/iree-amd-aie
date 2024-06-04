// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "iree-amdaie-normalize-loop-bounds"

namespace mlir::iree_compiler::AMDAIE {

/// NOTE: Copy of existing utility struct in llvm-project:
/// https://github.com/llvm/llvm-project/blob/172759492a162592da7ae9e03888661c108b1be4/mlir/lib/Dialect/SCF/Utils/Utils.cpp#L33C1-L40C15.
/// Can be replaced with that utility struct once it is exposed.
///
// This structure is to pass and return sets of loop parameters without
// confusing the order.
struct LoopParams {
  Value lowerBound;
  Value upperBound;
  Value step;
};

/// NOTE: Copy of existing utility function in llvm-project:
/// https://github.com/llvm/llvm-project/blob/172759492a162592da7ae9e03888661c108b1be4/mlir/lib/Dialect/SCF/Utils/Utils.cpp#L485.
/// Can be replaced with that utility function once it is exposed.
static LoopParams emitNormalizedLoopBounds(RewriterBase &rewriter, Location loc,
                                           Value lb, Value ub, Value step) {
  // For non-index types, generate `arith` instructions
  // Check if the loop is already known to have a constant zero lower bound or
  // a constant one step.
  bool isZeroBased = false;
  if (auto lbCst = getConstantIntValue(lb)) isZeroBased = lbCst.value() == 0;

  bool isStepOne = false;
  if (auto stepCst = getConstantIntValue(step))
    isStepOne = stepCst.value() == 1;

  // Compute the number of iterations the loop executes: ceildiv(ub - lb, step)
  // assuming the step is strictly positive.  Update the bounds and the step
  // of the loop to go from 0 to the number of iterations, if necessary.
  if (isZeroBased && isStepOne) return {lb, ub, step};

  Value diff =
      isZeroBased ? ub : rewriter.createOrFold<arith::SubIOp>(loc, ub, lb);
  Value newUpperBound =
      isStepOne ? diff
                : rewriter.createOrFold<arith::CeilDivSIOp>(loc, diff, step);

  Value newLowerBound = isZeroBased
                            ? lb
                            : rewriter.create<arith::ConstantOp>(
                                  loc, rewriter.getZeroAttr(lb.getType()));
  Value newStep = isStepOne
                      ? step
                      : rewriter.create<arith::ConstantOp>(
                            loc, rewriter.getIntegerAttr(step.getType(), 1));

  return {newLowerBound, newUpperBound, newStep};
}

/// Transform a `scf.for` loop with a strictly positive step
///   for %i = %lb to %ub step %s
/// into a 0-based loop with step 1
///   for %ii = 0 to ceildiv(%ub - %lb, %s) step 1
/// Insert an `affine.apply` operation to compute the denormalized index value.
LogicalResult normalizeLoopBounds(RewriterBase &rewriter, scf::ForOp forOp) {
  OpBuilder::InsertionGuard g(rewriter);
  // Return if already normalized
  std::optional<int64_t> lbInt = getConstantIntValue(forOp.getLowerBound());
  std::optional<int64_t> stepInt = getConstantIntValue(forOp.getStep());
  if (lbInt && stepInt && lbInt.value() == 0 && stepInt.value() == 1) {
    return success();
  }

  Value iv = forOp.getInductionVar();
  rewriter.setInsertionPoint(forOp);
  auto newLoopParams =
      emitNormalizedLoopBounds(rewriter, forOp.getLoc(), forOp.getLowerBound(),
                               forOp.getUpperBound(), forOp.getStep());

  rewriter.modifyOpInPlace(forOp, [&]() {
    forOp.setLowerBound(newLoopParams.lowerBound);
    forOp.setUpperBound(newLoopParams.upperBound);
    forOp.setStep(newLoopParams.step);
  });

  rewriter.setInsertionPointToStart(forOp.getBody());
  AffineExpr idx;
  bindDims(forOp.getContext(), idx);
  auto affineApply = rewriter.create<affine::AffineApplyOp>(
      forOp.getLoc(), idx * stepInt.value() + lbInt.value(),
      ValueRange{
          iv,
      });
  SmallPtrSet<Operation *, 2> preserve(
      {iv.getDefiningOp(), affineApply.getOperation()});
  rewriter.replaceAllUsesExcept(iv, affineApply.getResult(), preserve);
  return success();
}

/// Transform a `scf.forall` loop with a strictly positive steps
///   forall (%i, %j) = (%lb0, %lb1) to (%ub0, %ub1) step (%s0, %s1)
/// into a 0-based loop with step 1
///   forall (%i, %j) in (ceildiv(%ub0 - %lb0, %s0), ceildiv(%ub1 - %lb1, %s1))
/// Insert `affine.apply` operations to compute the denormalized index values.
LogicalResult normalizeLoopBounds(RewriterBase &rewriter,
                                  scf::ForallOp forallOp) {
  OpBuilder::InsertionGuard g(rewriter);
  if (forallOp.isNormalized()) return success();

  SmallVector<OpFoldResult> newLbs;
  SmallVector<OpFoldResult> newUbs;
  SmallVector<OpFoldResult> newSteps;
  for (auto &&[iv, lb, ub, step] : llvm::zip(
           forallOp.getInductionVars(), forallOp.getLowerBound(rewriter),
           forallOp.getUpperBound(rewriter), forallOp.getStep(rewriter))) {
    std::optional<int64_t> lbInt = getConstantIntValue(lb);
    std::optional<int64_t> stepInt = getConstantIntValue(step);
    if (!lbInt || !stepInt) return failure();

    rewriter.setInsertionPoint(forallOp);
    auto newLoopParams =
        emitNormalizedLoopBounds(rewriter, forallOp.getLoc(), lb, ub, step);

    newLbs.push_back(getAsOpFoldResult(newLoopParams.lowerBound));
    newUbs.push_back(getAsOpFoldResult(newLoopParams.upperBound));
    newSteps.push_back(getAsOpFoldResult(newLoopParams.step));

    rewriter.setInsertionPointToStart(forallOp.getBody());
    AffineExpr idx;
    bindDims(forallOp.getContext(), idx);
    auto affineApply = rewriter.create<affine::AffineApplyOp>(
        forallOp.getLoc(), idx * stepInt.value() + lbInt.value(),
        ValueRange{
            iv,
        });
    SmallPtrSet<Operation *, 2> preserve(
        {iv.getDefiningOp(), affineApply.getOperation()});
    rewriter.replaceAllUsesExcept(iv, affineApply.getResult(), preserve);
  }

  rewriter.setInsertionPointAfter(forallOp);
  SmallVector<Value> outputs;
  SmallVector<OpFoldResult> empty;
  auto newLoop =
      rewriter.create<scf::ForallOp>(rewriter.getUnknownLoc(), newLbs, newUbs,
                                     newSteps, outputs, forallOp.getMapping());

  // Map control operands.
  IRMapping mapping;
  mapping.map(forallOp.getInductionVars(), newLoop.getInductionVars());
  mapping.map(forallOp.getRegionIterArgs(), newLoop.getRegionIterArgs());

  rewriter.setInsertionPointToStart(newLoop.getBody());
  for (Operation &op : forallOp.getBody()->without_terminator())
    rewriter.clone(op, mapping);

  rewriter.replaceOp(forallOp, newLoop);
  return success();
}

namespace {
struct AMDAIENormalizeLoopBounds
    : public impl::AMDAIENormalizeLoopBoundsBase<AMDAIENormalizeLoopBounds> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect>();
  }

  void runOnOperation() override {
    Operation *parentOp = getOperation();
    IRRewriter rewriter(parentOp->getContext());

    parentOp->walk(
        [&](scf::ForOp forOp) { (void)normalizeLoopBounds(rewriter, forOp); });
    parentOp->walk([&](scf::ForallOp forallOp) {
      (void)normalizeLoopBounds(rewriter, forallOp);
    });
  }
};
}  // namespace

std::unique_ptr<Pass> createAMDAIENormalizeLoopBoundsPass() {
  return std::make_unique<AMDAIENormalizeLoopBounds>();
}

}  // namespace mlir::iree_compiler::AMDAIE
