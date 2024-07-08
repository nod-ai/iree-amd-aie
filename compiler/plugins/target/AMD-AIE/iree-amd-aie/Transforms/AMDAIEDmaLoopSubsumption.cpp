// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the transformation that subsumes a loop iteration into a
// DMA access pattern if possible. This adds an additional dimension to the
// DMA's access pattern and hoits the DMA operation out of the loop. This
// transformation is possible if:
//
// - The loop's bounds and step size are all constants.
// - The DMA is only operated on once within the loop's scope. Otherwise,
//   subsumbtion of the loop iteration into the DMA can change the temporal
//   behaviour of the program.
// - The DMA has additional available access pattern dimensions. This
//   information is retrieved from a target hardware model.
//
//===----------------------------------------------------------------------===//

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/IR/AMDAIETargetModel.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-dma-loop-subsumption"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Return an ancestor of 'op' in 'block', or nullptr if no such ancestor.
Operation *getAncestorInBlock(Operation *op, Block *block) {
  if (!op || !block) return nullptr;
  auto parent = op;
  while (parent && (parent->getBlock() != block))
    parent = parent->getParentOp();
  return parent;
}

/// Utility affine expression visitor to retrieve the stride from the
/// expression.
struct RetrieveStrideSize : public AffineExprVisitor<RetrieveStrideSize> {
  std::optional<int64_t> stride;
  void visitMulExpr(AffineBinaryOpExpr expr) {
    if (auto rhsSize = dyn_cast<AffineConstantExpr>(expr.getRHS());
        isa<AffineDimExpr>(expr.getLHS())) {
      stride = rhsSize.getValue();
    } else if (auto lhsSize = dyn_cast<AffineConstantExpr>(expr.getLHS());
               isa<AffineDimExpr>(expr.getRHS())) {
      stride = lhsSize.getValue();
    }
  }
};

/// Utility to clean up the DMA users after loop subsumption + hoisting. This
/// will hoist `amdaie.npu.dma_cpy_nd`'s users like `npu.dma_wait` as well.
LogicalResult moveUsersToHoistedDMAScope(Operation *parentOp) {
  IRRewriter rewriter(parentOp->getContext());
  // Move `amdaie.npu.dma_wait` operation after the parent op in the same block
  // as the input `amdaie.npu.dma_cpy_nd` operation. This parent op will
  // typically be a loop out of which the DMA operation has been hoisted. Moving
  // the wait operation after this loop is important to avoid a deadlock with
  // whatever operations are still remaining inside the loop's scope.
  WalkResult res = parentOp->walk([&](AMDAIE::NpuDmaWaitOp npuDmaWaitOp) {
    Operation *dmaOp = npuDmaWaitOp.getDma().getDefiningOp();
    Operation *ancestorInSameBlock =
        getAncestorInBlock(npuDmaWaitOp, dmaOp->getBlock());
    if (!ancestorInSameBlock) {
      npuDmaWaitOp->emitOpError(
          "doesn't have an ancestor in the same scope as the source DMA op");
      return WalkResult::interrupt();
    }
    rewriter.moveOpAfter(npuDmaWaitOp, ancestorInSameBlock);
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return failure();
  return success();
}

class SubsumeLoopIntoDMA
    : public OpInterfaceRewritePattern<AMDAIE::DoublyStridedOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  /// Utility to add a loop iteration to an offsets/sizes/strides access
  /// pattern.
  LogicalResult addIterationToAccessPattern(
      RewriterBase &rewriter, int64_t lowerBound, int64_t upperBound,
      int64_t step, const DenseSet<Value> &inductionValues,
      SmallVector<OpFoldResult> &newOffsets,
      SmallVector<OpFoldResult> &newSizes,
      SmallVector<OpFoldResult> &newStrides) const {
    SmallVector<OpFoldResult> insertOffsets;
    SmallVector<OpFoldResult> insertSizes;
    SmallVector<OpFoldResult> insertStrides;
    for (auto &&[i, offset] : llvm::enumerate(newOffsets)) {
      Value offsetValue = getValueOrCreateConstantIndexOp(
          rewriter, rewriter.getUnknownLoc(), offset);
      if (inductionValues.contains(offsetValue)) {
        // Initialize the offsetStride to 1. This handles the case where an
        // induction variable is directly used as an offset inside a strided
        // operation.
        int64_t offsetStride = 1;
        // If the offset value is determined by an affine expression, retrieve
        // the affine expression's stride.multiplier and calculate the actual
        // offset stride.
        if (offsetValue.getDefiningOp() &&
            isa<affine::AffineApplyOp>(offsetValue.getDefiningOp())) {
          auto applyOp =
              cast<affine::AffineApplyOp>(offsetValue.getDefiningOp());
          // Retrieve the stride from the affine map using an affine expression
          // visitor. This is the place where invalid maps are filtered out.
          // Invalid cases will have `retriever.stride == nullopt` after
          // visiting.
          AffineMap affineMap = applyOp.getAffineMap();
          RetrieveStrideSize retriever;
          retriever.visit(affineMap.getResult(0));
          if (!retriever.stride) return failure();
          offsetStride *= retriever.stride.value();
        }

        // Multiplying by step size handles the non-normalized case.
        int64_t stride =
            getConstantIntValue(newStrides[i]).value() * offsetStride * step;

        newOffsets[i] = getAsIndexOpFoldResult(rewriter.getContext(),
                                               lowerBound * offsetStride);
        insertOffsets.push_back(
            getAsIndexOpFoldResult(rewriter.getContext(), 0));

        // The step size is equal to the the number of iterations
        // (ceilDiv(upperBound - lowerBound, step))
        int64_t diff = upperBound - lowerBound;
        assert(diff > 0 &&
               "expected positive difference between upper bound and lower "
               "bound");
        assert(step > 0 && "expected positive step");
        int64_t newSize = 1 + ((diff - 1) / step);
        insertSizes.push_back(
            getAsIndexOpFoldResult(rewriter.getContext(), newSize));

        insertStrides.push_back(
            getAsIndexOpFoldResult(rewriter.getContext(), stride));
      }
    }
    newOffsets.insert(newOffsets.begin(), insertOffsets.begin(),
                      insertOffsets.end());
    newSizes.insert(newSizes.begin(), insertSizes.begin(), insertSizes.end());
    newStrides.insert(newStrides.begin(), insertStrides.begin(),
                      insertStrides.end());
    return success();
  }

  /// Rewrite function for a doubly strided operation with any loop-like parent
  /// operation.
  LogicalResult rewriteWithLoopLikeOpParent(
      AMDAIE::DoublyStridedOpInterface op, PatternRewriter &rewriter,
      size_t sourceMaxNbDims, size_t targetMaxNbDims,
      const SmallVector<int64_t> &lowerBounds,
      const SmallVector<int64_t> &upperBounds,
      const SmallVector<int64_t> &steps,
      const SmallVector<DenseSet<Value>> &inductionValues,
      const DenseSet<Value> &allInductionValues) const {
    auto loopOp = dyn_cast<LoopLikeOpInterface>(op->getParentOp());
    if (!loopOp) return failure();

    // Initialize new access pattern offsets/sizes/strides with current values.
    SmallVector<OpFoldResult> newSourceOffsets = op.getSourceMixedOffsets();
    SmallVector<OpFoldResult> newSourceSizes = op.getSourceMixedSizes();
    SmallVector<OpFoldResult> newSourceStrides = op.getSourceMixedStrides();
    SmallVector<OpFoldResult> newTargetOffsets = op.getTargetMixedOffsets();
    SmallVector<OpFoldResult> newTargetSizes = op.getTargetMixedSizes();
    SmallVector<OpFoldResult> newTargetStrides = op.getTargetMixedStrides();

    // Use source/target maxNbDims to check whether there are sufficient source
    // and target dimensions. Otherwise, abort.
    auto verifyNbDimsNeeded = [&](const SmallVector<Value> &dynamicOffsets,
                                  size_t nbOffsets,
                                  size_t maxNbDims) -> LogicalResult {
      size_t counter = 0;
      for (Value offset : dynamicOffsets)
        if (allInductionValues.contains(offset)) counter++;
      if (nbOffsets + counter > maxNbDims) return failure();
      return success();
    };
    SmallVector<Value> dynamicSourceOffsets = op.getSourceOffsets();
    SmallVector<Value> dynamicTargetOffsets = op.getTargetOffsets();
    if (failed(verifyNbDimsNeeded(dynamicSourceOffsets, newSourceOffsets.size(),
                                  sourceMaxNbDims)))
      return failure();
    if (failed(verifyNbDimsNeeded(dynamicTargetOffsets, newTargetOffsets.size(),
                                  targetMaxNbDims)))
      return failure();

    // Add the loop iterations to the DMA access patterns.
    for (auto &&[lb, ub, step, iterationIvValues] : llvm::reverse(
             llvm::zip(lowerBounds, upperBounds, steps, inductionValues))) {
      // Add loop iteration to the access pattern on the source side.
      if (failed(addIterationToAccessPattern(
              rewriter, lb, ub, step, iterationIvValues, newSourceOffsets,
              newSourceSizes, newSourceStrides))) {
        return failure();
      }
      // Add loop iteration to the access pattern on the target side.
      if (failed(addIterationToAccessPattern(
              rewriter, lb, ub, step, iterationIvValues, newTargetOffsets,
              newTargetSizes, newTargetStrides))) {
        return failure();
      }
    }

    assert(newSourceOffsets.size() == newSourceSizes.size() &&
           "expected same number of source offsets and sizes");
    assert(newSourceOffsets.size() == newSourceStrides.size() &&
           "expected same number of source offsets and strides");
    assert(newTargetOffsets.size() == newTargetSizes.size() &&
           "expected same number of target offsets and sizes");
    assert(newTargetOffsets.size() == newTargetStrides.size() &&
           "expected same number of target offsets and strides");

    // Create new doubly strided operation with the updated access pattern and
    // move it before the loop.
    rewriter.setInsertionPoint(loopOp);
    auto newDoublyStridedOp = op.createDoublyStridedOp(
        rewriter, newTargetOffsets, newTargetSizes, newTargetStrides,
        newSourceOffsets, newSourceSizes, newSourceStrides);
    rewriter.replaceOp(op, newDoublyStridedOp.getOperation());
    return success();
  }

  /// Main rewrite function for a doubly strided operation with a `scf.for`
  /// parent operation. Only handle a loop induction variable with an
  /// optional `affine.apply` user for now.
  LogicalResult rewriteWithForOpParent(AMDAIE::DoublyStridedOpInterface op,
                                       PatternRewriter &rewriter,
                                       size_t sourceMaxNbDims,
                                       size_t targetMaxNbDims) const {
    auto forOp = dyn_cast<scf::ForOp>(op->getParentOp());
    if (!forOp) return failure();

    // Dynamic bounds or step are not supported.
    std::optional<int64_t> lowerBound =
        getConstantIntValue(forOp.getLowerBound());
    std::optional<int64_t> upperBound =
        getConstantIntValue(forOp.getUpperBound());
    std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
    if (!lowerBound || !upperBound || !step) return failure();

    // Only handle loop induction variable with an optional `affine.apply` user
    // for now.
    Value iv = forOp.getInductionVar();
    DenseSet<Value> curIvValues = {iv};
    for (OpOperand &use : iv.getUses()) {
      if (!use.getOwner()) continue;
      if (auto userApplyOp = dyn_cast<affine::AffineApplyOp>(use.getOwner())) {
        curIvValues.insert(userApplyOp.getResult());
      }
    }
    if (!llvm::any_of(op->getOperands(), [&](Value operand) {
          return curIvValues.contains(operand);
        })) {
      return failure();
    }

    SmallVector<int64_t> lowerBounds = {lowerBound.value()};
    SmallVector<int64_t> upperBounds = {upperBound.value()};
    SmallVector<int64_t> steps = {step.value()};
    SmallVector<DenseSet<Value>> inductionValues = {curIvValues};
    return rewriteWithLoopLikeOpParent(
        op, rewriter, sourceMaxNbDims, targetMaxNbDims, lowerBounds,
        upperBounds, steps, inductionValues, curIvValues);
  }

  /// Main rewrite function for a doubly strided operation with a `scf.forall`
  /// parent operation. Only handle loop induction variables with an
  /// optional `affine.apply` user for now.
  LogicalResult rewriteWithForallOpParent(AMDAIE::DoublyStridedOpInterface op,
                                          PatternRewriter &rewriter,
                                          size_t sourceMaxNbDims,
                                          size_t targetMaxNbDims) const {
    auto forallOp = dyn_cast<scf::ForallOp>(op->getParentOp());
    if (!forallOp) return failure();

    // Dynamic bounds or step are not supported.
    std::optional<SmallVector<int64_t>> lowerBounds =
        getConstantIntValues(forallOp.getMixedLowerBound());
    std::optional<SmallVector<int64_t>> upperBounds =
        getConstantIntValues(forallOp.getMixedUpperBound());
    std::optional<SmallVector<int64_t>> steps =
        getConstantIntValues(forallOp.getMixedStep());
    if (!lowerBounds || !upperBounds || !steps) return failure();

    // A set of all induction variables and optional `affine.apply` user values
    // for easy verification whether any of the induction variables or
    // `affine.apply` values are being used.
    DenseSet<Value> allInductionValues;
    // A vector of all induction varilable dependent values for each induction
    // var. Includes the induction variable itself and any `affine.apply` users.
    SmallVector<DenseSet<Value>> inductionValues;
    for (Value iv : forallOp.getInductionVars()) {
      DenseSet<Value> curIvValues = {iv};
      allInductionValues.insert(iv);
      for (Operation *userOp : iv.getUsers()) {
        if (auto userApplyOp = dyn_cast<affine::AffineApplyOp>(userOp)) {
          curIvValues.insert(userApplyOp.getResult());
          allInductionValues.insert(userApplyOp.getResult());
        }
      }
      inductionValues.push_back(curIvValues);
    }
    // Return early if the strided operation doesn't use any of the
    // induction variable dependent values.
    if (!llvm::any_of(op->getOperands(), [&](Value operand) {
          return allInductionValues.contains(operand);
        })) {
      return failure();
    }
    return rewriteWithLoopLikeOpParent(op, rewriter, sourceMaxNbDims,
                                       targetMaxNbDims, lowerBounds.value(),
                                       upperBounds.value(), steps.value(),
                                       inductionValues, allInductionValues);
  }

  LogicalResult matchAndRewrite(AMDAIE::DoublyStridedOpInterface op,
                                PatternRewriter &rewriter) const override {
    // Depending on the DMA being targetted, there can be a different number of
    // max dimensions supported by the hardware. Consider the different cases
    // for Shim, MemTile and core DMAs:
    // - Shim DMAs: As the shim DMA typically isn't synchronized with other DMAs
    //   (through semaphore locks), the inter-iteration access pattern is
    //   typically used as an additional intra-iteration access pattern,
    //   resulting in 4 DMA dimensions which can be used to address global
    //   memory data.
    // - As the MemTile DMAs are typically synchronized with other DMAs for
    //   stream-through, double-buffering purposes, the inter-iteration can't
    //   typically be used in the same way as the intra-iteration dimensions.
    //   Therefore, for now, only the intra-iteration dimensions can be used for
    //   DMA access patterns.
    // - Core DMAs: As the core DMAs are typically synchronized with the core
    //   processor for data access purposes (read/write), the inter-iteration
    //   can't typically be used in the same way as the intra-iteration
    //   dimensions. Therefore, for now, only the intra-iteration dimensions can
    //   be used for DMA access patterns.
    size_t sourceMaxNbDims{0};
    size_t targetMaxNbDims{0};
    if (auto npuDmaOp = dyn_cast<AMDAIE::NpuDmaCpyNdOp>(op.getOperation())) {
      uint64_t sourceMemspaceInt = npuDmaOp.getSourceMemorySpaceAsUInt();
      uint64_t targetMemspaceInt = npuDmaOp.getTargetMemorySpaceAsUInt();
      if (sourceMemspaceInt == 0) {
        sourceMaxNbDims = kAMDAIEShimDmaNbIntraDims + kAMDAIEShimDmaNbInterDims;
      } else if (sourceMemspaceInt == 1) {
        sourceMaxNbDims = kAMDAIEMemTileDmaNbIntraDims;
      } else if (sourceMemspaceInt == 2) {
        sourceMaxNbDims = kAMDAIECoreDmaNbIntraDims;
      }
      if (targetMemspaceInt == 0) {
        targetMaxNbDims = kAMDAIEShimDmaNbIntraDims + kAMDAIEShimDmaNbInterDims;
      } else if (targetMemspaceInt == 1) {
        targetMaxNbDims = kAMDAIEMemTileDmaNbIntraDims;
      } else if (targetMemspaceInt == 2) {
        targetMaxNbDims = kAMDAIECoreDmaNbIntraDims;
      }

      // Check that the DMA this `amdaie.npu.dma_cpy_nd` operation is operating
      // on, is not being touched within the same scope. Otherwise, the rewrite
      // is not be valid in general as it would be changing the temporal usage
      // of the source DMA.
      Operation *parentOp = op->getParentOp();
      if (!parentOp) return failure();
      Value dma = npuDmaOp.getDma();
      for (Operation *userOp : dma.getUsers()) {
        if (userOp != op.getOperation() && parentOp->isProperAncestor(userOp)) {
          return failure();
        }
      }
    } else {
      return failure();
    }

    if (isa<scf::ForOp>(op->getParentOp())) {
      return rewriteWithForOpParent(op, rewriter, sourceMaxNbDims,
                                    targetMaxNbDims);
    } else if (isa<scf::ForallOp>(op->getParentOp())) {
      return rewriteWithForallOpParent(op, rewriter, sourceMaxNbDims,
                                       targetMaxNbDims);
    } else {
      return failure();
    }
  }
};

class AMDAIEDmaLoopSubsumptionPass
    : public impl::AMDAIEDmaLoopSubsumptionBase<AMDAIEDmaLoopSubsumptionPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEDmaLoopSubsumptionPass() = default;
  AMDAIEDmaLoopSubsumptionPass(const AMDAIEDmaLoopSubsumptionPass &pass){};
  void runOnOperation() override;
};

void AMDAIEDmaLoopSubsumptionPass::runOnOperation() {
  Operation *parentOp = getOperation();
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<SubsumeLoopIntoDMA>(context);
  if (failed(applyPatternsAndFoldGreedily(parentOp, std::move(patterns)))) {
    parentOp->emitOpError("failed to subsume some loops into DMA operations");
    return signalPassFailure();
  }

  if (failed(moveUsersToHoistedDMAScope(parentOp))) {
    parentOp->emitOpError(
        "failed to move DMA users to correct scope after loop subsumption");
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEDmaLoopSubsumptionPass() {
  return std::make_unique<AMDAIEDmaLoopSubsumptionPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
