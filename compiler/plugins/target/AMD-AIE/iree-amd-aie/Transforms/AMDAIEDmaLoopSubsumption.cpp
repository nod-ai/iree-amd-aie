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
#include "iree-amd-aie/Transforms/AMDAIEDmaUtils.h"
#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-dma-loop-subsumption"

namespace mlir::iree_compiler::AMDAIE {

using mlir::OpTrait::iree_compiler::AMDAIE::CircularDmaOp;

/// Utility to calculate the number of iterations of a loop with provided bounds
/// and step: `ceilDiv(upperBound - lowerBound, step)`.
int64_t calculateNbIterations(int64_t lowerBound, int64_t upperBound,
                              int64_t step) {
  int64_t diff = upperBound - lowerBound;
  assert(diff > 0 &&
         "expected positive difference between upper bound and lower "
         "bound");
  assert(step > 0 && "expected positive step");
  return 1 + ((diff - 1) / step);
}

namespace {

struct SubsumeLoopIntoDMA
    : public OpInterfaceRewritePattern<AMDAIE::DoublyStridedOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  SubsumeLoopIntoDMA(MLIRContext *context, AMDAIE::AMDAIEDeviceModel &&model,
                     bool onlyZeroStrideOnOuterDim)
      : OpInterfaceRewritePattern(context),
        deviceModel(model),
        onlyZeroStrideOnOuterDim(onlyZeroStrideOnOuterDim) {}

  /// Utility to add a loop iteration to a new offsets/sizes/strides access
  /// pattern to be inserted into the existing pattern later. The loop iteration
  /// is not inserted into the existing access pattern directly, so checks can
  /// be done on the validity of new strides and sizes to abort the rewrite
  /// before the IR is being rewritten.
  ///
  /// This function handles following cases:
  /// 1. If an offset which has an loop induction variable dependency can be
  /// found, calculate the stride and size based on the dependency, potentially
  /// taking into account an affine expression scale and bias.
  /// 2. If there is no loop induction variable dependency, the meaning of the
  /// iteration is that this strided operation is repeated `ceilDiv(upperBound -
  /// lowerBound, step)` number of times, so a new dimension is added to the
  /// access pattern with `stride == 0` and `size == ceilDiv(upperBound -
  /// lowerBound, step)`.
  LogicalResult addIterationToNewAccessPattern(
      RewriterBase &rewriter, int64_t lowerBound, int64_t upperBound,
      int64_t step, const DenseSet<Value> &inductionValues,
      const SmallVector<OpFoldResult> &offsets,
      const SmallVector<OpFoldResult> &strides,
      SmallVector<int64_t> &insertOffsets, SmallVector<int64_t> &insertSizes,
      SmallVector<int64_t> &insertStrides,
      SmallVector<std::pair<size_t, int64_t>> &updateOffsets) const {
    const int64_t nbIterations =
        calculateNbIterations(lowerBound, upperBound, step);

    bool loopDependency{false};
    for (auto &&[i, offset] : llvm::enumerate(offsets)) {
      if (auto offsetValue = dyn_cast_if_present<Value>(offset);
          inductionValues.contains(offsetValue)) {
        loopDependency = true;
        // Initialize the offsetStride to 1. This handles the case where an
        // induction variable is directly used as an offset inside a strided
        // operation.
        int64_t offsetStride = 1;
        int64_t offsetBase = 0;
        // If the offset value is determined by an affine expression, retrieve
        // the affine expression's stride scale and calculate the actual
        // offset stride.
        if (isa_and_present<affine::AffineApplyOp>(
                offsetValue.getDefiningOp())) {
          auto applyOp =
              cast<affine::AffineApplyOp>(offsetValue.getDefiningOp());
          // Retrieve the scale and optional bias from the affine map using an
          // affine expression visitor. This is the place where invalid maps are
          // filtered out. Invalid cases will have `retriever.scale == nullopt`
          // after visiting.
          AffineMap affineMap = applyOp.getAffineMap();
          RetrieveScaleAndBias retriever;
          if (failed(retriever.visit(affineMap.getResult(0)))) return failure();
          if (!retriever.scale) return failure();
          offsetStride *= retriever.scale.value();
          if (retriever.bias) offsetBase = retriever.bias.value();
        }

        // Multiplying by step size handles the non-normalized case.
        int64_t stride =
            getConstantIntValue(strides[i]).value() * offsetStride * step;

        updateOffsets.push_back(
            std::make_pair(i, offsetBase + lowerBound * offsetStride));

        // Don't add a unit iteration to better use available dimensions.
        // However, the current offset should be updated, therefore this check
        // is placed after `newOffsets[i]` has been updated.
        if (nbIterations == 1) continue;

        insertOffsets.push_back(0);
        insertSizes.push_back(nbIterations);
        insertStrides.push_back(stride);
      }
    }
    assert(insertOffsets.size() == insertSizes.size() &&
           "expected same number of offsets and sizes to be inserted");
    assert(insertOffsets.size() == insertStrides.size() &&
           "expected same number of offsets and strides to be inserted");
    // Handle the 'no loop dependency' case.
    if (!loopDependency && nbIterations != 1) {
      insertOffsets.push_back(0);
      insertSizes.push_back(nbIterations);
      insertStrides.push_back(0);
    }
    return success();
  }

  /// Rewrite function for a doubly strided operation with any loop-like parent
  /// operation.
  LogicalResult rewriteWithLoopLikeOpParent(
      AMDAIE::DoublyStridedOpInterface op, PatternRewriter &rewriter,
      const AMDAIE::DmaDimConfig &dmaDimConfig,
      const SmallVector<int64_t> &lowerBounds,
      const SmallVector<int64_t> &upperBounds,
      const SmallVector<int64_t> &steps,
      const SmallVector<DenseSet<Value>> &inductionValues,
      const DenseSet<Value> &allInductionValues) const {
    auto loopOp = dyn_cast<LoopLikeOpInterface>(op->getParentOp());
    if (!loopOp) return failure();

    // Check that the operands are not located within the scope as the
    // parentOp.
    if (llvm::any_of(op->getOperands(), [&](Value operand) {
          return !allInductionValues.contains(operand) &&
                 operand.getDefiningOp() &&
                 loopOp->isProperAncestor(operand.getDefiningOp());
        })) {
      return rewriter.notifyMatchFailure(
          op,
          "Has operands within the same scope, so the parent loop  op can't be "
          "subsumed as this transformation would move this op outside that "
          "parent op.");
    }

    // Handle ops with trait `CircularDmaOp`. If circular DMA ops don't have a
    // loop dependency, they can be hoisted outside the loop without any
    // changes as in principal they run indefinitely until the resource operated
    // on is dedicated to another operation. Therefore, care should be taken to
    // ensure that the resource being operated on is not touched within the same
    // scope.
    // TODO(jornt): find a way to annotate the DMA resource(s) being
    // operated on so that this touch checking logic can be implemented in
    // general for different ops.
    if (op->hasTrait<CircularDmaOp>() &&
        !hasLoopDependency(op, allInductionValues)) {
      rewriter.setInsertionPoint(loopOp);
      Operation *cloneOp = rewriter.clone(*op.getOperation());
      rewriter.replaceOp(op, cloneOp);
      return success();
    }

    // Initialize new access pattern offsets/sizes/strides with current values.
    SmallVector<OpFoldResult> newSourceOffsets = op.getSourceMixedOffsets();
    SmallVector<OpFoldResult> newSourceSizes = op.getSourceMixedSizes();
    SmallVector<OpFoldResult> newSourceStrides = op.getSourceMixedStrides();
    SmallVector<OpFoldResult> newTargetOffsets = op.getTargetMixedOffsets();
    SmallVector<OpFoldResult> newTargetSizes = op.getTargetMixedSizes();
    SmallVector<OpFoldResult> newTargetStrides = op.getTargetMixedStrides();

    // Verify number of dimensions needed to subsume this loop into the strided
    // access pattern and fail early if there aren't enough dimensions.
    size_t nbNonUnitIterations{0};
    for (auto &&[lb, ub, step] : llvm::zip(lowerBounds, upperBounds, steps)) {
      const int64_t nbIterations = calculateNbIterations(lb, ub, step);
      // We should not do any rewrite if we encounter a loop with no iterations.
      if (nbIterations == 0) return failure();
      if (nbIterations > 1) nbNonUnitIterations++;
    }
    if (newSourceOffsets.size() + nbNonUnitIterations >
        dmaDimConfig.sourceMaxNbDims)
      return failure();
    if (newTargetOffsets.size() + nbNonUnitIterations >
        dmaDimConfig.targetMaxNbDims)
      return failure();

    // Fail if zero stride is only supported on the outer dimension and adding
    // this loop to the strided access pattern would violate that.
    if (onlyZeroStrideOnOuterDim) {
      if (!newSourceStrides.empty()) {
        std::optional<int64_t> outerStride =
            getConstantIntValue(newSourceStrides[0]);
        if (outerStride && outerStride.value() == 0) return failure();
      }
      if (!newTargetStrides.empty()) {
        std::optional<int64_t> outerStride =
            getConstantIntValue(newTargetStrides[0]);
        if (outerStride && outerStride.value() == 0) return failure();
      }

      SmallVector<Value> dynamicSourceOffsets = op.getSourceOffsets();
      SmallVector<Value> dynamicTargetOffsets = op.getTargetOffsets();
      for (size_t i = 1; i < inductionValues.size(); i++) {
        // Skip unit iterations.
        int64_t lb = lowerBounds[i];
        int64_t ub = upperBounds[i];
        int64_t step = steps[i];
        if (calculateNbIterations(lb, ub, step) == 1) continue;
        const DenseSet<Value> &iterationIvValues = inductionValues[i];
        // If there is no dependency on the loop for non-initial iterations,
        // this will result in an non-outer stride of `0`, so fail.
        if (!dynamicSourceOffsets.empty() &&
            !llvm::any_of(dynamicSourceOffsets, [&](Value offset) {
              return iterationIvValues.contains(offset);
            })) {
          return failure();
        }
        if (!dynamicTargetOffsets.empty() &&
            !llvm::any_of(dynamicTargetOffsets, [&](Value offset) {
              return iterationIvValues.contains(offset);
            })) {
          return failure();
        }
      }
    }

    auto anyOutOfRange = [](const SmallVector<int64_t> &values,
                            const SmallVector<uint32_t> &maxValues,
                            size_t begin) -> bool {
      assert(maxValues.size() - begin >= values.size() &&
             "begin should be set so that the values don't exceed the max "
             "values slice");
      for (size_t i = 0; i < values.size(); ++i) {
        int64_t value = values[i];
        uint32_t maxValue = maxValues[begin + i];
        if (value < 0 || value > maxValue) return true;
      }
      return false;
    };

    SmallVector<int64_t> insertSourceOffsets;
    SmallVector<int64_t> insertSourceSizes;
    SmallVector<int64_t> insertSourceStrides;
    SmallVector<std::pair<size_t, int64_t>> updateSourceOffsets;
    SmallVector<int64_t> insertTargetOffsets;
    SmallVector<int64_t> insertTargetSizes;
    SmallVector<int64_t> insertTargetStrides;
    SmallVector<std::pair<size_t, int64_t>> updateTargetOffsets;

    // Add the loop iterations to the DMA access patterns.
    for (auto &&[lb, ub, step, iterationIvValues] :
         llvm::zip(lowerBounds, upperBounds, steps, inductionValues)) {
      // Add loop iteration to the access pattern on the source side.
      if (!newSourceOffsets.empty()) {
        if (failed(addIterationToNewAccessPattern(
                rewriter, lb, ub, step, iterationIvValues, newSourceOffsets,
                newSourceStrides, insertSourceOffsets, insertSourceSizes,
                insertSourceStrides, updateSourceOffsets))) {
          return failure();
        }
        SmallVector<uint32_t> maxSizes =
            dmaDimConfig.getMaxSizes<CopyOpOperateOn::Source>();
        SmallVector<uint32_t> maxStrides =
            dmaDimConfig.getMaxStrides<CopyOpOperateOn::Source>();
        assert(maxSizes.size() >=
                   insertSourceSizes.size() + newSourceSizes.size() &&
               "Max number of dimensions exceeded");
        size_t begin =
            maxSizes.size() - insertSourceSizes.size() - newSourceSizes.size();
        if (anyOutOfRange(insertSourceSizes, maxSizes, begin)) return failure();
        if (anyOutOfRange(insertSourceStrides, maxStrides, begin))
          return failure();
      }
      // Add loop iteration to the access pattern on the target side.
      if (!newTargetOffsets.empty()) {
        if (failed(addIterationToNewAccessPattern(
                rewriter, lb, ub, step, iterationIvValues, newTargetOffsets,
                newTargetStrides, insertTargetOffsets, insertTargetSizes,
                insertTargetStrides, updateTargetOffsets))) {
          return failure();
        }
        SmallVector<uint32_t> maxSizes =
            dmaDimConfig.getMaxSizes<CopyOpOperateOn::Target>();
        SmallVector<uint32_t> maxStrides =
            dmaDimConfig.getMaxStrides<CopyOpOperateOn::Target>();
        assert(maxSizes.size() >=
                   insertTargetSizes.size() + newTargetSizes.size() &&
               "Max number of dimensions exceeded");
        size_t begin =
            maxSizes.size() - insertTargetSizes.size() - newTargetSizes.size();
        if (anyOutOfRange(insertTargetSizes, maxSizes, begin)) return failure();
        if (anyOutOfRange(insertTargetStrides, maxStrides, begin))
          return failure();
      }
    }

    // Update the source and target access patterns.
    auto toOpFoldResult =
        [&](const SmallVector<int64_t> &values) -> SmallVector<OpFoldResult> {
      return llvm::map_to_vector(values, [&](int64_t v) -> OpFoldResult {
        return rewriter.getI64IntegerAttr(v);
      });
    };
    for (auto &&[index, value] : updateSourceOffsets)
      newSourceOffsets[index] = rewriter.getI64IntegerAttr(value);
    SmallVector<OpFoldResult> insertSourceOffsetsOFR =
        toOpFoldResult(insertSourceOffsets);
    SmallVector<OpFoldResult> insertSourceSizesOFR =
        toOpFoldResult(insertSourceSizes);
    SmallVector<OpFoldResult> insertSourceStridesOFR =
        toOpFoldResult(insertSourceStrides);
    newSourceOffsets.insert(newSourceOffsets.begin(),
                            insertSourceOffsetsOFR.begin(),
                            insertSourceOffsetsOFR.end());
    newSourceSizes.insert(newSourceSizes.begin(), insertSourceSizesOFR.begin(),
                          insertSourceSizesOFR.end());
    newSourceStrides.insert(newSourceStrides.begin(),
                            insertSourceStridesOFR.begin(),
                            insertSourceStridesOFR.end());

    for (auto &&[index, value] : updateTargetOffsets)
      newTargetOffsets[index] = rewriter.getI64IntegerAttr(value);
    SmallVector<OpFoldResult> insertTargetOffsetsOFR =
        toOpFoldResult(insertTargetOffsets);
    SmallVector<OpFoldResult> insertTargetSizesOFR =
        toOpFoldResult(insertTargetSizes);
    SmallVector<OpFoldResult> insertTargetStridesOFR =
        toOpFoldResult(insertTargetStrides);
    newTargetOffsets.insert(newTargetOffsets.begin(),
                            insertTargetOffsetsOFR.begin(),
                            insertTargetOffsetsOFR.end());
    newTargetSizes.insert(newTargetSizes.begin(), insertTargetSizesOFR.begin(),
                          insertTargetSizesOFR.end());
    newTargetStrides.insert(newTargetStrides.begin(),
                            insertTargetStridesOFR.begin(),
                            insertTargetStridesOFR.end());

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
  LogicalResult rewriteWithForOpParent(
      AMDAIE::DoublyStridedOpInterface op, PatternRewriter &rewriter,
      const AMDAIE::DmaDimConfig &dmaDimConfig) const {
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

    SmallVector<int64_t> lowerBounds = {lowerBound.value()};
    SmallVector<int64_t> upperBounds = {upperBound.value()};
    SmallVector<int64_t> steps = {step.value()};
    SmallVector<DenseSet<Value>> inductionValues = {curIvValues};
    return rewriteWithLoopLikeOpParent(op, rewriter, dmaDimConfig, lowerBounds,
                                       upperBounds, steps, inductionValues,
                                       curIvValues);
  }

  /// Main rewrite function for a doubly strided operation with a `scf.forall`
  /// parent operation. Only handle loop induction variables with an
  /// optional `affine.apply` user for now.
  LogicalResult rewriteWithForallOpParent(
      AMDAIE::DoublyStridedOpInterface op, PatternRewriter &rewriter,
      const AMDAIE::DmaDimConfig &dmaDimConfig) const {
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
    return rewriteWithLoopLikeOpParent(
        op, rewriter, dmaDimConfig, lowerBounds.value(), upperBounds.value(),
        steps.value(), inductionValues, allInductionValues);
  }

  LogicalResult matchAndRewrite(AMDAIE::DoublyStridedOpInterface op,
                                PatternRewriter &rewriter) const override {
    Operation *parentOp = op->getParentOp();
    if (!parentOp) return rewriter.notifyMatchFailure(op, "Has no parent");
    if (!isa<LoopLikeOpInterface>(parentOp))
      return rewriter.notifyMatchFailure(op, "Parent is not a loop-like op");

    auto hasUsersInSameScope = [&](Value result) -> bool {
      for (Operation *userOp : result.getUsers()) {
        if (userOp != op.getOperation() && parentOp->isProperAncestor(userOp)) {
          return true;
        }
      }
      return false;
    };

    uint8_t sourceMemspaceInt;
    uint8_t targetMemspaceInt;
    if (auto npuDmaOp = dyn_cast<AMDAIE::NpuDmaCpyNdOp>(op.getOperation())) {
      sourceMemspaceInt = npuDmaOp.getSourceMemorySpaceAsUInt();
      targetMemspaceInt = npuDmaOp.getTargetMemorySpaceAsUInt();

      // Check that the connection this `amdaie.npu.dma_cpy_nd` operation is
      // operating on, is not being touched within the same scope. Otherwise,
      // the rewrite is not valid in general as it would be changing the
      // temporal usage of the source connection.
      AMDAIE::ConnectionOp connectionOp = npuDmaOp.getConnectionOp();
      if (!connectionOp) {
        return rewriter.notifyMatchFailure(
            op, "should operate on an `amdaie.connection` op");
      }
      std::optional<AMDAIE::ConnectionType> connectionType =
          connectionOp.getConnectionType();
      if (connectionType &&
          connectionType.value() == AMDAIE::ConnectionType::Packet) {
        return rewriter.notifyMatchFailure(
            op,
            "operating on a packet connection, which can potentially still be "
            "merged with other connections, so abort loop subsumption as it "
            "could potentially lead to deadlocks");
      }
      if (hasUsersInSameScope(connectionOp.getResult())) {
        return rewriter.notifyMatchFailure(
            op,
            "Has users of same DMA in scope, analysis to check validity of "
            "subsumption unimplemented");
      }
    } else if (auto npuCircularDmaOp =
                   dyn_cast<AMDAIE::NpuCircularDmaCpyNdOp>(op.getOperation())) {
      // TODO(jornt): Consolidate with `NpuDmaCpyNdOp`.
      sourceMemspaceInt = npuCircularDmaOp.getSourceMemorySpaceAsUInt();
      targetMemspaceInt = npuCircularDmaOp.getTargetMemorySpaceAsUInt();

      // Check that the connection this `amdaie.npu.dma_cpy_nd` operation is
      // operating on, is not being touched within the same scope. Otherwise,
      // the rewrite is not valid in general as it would be changing the
      // temporal usage of the source connection.
      AMDAIE::ConnectionOp connectionOp = npuCircularDmaOp.getConnectionOp();
      if (!connectionOp) {
        return rewriter.notifyMatchFailure(
            op, "should operate on an `amdaie.connection` op");
      }
      if (hasUsersInSameScope(connectionOp.getResult())) {
        return rewriter.notifyMatchFailure(
            op,
            "Has users of same DMA in scope, analysis to check validity of "
            "subsumption unimplemented");
      }
    } else {
      return rewriter.notifyMatchFailure(
          op,
          "Is not an `amdaie.npu.dma_cpy_nd` or "
          "`amdaie.npu.circular_dma_cpy_nd` operation");
    }

    AMDAIE::DmaDimConfig dmaDimConfig(deviceModel, sourceMemspaceInt,
                                      targetMemspaceInt);

    if (isa<scf::ForOp>(parentOp)) {
      return rewriteWithForOpParent(op, rewriter, dmaDimConfig);
    } else if (isa<scf::ForallOp>(parentOp)) {
      return rewriteWithForallOpParent(op, rewriter, dmaDimConfig);
    } else {
      return rewriter.notifyMatchFailure(
          op, "Has parent operation of currently unsupported type");
    }
  }

 private:
  static bool hasLoopDependency(AMDAIE::DoublyStridedOpInterface op,
                                const DenseSet<Value> &inductionValues) {
    auto dependsOnInductionValue = [&](ArrayRef<Value> values) {
      return llvm::any_of(values,
                          [&](Value v) { return inductionValues.contains(v); });
    };
    SmallVector<Value> dynamicSourceOffsets = op.getSourceOffsets();
    SmallVector<Value> dynamicTargetOffsets = op.getTargetOffsets();
    if (dependsOnInductionValue(dynamicSourceOffsets)) return true;
    if (dependsOnInductionValue(dynamicTargetOffsets)) return true;
    return false;
  }

  // The device model to use for the DMA operation.
  AMDAIE::AMDAIEDeviceModel deviceModel;

  // In AIE2(+), a stride with value `0`, indicating a repeat of the subsequent
  // dimensions is only supported on the outer dimension through the use of a
  // buffer descriptor repeat count. To not bake this limitation to deeply into
  // the loop subsumption transformation, it's made an option with this flag.
  bool onlyZeroStrideOnOuterDim;
};

class AMDAIEDmaLoopSubsumptionPass
    : public impl::AMDAIEDmaLoopSubsumptionBase<AMDAIEDmaLoopSubsumptionPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEDmaLoopSubsumptionPass() = default;
  AMDAIEDmaLoopSubsumptionPass(const AMDAIEDmaLoopSubsumptionPass &pass){};
  AMDAIEDmaLoopSubsumptionPass(const AMDAIEDmaLoopSubsumptionOptions &options)
      : AMDAIEDmaLoopSubsumptionBase(options) {}
  void runOnOperation() override;
};

void AMDAIEDmaLoopSubsumptionPass::runOnOperation() {
  Operation *parentOp = getOperation();
  MLIRContext *context = &getContext();

  RewritePatternSet patterns(context);
  {
    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(parentOp);
    std::optional<AMDAIEDevice> maybeDevice = getConfigAMDAIEDevice(targetAttr);
    if (!maybeDevice) {
      parentOp->emitOpError()
          << "has no AMDAIEDevice in the target attribute configuration. This "
             "device-specific information is required to determine when loops "
             "can be subsumed into DMA operations, and must be attached to a "
             "containing ModuleOp.";
      return signalPassFailure();
    }
    AMDAIE::AMDAIEDeviceModel deviceModel =
        AMDAIE::getDeviceModel(maybeDevice.value());
    populateDmaLoopSubsumptionPattern(patterns, std::move(deviceModel),
                                      onlyZeroStrideOnOuterDim);
  }

  if (failed(applyPatternsAndFoldGreedily(parentOp, std::move(patterns)))) {
    parentOp->emitOpError("failed to subsume some loops into DMA operations");
    return signalPassFailure();
  }

  IRRewriter rewriter(parentOp->getContext());
  if (failed(moveNpuDmaSyncUsersAfterAncestorInSameBlock(rewriter, parentOp))) {
    parentOp->emitOpError(
        "failed to move DMA users to correct scope after loop subsumption");
    return signalPassFailure();
  }
}

}  // namespace

void populateDmaLoopSubsumptionPattern(RewritePatternSet &patterns,
                                       AMDAIE::AMDAIEDeviceModel &&deviceModel,
                                       bool onlyZeroStrideOnOuterDim) {
  SubsumeLoopIntoDMA pattern(patterns.getContext(), std::move(deviceModel),
                             onlyZeroStrideOnOuterDim);
  patterns.insert<SubsumeLoopIntoDMA>(std::move(pattern));
}

std::unique_ptr<Pass> createAMDAIEDmaLoopSubsumptionPass(
    AMDAIEDmaLoopSubsumptionOptions options) {
  return std::make_unique<AMDAIEDmaLoopSubsumptionPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
