// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIEDmaUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-canonicalize-doubly-strided-dma"

namespace mlir::iree_compiler::AMDAIE {

// namespace {

/// Recognize linear accesses across multiple DMA access dimensions and fold
/// them.
LogicalResult foldDmaOpLinearDims(RewriterBase &rewriter,
                                  AMDAIE::DoublyStridedOpInterface op) {
  OpBuilder::InsertionGuard guard(rewriter);
  SmallVector<OpFoldResult> sourceOffsets = op.getSourceMixedOffsets();
  SmallVector<OpFoldResult> sourceSizes = op.getSourceMixedSizes();
  SmallVector<OpFoldResult> sourceStrides = op.getSourceMixedStrides();
  SmallVector<OpFoldResult> targetOffsets = op.getTargetMixedOffsets();
  SmallVector<OpFoldResult> targetSizes = op.getTargetMixedSizes();
  SmallVector<OpFoldResult> targetStrides = op.getTargetMixedStrides();
  SmallVector<OpFoldResult> newSourceOffsets, newSourceSizes, newSourceStrides,
      newTargetOffsets, newTargetSizes, newTargetStrides;
  LogicalResult sourceRes =
      foldLinearDims(op.getContext(), sourceOffsets, sourceSizes, sourceStrides,
                     newSourceOffsets, newSourceSizes, newSourceStrides);
  LogicalResult targetRes =
      foldLinearDims(op.getContext(), targetOffsets, targetSizes, targetStrides,
                     newTargetOffsets, newTargetSizes, newTargetStrides);
  if (failed(sourceRes) && failed(targetRes)) {
    return failure();
  }

  rewriter.setInsertionPointAfter(op);
  auto newDoublyStridedOp = op.createDoublyStridedOp(
      rewriter, newTargetOffsets, newTargetSizes, newTargetStrides,
      newSourceOffsets, newSourceSizes, newSourceStrides);
  rewriter.replaceOp(op, newDoublyStridedOp.getOperation());
  return success();
}

/// Fold single dimension linear accesses and make them implicit.
LogicalResult foldDmaOpSingleDims(RewriterBase &rewriter,
                                  AMDAIE::DoublyStridedOpInterface op) {
  OpBuilder::InsertionGuard guard(rewriter);
  SmallVector<OpFoldResult> sourceOffsets = op.getSourceOffsets();
  SmallVector<OpFoldResult> sourceSizes = op.getSourceSizes();
  SmallVector<OpFoldResult> sourceStrides = op.getSourceStrides();
  SmallVector<OpFoldResult> targetOffsets = op.getTargetOffsets();
  SmallVector<OpFoldResult> targetSizes = op.getTargetSizes();
  SmallVector<OpFoldResult> targetStrides = op.getTargetStrides();
  LogicalResult sourceRes =
      foldSingleDim(sourceOffsets, sourceSizes, sourceStrides);
  LogicalResult targetRes =
      foldSingleDim(targetOffsets, targetSizes, targetStrides);
  if (failed(sourceRes) && failed(targetRes)) {
    return failure();
  }

  rewriter.setInsertionPointAfter(op);
  auto newDoublyStridedOp = op.createDoublyStridedOp(
      rewriter, targetOffsets, targetSizes, targetStrides, sourceOffsets,
      sourceSizes, sourceStrides);
  rewriter.replaceOp(op, newDoublyStridedOp.getOperation());
  return success();
}

/// Fold unit dimensions within a strided access pattern.
LogicalResult foldDmaOpUnitDims(RewriterBase &rewriter,
                                AMDAIE::DoublyStridedOpInterface op) {
  OpBuilder::InsertionGuard guard(rewriter);
  SmallVector<OpFoldResult> sourceOffsets = op.getSourceMixedOffsets();
  SmallVector<OpFoldResult> sourceSizes = op.getSourceMixedSizes();
  SmallVector<OpFoldResult> sourceStrides = op.getSourceMixedStrides();
  SmallVector<OpFoldResult> targetOffsets = op.getTargetMixedOffsets();
  SmallVector<OpFoldResult> targetSizes = op.getTargetMixedSizes();
  SmallVector<OpFoldResult> targetStrides = op.getTargetMixedStrides();
  SmallVector<OpFoldResult> newSourceOffsets, newSourceSizes, newSourceStrides,
      newTargetOffsets, newTargetSizes, newTargetStrides;
  LogicalResult sourceRes =
      foldUnitDims(sourceOffsets, sourceSizes, sourceStrides, newSourceOffsets,
                   newSourceSizes, newSourceStrides);
  LogicalResult targetRes =
      foldUnitDims(targetOffsets, targetSizes, targetStrides, newTargetOffsets,
                   newTargetSizes, newTargetStrides);
  if (failed(sourceRes) && failed(targetRes)) {
    return failure();
  }

  rewriter.setInsertionPointAfter(op);
  auto newDoublyStridedOp = op.createDoublyStridedOp(
      rewriter, newTargetOffsets, newTargetSizes, newTargetStrides,
      newSourceOffsets, newSourceSizes, newSourceStrides);
  rewriter.replaceOp(op, newDoublyStridedOp.getOperation());
  return success();
}

namespace {

class AMDAIECanonicalizeDoublyStridedOpPass
    : public impl::AMDAIECanonicalizeDoublyStridedOpBase<
          AMDAIECanonicalizeDoublyStridedOpPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIECanonicalizeDoublyStridedOpPass() = default;
  AMDAIECanonicalizeDoublyStridedOpPass(
      const AMDAIECanonicalizeDoublyStridedOpPass &pass){};
  void runOnOperation() override;
};

void AMDAIECanonicalizeDoublyStridedOpPass::runOnOperation() {
  Operation *parentOp = getOperation();
  IRRewriter rewriter(parentOp->getContext());

  // Fold linear dimensions within a DMA op.
  parentOp->walk([&](AMDAIE::DoublyStridedOpInterface dmaOp) {
    (void)foldDmaOpLinearDims(rewriter, dmaOp);
  });

  // Fold DMA unit dimensions.
  parentOp->walk([&](AMDAIE::DoublyStridedOpInterface dmaOp) {
    (void)foldDmaOpUnitDims(rewriter, dmaOp);
  });

  // Make DMA accesses with single dimension implicit.
  parentOp->walk([&](AMDAIE::DoublyStridedOpInterface dmaOp) {
    (void)foldDmaOpSingleDims(rewriter, dmaOp);
  });
}

}  // namespace

std::unique_ptr<Pass> createAMDAIECanonicalizeDoublyStridedOpPass() {
  return std::make_unique<AMDAIECanonicalizeDoublyStridedOpPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
