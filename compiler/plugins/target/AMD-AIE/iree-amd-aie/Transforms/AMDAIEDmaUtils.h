// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_AMDAIEDMAUTILS_H_
#define IREE_AMD_AIE_TRANSFORMS_AMDAIEDMAUTILS_H_

#include "iree-amd-aie/IR/AMDAIEAttrs.h"
#include "iree-amd-aie/IR/AMDAIEDmaOpInterface.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::AMDAIE {

/// Fold subsequent dimensions within a strided access pattern that describe a
/// single linear access. Returns `success` if folding took place.
LogicalResult foldLinearDims(MLIRContext *ctx,
                             const SmallVector<OpFoldResult> &offsets,
                             const SmallVector<OpFoldResult> &sizes,
                             const SmallVector<OpFoldResult> &strides,
                             SmallVector<OpFoldResult> &newOffsets,
                             SmallVector<OpFoldResult> &newSizes,
                             SmallVector<OpFoldResult> &newStrides);

/// Fold single dimension linear accesses and make them implicit. `This
/// operation happens in place. Returns `success` if folding took place.
LogicalResult foldSingleDim(SmallVector<OpFoldResult> &offsets,
                            SmallVector<OpFoldResult> &sizes,
                            SmallVector<OpFoldResult> &strides);

/// Fold unit dimensions within a strided access pattern. Returns `success` if
/// folding took place.
LogicalResult foldUnitDims(const SmallVector<OpFoldResult> &offsets,
                           const SmallVector<OpFoldResult> &strides,
                           const SmallVector<OpFoldResult> &sizes,
                           SmallVector<OpFoldResult> &newOffsets,
                           SmallVector<OpFoldResult> &newStrides,
                           SmallVector<OpFoldResult> &newSizes);

/// Utility to discard all non-zero offsets that have dimension equal to 1 on
/// the same index of the provided shape. This helps with updating DMA
/// operations for a shape change. If an empty shape is passed, all non-zero
/// offsets will be removed.
template <CopyOpOperateOn OperateOn>
AMDAIE::DoublyStridedOpInterface discardAllNonZeroOffsets(
    RewriterBase &rewriter, AMDAIE::DoublyStridedOpInterface op,
    SmallVector<int64_t> &shape) {
  SmallVector<OpFoldResult> newSourceOffsets;
  SmallVector<OpFoldResult> newSourceSizes;
  SmallVector<OpFoldResult> newSourceStrides;
  SmallVector<OpFoldResult> newTargetOffsets;
  SmallVector<OpFoldResult> newTargetSizes;
  SmallVector<OpFoldResult> newTargetStrides;
  if constexpr (OperateOn == CopyOpOperateOn::Source) {
    SmallVector<OpFoldResult> offsets = op.getSourceMixedOffsets();
    SmallVector<OpFoldResult> sizes = op.getSourceMixedSizes();
    SmallVector<OpFoldResult> strides = op.getSourceMixedStrides();
    // Set shape to a vector of ones as a default.
    if (shape.empty()) {
      SmallVector<int64_t> ones(offsets.size(), 1);
      shape = ones;
    }
    if (shape.size() != offsets.size()) return op;
    // Fill source offsets/sizes/strides.
    for (auto &&[offset, size, stride, dim] :
         llvm::zip(offsets, sizes, strides, shape)) {
      std::optional<int64_t> constantOffset = getConstantIntValue(offset);
      if (dim == 1 && !constantOffset) continue;
      if (dim == 1 && constantOffset && constantOffset.value() != 0) continue;
      newSourceOffsets.push_back(offset);
      newSourceSizes.push_back(size);
      newSourceStrides.push_back(stride);
    }
    newTargetOffsets = op.getTargetMixedOffsets();
    newTargetSizes = op.getTargetMixedSizes();
    newTargetStrides = op.getTargetMixedStrides();
  } else if constexpr (OperateOn == CopyOpOperateOn::Target) {
    SmallVector<OpFoldResult> offsets = op.getTargetMixedOffsets();
    SmallVector<OpFoldResult> sizes = op.getTargetMixedSizes();
    SmallVector<OpFoldResult> strides = op.getTargetMixedStrides();
    // Set shape to a vector of ones as a default.
    if (shape.empty()) {
      SmallVector<int64_t> ones(offsets.size(), 1);
      shape = ones;
    }
    if (shape.size() != offsets.size()) return op;
    // Fill source offsets/sizes/strides.
    for (auto &&[offset, size, stride, dim] :
         llvm::zip(offsets, sizes, strides, shape)) {
      std::optional<int64_t> constantOffset = getConstantIntValue(offset);
      if (dim == 1 && !constantOffset) continue;
      if (dim == 1 && constantOffset && constantOffset.value() != 0) continue;
      newTargetOffsets.push_back(offset);
      newTargetSizes.push_back(size);
      newTargetStrides.push_back(stride);
    }
    newSourceOffsets = op.getSourceMixedOffsets();
    newSourceSizes = op.getSourceMixedSizes();
    newSourceStrides = op.getSourceMixedStrides();
  }
  rewriter.setInsertionPointAfter(op);
  auto newDoublyStridedOp = op.createDoublyStridedOp(
      rewriter, newTargetOffsets, newTargetSizes, newTargetStrides,
      newSourceOffsets, newSourceSizes, newSourceStrides);
  rewriter.replaceOp(op, newDoublyStridedOp.getOperation());
  return newDoublyStridedOp;
}

}  // namespace mlir::iree_compiler::AMDAIE

#endif
