// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDmaOpInterface.h"

/// Include the definitions of the dma-like interfaces.
#include "iree-amd-aie/IR/AMDAIEDmaOpInterface.cpp.inc"

namespace mlir::iree_compiler::AMDAIE {

namespace detail {

LogicalResult verifyStaticOrDynamicEntryInvariant(Operation *op, StringRef name,
                                                  ArrayRef<int64_t> staticVals,
                                                  ValueRange values) {
  unsigned expectedNumDynamicEntries = llvm::count_if(
      staticVals,
      [&](int64_t staticVal) { return ShapedType::isDynamic(staticVal); });
  if (values.size() != expectedNumDynamicEntries)
    return op->emitError("expected ")
           << expectedNumDynamicEntries << " dynamic " << name << " values";
  return success();
}

LogicalResult verifyNonNegativeInvariant(Operation *op, StringRef name,
                                         ArrayRef<int64_t> staticVals) {
  for (int64_t val : staticVals) {
    if (val < 0 && !ShapedType::isDynamic(val))
      return op->emitError("expected ")
             << name << " to be non-negative, but got " << val;
  }
  return success();
}

LogicalResult verifyDoublyStridedOp(DoublyStridedOpInterface op) {
  // Checks whether source and target access descriptors (`offsets`, `sizes`,
  // `strides`) have the same number of dimensions.
  if (op.getTargetMixedOffsets().size() != op.getTargetMixedSizes().size()) {
    return op.emitError(
        "target sizes should have same number of dimensions as target offsets");
  }
  if (op.getTargetMixedOffsets().size() != op.getTargetMixedStrides().size()) {
    return op.emitError(
        "target strides should have same number of dimensions as target "
        "offsets");
  }
  if (op.getSourceMixedOffsets().size() != op.getSourceMixedSizes().size()) {
    return op.emitError(
        "source sizes should have same number of dimensions as source offsets");
  }
  if (op.getSourceMixedOffsets().size() != op.getSourceMixedStrides().size()) {
    return op.emitError(
        "source strides should have same number of dimensions as source "
        "offsets");
  }

  // If an entry of the static access pattern operands is equal to a special
  // sentinel value, namely `ShapedType::kDynamic`, then the corresponding entry
  // should be a dynamic value.
  if (failed(verifyStaticOrDynamicEntryInvariant(op, "target offsets",
                                                 op.getTargetStaticOffsets(),
                                                 op.getTargetOffsets())))
    return failure();
  if (failed(verifyStaticOrDynamicEntryInvariant(
          op, "target sizes", op.getTargetStaticSizes(), op.getTargetSizes())))
    return failure();
  if (failed(verifyStaticOrDynamicEntryInvariant(op, "target strides",
                                                 op.getTargetStaticStrides(),
                                                 op.getTargetStrides())))
    return failure();
  if (failed(verifyStaticOrDynamicEntryInvariant(op, "source offsets",
                                                 op.getSourceStaticOffsets(),
                                                 op.getSourceOffsets())))
    return failure();
  if (failed(verifyStaticOrDynamicEntryInvariant(
          op, "source sizes", op.getSourceStaticSizes(), op.getSourceSizes())))
    return failure();
  if (failed(verifyStaticOrDynamicEntryInvariant(op, "source strides",
                                                 op.getSourceStaticStrides(),
                                                 op.getSourceStrides())))
    return failure();

  // Check whether static offsets, sizes and strides are non-negative
  if (failed(verifyNonNegativeInvariant(op, "target offsets",
                                        op.getTargetStaticOffsets())))
    return failure();
  if (failed(verifyNonNegativeInvariant(op, "target sizes",
                                        op.getTargetStaticSizes())))
    return failure();
  if (failed(verifyNonNegativeInvariant(op, "target strides",
                                        op.getTargetStaticStrides())))
    return failure();
  if (failed(verifyNonNegativeInvariant(op, "source offsets",
                                        op.getSourceStaticOffsets())))
    return failure();
  if (failed(verifyNonNegativeInvariant(op, "source sizes",
                                        op.getSourceStaticSizes())))
    return failure();
  if (failed(verifyNonNegativeInvariant(op, "source strides",
                                        op.getSourceStaticStrides())))
    return failure();
  return success();
}

}  // namespace detail

}  // namespace mlir::iree_compiler::AMDAIE
