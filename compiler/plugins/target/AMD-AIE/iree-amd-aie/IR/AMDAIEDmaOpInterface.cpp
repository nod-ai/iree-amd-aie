// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>

#include "iree-amd-aie/IR/AMDAIEDmaOpInterface.h"

#include "iree-amd-aie/IR/AMDAIEAttrs.h"

/// Include the definitions of the dma-like interfaces.
#include "iree-amd-aie/IR/AMDAIEDmaOpInterface.cpp.inc"

namespace mlir::iree_compiler::AMDAIE {

namespace detail {

/// Utility to compute the static base offset on either the source or target
/// side of a doubly strided operation.
template <CopyOpOperateOn OperateOn>
std::optional<int64_t> getStaticBaseOffset(DoublyStridedOpInterface op) {
  int64_t baseOffset = 0;
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> strides;
  if constexpr (OperateOn == CopyOpOperateOn::Source) {
    offsets = op.getSourceMixedOffsets();
    strides = op.getSourceMixedStrides();
  } else if constexpr (OperateOn == CopyOpOperateOn::Target) {
    offsets = op.getTargetMixedOffsets();
    strides = op.getTargetMixedStrides();
  } else {
    assert(false && "Function can only operate on Source or Target");
  }
  for (auto &&[offset, stride] : llvm::zip(offsets, strides)) {
    std::optional<int64_t> constantOffset = getConstantIntValue(offset);
    std::optional<int64_t> constantStride = getConstantIntValue(stride);
    // If offset is zero, we can just continue to the next one. This enables
    // the case where the stride is dynamic.
    if (constantOffset && constantOffset.value() == 0) continue;
    if (constantOffset && constantStride) {
      baseOffset += (constantOffset.value() * constantStride.value());
    } else {
      return std::nullopt;
    }
  }
  return baseOffset;
}

template <CopyOpOperateOn OperateOn>
std::optional<int64_t> getStaticSize(DoublyStridedOpInterface op) {
  SmallVector<OpFoldResult> sizes;
  if constexpr (OperateOn == CopyOpOperateOn::Source) {
    sizes = op.getSourceMixedSizes();
  } else if constexpr (OperateOn == CopyOpOperateOn::Target) {
    sizes = op.getTargetMixedSizes();
  } else {
    assert(false && "Function can only operate on Source or Target");
  }
  if (sizes.size() == 0) return 0;
  std::optional<SmallVector<int64_t>> staticSizes = getConstantIntValues(sizes);
  if (!staticSizes) return std::nullopt;
  return std::accumulate(staticSizes->begin(), staticSizes->end(), 1,
                         std::multiplies<>());
}

/// Return the static base offset on the source side if it can be computed.
/// Otherwise, returns nullopt.
std::optional<int64_t> getSourceStaticBaseOffset(DoublyStridedOpInterface op) {
  return getStaticBaseOffset<CopyOpOperateOn::Source>(op);
}


std::optional<int64_t> getSourceStaticSize(DoublyStridedOpInterface op) {
  return getStaticSize<CopyOpOperateOn::Source>(op);
}

/// Return the static base offset on the target side if it can be computed.
/// Otherwise, returns nullopt.
std::optional<int64_t> getTargetStaticBaseOffset(DoublyStridedOpInterface op) {
  return getStaticBaseOffset<CopyOpOperateOn::Target>(op);
}


std::optional<int64_t> getTargetStaticSize(DoublyStridedOpInterface op) {
  return getStaticSize<CopyOpOperateOn::Target>(op);
}

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
