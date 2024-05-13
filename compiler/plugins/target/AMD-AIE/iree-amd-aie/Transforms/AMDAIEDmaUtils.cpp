// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AMDAIEDmaUtils.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"

namespace mlir::iree_compiler::AMDAIE {

/// Utility to retrieve a constant index from an OpFoldResult.
int64_t getConstantIndexOrAssert(OpFoldResult dim) {
  std::optional<int64_t> size = getConstantIntValue(dim);
  assert(size.has_value() && "expect constant index");
  return size.value();
}

/// Fold subsequent dimensions within a strided access pattern that describe a
/// single linear access.
///
/// Example:
///
/// `offsets: [0, 0], sizes: [2, 8], strides: [8, 1]`
///
/// This describes accessing two times eight elements with a stride of eight in
/// between. For example if the input is: [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
/// 2, 2, 2, 2] this will access [1, 1, 1, 1, 1, 1, 1, 1] and then [2, 2, 2, 2,
/// 2, 2, 2, 2].
///
/// This is the same access pattern as accessing
/// [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2] in one go, which is
/// represented by:
///
/// `offsets: [0], sizes: [16], strides: [1]`
LogicalResult foldLinearDims(MLIRContext *ctx,
                             const SmallVector<OpFoldResult> &offsets,
                             const SmallVector<OpFoldResult> &sizes,
                             const SmallVector<OpFoldResult> &strides,
                             SmallVector<OpFoldResult> &newOffsets,
                             SmallVector<OpFoldResult> &newSizes,
                             SmallVector<OpFoldResult> &newStrides) {
  bool foldableLinearDimsFound = false;

  if (offsets.size() == 0) {
    return success(foldableLinearDimsFound);
  }

  newOffsets.push_back(offsets[0]);
  newStrides.push_back(strides[0]);
  newSizes.push_back(sizes[0]);

  for (int i = 1; i < offsets.size(); i++) {
    // Conditions for folding a dim.
    // 1. size(i) x stride(i) == stride(i-1), with this we can have new size(i-1)
    // = size(i-1) * size(i), stride(i-1) = stride(i) and then fold away the i
    // dimension
    // 2. Offset(i-1) = 0. This is required because we are dropping the offset
    // of the i-1 dimension and doing offset(i-1) = offset(i)
    int vecSize = newOffsets.size();
    if (isConstantIntValue(newOffsets[vecSize - 1], 0) &&
        getConstantIndexOrAssert(sizes[i]) *
                getConstantIndexOrAssert(strides[i]) ==
            getConstantIndexOrAssert(newStrides[vecSize - 1])) {
      foldableLinearDimsFound = true;
      int vecSize = newOffsets.size();
      newOffsets[vecSize - 1] = offsets[i];
      newStrides[vecSize - 1] = strides[i];
      newSizes[vecSize - 1] = getAsIndexOpFoldResult(
          ctx, getConstantIndexOrAssert(sizes[i]) *
                   getConstantIndexOrAssert(newSizes[vecSize - 1]));

      continue;
    }
    newOffsets.push_back(offsets[i]);
    newStrides.push_back(strides[i]);
    newSizes.push_back(sizes[i]);
  }
  return success(foldableLinearDimsFound);
}

/// Fold single dimension linear accesses and make them implicit. This requires
/// offset 0 and stride 1.
///
/// Example:
///   offsets: [%c0], sizes: [%c1024], strides: [%c1]
/// becomes:
///   offsets: [], sizes: [], strides: []
LogicalResult foldSingleDim(SmallVector<OpFoldResult> &offsets,
                            SmallVector<OpFoldResult> &sizes,
                            SmallVector<OpFoldResult> &strides) {
  if (offsets.size() == 0) {
    return failure();
  }
  if (offsets.size() == 1 && getConstantIntValue(offsets[0]) &&
      getConstantIntValue(offsets[0]).value() == 0 &&
      getConstantIntValue(strides[0]) &&
      getConstantIntValue(strides[0]).value() == 1) {
    offsets.clear();
    sizes.clear();
    strides.clear();
    return success();
  }
  return failure();
}

/// Fold unit dimensions within a strided access pattern.
LogicalResult foldUnitDims(const SmallVector<OpFoldResult> &offsets,
                           const SmallVector<OpFoldResult> &sizes,
                           const SmallVector<OpFoldResult> &strides,
                           SmallVector<OpFoldResult> &newOffsets,
                           SmallVector<OpFoldResult> &newSizes,
                           SmallVector<OpFoldResult> &newStrides) {
  bool foldableUnitDimsFound = false;

  for (int i = 0; i < offsets.size(); i++) {
    // Dim can be folded if offset is 0 and size is 1
    if (isConstantIntValue(offsets[i], 0) && isConstantIntValue(sizes[i], 1)) {
      foldableUnitDimsFound = true;
      continue;
    }
    newOffsets.push_back(offsets[i]);
    newStrides.push_back(strides[i]);
    newSizes.push_back(sizes[i]);
  }
  return success(foldableUnitDimsFound);
}

}  // namespace mlir::iree_compiler::AMDAIE
