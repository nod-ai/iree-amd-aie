// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_AMDAIEDMAUTILS_H_
#define IREE_AMD_AIE_TRANSFORMS_AMDAIEDMAUTILS_H_

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"

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

}  // namespace mlir::iree_compiler::AMDAIE

#endif
