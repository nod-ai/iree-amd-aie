// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_AMDAIE_UTILS_DMA_UTILS_H_
#define IREE_COMPILER_AMDAIE_UTILS_DMA_UTILS_H_

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::iree_compiler::AMDAIE {

/// Utility to compute the extent of the range of values by the provided size
/// and stride access pattern. Returns std::nullopt if the extent can't be
/// computed due to negative values or zeros. Returns 0 in case of empty
/// offsets/strides/sizes.
///
/// The extent can be computed using the following formula:
///
/// product_{dim} {(sizes[dim] - 1) * strides[dim]} + 1
///
/// The addition of 1 at the end is to return the size of the range in which
/// elements are being accessed instead of the index of the last accessed
/// element.
///
/// Example:
///
/// Access pattern: (sizes: [2, 2], strides: [4, 1])
///
/// With this access pattern, following elements are accessed (`1`:
/// accessed, `-`: not accessed):
///
/// 1 1 - - 1 1
///
/// Here, the size of the range of elements being accessed is 6.
std::optional<int64_t> getAccessRangeExtent(
    const SmallVector<int64_t> &sizes, const SmallVector<int64_t> &strides);

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_COMPILER_AMDAIE_UTILS_DMA_UTILS_H_
