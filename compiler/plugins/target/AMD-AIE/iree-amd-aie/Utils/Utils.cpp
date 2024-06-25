// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Utils/Utils.h"

namespace mlir::iree_compiler::AMDAIE {

std::optional<int64_t> getAccessRangeExtent(
    const SmallVector<int64_t> &sizes, const SmallVector<int64_t> &strides) {
  assert(sizes.size() == strides.size() &&
         "sizes and strides are expected to have the same size");
  if (llvm::any_of(sizes, [](int64_t size) { return size <= 0; }))
    return std::nullopt;
  if (llvm::any_of(strides, [](int64_t stride) { return stride <= 0; }))
    return std::nullopt;
  int64_t extent = 0;
  for (auto &&[size, stride] : llvm::zip(sizes, strides))
    extent += ((size - 1) * stride);
  // In the end add 1 to get the final extent, not the last access index. For
  // example, last access index 1023 would return an extent of 1024 as 1024
  // elements are part of the access pattern's range.
  if (!strides.empty()) extent += 1;
  return extent;
}

}  // namespace mlir::iree_compiler::AMDAIE
