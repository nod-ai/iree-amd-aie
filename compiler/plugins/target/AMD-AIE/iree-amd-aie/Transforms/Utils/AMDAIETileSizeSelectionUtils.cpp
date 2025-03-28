// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AMDAIETileSizeSelectionUtils.h"

#include "AMDAIEUtils.h"

namespace mlir::iree_compiler::AMDAIE {

void findLargestTileSizes(int64_t m, int64_t n, int64_t k, uint32_t& curMax,
                          const TileParams& params, TileSize& best) {
  if (m < 16 || n < 16 || k < 16) return;

  uint32_t A = params.numBytesA * m * k * params.isDoubleBufferA;
  uint32_t B = params.numBytesB * n * k * params.isDoubleBufferB;
  uint32_t C = params.numBytesC * m * n * params.isDoubleBufferC;
  uint32_t Acc = params.numBytesAcc * m * n * params.isDoubleBufferAcc;
  int64_t memoryUsage = A + B + C + Acc;

  bool isDivisible = (params.inputM % m == 0) && (params.inputN % n == 0) &&
                     (params.inputK % k == 0);

  if (isDivisible && memoryUsage < params.memoryLimit && memoryUsage > curMax) {
    curMax = memoryUsage;
    best = {m, n, k};
  }

  // Reduce m first, then n and k. k is preferred to be larger (reduce at last),
  // because there is only one level of tiling of the reduction dimension.
  // If the input type has less bitwidth than output type, we prefer k size at
  // least double size of the m/n tile size.
  if (m >= n && m >= k / 2) {
    findLargestTileSizes(m - 16, n, k, curMax, params, best);
  } else if (n >= k / 2) {
    findLargestTileSizes(m, n - 16, k, curMax, params, best);
  } else {
    findLargestTileSizes(m, n, k - 16, curMax, params, best);
  }
}

TileSize selectL1TileSizes(const TileParams& params) {
  uint32_t curMax = 0;
  TileSize best = {16, 16, 16};
  int64_t mStart = detail::findLargestFactor(params.inputM, 128, 4);
  int64_t nStart = detail::findLargestFactor(params.inputN, 128, 4);
  int64_t kStart = detail::findLargestFactor(params.inputK, 128, 8);
  findLargestTileSizes(mStart, nStart, kStart, curMax, params, best);
  return best;
}

}  // namespace mlir::iree_compiler::AMDAIE
