// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AMDAIETileSizeSelectionUtils.h"

#include "AMDAIEUtils.h"

namespace mlir::iree_compiler::AMDAIE {

void findLargestTileSizes(uint32_t m, uint32_t n, uint32_t k, uint32_t& curMax,
                          const TileParams& params, TileSize& best) {
  if (m < 16 || n < 16 || k < 16) return;

  bool isInputDivisible = (params.inputM % m == 0) &&
                          (params.inputN % n == 0) && (params.inputK % k == 0);
  bool isIntrinsicDivisible = (m % params.vectorM == 0) &&
                              (n % params.vectorN == 0) &&
                              (k % params.vectorK == 0);

  if (isInputDivisible && isIntrinsicDivisible) {
    uint32_t A = params.numBytesA * m * k * params.bufferDepthA;
    uint32_t B = params.numBytesB * n * k * params.bufferDepthB;
    uint32_t C = params.numBytesC * m * n * params.bufferDepthC;
    uint32_t Acc = params.numBytesAcc * m * n * params.bufferDepthAcc;
    int64_t memoryUsage = A + B + C + Acc;

    if (memoryUsage < params.memoryLimit && memoryUsage > curMax) {
      curMax = memoryUsage;
      best = {m, n, k};
    }
  }

  // Reduce n first, then m and k. n is reduced first because we expect more
  // columns than rows on Strix, and this would balance the L2 M and N tile
  // sizes. k is preferred to be larger (reduce at last), because there is only
  // one level of tiling of the reduction dimension. If the input type has less
  // bitwidth than output type, we prefer k size at least the double size of
  // m or n tile size.
  if (n >= m && n >= k / 2) {
    findLargestTileSizes(m, n / 2, k, curMax, params, best);
  } else if (m >= k / 2) {
    findLargestTileSizes(m / 2, n, k, curMax, params, best);
  } else {
    findLargestTileSizes(m, n, k / 2, curMax, params, best);
  }
}

TileSize selectL1TileSizes(const TileParams& params) {
  uint32_t curMax = 0;
  TileSize best = {16, 16, 16};
  uint32_t mStart =
      detail::findLargestFactor(params.inputM, 128, params.vectorM);
  uint32_t nStart =
      detail::findLargestFactor(params.inputN, 128, params.vectorN);
  uint32_t kStart =
      detail::findLargestFactor(params.inputK, 128, params.vectorK);
  findLargestTileSizes(mStart, nStart, kStart, curMax, params, best);
  return best;
}

}  // namespace mlir::iree_compiler::AMDAIE
