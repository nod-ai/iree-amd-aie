// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AMDAIETileSizeSelectionUtils.h"

#include "AMDAIEUtils.h"

namespace mlir::iree_compiler::AMDAIE {

constexpr unsigned minL1TileSize = 16;
constexpr unsigned maxL1TileSize = 128;

void findLargestL1TileSizes(uint32_t m, uint32_t n, uint32_t k,
                            uint32_t& curMax, const TileParams& params,
                            TileSize& best) {
  if (m < minL1TileSize || n < minL1TileSize || k < minL1TileSize) return;

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
    findLargestL1TileSizes(m, n / 2, k, curMax, params, best);
  } else if (m >= k / 2) {
    findLargestL1TileSizes(m / 2, n, k, curMax, params, best);
  } else {
    findLargestL1TileSizes(m, n, k / 2, curMax, params, best);
  }
}

TileSize selectL1TileSizes(const TileParams& params) {
  uint32_t curMax = 0;
  TileSize best = {minL1TileSize, minL1TileSize, minL1TileSize};
  uint32_t mStart =
      detail::findLargestFactor(params.inputM, maxL1TileSize, params.vectorM);
  uint32_t nStart =
      detail::findLargestFactor(params.inputN, maxL1TileSize, params.vectorN);
  uint32_t kStart =
      detail::findLargestFactor(params.inputK, maxL1TileSize, params.vectorK);
  findLargestL1TileSizes(mStart, nStart, kStart, curMax, params, best);
  return best;
}

void findLargestL2TileSizes(uint32_t m, uint32_t n, const uint32_t k,
                            uint32_t& curMax, const TileParams& params,
                            TileSize& best) {
  if (m > params.inputM || n > params.inputN || curMax > params.memoryLimit)
    return;

  bool isInputDivisible = (params.inputM % m == 0) && (params.inputN % n == 0);
  if (isInputDivisible) {
    uint32_t A = params.numBytesA * m * k * params.bufferDepthA;
    uint32_t B = params.numBytesB * n * k * params.bufferDepthB;
    uint32_t C = params.numBytesC * m * n * params.bufferDepthC;
    uint32_t Acc = params.numBytesAcc * m * n * params.bufferDepthAcc;
    int64_t memoryUsage = A + B + C + Acc;

    if (memoryUsage <= params.memoryLimit && memoryUsage > curMax) {
      curMax = memoryUsage;
      best = {m, n, k};
    }
  }

  if (m * 2 > params.inputM) {
    findLargestL2TileSizes(m, n * 2, k, curMax, params, best);
  } else if (n * 2 > params.inputN) {
    findLargestL2TileSizes(m * 2, n, k, curMax, params, best);
  } else if (n >= m) {
    findLargestL2TileSizes(m * 2, n, k, curMax, params, best);
  } else {
    findLargestL2TileSizes(m, n * 2, k, curMax, params, best);
  }
}

TileSize selectL2TileSizes(const TileParams& params, const uint32_t maxL1TileM,
                           const uint32_t maxL1TileN) {
  uint32_t curMax = 0;
  // Start with the L1 tile sizes for m, n dimension. The k dimension is kept
  // as constant (for now it's the input K size).
  TileSize best = {maxL1TileM, maxL1TileN, params.inputK};
  findLargestL2TileSizes(maxL1TileM, maxL1TileN, params.inputK, curMax, params,
                         best);
  return best;
}

}  // namespace mlir::iree_compiler::AMDAIE
