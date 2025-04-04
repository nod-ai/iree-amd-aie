// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_AMDAIETILESIZESELECTIONUTILS_H_
#define IREE_AMD_AIE_TRANSFORMS_AMDAIETILESIZESELECTIONUTILS_H_

#include <stdint.h>

namespace mlir::iree_compiler::AMDAIE {

struct TileParams {
  int64_t memoryLimit;
  uint32_t numBytesA, numBytesB, numBytesC, numBytesAcc;
  uint32_t isDoubleBufferA, isDoubleBufferB, isDoubleBufferC, isDoubleBufferAcc;
  uint32_t inputM, inputN, inputK;
  uint32_t vectorM, vectorN, vectorK;
};

struct TileSize {
  uint32_t M, N, K;
  bool operator==(const TileSize& tileSize) const {
    return M == tileSize.M && N == tileSize.N && K == tileSize.K;
  }
};

TileSize selectL1TileSizes(const TileParams& params);

}  // namespace mlir::iree_compiler::AMDAIE

#endif
