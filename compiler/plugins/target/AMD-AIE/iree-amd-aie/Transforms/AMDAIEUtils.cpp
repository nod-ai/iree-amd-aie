// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AMDAIEUtils.h"

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::iree_compiler::AMDAIE {

namespace {

// Generate a DenseMap key we can use for the element types (alternatives
// considered: implement tombstone for std::array, or use std::map instead of
// DenseMap).
constexpr uint32_t getElementTypeKey(uint32_t a, uint32_t b, uint32_t c) {
  return a + (b << 8) + (c << 16);
}

// Map from (LHS bitwidth, RHS bitwidth, Accumulator bitwidth) to the AIE
// instruction size (m, n, k) for the integer types with those bitwidths.
const auto& getIntegerMatmulInstructionSizeMap() {
  // Sanity check.
  static_assert(getElementTypeKey(1, 2, 3) == 1 + 2 * 256 + 3 * 65536);

  static DenseMap<uint32_t, std::array<uint32_t, 3>> matmulIntSizes{

      // `vector<4x16xi8>`  | `vector<16x8xi4>`  | `vector<4x8xi32>`
      {getElementTypeKey(8, 4, 32), {4, 8, 16}},

      // `vector<4x8xi8>`   | `vector<8x8xi8>`   | `vector<4x8xi32>`
      {getElementTypeKey(8, 8, 32), {4, 8, 8}},

      // `vector<4x4xi16>`  | `vector<4x8xi8>`   | `vector<4x8xi32>`
      {getElementTypeKey(16, 8, 32), {4, 8, 4}},

      // `vector<4x2xi16>`  | `vector<2x8xi16>`  | `vector<4x8xi32>`
      {getElementTypeKey(16, 16, 32), {4, 8, 2}},

      // `vector<2x8xi16>`  | `vector<8x8xi8>`   | `vector<2x8xi64>`
      // `vector<4x8xi16>`  | `vector<8x4xi8>`   | `vector<4x4xi64>`
      //   choosing the first i16 x i8 -> i64 instruction (arbitrarily)
      {getElementTypeKey(16, 8, 64), {2, 8, 8}},

      // `vector<2x4xi16>`  | `vector<4x8xi16>`  | `vector<2x8xi64>`
      // `vector<4x4xi16>`  | `vector<4x4xi16>`  | `vector<4x4xi64>`
      //   choosing the first i16 x i16 -> i64 instruction (arbitrarily)
      {getElementTypeKey(16, 16, 64), {2, 8, 4}},

      // `vector<4x2xi32>`  | `vector<2x4xi16>`  | `vector<4x4xi64>`
      {getElementTypeKey(32, 16, 64), {4, 4, 2}},
  };
  return matmulIntSizes;
}
}  // namespace

FailureOr<std::array<uint32_t, 3>> getAIEIntegerMatmulInstructionSize(
    uint32_t nBitsLhs, uint32_t nBitsRhs, uint32_t nBitsAcc) {
  const auto& mapForIntTypes = getIntegerMatmulInstructionSizeMap();
  auto it =
      mapForIntTypes.find(getElementTypeKey(nBitsLhs, nBitsRhs, nBitsAcc));
  if (it == mapForIntTypes.end()) {
    return failure();
  }
  return it->second;
}

FailureOr<std::array<uint32_t, 3>> getAIEMatmulInstructionSize(Type elTypeLhs,
                                                               Type elTypeRhs,
                                                               Type elTypeAcc) {
  bool allFloatingPoint = elTypeLhs.isa<FloatType>() &&
                          elTypeRhs.isa<FloatType>() &&
                          elTypeAcc.isa<FloatType>();

  bool allInteger = elTypeLhs.isa<IntegerType>() &&
                    elTypeRhs.isa<IntegerType>() &&
                    elTypeAcc.isa<IntegerType>();

  if (!allInteger && !allFloatingPoint) {
    return failure();
  }

  auto nBitsLhs = elTypeLhs.getIntOrFloatBitWidth();
  auto nBitsRhs = elTypeRhs.getIntOrFloatBitWidth();
  auto nBitsAcc = elTypeAcc.getIntOrFloatBitWidth();

  if (allFloatingPoint) {
    if (nBitsLhs == 16 && nBitsRhs == 16 && nBitsAcc == 32) {
      return std::array<uint32_t, 3>{4, 4, 8};
    }
    // There is only 1 floating point case in the table (handled above).
    return failure();
  }

  assert(allInteger);

  return getAIEIntegerMatmulInstructionSize(nBitsLhs, nBitsRhs, nBitsAcc);
}

}  // namespace mlir::iree_compiler::AMDAIE
