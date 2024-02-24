// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_AMDAIEUTILS_H_
#define IREE_AMD_AIE_TRANSFORMS_AMDAIEUTILS_H_

#include <array>

#include "mlir/IR/Types.h"

namespace mlir::iree_compiler::AMDAIE {

// This function is based on the following table pulled from the
// AIEVec_MatMulOp documentation in
// mlir-aie/include/aie/Dialect/AIEVec/IR/AIEVecOps.td
//
//   lhs                | rhs                | accumulator
//  :------------------:|:------------------:|:-----------------:
//   `vector<4x16xi8>`  | `vector<16x8xi4>`  | `vector<4x8xi32>`
//   `vector<4x8xi8>`   | `vector<8x8xi8>`   | `vector<4x8xi32>`
//   `vector<4x4xi16>`  | `vector<4x8xi8>`   | `vector<4x8xi32>`
//   `vector<4x2xi16>`  | `vector<2x8xi16>`  | `vector<4x8xi32>`
//   `vector<2x8xi16>`  | `vector<8x8xi8>`   | `vector<2x8xi64>`
//   `vector<4x8xi16>`  | `vector<8x4xi8>`   | `vector<4x4xi64>`
//   `vector<2x4xi16>`  | `vector<4x8xi16>`  | `vector<2x8xi64>`
//   `vector<4x4xi16>`  | `vector<4x4xi16>`  | `vector<4x4xi64>`
//   `vector<4x2xi32>`  | `vector<2x4xi16>`  | `vector<4x4xi64>`
//   `vector<4x8xbf16>` | `vector<8x4xbf16>` | `vector<4x4xf32>`
//
// An instruction size (m, n, k) is returned for each combination of element
// type in the table. Combinations of element type that are not covered by the
// table return failure.
//
// Example: consider the first line of the table:
//   `vector<4x16xi8>`  | `vector<16x8xi4>`  | `vector<4x8xi32>`
//
// This first line says that if 'lhs' is an i8 tensor, 'rhs' is an i4 tensor
// and 'accumulator' is an i32 tensor, then there is an AIE instruction for
// matmul with m = 4, n = 8, k = 16.
FailureOr<std::array<uint32_t, 3>> getAIEMatmulInstructionSize(
    Type elTypeLhs, Type elTypeRhs, Type elTypeAcc);

// Return the AIE instruction size (m, n, k) for the integer types with
// bitwidths nBitsLhs, nBitsRhs, and nBitsAcc. Based on the table above.
FailureOr<std::array<uint32_t, 3>> getAIEIntegerMatmulInstructionSize(
    uint32_t nBitsLhs, uint32_t nBitsRhs, uint32_t nBitsAcc);

}  // namespace mlir::iree_compiler::AMDAIE

#endif
