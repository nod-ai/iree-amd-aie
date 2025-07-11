// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include <aie_api/aie.hpp>
#include <type_traits>

template <typename T, int M, int N>
void zero_fill_vectorized(T *__restrict pC, unsigned offsetC) {
  // 512 bit store units for npu4.
  constexpr int r = 512 / (sizeof(T) * 8);
  static_assert(M * N / r > 0);
  const aie::vector<T, r> zeros = aie::zeros<T, r>();
  T *__restrict pC1 = pC + offsetC;
  const T *__restrict c_end = pC1 + M * N;
  for (; pC1 + r < c_end; pC1 += r) {
    aie::store_v(pC1, zeros);
  }
  // Do a scalar write for any remainder not divisible by vector instruction
  // size r
  for (; pC1 < c_end; pC1++) {
    *pC1 = 0;
  }
}

// clang-format off
extern "C" {

#define zero_fill_combos_bf16(X, M, N)  \
  X(bfloat16, bf16, M, N)

#define zero_fill_combos_f32(X, M, N)  \
  X(float, f32, M, N)

#define zero_fill_combos_i32(X, M, N)  \
  X(int32, i32, M, N)

#define zero_fill_vectorized_c_func(ctype_out, mlir_type_out, M, N)             \
  void zero_fill_##mlir_type_out##_##M##x##N(ctype_out *c_out, unsigned offsetC) { \
    zero_fill_vectorized<ctype_out, M, N>(c_out, offsetC);                      \
  }

zero_fill_combos_bf16(zero_fill_vectorized_c_func, 4, 1024)
zero_fill_combos_i32(zero_fill_vectorized_c_func, 64, 64)

}  // extern "C"
// clang-format on
