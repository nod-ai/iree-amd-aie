// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define NOCPP

#include <stdint.h>

#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#ifndef __chess__
#include "aiebase_chess.h"
#include "aiebase_typedefs.h"
#endif

#include <aie_api/aie.hpp>

template <typename T, int M, int N, int r>
void zero_fill_vectorized(T *__restrict pC, unsigned offsetC) {
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

#define zero_fill_combos(X, M, N)  \
  X(bfloat16, bf16, M, N, N)     \
  X(float, f32, M, N, N/2)         \
  X(int32, i32, M, N, N/2)

#define zero_fill_vectorized_c_func(ctype_out, mlir_type_out, M, N, r)             \
  void zero_fill_##mlir_type_out##_##M##x##N(ctype_out *c_out, unsigned offsetC) { \
    zero_fill_vectorized<ctype_out, M, N, r>(c_out, offsetC);                      \
  }

zero_fill_combos(zero_fill_vectorized_c_func, 16, 8)
zero_fill_combos(zero_fill_vectorized_c_func, 16, 16)
zero_fill_combos(zero_fill_vectorized_c_func, 32, 16)
zero_fill_combos(zero_fill_vectorized_c_func, 32, 32)
zero_fill_combos(zero_fill_vectorized_c_func, 64, 32)
zero_fill_combos(zero_fill_vectorized_c_func, 64, 64)

}  // extern "C"
// clang-format on
