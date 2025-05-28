// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

template <int M, int N, int r>
void zero_fill_vectorized(v16int32 *__restrict pC, unsigned offsetC) {
  v16int32 zeros = broadcast_zero_to_v16int32();
  for (unsigned i = offsetC / r; i < offsetC / r + M * N / r; i++) {
    pC[i] = zeros;
  }
}

template <int M, int N, int r>
void zero_fill_vectorized(v16float *__restrict pC, unsigned offsetC) {
  v16float zeros = broadcast_zero_to_v16float();
  for (unsigned i = offsetC / r; i < offsetC / r + M * N / r; i++) {
    pC[i] = zeros;
  }
}

// clang-format off
extern "C" {

#define zero_fill_combos_i32(X, M, N)  \
  X(v16int32, i32, M, N, 16)

#define zero_fill_combos_f32(X, M, N)  \
  X(v16float, f32, M, N, 16)

#define zero_fill_vectorized_c_func(ctype_out, mlir_type_out, M, N, r)             \
  void zero_fill_##mlir_type_out##_##M##x##N(ctype_out *c_out, unsigned offsetC) { \
    zero_fill_vectorized<M, N, r>(c_out, offsetC);                      \
  }

zero_fill_combos_i32(zero_fill_vectorized_c_func, 32, 32)
zero_fill_combos_i32(zero_fill_vectorized_c_func, 64, 32)
zero_fill_combos_i32(zero_fill_vectorized_c_func, 64, 64)

zero_fill_combos_f32(zero_fill_vectorized_c_func, 16, 8)
zero_fill_combos_f32(zero_fill_vectorized_c_func, 16, 16)
zero_fill_combos_f32(zero_fill_vectorized_c_func, 32, 32)

}  // extern "C"
// clang-format on
