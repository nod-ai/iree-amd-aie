// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

template <int M, int N, int r, int u>
void trunci_vectorized(v32int32 *__restrict in, int64_t offsetIn, int64_t shift,
                       v32int8 *__restrict out, int64_t offsetOut) {
  static_assert(M * N / r / u > 0);
#pragma clang loop unroll_count(M *N / r / u)
  for (unsigned i = 0; i < M * N / r; i++) {
    out[offsetOut + i] = ssrs((v32acc32)in[offsetIn + i], shift, 0);
  }
}

// clang-format off
extern "C" {

#define trunci_combos_i32_i8(X, M, N)  \
  X(v32int32, i32, v32int8, i8, M, N, 32, 8)

// A vectorized truncation function on a buffer of size M * N with vectorization size
// 'r' and unrolling factor 'u'. The unrolling factor is typically found experimentally.
#define trunci_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out, M, N, r, u)                  \
  void trunci_##mlir_type_in##_##mlir_type_out##_##M##x##N(                                       \
      ctype_in *in, int64_t offsetIn, int64_t shift, ctype_out *out, int64_t offsetOut) {         \
    trunci_vectorized<M, N, r, u>(in, offsetIn, shift, out, offsetOut);                              \
  }

trunci_combos_i32_i8(trunci_c_func, 32, 32)
trunci_combos_i32_i8(trunci_c_func, 64, 32)
trunci_combos_i32_i8(trunci_c_func, 64, 64)

}  // extern "C"
// clang-format on
