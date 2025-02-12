// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

R"peano(

template<typename T, int M, int N, int val>
__attribute__((noinline)) void zero_vectorized(T *__restrict pC, unsigned offsetC)
{
  T *__restrict pC1 = pC + offsetC;
  for (unsigned r = 0; r < M; r += 1) {
    for (unsigned c = 0; c < N; c += 1) {
      unsigned o0 = N * r + c;
      pC1[o0] = T(0);
    }
  }
}

template<unsigned rowA, unsigned colA, unsigned colB, unsigned L0_M, unsigned L0_K, unsigned L0_N>
void matmul_vectorized_i8_i32(const int8 * __restrict pA, unsigned offsetA, const int8 * __restrict pB, unsigned offsetB, int32 * __restrict pC, unsigned offsetC)
{
  const unsigned size_A = L0_M * L0_K;
  const unsigned size_B = L0_K * L0_N;
  const unsigned size_C = L0_M * L0_N;

  for (unsigned z = 0; z < rowA; z += 2) {
      v64acc32 *__restrict pC0 = (v64acc32 *)(pC + offsetC + (z)*size_C);
      v64acc32 *__restrict pC1 = (v64acc32 *)(pC + offsetC + ((z + 1)) * size_C);

      for (unsigned j = 0; j < colB; j += 2) {
          const v64int8 *__restrict pA0 = (v64int8 *)(pA + offsetA + (z)*size_A);
          const v64int8 *__restrict pA1 = (v64int8 *)(pA + offsetA + ((z + 1)) * size_A);

          const v64int8 *__restrict pB0 = (v64int8 *)(pB + offsetB + (j)*colA*size_B);
          const v64int8 *__restrict pB1 = (v64int8 *)(pB + offsetB + ((j + 1))*colA * size_B);

          v64int8 A0 = *pA0;
          pA0 += rowA;
          v64int8 A1 = *pA1;
          pA1 += rowA;

          v64int8 B0 = *pB0++;
          v64int8 B1 = *pB1++;

          v64acc32 acc_C00 = *pC0;
          v64acc32 acc_C01 = *(pC0 + rowA);

          v64acc32 acc_C10 = *pC1;
          v64acc32 acc_C11 = *(pC1 + rowA);

          acc_C00 = mac_8x8_8x8(A0, B0, acc_C00);
          acc_C01 = mac_8x8_8x8(A0, B1, acc_C01);
          acc_C10 = mac_8x8_8x8(A1, B0, acc_C10);
          acc_C11 = mac_8x8_8x8(A1, B1, acc_C11);

          // chess_prepare_for_pipelining chess_loop_range(7, )
          for (unsigned i = 1; i < colA; ++i) {
              v64int8 A0 = *pA0;
              pA0 += rowA;
              v64int8 A1 = *pA1;
              pA1 += rowA;

              v64int8 B0 = *pB0++;
              v64int8 B1 = *pB1++;

              acc_C00 = mac_8x8_8x8(A0, B0, acc_C00);
              acc_C01 = mac_8x8_8x8(A0, B1, acc_C01);
              acc_C10 = mac_8x8_8x8(A1, B0, acc_C10);
              acc_C11 = mac_8x8_8x8(A1, B1, acc_C11);
          }

          // -----

          v64acc32 * __restrict pOut00 = pC0;
          *pOut00 = acc_C00;
          pC0 += rowA;

          v64acc32 * __restrict pOut01 = pC0;
          *pOut01 = acc_C01;
          pC0 += rowA;

          // -----

          v64acc32 * __restrict pOut10 = pC1;
          *pOut10 = acc_C10;
          pC1 += rowA;

          v64acc32 * __restrict pOut11 = pC1;
          *pOut11 = acc_C11;
          pC1 += rowA;
        }
    }
}

template <unsigned m, unsigned k, unsigned n>
void matmul_vectorized_8x8x8_i8_i8_i32(const int8 *__restrict pA,
                                      unsigned offsetA,
                                      const int8 *__restrict pB,
                                      unsigned offsetB, int32 *__restrict pC,
                                      unsigned offsetC) {
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;
  static_assert(m / r > 0);
  static_assert(k / s > 0);
  static_assert(n / t > 0);
  return matmul_vectorized_i8_i32<m / r, k / s, n / t, r, s, t>
      (pA, offsetA, pB, offsetB, pC, offsetC);
}

extern "C" {

#define matmul_combos_i8(X, M, N, K)                                  \
  X(int8, i8, int8, i8, int32, i32, M, N, K, 8, 8, 8)

#define zero_fill_combos(X, M, N)  \
  X(int32, i32, M, N, 16)

#define matmul_vectorized_c_func(lhs_ctype_in, lhs_mlir_type_in,                                                 \
                                 rhs_ctype_in, rhs_mlir_type_in,                                                 \
                                 acc_ctype_out, acc_mlir_type_out, M, N, K, r, s, t)                             \
  void matmul_##lhs_mlir_type_in##_##rhs_mlir_type_in##_##acc_mlir_type_out##_##M##x##N##x##K##_##r##x##s##x##t( \
      lhs_ctype_in *a_in, unsigned offsetA, rhs_ctype_in *b_in, unsigned offsetB,                                \
      acc_ctype_out *c_out, unsigned offsetC) {                                                                  \
    matmul_vectorized_##r##x##s##x##t##_##lhs_mlir_type_in##_##rhs_mlir_type_in##_##acc_mlir_type_out<           \
        M, K, N>(a_in, offsetA, b_in, offsetB, c_out, offsetC);                                                  \
  }

#define zero_vectorized_c_func(ctype_out, mlir_type_out, M, N, r)             \
  void zero_##mlir_type_out##_##M##x##N(ctype_out *c_out, unsigned offsetC) { \
    zero_vectorized<ctype_out, M, N, r>(c_out, offsetC);                      \
  }

matmul_combos_i8(matmul_vectorized_c_func, 32, 32, 32)
matmul_combos_i8(matmul_vectorized_c_func, 32, 32, 64)

zero_fill_combos(zero_vectorized_c_func, 32, 32)

}  // extern "C"
)peano"
