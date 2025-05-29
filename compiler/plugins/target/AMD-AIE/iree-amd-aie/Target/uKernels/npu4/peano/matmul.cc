// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

template <unsigned rowA, unsigned colA, unsigned colB, unsigned L0_M,
          unsigned L0_K, unsigned L0_N>
void matmul_vectorized_i8_i32(const int8 *__restrict pA, unsigned offsetA,
                              const int8 *__restrict pB, unsigned offsetB,
                              int32 *__restrict pC, unsigned offsetC) {
  const unsigned size_A = L0_M * L0_K;
  const unsigned size_B = L0_K * L0_N;
  const unsigned size_C = L0_M * L0_N;

  // Use a double set of A/B variables to help the compiler in finding better
  // pipelining solutions. The number of registers needed among them should stay
  // within vector register limits, sp we expect these to be mapped to actual
  // vector registers.
  v64int8 A0;
  v64int8 A1;
  v64int8 B0;
  v64int8 B1;

  v64int8 A0_1;
  v64int8 A1_1;
  v64int8 B0_1;
  v64int8 B1_1;

  // The accumulator variables for calculating the results of the AxB MAC
  // operations. This should fit within the number of available accumulator
  // registers, so we expect these to be mapped to actual accumulator registers.
  v64acc32 acc_C00;
  v64acc32 acc_C01;
  v64acc32 acc_C10;
  v64acc32 acc_C11;

  for (unsigned z = 0; z < rowA; z += 2) {
    v64acc32 *__restrict pC0 = (v64acc32 *)(pC + offsetC + (z)*size_C);
    v64acc32 *__restrict pC1 = (v64acc32 *)(pC + offsetC + ((z + 1)) * size_C);

    for (unsigned j = 0; j < colB; j += 2) {
      const v64int8 *__restrict pA0 = (v64int8 *)(pA + offsetA + (z)*size_A);
      const v64int8 *__restrict pA1 =
          (v64int8 *)(pA + offsetA + ((z + 1)) * size_A);

      const v64int8 *__restrict pB0 =
          (v64int8 *)(pB + offsetB + (j)*colA * size_B);
      const v64int8 *__restrict pB1 =
          (v64int8 *)(pB + offsetB + ((j + 1)) * colA * size_B);

      acc_C00 = *pC0;
      acc_C01 = *(pC0 + rowA);

      acc_C10 = *pC1;
      acc_C11 = *(pC1 + rowA);

      A0 = *pA0;
      pA0 += rowA;
      A1 = *pA1;
      pA1 += rowA;

      B0 = *pB0++;
      B1 = *pB1++;

      A0_1 = *pA0;
      pA0 += rowA;
      A1_1 = *pA1;
      pA1 += rowA;

      B0_1 = *pB0++;
      B1_1 = *pB1++;
      acc_C00 = mac_8x8_8x8(A0, B0, acc_C00);
      acc_C01 = mac_8x8_8x8(A0, B1, acc_C01);
      acc_C10 = mac_8x8_8x8(A1, B0, acc_C10);
      acc_C11 = mac_8x8_8x8(A1, B1, acc_C11);

      for (unsigned i = 1; i < colA - 1; i += 2) {
        A0 = *pA0;
        pA0 += rowA;
        A1 = *pA1;
        pA1 += rowA;

        B0 = *pB0++;
        B1 = *pB1++;

        acc_C00 = mac_8x8_8x8(A0_1, B0_1, acc_C00);
        acc_C01 = mac_8x8_8x8(A0_1, B1_1, acc_C01);
        acc_C10 = mac_8x8_8x8(A1_1, B0_1, acc_C10);
        acc_C11 = mac_8x8_8x8(A1_1, B1_1, acc_C11);

        A0_1 = *pA0;
        pA0 += rowA;
        A1_1 = *pA1;
        pA1 += rowA;

        B0_1 = *pB0++;
        B1_1 = *pB1++;

        acc_C00 = mac_8x8_8x8(A0, B0, acc_C00);
        acc_C01 = mac_8x8_8x8(A0, B1, acc_C01);
        acc_C10 = mac_8x8_8x8(A1, B0, acc_C10);
        acc_C11 = mac_8x8_8x8(A1, B1, acc_C11);
      }

      acc_C00 = mac_8x8_8x8(A0_1, B0_1, acc_C00);
      acc_C01 = mac_8x8_8x8(A0_1, B1_1, acc_C01);
      acc_C10 = mac_8x8_8x8(A1_1, B0_1, acc_C10);
      acc_C11 = mac_8x8_8x8(A1_1, B1_1, acc_C11);

      // -----

      v64acc32 *__restrict pOut00 = pC0;
      *pOut00 = acc_C00;
      pC0 += rowA;

      v64acc32 *__restrict pOut01 = pC0;
      *pOut01 = acc_C01;
      pC0 += rowA;

      // -----

      v64acc32 *__restrict pOut10 = pC1;
      *pOut10 = acc_C10;
      pC1 += rowA;

      v64acc32 *__restrict pOut11 = pC1;
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
  static_assert(m / r > 0,
                "There should be at least one iteration on the M dimension.");
  static_assert(
      k / s % 2 == 0,
      "The ukernel is manually unrolled in the K iteration with a factor of 2,"
      "so the number of K iterations should be a factor of 2.");
  static_assert(n / t > 0,
                "There should be at least one iteration on the N dimension.");
  return matmul_vectorized_i8_i32<m / r, k / s, n / t, r, s, t>(
      pA, offsetA, pB, offsetB, pC, offsetC);
}

v64bfp16ebs8 load_v64bf16_as_bfp16(const bfloat16 *__restrict p) {
  v32bfloat16 v0 = *(v32bfloat16 *)(p);
  v32bfloat16 v1 = *(v32bfloat16 *)(p + 32);
  v32accfloat accum0 = ups(v0);
  v32accfloat accum1 = ups(v1);
  v64accfloat accum = concat(accum0, accum1);
  return to_v64bfp16ebs8(accum);
}

v64bfp16ebs8 load_v64bf16_as_bfp16_T(const bfloat16 *__restrict p) {
  v32bfloat16 v0 = *(v32bfloat16 *)(p);
  v32bfloat16 v1 = *(v32bfloat16 *)(p + 32);
  v32bfloat16 v0_shuffed = shuffle(v0, 29);
  v32bfloat16 v1_shuffed = shuffle(v1, 29);
  v32bfloat16 v_shuffed_lo = shuffle(v0_shuffed, v1_shuffed, 14);
  v32bfloat16 v_shuffed_hi = shuffle(v0_shuffed, v1_shuffed, 15);
  v32accfloat accum0 = ups(v_shuffed_lo);
  v32accfloat accum1 = ups(v_shuffed_hi);
  v64accfloat accum = concat(accum0, accum1);
  return to_v64bfp16ebs8(accum);
}

template <unsigned rowA, unsigned colA, unsigned colB, unsigned L0_M,
          unsigned L0_K, unsigned L0_N>
void matmul_vectorized_bf16_f32(const bfloat16 *__restrict pA, unsigned offsetA,
                                const bfloat16 *__restrict pB, unsigned offsetB,
                                float *__restrict pC, unsigned offsetC) {
  const unsigned size_A = L0_M * L0_K;
  const unsigned size_B = L0_K * L0_N;
  const unsigned size_C = L0_M * L0_N;

  v64bfp16ebs8 A0;
  v64bfp16ebs8 A1;
  v64bfp16ebs8 B0;
  v64bfp16ebs8 B1;
  v64accfloat acc_C00;
  v64accfloat acc_C01;
  v64accfloat acc_C10;
  v64accfloat acc_C11;

  for (unsigned z = 0; z < rowA; z += 2) {
    v64accfloat *__restrict pC0 = (v64accfloat *)(pC + offsetC + (z)*size_C);
    v64accfloat *__restrict pC1 =
        (v64accfloat *)(pC + offsetC + ((z + 1)) * size_C);

    for (unsigned j = 0; j < colB; j += 2) {
      const bfloat16 *__restrict pA0 = (bfloat16 *)(pA + offsetA + (z)*size_A);
      const bfloat16 *__restrict pA1 =
          (bfloat16 *)(pA + offsetA + ((z + 1)) * size_A);

      const bfloat16 *__restrict pB0 =
          (bfloat16 *)(pB + offsetB + (j)*colA * size_B);
      const bfloat16 *__restrict pB1 =
          (bfloat16 *)(pB + offsetB + ((j + 1)) * colA * size_B);

      A0 = load_v64bf16_as_bfp16(pA0);
      pA0 += rowA * size_A;
      A1 = load_v64bf16_as_bfp16(pA1);
      pA1 += rowA * size_A;

      B0 = load_v64bf16_as_bfp16_T(pB0);
      pB0 += size_B;
      B1 = load_v64bf16_as_bfp16_T(pB1);
      pB1 += size_B;

      acc_C00 = *pC0;
      acc_C01 = *(pC0 + rowA);

      acc_C10 = *pC1;
      acc_C11 = *(pC1 + rowA);

      acc_C00 = mac_8x8_8x8T(A0, B0, acc_C00);
      acc_C01 = mac_8x8_8x8T(A0, B1, acc_C01);
      acc_C10 = mac_8x8_8x8T(A1, B0, acc_C10);
      acc_C11 = mac_8x8_8x8T(A1, B1, acc_C11);

      for (unsigned i = 1; i < colA; ++i) {
        A0 = load_v64bf16_as_bfp16(pA0);
        pA0 += rowA * size_A;
        A1 = load_v64bf16_as_bfp16(pA1);
        pA1 += rowA * size_A;

        B0 = load_v64bf16_as_bfp16_T(pB0);
        pB0 += size_B;
        B1 = load_v64bf16_as_bfp16_T(pB1);
        pB1 += size_B;

        acc_C00 = mac_8x8_8x8T(A0, B0, acc_C00);
        acc_C01 = mac_8x8_8x8T(A0, B1, acc_C01);
        acc_C10 = mac_8x8_8x8T(A1, B0, acc_C10);
        acc_C11 = mac_8x8_8x8T(A1, B1, acc_C11);
      }

      // -----

      v64accfloat *__restrict pOut00 = pC0;
      *pOut00 = acc_C00;
      pC0 += rowA;

      v64accfloat *__restrict pOut01 = pC0;
      *pOut01 = acc_C01;
      pC0 += rowA;

      // -----

      v64accfloat *__restrict pOut10 = pC1;
      *pOut10 = acc_C10;
      pC1 += rowA;

      v64accfloat *__restrict pOut11 = pC1;
      *pOut11 = acc_C11;
      pC1 += rowA;
    }
  }
}

template <unsigned m, unsigned k, unsigned n>
void matmul_vectorized_8x8x8_bf16_bf16_f32(const bfloat16 *__restrict pA,
                                           unsigned offsetA,
                                           const bfloat16 *__restrict pB,
                                           unsigned offsetB,
                                           float *__restrict pC,
                                           unsigned offsetC) {
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;
  static_assert(m / r > 0);
  static_assert(k / s > 0);
  static_assert(n / t > 0);
  return matmul_vectorized_bf16_f32<m / r, k / s, n / t, r, s, t>(
      pA, offsetA, pB, offsetB, pC, offsetC);
}

// clang-format off
extern "C" {

#define matmul_combos_i8(X, M, N, K)                                  \
  X(int8, i8, int8, i8, int32, i32, M, N, K, 8, 8, 8)

#define matmul_combos_bfp16(X, M, N, K)                                     \
  X(bfloat16, bf16, bfloat16, bf16, float, f32, M, N, K, 8, 8, 8)

#define matmul_vectorized_c_func(lhs_ctype_in, lhs_mlir_type_in,                                                 \
                                 rhs_ctype_in, rhs_mlir_type_in,                                                 \
                                 acc_ctype_out, acc_mlir_type_out, M, N, K, r, s, t)                             \
  void matmul_##lhs_mlir_type_in##_##rhs_mlir_type_in##_##acc_mlir_type_out##_##M##x##N##x##K##_##r##x##s##x##t( \
      lhs_ctype_in *a_in, unsigned offsetA, rhs_ctype_in *b_in, unsigned offsetB,                                \
      acc_ctype_out *c_out, unsigned offsetC) {                                                                  \
    matmul_vectorized_##r##x##s##x##t##_##lhs_mlir_type_in##_##rhs_mlir_type_in##_##acc_mlir_type_out<           \
        M, K, N>(a_in, offsetA, b_in, offsetB, c_out, offsetC);                                                  \
  }

matmul_combos_i8(matmul_vectorized_c_func, 32, 32, 32)
matmul_combos_i8(matmul_vectorized_c_func, 32, 32, 64)
matmul_combos_i8(matmul_vectorized_c_func, 64, 32, 128)
matmul_combos_i8(matmul_vectorized_c_func, 32, 32, 128)
matmul_combos_i8(matmul_vectorized_c_func, 64, 64, 128)

matmul_combos_bfp16(matmul_vectorized_c_func, 16, 8, 32)
matmul_combos_bfp16(matmul_vectorized_c_func, 16, 8, 64)
matmul_combos_bfp16(matmul_vectorized_c_func, 16, 16, 32)
matmul_combos_bfp16(matmul_vectorized_c_func, 32, 32, 32)
matmul_combos_bfp16(matmul_vectorized_c_func, 32, 32, 64)
matmul_combos_bfp16(matmul_vectorized_c_func, 32, 16, 128)
matmul_combos_bfp16(matmul_vectorized_c_func, 32, 32, 128)

}  // extern "C"
// clang-format on
