// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

R"chess(
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

template<typename T, int M, int N, int val>
void zero_vectorized(T *__restrict pC, unsigned offsetC)
{
  T *__restrict pC1 = pC + offsetC;
  for (unsigned r = 0; r < M; r += 1) {
    for (unsigned c = 0; c < N; c += 1) {
      unsigned o0 = N * r + c;
      pC1[o0] = T(0);
    }
  }
}


template <typename T_in> aie::accum<accfloat, 64> load_v64(const T_in *__restrict p) {
  aie::vector<T_in, 32> v0 = aie::load_v<32>(p);
  aie::accum<accfloat, 32> accum0 = aie::accum<accfloat, 32>(v0);
  aie::vector<T_in, 32> v1 = aie::load_v<32>(p + 32);
  aie::accum<accfloat, 32> accum1 = aie::accum<accfloat, 32>(v1);
  aie::accum<accfloat, 64>  accum = aie::concat(accum0, accum1);
  return accum;
}

template <typename T_in> aie::accum<accfloat, 64> load_v64_T(const T_in *__restrict p) {
  aie::vector<T_in, 32> v0 = aie::load_v<32>(p);
  aie::vector<T_in, 32> v1 = aie::load_v<32>(p + 32);
  aie::vector<T_in, 32> v0Shuffled = ::shuffle(v0, 29);//shuffle_T16_4x8
  aie::vector<T_in, 32> v1Shuffled = ::shuffle(v1, 29);//shuffle_T16_4x8
  aie::vector<T_in, 32> vShuffledLo = ::shuffle(v0Shuffled, v1Shuffled, 14);//shuffle_T64_2x8_lo
  aie::vector<T_in, 32> vShuffledHi = ::shuffle(v0Shuffled, v1Shuffled, 15);//shuffle_T64_2x8_hi
  aie::accum<accfloat, 32> accum0 = aie::accum<accfloat, 32>(vShuffledLo);
  aie::accum<accfloat, 32> accum1 = aie::accum<accfloat, 32>(vShuffledHi);
  aie::accum<accfloat, 64>  accum = aie::concat(accum0, accum1);
  return accum;
}

template<unsigned rowA, unsigned colA, unsigned colB, unsigned L0_M, unsigned L0_K, unsigned L0_N>
void matmul_vectorized_bf16_f32(const bfloat16 * __restrict pA, unsigned offsetA, const bfloat16 * __restrict pB, unsigned offsetB, float * __restrict pC, unsigned offsetC)
{
  const unsigned size_A = L0_M * L0_K;
  const unsigned size_B = L0_K * L0_N;
  const unsigned size_C = L0_M * L0_N;
  using MMUL = aie::detail::mmul_bfp16_bfp16<L0_M, L0_K, L0_N, bfp16ebs8, bfp16ebs8, 32>;

  v32accfloat * restrict pOut = (v32accfloat *) (pC + offsetC);

  for (unsigned z = 0; z < rowA; z += 2)
    chess_loop_range(4, ) {
      float *__restrict pC1 = pC + offsetC + (z)*size_C;
      float *__restrict pC2 = pC + offsetC + ((z + 1)) * size_C;


      for (unsigned j = 0; j < colB; j += 2)
        chess_prepare_for_pipelining chess_loop_range(4, ) {
          const bfloat16 *__restrict pA1 = pA + offsetA + (z)*size_A;
          const bfloat16 *__restrict pA2 = pA + offsetA + ((z + 1)) * size_A;
          const bfloat16 *__restrict pB1 = pB + offsetB + (j)*colA*size_B;
          const bfloat16 *__restrict pB2 = pB + offsetB + ((j + 1))*colA * size_B;

          aie::accum<accfloat, size_A>  accumA0 = load_v64<bfloat16>(pA1);
          aie::bfp_vector<bfp16ebs8, size_A> A0 = accumA0.template to_vector<bfp16ebs8>();
          pA1 += rowA * size_A;
          aie::accum<accfloat, size_A>  accumA1 = load_v64<bfloat16>(pA2);
          aie::bfp_vector<bfp16ebs8, size_A> A1 = accumA1.template to_vector<bfp16ebs8>();
          pA2 += rowA * size_A;
          aie::accum<accfloat, size_B>  accumB0 = load_v64_T<bfloat16>(pB1);
          aie::bfp_vector<bfp16ebs8, size_B> B0 = accumB0.template to_vector<bfp16ebs8>();
          pB1 += size_B;
          aie::accum<accfloat, size_B>  accumB1 = load_v64_T<bfloat16>(pB2);
          aie::bfp_vector<bfp16ebs8, size_B> B1 = accumB1.template to_vector<bfp16ebs8>();
          pB2 += size_B;

          aie::accum<accfloat, size_C>  acc_C00 = load_v64<float>(pC1);
          aie::accum<accfloat, size_C>  acc_C01 = load_v64<float>(pC1 + size_C * rowA);
          aie::accum<accfloat, size_C>  acc_C10 = load_v64<float>(pC2);
          aie::accum<accfloat, size_C>  acc_C11 = load_v64<float>(pC2 + size_C * rowA);

          acc_C00 = mac_8x8_8x8T_conf( A0, B0, acc_C00, 0, 0, 0 );
          acc_C01 = mac_8x8_8x8T_conf( A0, B1, acc_C01, 0, 0, 0 );
          acc_C10 = mac_8x8_8x8T_conf( A1, B0, acc_C10, 0, 0, 0 );
          acc_C11 = mac_8x8_8x8T_conf( A1, B1, acc_C11, 0, 0, 0 );

          for (unsigned i = 1; i < colA; ++i)
            chess_prepare_for_pipelining chess_loop_range(7, ) {
              accumA0 = load_v64<bfloat16>(pA1);
              A0 = accumA0.template to_vector<bfp16ebs8>();
              pA1 += rowA * size_A;
              accumA1 = load_v64<bfloat16>(pA2);
              A1 = accumA1.template to_vector<bfp16ebs8>();
              pA2 += rowA * size_A;
              accumB0 = load_v64_T<bfloat16>(pB1);
              B0 = accumB0.template to_vector<bfp16ebs8>();
              pB1 += size_B;
              accumB1 = load_v64_T<bfloat16>(pB2);
              B1 = accumB1.template to_vector<bfp16ebs8>();
              pB2 += size_B;

              acc_C00 = mac_8x8_8x8T_conf( A0, B0, acc_C00, 0, 0, 0 );
              acc_C01 = mac_8x8_8x8T_conf( A0, B1, acc_C01, 0, 0, 0 );
              acc_C10 = mac_8x8_8x8T_conf( A1, B0, acc_C10, 0, 0, 0 );
              acc_C11 = mac_8x8_8x8T_conf( A1, B1, acc_C11, 0, 0, 0 );
            }


          v64float * __restrict pOut00 = (v64float *) pC1;
          *pOut00 = (v64float)(acc_C00);
          pC1 += size_C * rowA;

          v64float * __restrict pOut01 = (v64float *) pC1;
          *pOut01 = (v64float)(acc_C01);
          pC1 += size_C * rowA;

          v64float * __restrict pOut10 = (v64float *) pC2;
          *pOut10 = (v64float)(acc_C10);
          pC2 += size_C * rowA;

          v64float * __restrict pOut11 = (v64float *) pC2;
          *pOut11 = (v64float)(acc_C11);
          pC2 += size_C * rowA;
        }
    }
}

template <unsigned m, unsigned k, unsigned n>
void matmul_vectorized_8x8x8_bf16_bf16_f32(const bfloat16 *__restrict pA,
                                      unsigned offsetA,
                                      const bfloat16 *__restrict pB,
                                      unsigned offsetB, float *__restrict pC,
                                      unsigned offsetC) {
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;
  static_assert(m / r > 0);
  static_assert(k / s > 0);
  static_assert(n / t > 0);
  return matmul_vectorized_bf16_f32<m / r, k / s, n / t, r, s, t>
      (pA, offsetA, pB, offsetB, pC, offsetC);
}

extern "C" {

#define matmul_combos(X, M, N, K)                                     \
  X(bfloat16, bf16, bfloat16, bf16, float, f32, M, N, K, 8, 8, 8)

#define zero_fill_combos(X, M, N)  \
  X(bfloat16, bf16, M, N, N/2)     \
  X(float, f32, M, N, N/2)

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

matmul_combos(matmul_vectorized_c_func, 16, 16, 32)
matmul_combos(matmul_vectorized_c_func, 32, 32, 32)
matmul_combos(matmul_vectorized_c_func, 32, 32, 64)
matmul_combos(matmul_vectorized_c_func, 64, 64, 64)

zero_fill_combos(zero_vectorized_c_func, 16, 16)
zero_fill_combos(zero_vectorized_c_func, 32, 32)
zero_fill_combos(zero_vectorized_c_func, 64, 64)

}  // extern "C"
)chess"
