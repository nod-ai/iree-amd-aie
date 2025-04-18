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

template<int M, int N, int r>
void trunci_vectorized(v32int32 *__restrict in, int64_t offsetIn, int64_t shift,
                        v32int8 *__restrict out, int64_t offsetOut) {
  for (unsigned i = 0; i < M * N / r; i++) {
    out[offsetOut + i] = ssrs((v32acc32)in[offsetIn + i], shift, 0);
  }
}

template<typename T, int M, int N, int r>
void zero_vectorized(T *__restrict pC, unsigned offsetC)
{
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


template <typename T_in, typename T_out>
aie::accum<T_out, 64> load_v64(const T_in *__restrict p) {
  aie::vector<T_in, 32> v0 = aie::load_v<32>(p);
  aie::accum<T_out, 32> accum0 = aie::accum<T_out, 32>(v0);
  aie::vector<T_in, 32> v1 = aie::load_v<32>(p + 32);
  aie::accum<T_out, 32> accum1 = aie::accum<T_out, 32>(v1);
  aie::accum<T_out, 64>  accum = aie::concat(accum0, accum1);
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

          aie::accum<accfloat, size_A>  accumA0 = load_v64<bfloat16, accfloat>(pA1);
          aie::bfp_vector<bfp16ebs8, size_A> A0 = accumA0.template to_vector<bfp16ebs8>();
          pA1 += rowA * size_A;
          aie::accum<accfloat, size_A>  accumA1 = load_v64<bfloat16, accfloat>(pA2);
          aie::bfp_vector<bfp16ebs8, size_A> A1 = accumA1.template to_vector<bfp16ebs8>();
          pA2 += rowA * size_A;
          aie::accum<accfloat, size_B>  accumB0 = load_v64_T<bfloat16>(pB1);
          aie::bfp_vector<bfp16ebs8, size_B> B0 = accumB0.template to_vector<bfp16ebs8>();
          pB1 += size_B;
          aie::accum<accfloat, size_B>  accumB1 = load_v64_T<bfloat16>(pB2);
          aie::bfp_vector<bfp16ebs8, size_B> B1 = accumB1.template to_vector<bfp16ebs8>();
          pB2 += size_B;

          aie::accum<accfloat, size_C>  acc_C00 = load_v64<float, accfloat>(pC1);
          aie::accum<accfloat, size_C>  acc_C01 = load_v64<float, accfloat>(pC1 + size_C * rowA);
          aie::accum<accfloat, size_C>  acc_C10 = load_v64<float, accfloat>(pC2);
          aie::accum<accfloat, size_C>  acc_C11 = load_v64<float, accfloat>(pC2 + size_C * rowA);

          acc_C00 = mac_8x8_8x8T_conf( A0, B0, acc_C00, 0, 0, 0 );
          acc_C01 = mac_8x8_8x8T_conf( A0, B1, acc_C01, 0, 0, 0 );
          acc_C10 = mac_8x8_8x8T_conf( A1, B0, acc_C10, 0, 0, 0 );
          acc_C11 = mac_8x8_8x8T_conf( A1, B1, acc_C11, 0, 0, 0 );

          for (unsigned i = 1; i < colA; ++i)
            chess_prepare_for_pipelining chess_loop_range(7, ) {
              accumA0 = load_v64<bfloat16, accfloat>(pA1);
              A0 = accumA0.template to_vector<bfp16ebs8>();
              pA1 += rowA * size_A;
              accumA1 = load_v64<bfloat16, accfloat>(pA2);
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

template<unsigned rowA, unsigned colA, unsigned colB, unsigned L0_M, unsigned L0_K, unsigned L0_N>
void matmul_vectorized_i8_i32(const int8 * __restrict pA, unsigned offsetA, const int8 * __restrict pB, unsigned offsetB, int32 * __restrict pC, unsigned offsetC)
{
  const unsigned size_A = L0_M * L0_K;
  const unsigned size_B = L0_K * L0_N;
  const unsigned size_C = L0_M * L0_N;
  using MMUL = aie::mmul<L0_M, L0_K, L0_N, int8, int8, acc32>;

  for (unsigned z = 0; z < rowA; z += 2)
    chess_loop_range(4, ) {
      int32 *__restrict pC0 = pC + offsetC + (z)*size_C;
      int32 *__restrict pC1 = pC + offsetC + ((z + 1)) * size_C;

      for (unsigned j = 0; j < colB; j += 2)
        chess_prepare_for_pipelining chess_loop_range(4, ) {
          const int8 *__restrict pA0 = pA + offsetA + (z)*size_A;
          const int8 *__restrict pA1 = pA + offsetA + ((z + 1)) * size_A;
          const int8 *__restrict pB0 = pB + offsetB + (j)*colA*size_B;
          const int8 *__restrict pB1 = pB + offsetB + ((j + 1))*colA * size_B;

          aie::vector<int8, size_A> A0 = aie::load_v<size_A>(pA0);
          pA0 += rowA * size_A;
          aie::vector<int8, size_A> A1 = aie::load_v<size_A>(pA1);
          pA1 += rowA * size_A;
          aie::vector<int8, size_A> B0 = aie::load_v<size_A>(pB0);
          pB0 += size_B;
          aie::vector<int8, size_B> B1 = aie::load_v<size_A>(pB1);
          pB1 += size_B;

          aie::accum<acc32, size_C>  acc_C00 = load_v64<int32, acc32>(pC0);
          aie::accum<acc32, size_C>  acc_C01 = load_v64<int32, acc32>(pC0 + size_C * rowA);
          aie::accum<acc32, size_C>  acc_C10 = load_v64<int32, acc32>(pC1);
          aie::accum<acc32, size_C>  acc_C11 = load_v64<int32, acc32>(pC1 + size_C * rowA);

          MMUL C00(acc_C00);
          MMUL C01(acc_C01);
          MMUL C10(acc_C10);
          MMUL C11(acc_C11);

          C00.mac(A0, B0);
          C01.mac(A0, B1);
          C10.mac(A1, B0);
          C11.mac(A1, B1);

          for (unsigned i = 1; i < colA; ++i)
            chess_prepare_for_pipelining chess_loop_range(7, ) {
              A0 = aie::load_v<size_A>(pA0);
              pA0 += rowA * size_A;
              A1 = aie::load_v<size_A>(pA1);
              pA1 += rowA * size_A;
              B0 = aie::load_v<size_A>(pB0);
              pB0 += size_B;
              B1 = aie::load_v<size_A>(pB1);
              pB1 += size_B;

              C00.mac(A0, B0);
              C01.mac(A0, B1);
              C10.mac(A1, B0);
              C11.mac(A1, B1);
          }

          v32int32 * __restrict pOut00 = (v32int32 *) pC0;
          *pOut00++ = (v32int32)(C00.to_accum().template extract<32>(0));
          *pOut00++ = (v32int32)(C00.to_accum().template extract<32>(1));
          pC0 += size_C * rowA;

          v32int32 * __restrict pOut01 = (v32int32 *) pC0;
          *pOut01++ = (v32int32)(C01.to_accum().template extract<32>(0));
          *pOut01++ = (v32int32)(C01.to_accum().template extract<32>(1));
          pC0 += size_C * rowA;

          v32int32 * __restrict pOut10 = (v32int32 *) pC1;
          *pOut10++ = (v32int32)(C10.to_accum().template extract<32>(0));
          *pOut10++ = (v32int32)(C10.to_accum().template extract<32>(1));
          pC1 += size_C * rowA;

          v32int32 * __restrict pOut11 = (v32int32 *) pC1;
          *pOut11++ = (v32int32)(C11.to_accum().template extract<32>(0));
          *pOut11++ = (v32int32)(C11.to_accum().template extract<32>(1));
          pC1 += size_C * rowA;
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

#define matmul_combos(X, M, N, K)                                     \
  X(bfloat16, bf16, bfloat16, bf16, float, f32, M, N, K, 8, 8, 8)

#define matmul_combos_i8(X, M, N, K)                                  \
  X(int8, i8, int8, i8, int32, i32, M, N, K, 8, 8, 8)

#define zero_fill_combos(X, M, N)  \
  X(bfloat16, bf16, M, N, N)     \
  X(float, f32, M, N, N/2)         \
  X(int32, i32, M, N, N/2)

#define trunci_combos_i32_i8(X, M, N)  \
  X(v32int32, i32, v32int8, i8, M, N, 32)

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

#define trunci_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out, M, N, r)                  \
  void trunci_##mlir_type_in##_##mlir_type_out##_##M##x##N(                                       \
      ctype_in *in, int64_t offsetIn, int64_t shift, ctype_out *out, int64_t offsetOut) {         \
    trunci_vectorized<M, N, r>(in, offsetIn, shift, out, offsetOut);                              \
  }

matmul_combos(matmul_vectorized_c_func, 16, 8, 32)
matmul_combos(matmul_vectorized_c_func, 16, 8, 64)
matmul_combos(matmul_vectorized_c_func, 16, 16, 32)
matmul_combos(matmul_vectorized_c_func, 32, 32, 32)
matmul_combos(matmul_vectorized_c_func, 32, 32, 64)
matmul_combos(matmul_vectorized_c_func, 32, 16, 128)
matmul_combos(matmul_vectorized_c_func, 32, 32, 128)
matmul_combos(matmul_vectorized_c_func, 64, 64, 64)

matmul_combos_i8(matmul_vectorized_c_func, 16, 16, 32)
matmul_combos_i8(matmul_vectorized_c_func, 32, 16, 32)
matmul_combos_i8(matmul_vectorized_c_func, 32, 16, 64)
matmul_combos_i8(matmul_vectorized_c_func, 32, 32, 8)
matmul_combos_i8(matmul_vectorized_c_func, 32, 32, 32)
matmul_combos_i8(matmul_vectorized_c_func, 32, 32, 64)
matmul_combos_i8(matmul_vectorized_c_func, 32, 32, 128)
matmul_combos_i8(matmul_vectorized_c_func, 64, 64, 64)
matmul_combos_i8(matmul_vectorized_c_func, 64, 32, 128)
matmul_combos_i8(matmul_vectorized_c_func, 64, 64, 128)

trunci_combos_i32_i8(trunci_c_func, 32, 32)
trunci_combos_i32_i8(trunci_c_func, 64, 64)
trunci_combos_i32_i8(trunci_c_func, 32, 64)
trunci_combos_i32_i8(trunci_c_func, 64, 32)

zero_fill_combos(zero_vectorized_c_func, 16, 8)
zero_fill_combos(zero_vectorized_c_func, 16, 16)
zero_fill_combos(zero_vectorized_c_func, 32, 16)
zero_fill_combos(zero_vectorized_c_func, 32, 32)
zero_fill_combos(zero_vectorized_c_func, 64, 32)
zero_fill_combos(zero_vectorized_c_func, 64, 64)

}  // extern "C"
)chess"
