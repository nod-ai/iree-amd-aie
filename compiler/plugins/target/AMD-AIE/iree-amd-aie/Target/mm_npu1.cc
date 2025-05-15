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

template <typename T, int M, int N, int r>
void zero_vectorized(T *__restrict pC, unsigned offsetC) {
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

// Suppose A is a 64x64 tensor and B is a 64x64 tensor, and r=4, s=8, t=4.
//
// Let A[i,j] be the element at row i and column j of A, and
//     B[i,j] be the element at row i and column j of B.
//
// The expectations of this function on the points pA, pB, and pC are:
//
// 1) all elements of A are contiguous in memory, starting from pA + offsetA
// 2) all elements of B are contiguous in memory, starting from pB + offsetB
// 3) all elements of C are contiguous in memory, starting from pC + offsetC
// 4) element A[i,j] is at pA[offsetA + i*8 + (64*8)*(j/8) + j%8]
// 5) element B[i,j] is at pB[offsetB + i*4 + (64*4)*(j/4) + j%4]
//
// 4) and 5) describe vertical stripes of A and B that are stored contiguously,
// with a row-major order within each stripe. i.e. elements starting at ptrA +
// offsetA are:
//
// [A[0,0], ..., A[0,7], A[1,0], ..., A[1,7], A[2,0], ..., A[2,7], ... A[63,0],
// ..., A[63,7], A[0,8], ..., A[0,15], ..., A[63, 64]]
//

template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
void matmul_vectorized(const T_in *__restrict pA, unsigned offsetA,
                       const T_in *__restrict pB, unsigned offsetB,
                       T_out *__restrict pC, unsigned offsetC) {
  using MMUL = aie::mmul<r, s, t, T_in, T_in, accfloat>;

  event0();

  for (unsigned z = 0; z < rowA; z += 4) chess_loop_range(2, ) {
      T_out *__restrict pC1 = pC + offsetC + (z)*MMUL::size_C;
      T_out *__restrict pC2 = pC + offsetC + ((z + 1)) * MMUL::size_C;
      T_out *__restrict pC3 = pC + offsetC + ((z + 2)) * MMUL::size_C;
      T_out *__restrict pC4 = pC + offsetC + ((z + 3)) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 4)
        chess_prepare_for_pipelining chess_loop_range(8, ) {
          const T_in *__restrict pA1 = pA + offsetA + (z)*MMUL::size_A;
          const T_in *__restrict pA2 = pA + offsetA + ((z + 1)) * MMUL::size_A;
          const T_in *__restrict pA3 = pA + offsetA + ((z + 2)) * MMUL::size_A;
          const T_in *__restrict pA4 = pA + offsetA + ((z + 3)) * MMUL::size_A;

          const T_in *__restrict pB1 =
              pB + offsetB + ((j + 0)) * colA * MMUL::size_B;
          const T_in *__restrict pB2 =
              pB + offsetB + ((j + 1)) * colA * MMUL::size_B;
          const T_in *__restrict pB3 =
              pB + offsetB + ((j + 2)) * colA * MMUL::size_B;
          const T_in *__restrict pB4 =
              pB + offsetB + ((j + 3)) * colA * MMUL::size_B;

          aie::vector<T_in, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
          pA1 += rowA * MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2);
          pA2 += rowA * MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A2 = aie::load_v<MMUL::size_A>(pA3);
          pA3 += rowA * MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A3 = aie::load_v<MMUL::size_A>(pA4);
          pA4 += rowA * MMUL::size_A;
          aie::vector<T_in, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += MMUL::size_B;
          aie::vector<T_in, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2);
          pB2 += MMUL::size_B;
          aie::vector<T_in, MMUL::size_B> B2 = aie::load_v<MMUL::size_B>(pB3);
          pB3 += MMUL::size_B;
          aie::vector<T_in, MMUL::size_B> B3 = aie::load_v<MMUL::size_B>(pB4);
          pB4 += MMUL::size_B;

          aie::vector<T_out, MMUL::size_C> acc_C00 =
              aie::load_v<MMUL::size_C>(pC1);
          aie::vector<T_out, MMUL::size_C> acc_C01 =
              aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C02 =
              aie::load_v<MMUL::size_C>(pC1 + 2 * MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C03 =
              aie::load_v<MMUL::size_C>(pC1 + 3 * MMUL::size_C * rowA);

          aie::vector<T_out, MMUL::size_C> acc_C10 =
              aie::load_v<MMUL::size_C>(pC2);
          aie::vector<T_out, MMUL::size_C> acc_C11 =
              aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C12 =
              aie::load_v<MMUL::size_C>(pC2 + 2 * MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C13 =
              aie::load_v<MMUL::size_C>(pC2 + 3 * MMUL::size_C * rowA);

          aie::vector<T_out, MMUL::size_C> acc_C20 =
              aie::load_v<MMUL::size_C>(pC3);
          aie::vector<T_out, MMUL::size_C> acc_C21 =
              aie::load_v<MMUL::size_C>(pC3 + MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C22 =
              aie::load_v<MMUL::size_C>(pC3 + 2 * MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C23 =
              aie::load_v<MMUL::size_C>(pC3 + 3 * MMUL::size_C * rowA);

          aie::vector<T_out, MMUL::size_C> acc_C30 =
              aie::load_v<MMUL::size_C>(pC4);
          aie::vector<T_out, MMUL::size_C> acc_C31 =
              aie::load_v<MMUL::size_C>(pC4 + MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C32 =
              aie::load_v<MMUL::size_C>(pC4 + 2 * MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C33 =
              aie::load_v<MMUL::size_C>(pC4 + 3 * MMUL::size_C * rowA);

          MMUL C00(acc_C00);
          MMUL C01(acc_C01);
          MMUL C02(acc_C02);
          MMUL C03(acc_C03);

          MMUL C10(acc_C10);
          MMUL C11(acc_C11);
          MMUL C12(acc_C12);
          MMUL C13(acc_C13);

          MMUL C20(acc_C20);
          MMUL C21(acc_C21);
          MMUL C22(acc_C22);
          MMUL C23(acc_C23);

          MMUL C30(acc_C30);
          MMUL C31(acc_C31);
          MMUL C32(acc_C32);
          MMUL C33(acc_C33);

          C00.mac(A0, B0);
          C01.mac(A0, B1);
          C10.mac(A1, B0);
          C11.mac(A1, B1);

          C02.mac(A0, B2);
          C03.mac(A0, B3);
          C12.mac(A1, B2);
          C13.mac(A1, B3);

          C20.mac(A2, B0);
          C21.mac(A2, B1);
          C30.mac(A3, B0);
          C31.mac(A3, B1);

          C22.mac(A2, B2);
          C23.mac(A2, B3);
          C32.mac(A3, B2);
          C33.mac(A3, B3);

          for (unsigned i = 1; i < colA; ++i)
            chess_prepare_for_pipelining chess_loop_range(7, ) {
              A0 = aie::load_v<MMUL::size_A>(pA1);
              pA1 += rowA * MMUL::size_A;
              A1 = aie::load_v<MMUL::size_A>(pA2);
              pA2 += rowA * MMUL::size_A;
              A2 = aie::load_v<MMUL::size_A>(pA3);
              pA3 += rowA * MMUL::size_A;
              A3 = aie::load_v<MMUL::size_A>(pA4);
              pA4 += rowA * MMUL::size_A;

              B0 = aie::load_v<MMUL::size_B>(pB1);
              pB1 += MMUL::size_B;
              B1 = aie::load_v<MMUL::size_B>(pB2);
              pB2 += MMUL::size_B;
              B2 = aie::load_v<MMUL::size_B>(pB3);
              pB3 += MMUL::size_B;
              B3 = aie::load_v<MMUL::size_B>(pB4);
              pB4 += MMUL::size_B;

              C00.mac(A0, B0);
              C01.mac(A0, B1);
              C10.mac(A1, B0);
              C11.mac(A1, B1);

              C02.mac(A0, B2);
              C03.mac(A0, B3);
              C12.mac(A1, B2);
              C13.mac(A1, B3);

              C20.mac(A2, B0);
              C21.mac(A2, B1);
              C30.mac(A3, B0);
              C31.mac(A3, B1);

              C22.mac(A2, B2);
              C23.mac(A2, B3);
              C32.mac(A3, B2);
              C33.mac(A3, B3);
            }

          aie::store_v(pC1, C00.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;
          aie::store_v(pC1, C01.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;
          aie::store_v(pC1, C02.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;
          aie::store_v(pC1, C03.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;

          aie::store_v(pC2, C10.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;
          aie::store_v(pC2, C11.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;
          aie::store_v(pC2, C12.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;
          aie::store_v(pC2, C13.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;

          aie::store_v(pC3, C20.template to_vector<T_out>());
          pC3 += MMUL::size_C * rowA;
          aie::store_v(pC3, C21.template to_vector<T_out>());
          pC3 += MMUL::size_C * rowA;
          aie::store_v(pC3, C22.template to_vector<T_out>());
          pC3 += MMUL::size_C * rowA;
          aie::store_v(pC3, C23.template to_vector<T_out>());
          pC3 += MMUL::size_C * rowA;

          aie::store_v(pC4, C30.template to_vector<T_out>());
          pC4 += MMUL::size_C * rowA;
          aie::store_v(pC4, C31.template to_vector<T_out>());
          pC4 += MMUL::size_C * rowA;
          aie::store_v(pC4, C32.template to_vector<T_out>());
          pC4 += MMUL::size_C * rowA;
          aie::store_v(pC4, C33.template to_vector<T_out>());
          pC4 += MMUL::size_C * rowA;
        }
    }

  event1();
}

template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
static inline void matmul_vectorized_4x2(const T_in *__restrict pA,
                                         unsigned offsetA,
                                         const T_in *__restrict pB,
                                         unsigned offsetB,
                                         T_out *__restrict pC,
                                         unsigned offsetC) {

  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  for (unsigned z = 0; z < rowA; z += 4)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      T_out *__restrict pC1 = pC + offsetC + (z * colB + 0) * MMUL::size_C;
      T_out *__restrict pC2 = pC + offsetC + ((z + 1) * colB + 0) * MMUL::size_C;
      T_out *__restrict pC3 = pC + offsetC + ((z + 2) * colB + 0) * MMUL::size_C;
      T_out *__restrict pC4 = pC + offsetC + ((z + 3) * colB + 0) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
        chess_flatten_loop
#endif
        {
          const T_in *__restrict pA1 = pA + offsetA + (z * colA + 0) * MMUL::size_A;
          const T_in *__restrict pA2 = pA + offsetA + ((z + 1) * colA + 0) * MMUL::size_A;
          const T_in *__restrict pA3 = pA + offsetA + ((z + 2) * colA + 0) * MMUL::size_A;
          const T_in *__restrict pA4 = pA + offsetA + ((z + 3) * colA + 0) * MMUL::size_A;

          const T_in *__restrict pB1 = pB + offsetB + (0 * colB + j) * MMUL::size_B;
          const T_in *__restrict pB2 = pB + offsetB + (0 * colB + (j + 1)) * MMUL::size_B;

          aie::vector<T_in, MMUL::size_A> A01 = aie::load_v<MMUL::size_A>(pA1);
          pA1 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A11 = aie::load_v<MMUL::size_A>(pA2);
          pA2 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A21 = aie::load_v<MMUL::size_A>(pA3);
          pA3 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A31 = aie::load_v<MMUL::size_A>(pA4);
          pA4 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_B> B01 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += (MMUL::size_B * colB);
          aie::vector<T_in, MMUL::size_B> B11 = aie::load_v<MMUL::size_B>(pB2);
          pB2 += (MMUL::size_B * colB);

          aie::vector<T_out, MMUL::size_C> acc_C00 =
              aie::load_v<MMUL::size_C>(pC1);
          aie::vector<T_out, MMUL::size_C> acc_C01 =
              aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C10 =
              aie::load_v<MMUL::size_C>(pC2);
          aie::vector<T_out, MMUL::size_C> acc_C11 =
              aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C20 =
              aie::load_v<MMUL::size_C>(pC3);
          aie::vector<T_out, MMUL::size_C> acc_C21 =
              aie::load_v<MMUL::size_C>(pC3 + MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C30 =
              aie::load_v<MMUL::size_C>(pC4);
          aie::vector<T_out, MMUL::size_C> acc_C31 =
              aie::load_v<MMUL::size_C>(pC4 + MMUL::size_C);

          MMUL C00(acc_C00);
          MMUL C01(acc_C01);
          MMUL C10(acc_C10);
          MMUL C11(acc_C11);
          MMUL C20(acc_C20);
          MMUL C21(acc_C21);
          MMUL C30(acc_C30);
          MMUL C31(acc_C31);

          C00.mac(A01, B01);
          C01.mac(A01, B11);
          C10.mac(A11, B01);
          C11.mac(A11, B11);
          C20.mac(A21, B01);
          C21.mac(A21, B11);
          C30.mac(A31, B01);
          C31.mac(A31, B11);

          for (unsigned i = 1; i < colA; i += 1)
#ifdef OPT_PERF_ENABLED
            chess_flatten_loop
#endif
            {
              A01 = aie::load_v<MMUL::size_A>(pA1);
              pA1 += MMUL::size_A;
              A11 = aie::load_v<MMUL::size_A>(pA2);
              pA2 += MMUL::size_A;
              A21 = aie::load_v<MMUL::size_A>(pA3);
              pA3 += MMUL::size_A;
              A31 = aie::load_v<MMUL::size_A>(pA4);
              pA4 += MMUL::size_A;
              B01 = aie::load_v<MMUL::size_B>(pB1);
              pB1 += (MMUL::size_B * colB);
              B11 = aie::load_v<MMUL::size_B>(pB2);
              pB2 += (MMUL::size_B * colB);

              C00.mac(A01, B01);
              C01.mac(A01, B11);
              C10.mac(A11, B01);
              C11.mac(A11, B11);
              C20.mac(A21, B01);
              C21.mac(A21, B11);
              C30.mac(A31, B01);
              C31.mac(A31, B11);
            }

          aie::store_v(pC1, C00.template to_vector<T_out>());
          pC1 += MMUL::size_C;
          aie::store_v(pC1, C01.template to_vector<T_out>());
          pC1 += MMUL::size_C;
          aie::store_v(pC2, C10.template to_vector<T_out>());
          pC2 += MMUL::size_C;
          aie::store_v(pC2, C11.template to_vector<T_out>());
          pC2 += MMUL::size_C;
          aie::store_v(pC3, C20.template to_vector<T_out>());
          pC3 += MMUL::size_C;
          aie::store_v(pC3, C21.template to_vector<T_out>());
          pC3 += MMUL::size_C;
          aie::store_v(pC4, C30.template to_vector<T_out>());
          pC4 += MMUL::size_C;
          aie::store_v(pC4, C31.template to_vector<T_out>());
          pC4 += MMUL::size_C;
        }
    }

  event1();
}

template <unsigned m, unsigned k, unsigned n>
void matmul_vectorized_4x8x4_bf16_bf16_bf16(const bfloat16 *__restrict pA,
                                       unsigned offsetA,
                                       const bfloat16 *__restrict pB,
                                       unsigned offsetB,
                                       bfloat16 *__restrict pC,
                                       unsigned offsetC) {
  constexpr int r = 4;
  constexpr int s = 8;
  constexpr int t = 4;
  static_assert(m % (2 * r) == 0 && m / (2 * r) > 0);
  static_assert(k % (2 * s) == 0 && k / (2 * s) > 0);
  static_assert(n % (2 * t) == 0 && n / (2 * t) > 0);
  return matmul_vectorized<bfloat16, bfloat16, m / r, k / s, n / t, r, s, t>(
      pA, offsetA, pB, offsetB, pC, offsetC);
}

template <unsigned m, unsigned k, unsigned n>
void matmul_vectorized_4x8x4_bf16_bf16_f32(const bfloat16 *__restrict pA,
                                      unsigned offsetA,
                                      const bfloat16 *__restrict pB,
                                      unsigned offsetB, float *__restrict pC,
                                      unsigned offsetC) {
  constexpr int r = 4;
  constexpr int s = 8;
  constexpr int t = 4;
  static_assert(m % (2 * r) == 0 && m / (2 * r) > 0);
  static_assert(k % (2 * s) == 0 && k / (2 * s) > 0);
  static_assert(n % (2 * t) == 0 && n / (2 * t) > 0);
  return matmul_vectorized<bfloat16, float, m / r, k / s, n / t, r, s, t>(
      pA, offsetA, pB, offsetB, pC, offsetC);
}

template <unsigned m, unsigned k, unsigned n>
void matmul_vectorized_4x8x8_i8_i8_i32(const int8 *__restrict pA,
                                      unsigned offsetA,
                                      const int8 *__restrict pB,
                                      unsigned offsetB, int32 *__restrict pC,
                                      unsigned offsetC) {
  constexpr int r = 4;
  constexpr int s = 8;
  constexpr int t = 8;
  static_assert(m % (4 * r) == 0); // 'm' dimension
  static_assert(k % s == 0);       // 'k' dimension
  static_assert(n % (2 * t) == 0); // 'n' dimension
  return matmul_vectorized_4x2<int8, int32, m / r, k / s, n / t, r, s, t>(
      pA, offsetA, pB, offsetB, pC, offsetC);
}


alignas(aie::vector_decl_align) int16 exp_ilut_ab[512] = {
    16256, 16430, 16620, 16801, 16986, 17172, 17354, 17545, 16256, 16430, 16620,
    16801, 16986, 17172, 17354, 17545, 17722, 17917, 18092, 18282, 18463, 18648,
    18835, 19016, 17722, 17917, 18092, 18282, 18463, 18648, 18835, 19016, 19208,
    19384, 19578, 19754, 19943, 20125, 20310, 20497, 19208, 19384, 19578, 19754,
    19943, 20125, 20310, 20497, 20677, 20870, 21046, 21240, 21416, 21605, 21788,
    21971, 20677, 20870, 21046, 21240, 21416, 21605, 21788, 21971, 22160, 22339,
    22533, 22708, 22901, 23079, 23266, 23450, 22160, 22339, 22533, 22708, 22901,
    23079, 23266, 23450, 23633, 23822, 24001, 24195, 24370, 24562, 24741, 24928,
    23633, 23822, 24001, 24195, 24370, 24562, 24741, 24928, 25112, 25295, 25485,
    25663, 25858, 26032, 26224, 26403, 25112, 25295, 25485, 25663, 25858, 26032,
    26224, 26403, 26589, 26774, 26957, 27147, 27325, 27520, 27695, 27885, 26589,
    26774, 26957, 27147, 27325, 27520, 27695, 27885, 28065, 28251, 28437, 28618,
    28809, 28987, 29182, 29357, 28065, 28251, 28437, 28618, 28809, 28987, 29182,
    29357, 29547, 29727, 29913, 30099, 30280, 30472, 30649, 30843, 29547, 29727,
    29913, 30099, 30280, 30472, 30649, 30843, 31019, 31208, 31390, 31574, 31762,
    31942, 32135, 32311, 31019, 31208, 31390, 31574, 31762, 31942, 32135, 32311,
    32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505,
    32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505,
    32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505,
    32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505,
    32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505,
    32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505,
    32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505,
    32505, 32505, 32505, 0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     1,     3,     9,     24,    0,     0,
    0,     0,     1,     3,     9,     24,    66,    179,   372,   550,   737,
    921,   1104,  1293,  66,    179,   372,   550,   737,   921,   1104,  1293,
    1472,  1666,  1841,  2033,  2212,  2399,  2583,  2766,  1472,  1666,  1841,
    2033,  2212,  2399,  2583,  2766,  2956,  3134,  3329,  3503,  3694,  3874,
    4060,  4246,  2956,  3134,  3329,  3503,  3694,  3874,  4060,  4246,  4427,
    4618,  4796,  4991,  5165,  5356,  5536,  5722,  4427,  4618,  4796,  4991,
    5165,  5356,  5536,  5722,  5908,  6089,  6281,  6458,  6652,  6828,  7017,
    7198,  5908,  6089,  6281,  6458,  6652,  6828,  7017,  7198,  7383,  7570,
    7751,  7943,  8120,  8314,  8490,  8679,  7383,  7570,  7751,  7943,  8120,
    8314,  8490,  8679,  8861,  9045,  9233,  9413,  9606,  9782,  9975,  10152,
    8861,  9045,  9233,  9413,  9606,  9782,  9975,  10152, 10340, 10523, 10707,
    10895, 11075, 11268, 11444, 11636, 10340, 10523, 10707, 10895, 11075, 11268,
    11444, 11636, 11814, 12002, 12185, 12368, 12558, 12737, 12931, 13106, 11814,
    12002, 12185, 12368, 12558, 12737, 12931, 13106, 13298, 13476, 13663, 13848,
    14030, 14220, 14398, 14593, 13298, 13476, 13663, 13848, 14030, 14220, 14398,
    14593, 14768, 14959, 15138, 15325, 15510, 15692, 15883, 16060, 14768, 14959,
    15138, 15325, 15510, 15692, 15883, 16060};

alignas(aie::vector_decl_align) int16 exp_ilut_cd[512] = {
    16256, 16430, 16620, 16801, 16986, 17172, 17354, 17545, 16256, 16430, 16620,
    16801, 16986, 17172, 17354, 17545, 17722, 17917, 18092, 18282, 18463, 18648,
    18835, 19016, 17722, 17917, 18092, 18282, 18463, 18648, 18835, 19016, 19208,
    19384, 19578, 19754, 19943, 20125, 20310, 20497, 19208, 19384, 19578, 19754,
    19943, 20125, 20310, 20497, 20677, 20870, 21046, 21240, 21416, 21605, 21788,
    21971, 20677, 20870, 21046, 21240, 21416, 21605, 21788, 21971, 22160, 22339,
    22533, 22708, 22901, 23079, 23266, 23450, 22160, 22339, 22533, 22708, 22901,
    23079, 23266, 23450, 23633, 23822, 24001, 24195, 24370, 24562, 24741, 24928,
    23633, 23822, 24001, 24195, 24370, 24562, 24741, 24928, 25112, 25295, 25485,
    25663, 25858, 26032, 26224, 26403, 25112, 25295, 25485, 25663, 25858, 26032,
    26224, 26403, 26589, 26774, 26957, 27147, 27325, 27520, 27695, 27885, 26589,
    26774, 26957, 27147, 27325, 27520, 27695, 27885, 28065, 28251, 28437, 28618,
    28809, 28987, 29182, 29357, 28065, 28251, 28437, 28618, 28809, 28987, 29182,
    29357, 29547, 29727, 29913, 30099, 30280, 30472, 30649, 30843, 29547, 29727,
    29913, 30099, 30280, 30472, 30649, 30843, 31019, 31208, 31390, 31574, 31762,
    31942, 32135, 32311, 31019, 31208, 31390, 31574, 31762, 31942, 32135, 32311,
    32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505,
    32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505,
    32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505,
    32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505,
    32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505,
    32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505,
    32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505, 32505,
    32505, 32505, 32505, 0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     1,     3,     9,     24,    0,     0,
    0,     0,     1,     3,     9,     24,    66,    179,   372,   550,   737,
    921,   1104,  1293,  66,    179,   372,   550,   737,   921,   1104,  1293,
    1472,  1666,  1841,  2033,  2212,  2399,  2583,  2766,  1472,  1666,  1841,
    2033,  2212,  2399,  2583,  2766,  2956,  3134,  3329,  3503,  3694,  3874,
    4060,  4246,  2956,  3134,  3329,  3503,  3694,  3874,  4060,  4246,  4427,
    4618,  4796,  4991,  5165,  5356,  5536,  5722,  4427,  4618,  4796,  4991,
    5165,  5356,  5536,  5722,  5908,  6089,  6281,  6458,  6652,  6828,  7017,
    7198,  5908,  6089,  6281,  6458,  6652,  6828,  7017,  7198,  7383,  7570,
    7751,  7943,  8120,  8314,  8490,  8679,  7383,  7570,  7751,  7943,  8120,
    8314,  8490,  8679,  8861,  9045,  9233,  9413,  9606,  9782,  9975,  10152,
    8861,  9045,  9233,  9413,  9606,  9782,  9975,  10152, 10340, 10523, 10707,
    10895, 11075, 11268, 11444, 11636, 10340, 10523, 10707, 10895, 11075, 11268,
    11444, 11636, 11814, 12002, 12185, 12368, 12558, 12737, 12931, 13106, 11814,
    12002, 12185, 12368, 12558, 12737, 12931, 13106, 13298, 13476, 13663, 13848,
    14030, 14220, 14398, 14593, 13298, 13476, 13663, 13848, 14030, 14220, 14398,
    14593, 14768, 14959, 15138, 15325, 15510, 15692, 15883, 16060, 14768, 14959,
    15138, 15325, 15510, 15692, 15883, 16060};

alignas(aie::vector_decl_align) int16 exp_flut_ab[512] = {
    16256, 16257, 16257, 16258, 16258, 16259, 16259, 16260, 16256, 16257, 16257,
    16258, 16258, 16259, 16259, 16260, 16260, 16261, 16261, 16262, 16262, 16263,
    16263, 16264, 16260, 16261, 16261, 16262, 16262, 16263, 16263, 16264, 16264,
    16265, 16265, 16266, 16266, 16267, 16267, 16268, 16264, 16265, 16265, 16266,
    16266, 16267, 16267, 16268, 16269, 16269, 16270, 16270, 16271, 16271, 16272,
    16272, 16269, 16269, 16270, 16270, 16271, 16271, 16272, 16272, 16273, 16274,
    16274, 16275, 16275, 16276, 16276, 16277, 16273, 16274, 16274, 16275, 16275,
    16276, 16276, 16277, 16278, 16278, 16279, 16279, 16280, 16281, 16281, 16282,
    16278, 16278, 16279, 16279, 16280, 16281, 16281, 16282, 16282, 16283, 16284,
    16284, 16285, 16285, 16286, 16287, 16282, 16283, 16284, 16284, 16285, 16285,
    16286, 16287, 16287, 16288, 16289, 16289, 16290, 16290, 16291, 16292, 16287,
    16288, 16289, 16289, 16290, 16290, 16291, 16292, 16292, 16293, 16294, 16294,
    16295, 16296, 16296, 16297, 16292, 16293, 16294, 16294, 16295, 16296, 16296,
    16297, 16298, 16298, 16299, 16300, 16300, 16301, 16302, 16302, 16298, 16298,
    16299, 16300, 16300, 16301, 16302, 16302, 16303, 16304, 16304, 16305, 16306,
    16306, 16307, 16308, 16303, 16304, 16304, 16305, 16306, 16306, 16307, 16308,
    16309, 16309, 16310, 16311, 16311, 16312, 16313, 16314, 16309, 16309, 16310,
    16311, 16311, 16312, 16313, 16314, 16314, 16315, 16316, 16316, 16317, 16318,
    16319, 16319, 16314, 16315, 16316, 16316, 16317, 16318, 16319, 16319, 16320,
    16321, 16322, 16322, 16323, 16324, 16325, 16325, 16320, 16321, 16322, 16322,
    16323, 16324, 16325, 16325, 16326, 16327, 16328, 16329, 16329, 16330, 16331,
    16332, 16326, 16327, 16328, 16329, 16329, 16330, 16331, 16332, 16333, 16333,
    16334, 16335, 16336, 16337, 16337, 16338, 16333, 16333, 16334, 16335, 16336,
    16337, 16337, 16338, 16339, 16340, 16341, 16342, 16342, 16343, 16344, 16345,
    16339, 16340, 16341, 16342, 16342, 16343, 16344, 16345, 16346, 16347, 16347,
    16348, 16349, 16350, 16351, 16352, 16346, 16347, 16347, 16348, 16349, 16350,
    16351, 16352, 16353, 16354, 16354, 16355, 16356, 16357, 16358, 16359, 16353,
    16354, 16354, 16355, 16356, 16357, 16358, 16359, 16360, 16361, 16362, 16363,
    16363, 16364, 16365, 16366, 16360, 16361, 16362, 16363, 16363, 16364, 16365,
    16366, 16367, 16368, 16369, 16370, 16371, 16372, 16373, 16374, 16367, 16368,
    16369, 16370, 16371, 16372, 16373, 16374, 16375, 16376, 16377, 16378, 16379,
    16380, 16381, 16382, 16375, 16376, 16377, 16378, 16379, 16380, 16381, 16382,
    16383, 16384, 16384, 16385, 16385, 16386, 16386, 16387, 16383, 16384, 16384,
    16385, 16385, 16386, 16386, 16387, 16387, 16388, 16388, 16389, 16389, 16390,
    16390, 16391, 16387, 16388, 16388, 16389, 16389, 16390, 16390, 16391, 16391,
    16392, 16393, 16393, 16394, 16394, 16395, 16395, 16391, 16392, 16393, 16393,
    16394, 16394, 16395, 16395, 16396, 16396, 16397, 16397, 16398, 16399, 16399,
    16400, 16396, 16396, 16397, 16397, 16398, 16399, 16399, 16400, 16400, 16401,
    16401, 16402, 16402, 16403, 16404, 16404, 16400, 16401, 16401, 16402, 16402,
    16403, 16404, 16404, 16405, 16405, 16406, 16407, 16407, 16408, 16408, 16409,
    16405, 16405, 16406, 16407, 16407, 16408, 16408, 16409, 16410, 16410, 16411,
    16411, 16412, 16413, 16413, 16414, 16410, 16410, 16411, 16411, 16412, 16413,
    16413, 16414, 16414, 16415, 16416, 16416, 16417, 16418, 16418, 16419, 16414,
    16415, 16416, 16416, 16417, 16418, 16418, 16419, 16419, 16420, 16421, 16421,
    16422, 16423, 16423, 16424, 16419, 16420, 16421, 16421, 16422, 16423, 16423,
    16424, 16425, 16425, 16426, 16427, 16427, 16428, 16429, 16429, 16425, 16425,
    16426, 16427, 16427, 16428, 16429, 16429};

alignas(aie::vector_decl_align) int16 exp_flut_cd[512] = {
    16256, 16257, 16257, 16258, 16258, 16259, 16259, 16260, 16256, 16257, 16257,
    16258, 16258, 16259, 16259, 16260, 16260, 16261, 16261, 16262, 16262, 16263,
    16263, 16264, 16260, 16261, 16261, 16262, 16262, 16263, 16263, 16264, 16264,
    16265, 16265, 16266, 16266, 16267, 16267, 16268, 16264, 16265, 16265, 16266,
    16266, 16267, 16267, 16268, 16269, 16269, 16270, 16270, 16271, 16271, 16272,
    16272, 16269, 16269, 16270, 16270, 16271, 16271, 16272, 16272, 16273, 16274,
    16274, 16275, 16275, 16276, 16276, 16277, 16273, 16274, 16274, 16275, 16275,
    16276, 16276, 16277, 16278, 16278, 16279, 16279, 16280, 16281, 16281, 16282,
    16278, 16278, 16279, 16279, 16280, 16281, 16281, 16282, 16282, 16283, 16284,
    16284, 16285, 16285, 16286, 16287, 16282, 16283, 16284, 16284, 16285, 16285,
    16286, 16287, 16287, 16288, 16289, 16289, 16290, 16290, 16291, 16292, 16287,
    16288, 16289, 16289, 16290, 16290, 16291, 16292, 16292, 16293, 16294, 16294,
    16295, 16296, 16296, 16297, 16292, 16293, 16294, 16294, 16295, 16296, 16296,
    16297, 16298, 16298, 16299, 16300, 16300, 16301, 16302, 16302, 16298, 16298,
    16299, 16300, 16300, 16301, 16302, 16302, 16303, 16304, 16304, 16305, 16306,
    16306, 16307, 16308, 16303, 16304, 16304, 16305, 16306, 16306, 16307, 16308,
    16309, 16309, 16310, 16311, 16311, 16312, 16313, 16314, 16309, 16309, 16310,
    16311, 16311, 16312, 16313, 16314, 16314, 16315, 16316, 16316, 16317, 16318,
    16319, 16319, 16314, 16315, 16316, 16316, 16317, 16318, 16319, 16319, 16320,
    16321, 16322, 16322, 16323, 16324, 16325, 16325, 16320, 16321, 16322, 16322,
    16323, 16324, 16325, 16325, 16326, 16327, 16328, 16329, 16329, 16330, 16331,
    16332, 16326, 16327, 16328, 16329, 16329, 16330, 16331, 16332, 16333, 16333,
    16334, 16335, 16336, 16337, 16337, 16338, 16333, 16333, 16334, 16335, 16336,
    16337, 16337, 16338, 16339, 16340, 16341, 16342, 16342, 16343, 16344, 16345,
    16339, 16340, 16341, 16342, 16342, 16343, 16344, 16345, 16346, 16347, 16347,
    16348, 16349, 16350, 16351, 16352, 16346, 16347, 16347, 16348, 16349, 16350,
    16351, 16352, 16353, 16354, 16354, 16355, 16356, 16357, 16358, 16359, 16353,
    16354, 16354, 16355, 16356, 16357, 16358, 16359, 16360, 16361, 16362, 16363,
    16363, 16364, 16365, 16366, 16360, 16361, 16362, 16363, 16363, 16364, 16365,
    16366, 16367, 16368, 16369, 16370, 16371, 16372, 16373, 16374, 16367, 16368,
    16369, 16370, 16371, 16372, 16373, 16374, 16375, 16376, 16377, 16378, 16379,
    16380, 16381, 16382, 16375, 16376, 16377, 16378, 16379, 16380, 16381, 16382,
    16383, 16384, 16384, 16385, 16385, 16386, 16386, 16387, 16383, 16384, 16384,
    16385, 16385, 16386, 16386, 16387, 16387, 16388, 16388, 16389, 16389, 16390,
    16390, 16391, 16387, 16388, 16388, 16389, 16389, 16390, 16390, 16391, 16391,
    16392, 16393, 16393, 16394, 16394, 16395, 16395, 16391, 16392, 16393, 16393,
    16394, 16394, 16395, 16395, 16396, 16396, 16397, 16397, 16398, 16399, 16399,
    16400, 16396, 16396, 16397, 16397, 16398, 16399, 16399, 16400, 16400, 16401,
    16401, 16402, 16402, 16403, 16404, 16404, 16400, 16401, 16401, 16402, 16402,
    16403, 16404, 16404, 16405, 16405, 16406, 16407, 16407, 16408, 16408, 16409,
    16405, 16405, 16406, 16407, 16407, 16408, 16408, 16409, 16410, 16410, 16411,
    16411, 16412, 16413, 16413, 16414, 16410, 16410, 16411, 16411, 16412, 16413,
    16413, 16414, 16414, 16415, 16416, 16416, 16417, 16418, 16418, 16419, 16414,
    16415, 16416, 16416, 16417, 16418, 16418, 16419, 16419, 16420, 16421, 16421,
    16422, 16423, 16423, 16424, 16419, 16420, 16421, 16421, 16422, 16423, 16423,
    16424, 16425, 16425, 16426, 16427, 16427, 16428, 16429, 16429, 16425, 16425,
    16426, 16427, 16427, 16428, 16429, 16429};

extern "C" {
__attribute__((always_inline)) v16accfloat getExpBf16(v16bfloat16 x) {
  bfloat16 __aie_dm_resource_a *ilut_ab =
      (bfloat16 __aie_dm_resource_a *)exp_ilut_ab;
  bfloat16 __aie_dm_resource_b *ilut_cd =
      (bfloat16 __aie_dm_resource_b *)exp_ilut_cd;
  bfloat16 __aie_dm_resource_a *flut_ab =
      (bfloat16 __aie_dm_resource_a *)exp_flut_ab;
  bfloat16 __aie_dm_resource_b *flut_cd =
      (bfloat16 __aie_dm_resource_b *)exp_flut_cd;

  using lut_type = aie::lut<4, bfloat16, bfloat16>;
  const int LUT_elems = 256;
  const int step_i = 8;
  const int step_f = 0;

  lut_type lut_i(LUT_elems, ilut_ab, ilut_cd);
  lut_type lut_f(LUT_elems, flut_ab, flut_cd);
  aie::parallel_lookup<uint16, lut_type, aie::lut_oor_policy::truncate>
      lookup_i(lut_i, step_i);
  aie::parallel_lookup<uint16, lut_type, aie::lut_oor_policy::truncate>
      lookup_f(lut_f, step_f);

  aie::vector<bfloat16, 16> I_val_vec, F_val_vec;
  aie::accum<accfloat, 16> exp_val;
  aie::vector<bfloat16, 16> input_bf16 = x;

  // position of output decimal point = 8, making input become 8 bits, and for
  // LUT_elems = 256 lookup. aie::vector<int16, 16>
  // input=aie::to_fixed<int16>(input_bf16,8);
  aie::vector<int16, 32> input0 = v32int16(bfloat16_to_int(input_bf16, 8));
  aie::vector<int16, 16> input = aie::filter_even(input0);

  I_val_vec = lookup_i.fetch(input.cast_to<uint16>());
  F_val_vec = lookup_f.fetch(input.cast_to<uint16>());
  exp_val = aie::mul(I_val_vec, F_val_vec);
  return v16accfloat(exp_val);
}
}  // extern "C"

void softmax_simple_bf16(bfloat16 *restrict input_vector,
                         bfloat16 *restrict output_vector,
                         const int32_t vector_size) {
  event0();

  int num_elems = vector_size;
  float accum_exp_val;
  auto it_exp_in = aie::cbegin_vector<16>((bfloat16 *)input_vector);
  auto it_exp_out = aie::begin_vector<16>((bfloat16 *)output_vector);
  auto it_scale = aie::cbegin_restrict_vector<16>((bfloat16 *)output_vector);
  auto it_soft_out = aie::begin_restrict_vector<16>((bfloat16 *)output_vector);

  bfloat16 col_sum_inv;
  aie::vector<bfloat16, 16> in_elems, va;
  aie::accum<accfloat, 16> out_vals;
  int col_iters = num_elems >> 4;
  accum_exp_val = 0;

  /////////////////////
  //// Compute exp ////
  /////////////////////
  aie::vector<bfloat16, 16> exp_val;
  aie::vector<float, 16> input_fp32;

  const int elem_iters = num_elems / 16;
  aie::vector<bfloat16, 16> input_bf16;
  aie::accum<accfloat, 16> exp_val_accum;
  exp_val_accum = aie::zeros<accfloat, 16>();
  for (int i = 0; i < elem_iters; i++) {
    input_bf16 = *it_exp_in++;
    exp_val = to_v16bfloat16(getExpBf16(input_bf16));
    exp_val_accum = add(exp_val_accum, exp_val);
    *it_exp_out++ = exp_val;
  }
  aie::vector<float, 16> reduce = exp_val_accum.to_vector<float>();
  accum_exp_val = aie::reduce_add(reduce);
  /////////////////////

  col_sum_inv = (bfloat16)aie::inv(accum_exp_val);
  for (int c = 0; c < col_iters; c++) {
    in_elems = *it_scale++;
    out_vals = aie::mul(in_elems, col_sum_inv);
    *it_soft_out++ = out_vals.to_vector<bfloat16>();
  }

  event1();

  return;
}

extern "C" {

#define matmul_combos(X, M, N, K)                                     \
  X(bfloat16, bf16, bfloat16, bf16, bfloat16, bf16, M, N, K, 4, 8, 4) \
  X(bfloat16, bf16, bfloat16, bf16, float, f32, M, N, K, 4, 8, 4)

#define matmul_combos_i8(X, M, N, K)                                  \
  X(int8, i8, int8, i8, int32, i32, M, N, K, 4, 8, 8)

#define zero_fill_combos(X, M, N)  \
  X(bfloat16, bf16, M, N, N/2)     \
  X(float, f32, M, N, N/2)         \
  X(int32, i32, M, N, N/2)

#define matmul_vectorized_c_func(lhs_ctype_in, lhs_mlir_type_in,                                             \
                                 rhs_ctype_in, rhs_mlir_type_in,                                             \
                                 acc_ctype_out, acc_mlir_type_out, M, N, K, r, s, t)                         \
  void matmul_##lhs_mlir_type_in##_##rhs_mlir_type_in##_##acc_mlir_type_out##_##M##x##N##x##K##_##r##x##s##x##t( \
      lhs_ctype_in *a_in, unsigned offsetA, rhs_ctype_in *b_in, unsigned offsetB,                            \
      acc_ctype_out *c_out, unsigned offsetC) {                                                              \
    matmul_vectorized_##r##x##s##x##t##_##lhs_mlir_type_in##_##rhs_mlir_type_in##_##acc_mlir_type_out<       \
        M, K, N>(a_in, offsetA, b_in, offsetB, c_out, offsetC);                                              \
  }

#define zero_vectorized_c_func(ctype_out, mlir_type_out, M, N, r)             \
  void zero_##mlir_type_out##_##M##x##N(ctype_out *c_out, unsigned offsetC) { \
    zero_vectorized<ctype_out, M, N, r>(c_out, offsetC);                      \
  }

#define softmax_c_func(ctype, mlir_type, M, N) \
  void softmax_##mlir_type##_##M##x##N(ctype *input, ctype *output) { \
    softmax_simple_bf16(input, output, N); \
  }

matmul_combos(matmul_vectorized_c_func, 16, 16, 32)
matmul_combos(matmul_vectorized_c_func, 16, 16, 64)
matmul_combos(matmul_vectorized_c_func, 32, 32, 32)
matmul_combos(matmul_vectorized_c_func, 64, 64, 64)
matmul_combos(matmul_vectorized_c_func, 32, 32, 64)
matmul_combos(matmul_vectorized_c_func, 32, 32, 128)
matmul_combos(matmul_vectorized_c_func, 64, 32, 128)
matmul_combos_i8(matmul_vectorized_c_func, 16, 16, 32)
matmul_combos_i8(matmul_vectorized_c_func, 16, 16, 64)
matmul_combos_i8(matmul_vectorized_c_func, 32, 32, 8)
matmul_combos_i8(matmul_vectorized_c_func, 32, 32, 16)
matmul_combos_i8(matmul_vectorized_c_func, 32, 32, 32)
matmul_combos_i8(matmul_vectorized_c_func, 32, 32, 64)
matmul_combos_i8(matmul_vectorized_c_func, 64, 64, 64)
matmul_combos_i8(matmul_vectorized_c_func, 64, 32, 128)
matmul_combos_i8(matmul_vectorized_c_func, 64, 64, 128)

zero_fill_combos(zero_vectorized_c_func, 16, 16)
zero_fill_combos(zero_vectorized_c_func, 32, 32)
zero_fill_combos(zero_vectorized_c_func, 64, 32)
zero_fill_combos(zero_vectorized_c_func, 64, 64)


zero_fill_combos(zero_vectorized_c_func, 1, 32)
softmax_c_func(bfloat16, bf16, 1, 32)

}  // extern "C"
)chess"
