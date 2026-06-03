// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

namespace {

constexpr int kVecLanes = 32;

// Horizontal max-reduce a v32bfloat16 to a single bf16.
INTRINSIC(bfloat16) reduce_max_v32bf16(v32bfloat16 v) {
  bfloat16 m = ext_elem(v, 0);
  for (int i = 1; i < kVecLanes; ++i) {
    bfloat16 e = ext_elem(v, i);
    m = (e > m) ? e : m;
  }
  return m;
}

// Horizontal add-reduce a v32accfloat to a scalar float.
INTRINSIC(float) reduce_add_v32accf(v32accfloat acc) {
  v16accfloat r16 =
      add(extract_v16accfloat(acc, 0), extract_v16accfloat(acc, 1));
  v16bfloat16 r16b = to_v16bfloat16(r16);
  v32bfloat16 r32b = set_v32bfloat16(0, r16b);
  float s = 0.0f;
  for (int i = 0; i < 16; ++i) {
    s += (float)ext_elem(r32b, i);
  }
  return s;
}

// Three-pass softmax over a contiguous bf16 row of length `n`, where
// `n % 32 == 0`. Math involved in this implementation:
//   pass 1: m = max_i(x_i * log2e)
//   pass 2: e_i = 2^(x_i * log2e - m) ; sum = sum_i e_i ; out[i] = e_i
//   pass 3: out[i] = e_i * (1 / sum)
inline void softmax_bf16_impl(bfloat16 *restrict in, bfloat16 *restrict out,
                              int32_t n) {
  event0();

  const int32_t iters = n / kVecLanes;

  // bf16 representable value of log2(e).
  const v32bfloat16 log2e_v = broadcast_to_v32bfloat16((bfloat16)1.4453125f);

  // ---- Pass 1: max-track of (x_i * log2e), in bf16 ---------------------
  v32bfloat16 max_v = broadcast_to_v32bfloat16((bfloat16)-32768.0f);
  {
    const v32bfloat16 *restrict pIn = (const v32bfloat16 *)in;
    for (int32_t i = 0; i < iters; ++i) {
      v32bfloat16 x = *pIn++;
      v32accfloat scaled_acc = mul_elem_32(x, /*sgn_x=*/1, log2e_v,
                                           /*sgn_y=*/1);
      v32bfloat16 scaled_bf = to_v32bfloat16(scaled_acc);
      max_v = max(max_v, scaled_bf);
    }
  }
  bfloat16 max_scalar = reduce_max_v32bf16(max_v);
  v32accfloat max_acc = ups(broadcast_to_v32bfloat16(max_scalar));

  // ---- Pass 2: e_i = exp2(x_i * log2e - m); accumulate sum; store e_i --
  v32accfloat sum_acc = ups(broadcast_to_v32bfloat16((bfloat16)0.0f));
  {
    const v32bfloat16 *restrict pIn = (const v32bfloat16 *)in;
    v32bfloat16 *restrict pOut = (v32bfloat16 *)out;
    for (int32_t i = 0; i < iters; ++i) {
      v32bfloat16 x = *pIn++;
      v32accfloat scaled = mul_elem_32(x, /*sgn_x=*/1, log2e_v,
                                       /*sgn_y=*/1);
      v32accfloat shifted = sub(scaled, max_acc);
      v32bfloat16 e = exp2(shifted);
      *pOut++ = e;
      sum_acc = add(sum_acc, ups(e));
    }
  }

  // ---- Pass 3: out_i = e_i * (1 / sum) ---------------------------------
  float total = reduce_add_v32accf(sum_acc);
  bfloat16 inv_sum = (bfloat16)inv(total);
  v32bfloat16 inv_sum_v = broadcast_to_v32bfloat16(inv_sum);
  {
    const v32bfloat16 *restrict pIn = (const v32bfloat16 *)out;
    v32bfloat16 *restrict pOut = (v32bfloat16 *)out;
    for (int32_t i = 0; i < iters; ++i) {
      v32bfloat16 e = *pIn++;
      v32accfloat scaled = mul_elem_32(e, /*sgn_x=*/1, inv_sum_v,
                                       /*sgn_y=*/1);
      *pOut++ = to_v32bfloat16(scaled);
    }
  }

  event1();
}

}  // namespace

// Entry-point factory.
// TODO(avarma): Revisit this to have just one ABI.
//
// Two ABIs are exposed here, so the same softmax.o serves both the `.aiec`
// per-row flow and the linalg.softmax -> ukernel matcher flow.
//
//   1. Bareptr-friendly per-N row specialisations.
//
//   2. Per-(M,N) tile specialisations.
extern "C" {

#define SOFTMAX_BF16_PER_N(N)                                                \
  void softmax_bf16_##N(bfloat16 *restrict input, unsigned input_offset,     \
                        bfloat16 *restrict output, unsigned output_offset) { \
    softmax_bf16_impl(input + input_offset, output + output_offset, (N));    \
  }

SOFTMAX_BF16_PER_N(32)
SOFTMAX_BF16_PER_N(64)
SOFTMAX_BF16_PER_N(128)
SOFTMAX_BF16_PER_N(256)
SOFTMAX_BF16_PER_N(512)
SOFTMAX_BF16_PER_N(1024)
SOFTMAX_BF16_PER_N(2048)

#undef SOFTMAX_BF16_PER_N

#define SOFTMAX_BF16_PER_MxN(M, N)                                             \
  void softmax_bf16_##M##x##N(bfloat16 *restrict input, unsigned input_offset, \
                              bfloat16 *restrict output,                       \
                              unsigned output_offset) {                        \
    bfloat16 *restrict in = input + input_offset;                              \
    bfloat16 *restrict out = output + output_offset;                           \
    for (int32_t r = 0; r < (M); ++r) {                                        \
      softmax_bf16_impl(in + r * (N), out + r * (N), (N));                     \
    }                                                                          \
  }

SOFTMAX_BF16_PER_MxN(4, 128)

#undef SOFTMAX_BF16_PER_MxN

}  // extern "C"
