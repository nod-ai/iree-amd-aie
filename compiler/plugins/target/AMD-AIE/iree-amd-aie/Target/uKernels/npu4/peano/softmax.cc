// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

namespace {

constexpr int kVecLanes = 32;

// Horizontal max-reduce a v32bfloat16 to a single bf16.
INTRINSIC(bfloat16) reduce_max_v32bf16(v32bfloat16 v) {
  const v32bfloat16 z = broadcast_zero_to_v32bfloat16();
  v = max(v, shift(v, z, 16));
  v = max(v, shift(v, z, 8));
  v = max(v, shift(v, z, 4));
  v = max(v, shift(v, z, 2));
  v = max(v, shift(v, z, 1));
  return ext_elem(v, 0);
}

// Horizontal add-reduce a v32accfloat to a scalar float.
INTRINSIC(float) reduce_add_v32accf(v32accfloat acc) {
  const v16accfloat z = broadcast_zero_to_v16accfloat();
  // Fold 32 -> 16 lanes (lossless), then a log2(16) = 4-step shift/add
  // butterfly down to lane 0, all in accfloat.
  v16accfloat r = add(extract_v16accfloat(acc, 0), extract_v16accfloat(acc, 1));
  r = add(r, shift(r, z, 8));
  r = add(r, shift(r, z, 4));
  r = add(r, shift(r, z, 2));
  r = add(r, shift(r, z, 1));
  // Lane 0 holds the full f32 sum; narrow just that scalar to read it out.
  return (float)ext_elem(set_v32bfloat16(0, to_v16bfloat16(r)), 0);
}

// Three-pass softmax over a contiguous bf16 row of length `n`, where
// `n % 32 == 0`. Math:
//   pass 1: m = max_i(x_i) * log2e
//   pass 2: e_i = 2^(x_i * log2e - m) ; sum = sum_i e_i ; out[i] = e_i
//   pass 3: out[i] = e_i * (1 / sum)
inline void softmax_bf16_impl(bfloat16 *restrict in, bfloat16 *restrict out,
                              int32_t n) {
  event0();

  const int32_t iters = n / kVecLanes;

  // bf16-representable value of log2(e). 1.4453125 = 0x3FB9 in bf16, same
  // constant mlir-aie's aie_kernels/aie2p/softmax.cc uses.
  const v32bfloat16 log2e_v = broadcast_to_v32bfloat16((bfloat16)1.4453125f);

  // ---- Pass 1: running max of the raw row -----------------------------
  // 2^(x*log2e) is monotonic in x and log2e > 0, so
  //   max_i(x_i * log2e) = log2e * max_i(x_i).
  // Take the max in the raw x domain (no per-element multiply / demote)
  // and fold the log2e factor into the single scalar multiply below.
  v32bfloat16 max_v = broadcast_to_v32bfloat16((bfloat16)-32768.0f);
  {
    const v32bfloat16 *restrict pIn = (const v32bfloat16 *)in;
    for (int32_t i = 0; i < iters; ++i) {
      max_v = max(max_v, *pIn++);
    }
  }
  // m = max_i(x_i) * log2e (one scalar multiply, kept in f32). Pre-negate
  // and broadcast so pass 2 can fuse the scale-and-shift into one MAC.
  const float neg_m = -((float)reduce_max_v32bf16(max_v) * 1.4453125f);
  const v16accfloat neg_m16 = broadcast_to_v16accfloat(neg_m);
  const v32accfloat neg_m_acc = concat(neg_m16, neg_m16);

  // ---- Pass 2: e_i = exp2(x_i * log2e - m); accumulate sum; store e_i --
  // shifted = x_i * log2e - m in a single fused multiply-accumulate:
  // mac_elem_32 computes neg_m_acc + x * log2e.
  v32accfloat sum_acc = ups(broadcast_zero_to_v32bfloat16());
  {
    const v32bfloat16 *restrict pIn = (const v32bfloat16 *)in;
    v32bfloat16 *restrict pOut = (v32bfloat16 *)out;
    for (int32_t i = 0; i < iters; ++i) {
      v32bfloat16 x = *pIn++;
      v32accfloat shifted = mac_elem_32(x, /*sgn_x=*/1, log2e_v,
                                        /*sgn_y=*/1, neg_m_acc);
      v32bfloat16 e = exp2(shifted);
      *pOut++ = e;
      sum_acc = add(sum_acc, ups(e));
    }
  }

  // ---- Pass 3: out_i = e_i * (1 / sum) --------------------------------
  bfloat16 inv_sum = (bfloat16)inv(reduce_add_v32accf(sum_acc));
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
