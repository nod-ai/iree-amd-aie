// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Unit tests for deriveHostPatchTableFromTransaction — the parser that walks a
// serialized XAie transaction binary, correlates each DDR_PATCH custom op to
// the BLOCKWRITE whose register span contains its BD base, and emits the flat
// (offset, arg_idx, arg_plus) triples the HAL uses to host-patch shim-DMA
// buffer addresses on the ERT_CMD_CHAIN path. This is the single place that
// understands the TXN binary layout (op headers, custom-op size fields, the
// BLOCKWRITE-payload-span trick); a bug here either silently drops patches
// (output buffers stay zero) or scribbles outside the control code.

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIETransactionBuilder.h"

namespace {

using namespace mlir::iree_compiler::AMDAIE;

// Little-endian byte writers — the XAie TXN binary is built as bytes and we
// then reinterpret it as u32. Construction in bytes is what the parser reads.
struct Bytes {
  std::vector<uint8_t> data;
  size_t size() const { return data.size(); }
  void u8(uint8_t v) { data.push_back(v); }
  void u32_at(size_t off, uint32_t v) {
    if (data.size() < off + 4) data.resize(off + 4, 0);
    for (int i = 0; i < 4; ++i) data[off + i] = (v >> (8 * i)) & 0xff;
  }
  void pad_to(size_t off) {
    if (data.size() < off) data.resize(off, 0);
  }
  std::vector<uint32_t> as_u32() const {
    std::vector<uint32_t> out(data.size() / 4);
    std::memcpy(out.data(), data.data(), out.size() * 4);
    return out;
  }
};

// Sets up a 16-byte XAie_TxnHeader; NumOps and TxnSize get filled in once the
// ops are appended.
static void WriteTxnHeader(Bytes &b) {
  b.pad_to(16);  // Major/Minor/DevGen/etc. stay 0 — the parser ignores them.
}

// Appends a BLOCKWRITE op (opcode 1) programming `payload_bytes` bytes of data
// starting at register `reg`. Header layout: byte0=opcode, reg at p+8, size at
// p+12 (total bytes = 16 + payload_bytes). The payload itself is zero-filled —
// the parser only cares about its span [reg, reg+payload_bytes), not contents.
static size_t AppendBlockWrite(Bytes &b, uint32_t reg, uint32_t payload_bytes) {
  size_t op_off = b.size();
  size_t total = 16 + payload_bytes;
  b.pad_to(op_off + total);
  b.data[op_off] = /*opcode=*/1;
  b.u32_at(op_off + 8, reg);
  b.u32_at(op_off + 12, static_cast<uint32_t>(total));
  return op_off;
}

// Appends a DDR_PATCH op (opcode 129, size 44). Field offsets follow XRT's
// patch_op_t / XAie_CustomOpHdr: size@p+4, regaddr@p+24, argidx@p+32,
// argplus@p+40.
static void AppendDdrPatch(Bytes &b, uint32_t regaddr, uint32_t argidx,
                           uint32_t argplus) {
  size_t op_off = b.size();
  size_t total = 44;
  b.pad_to(op_off + total);
  b.data[op_off] = /*opcode=*/129;
  b.u32_at(op_off + 4, static_cast<uint32_t>(total));
  b.u32_at(op_off + 24, regaddr);
  b.u32_at(op_off + 32, argidx);
  b.u32_at(op_off + 40, argplus);
}

// Sets NumOps@byte8 and TxnSize@byte12 once all ops are appended.
static std::vector<uint32_t> FinalizeTxn(Bytes &b, uint32_t num_ops) {
  b.u32_at(8, num_ops);
  b.u32_at(12, static_cast<uint32_t>(b.size()));
  return b.as_u32();
}

TEST(DeriveHostPatchTableTest, EmptyTxnReturnsEmptyTable) {
  // Header-only TXN with NumOps=0 → no DDR_PATCH ops → no triples.
  Bytes b;
  WriteTxnHeader(b);
  auto txn = FinalizeTxn(b, /*num_ops=*/0);
  auto table = deriveHostPatchTableFromTransaction(txn);
  EXPECT_TRUE(table.empty());
}

TEST(DeriveHostPatchTableTest, SingleBdProducesOneTriple) {
  // One BLOCKWRITE at reg 0x1D000 with payload 16 bytes (one BD), followed by
  // a DDR_PATCH that targets bd[1] (BD base = regaddr & ~0xF = 0x1D000).
  // Expected: offset = blockwrite_payload_start + (0x1D000 - 0x1D000) = the
  // first payload word of the BLOCKWRITE.
  Bytes b;
  WriteTxnHeader(b);
  size_t bw_off = AppendBlockWrite(b, /*reg=*/0x1D000, /*payload_bytes=*/16);
  AppendDdrPatch(b, /*regaddr=*/0x1D004, /*argidx=*/0, /*argplus=*/0);
  auto txn = FinalizeTxn(b, /*num_ops=*/2);
  auto table = deriveHostPatchTableFromTransaction(txn);
  ASSERT_EQ(table.size(), 3u);
  // payload_off = bw_off + 16 (BLOCKWRITE header). The bd base 0x1D000 is at
  // bw.reg exactly, so the offset into the patch table is payload_off + 0.
  EXPECT_EQ(table[0], static_cast<uint32_t>(bw_off + 16));
  EXPECT_EQ(table[1], 0u);  // arg_idx
  EXPECT_EQ(table[2], 0u);  // arg_plus
}

TEST(DeriveHostPatchTableTest, WideBlockWriteSpansMultipleBdsAndPatchesMiddle) {
  // The bug-prone case: a single BLOCKWRITE of 3 BDs (payload 0x60 = 96 bytes,
  // i.e. 3 × 0x20-stride BDs) at reg 0x1D000. A DDR_PATCH targets the MIDDLE
  // BD (regaddr 0x1D024 → bd base 0x1D020). The parser must match the
  // covering BLOCKWRITE by SPAN (reg ≤ bd_base < reg + payload_bytes) and
  // index into the payload by (bd_base - reg) = 0x20, NOT match the
  // BLOCKWRITE's exact starting register.
  Bytes b;
  WriteTxnHeader(b);
  size_t bw_off = AppendBlockWrite(b, /*reg=*/0x1D000, /*payload_bytes=*/0x60);
  AppendDdrPatch(b, /*regaddr=*/0x1D024, /*argidx=*/2, /*argplus=*/0x100);
  auto txn = FinalizeTxn(b, /*num_ops=*/2);
  auto table = deriveHostPatchTableFromTransaction(txn);
  ASSERT_EQ(table.size(), 3u);
  // Expected offset = payload_off + (bd_base - reg) = (bw_off + 16) + 0x20.
  EXPECT_EQ(table[0], static_cast<uint32_t>(bw_off + 16 + 0x20));
  EXPECT_EQ(table[1], 2u);      // arg_idx
  EXPECT_EQ(table[2], 0x100u);  // arg_plus (cumulative byte offset)
}

TEST(DeriveHostPatchTableDeathTest, DdrPatchWithNoCoveringBlockWriteIsFatal) {
  // A DDR_PATCH whose BD base falls outside any BLOCKWRITE's span is a
  // compiler bug — silently dropping it would leave the BD's shim-DMA address
  // at zero at runtime (output buffer reads as zeros). The parser must abort
  // loudly via llvm::report_fatal_error so the bug surfaces at compile time.
  Bytes b;
  WriteTxnHeader(b);
  AppendBlockWrite(b, /*reg=*/0x1D000, /*payload_bytes=*/16);
  AppendDdrPatch(b, /*regaddr=*/0x1E004, /*argidx=*/0,
                 /*argplus=*/0);  // 0x1E000 is outside [0x1D000, 0x1D010).
  auto txn = FinalizeTxn(b, /*num_ops=*/2);
  EXPECT_DEATH(
      { (void)deriveHostPatchTableFromTransaction(txn); },
      "DDR_PATCH op has no covering BLOCKWRITE");
}

TEST(DeriveHostPatchTableTest,
     MultipleBlockWritesAndDdrPatchesProduceOrderedTable) {
  // Two BLOCKWRITEs at distinct register ranges plus three DDR_PATCHes that
  // target one BD in the first BLOCKWRITE, one in the second, and one in the
  // first again. The output triples must appear in the order the DDR_PATCH
  // ops appear in the TXN stream (parser walks ops linearly).
  Bytes b;
  WriteTxnHeader(b);
  size_t bw1 = AppendBlockWrite(b, /*reg=*/0x1D000, /*payload_bytes=*/0x40);
  size_t bw2 = AppendBlockWrite(b, /*reg=*/0x1D100, /*payload_bytes=*/0x40);
  AppendDdrPatch(b, /*regaddr=*/0x1D004, /*argidx=*/0, /*argplus=*/0x0);
  AppendDdrPatch(b, /*regaddr=*/0x1D124, /*argidx=*/1, /*argplus=*/0x20);
  AppendDdrPatch(b, /*regaddr=*/0x1D024, /*argidx=*/0, /*argplus=*/0x40);
  auto txn = FinalizeTxn(b, /*num_ops=*/5);
  auto table = deriveHostPatchTableFromTransaction(txn);
  ASSERT_EQ(table.size(), 9u);  // three triples
  // Triple 0: BD 0x1D000 in BLOCKWRITE-1 → bw1 payload + 0x00.
  EXPECT_EQ(table[0], static_cast<uint32_t>(bw1 + 16));
  EXPECT_EQ(table[1], 0u);
  EXPECT_EQ(table[2], 0x0u);
  // Triple 1: BD 0x1D120 in BLOCKWRITE-2 → bw2 payload + 0x20.
  EXPECT_EQ(table[3], static_cast<uint32_t>(bw2 + 16 + 0x20));
  EXPECT_EQ(table[4], 1u);
  EXPECT_EQ(table[5], 0x20u);
  // Triple 2: BD 0x1D020 in BLOCKWRITE-1 → bw1 payload + 0x20.
  EXPECT_EQ(table[6], static_cast<uint32_t>(bw1 + 16 + 0x20));
  EXPECT_EQ(table[7], 0u);
  EXPECT_EQ(table[8], 0x40u);
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
