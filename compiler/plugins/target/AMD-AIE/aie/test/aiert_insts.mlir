// RUN: iree-opt --amdaie-dma-to-npu %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @of_toMem : memref<32xi32>
// CHECK:           memref.global "public" @of_fromMem : memref<32xi32>
// CHECK:           aie.shim_dma_allocation @of_fromMem(MM2S, 0, 0)
// CHECK:           aie.shim_dma_allocation @of_toMem(S2MM, 0, 0)
// CHECK:         } {npu_instructions = dense_resource<npu_instructions> : tensor<64xui32>, runtime_sequence_name = "sequence"}

// CHECK:         {-#
// CHECK:           dialect_resources: {
// CHECK:             builtin: {
// CHECK:               npu_instructions: "0x0400000000010306050100000600000000010000010000000000000020D0010030000000200000000000000000000000000000000000008000000000000000000000000281000000300000000000000000000000000000000000000024D001000000000002000000000000000000000000000000000000000000000004D20100000000000100008018000000010000000000000000D001003000000020000000800000000000000000008000070020800F000000000000000000000281000000300000000000000000000000000000000000000004D001000000000000000000000000008000000000000000000000000000000014D20100000000000000000018000000"
// CHECK:             }
// CHECK:           }
// CHECK:         #-}


module {
  aie.device(npu1_4col) {
    memref.global "public" @of_toMem : memref<32xi32>
    memref.global "public" @of_fromMem : memref<32xi32>
    aiex.runtime_sequence @sequence(%in : memref<4x2x8xi32>, %buf : memref<32xi32>, %out : memref<64xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      %c4 = arith.constant 4 : i64
      %c8 = arith.constant 8 : i64
      %c16 = arith.constant 16 : i64
      %c32 = arith.constant 32 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c32][%c0,%c0,%c0,%c1]) { metadata = @of_toMem, id = 1 : i64 } : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in[%c0,%c2,%c0,%c0][%c1,%c2,%c2,%c8][%c0,%c16,%c8,%c1]) { metadata = @of_fromMem, id = 0 : i64 } : memref<4x2x8xi32>
    }
    aie.shim_dma_allocation @of_fromMem (MM2S, 0, 0)
    aie.shim_dma_allocation @of_toMem (S2MM, 0, 0)
  }
}
