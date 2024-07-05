// RUN: iree-opt --amdaie-dma-to-npu %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @of_toMem : memref<32xi32>
// CHECK:           memref.global "public" @of_fromMem : memref<32xi32>
// CHECK:           func.func @sequence(%[[ARG0:.*]]: memref<4x2x8xi32>, %[[ARG1:.*]]: memref<32xi32>, %[[ARG2:.*]]: memref<64xi32>) {
// CHECK:             %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK:             %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:             %[[C2_I64:.*]] = arith.constant 2 : i64
// CHECK:             %[[C4_I64:.*]] = arith.constant 4 : i64
// CHECK:             %[[C8_I64:.*]] = arith.constant 8 : i64
// CHECK:             %[[C16_I64:.*]] = arith.constant 16 : i64
// CHECK:             %[[C32_I64:.*]] = arith.constant 32 : i64
// CHECK:             return
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @of_fromMem(MM2S, 0, 0)
// CHECK:           aie.shim_dma_allocation @of_toMem(S2MM, 0, 0)
// CHECK:         } {npu_instructions = array<i32: 100860160, 261, 6, 256, 1, 0, 118816, 48, 32, 0, 0, 0, -2147483648, 0, 0, 33554432, 129, 48, 0, 0, 0, 0, 118820, 0, 2, 0, 0, 0, 0, 0, 119300, 0, -2147483647, 24, 1, 0, 118784, 48, 32, 128, 0, 8388608, -2145386489, 15, 0, 33554432, 129, 48, 0, 0, 0, 0, 118788, 0, 0, 0, 128, 0, 0, 0, 119316, 0, 0, 24>}

module {
  aie.device(npu1_4col) {
    memref.global "public" @of_toMem : memref<32xi32>
    memref.global "public" @of_fromMem : memref<32xi32>
    func.func @sequence(%in : memref<4x2x8xi32>, %buf : memref<32xi32>, %out : memref<64xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      %c4 = arith.constant 4 : i64
      %c8 = arith.constant 8 : i64
      %c16 = arith.constant 16 : i64
      %c32 = arith.constant 32 : i64
      aiex.npu.dma_memcpy_nd(0, 0, %out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c32][%c0,%c0,%c0,%c1]) { metadata = @of_toMem, id = 1 : i64 } : memref<64xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %in[%c0,%c2,%c0,%c0][%c1,%c2,%c2,%c8][%c0,%c16,%c8,%c1]) { metadata = @of_fromMem, id = 0 : i64 } : memref<4x2x8xi32>
      return
    }
    aie.shim_dma_allocation @of_fromMem (MM2S, 0, 0)
    aie.shim_dma_allocation @of_toMem (S2MM, 0, 0)
  }
}
