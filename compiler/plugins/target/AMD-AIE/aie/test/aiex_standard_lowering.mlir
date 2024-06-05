
// RUN: iree-opt --aiex-standard-lowering %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @toMem : memref<16xi32>
// CHECK:           func.func @dma_and_wait(%[[ARG0:.*]]: memref<16xi32>, %[[ARG1:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @toMem(MM2S, 1, 1)
// CHECK:         }

module  {
  aie.device(npu1_4col) {
    memref.global "public" @toMem : memref<16xi32>
    func.func @dma_and_wait(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 16, 16][0, 0, 64]) { metadata = @toMem, id = 1 : i64 } : memref<16xi32>
      aiex.npu.dma_wait {symbol = @toMem}
      return
    }
    aie.shim_dma_allocation @toMem (MM2S, 1, 1)
  }
}
