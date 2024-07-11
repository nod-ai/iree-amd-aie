
// RUN: iree-opt --amdaie-dma-to-npu %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @toMem : memref<16xi32>
// CHECK:           memref.global "public" @fromMem : memref<16xi32>
// CHECK:           aie.shim_dma_allocation @fromMem(MM2S, 0, 0)
// CHECK:           aie.shim_dma_allocation @toMem(S2MM, 0, 0)
// CHECK:         } {npu_instructions = array<i32: 100860160, 261, 6, 256, 1, 0, 118816, 48, 256, 0, 0, 16777216, -2147483585, 0, 0, 33554432, 129, 48, 0, 0, 0, 0, 118820, 0, 0, 0, 0, 0, 0, 0, 119300, 0, -2147483647, 24, 1, 0, 118784, 48, 256, 64, 0, 16777216, -2147483585, 0, 0, 33554432, 129, 48, 0, 0, 0, 0, 118788, 0, 1, 0, 64, 0, 0, 0, 119316, 0, 0, 24>}

module  {
  aie.device(npu1_4col) {
    memref.global "public" @toMem : memref<16xi32>
    memref.global "public" @fromMem : memref<16xi32>
    func.func @test1(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
        aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 16, 16][0, 0, 64, 1]) { metadata = @toMem, id = 1 : i64, issue_token = true } : memref<16xi32>
        aiex.npu.dma_memcpy_nd(0, 1, %arg1[0, 0, 0, 16][1, 1, 16, 16][0, 0, 64, 1]) { metadata = @fromMem, id = 0 : i64 } : memref<16xi32>
        return
    }
    aie.shim_dma_allocation @fromMem (MM2S, 0, 0)
    aie.shim_dma_allocation @toMem (S2MM, 0, 0)
  }
}
