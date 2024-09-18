
// RUN: iree-opt --split-input-file --amdaie-dma-to-npu %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @toMem : memref<16xi32>
// CHECK:           memref.global "public" @fromMem : memref<16xi32>
// CHECK:           aie.shim_dma_allocation @fromMem(MM2S, 0, 0)
// CHECK:           aie.shim_dma_allocation @toMem(S2MM, 0, 0)

module  {
  aie.device(npu1_4col) {
    memref.global "public" @toMem : memref<16xi32>
    memref.global "public" @fromMem : memref<16xi32>
    aiex.runtime_sequence @dma_memcpy_nd_0(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 16, 16][0, 0, 64, 1]) { metadata = @toMem, id = 1 : i64 } : memref<16xi32>
      aiex.npu.dma_memcpy_nd(0, 1, %arg1[0, 0, 0, 16][1, 1, 16, 16][0, 0, 64, 1]) { metadata = @fromMem, id = 0 : i64 } : memref<16xi32>
    }
    aie.shim_dma_allocation @fromMem (MM2S, 0, 0)
    aie.shim_dma_allocation @toMem (S2MM, 0, 0)
  }
}

// -----

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @toMem : memref<16xi32>
// CHECK:           aie.shim_dma_allocation @toMem(S2MM, 0, 0)

module  {
  aie.device(npu1_4col) {
    memref.global "public" @toMem : memref<16xi32>
    aiex.runtime_sequence @dma_wait_s2mm(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 16, 16][0, 0, 64, 1]) { metadata = @toMem, id = 1 : i64 } : memref<16xi32>
      aiex.npu.dma_wait {symbol = @toMem}
    }
    aie.shim_dma_allocation @toMem (S2MM, 0, 0)
  }
}

// -----

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @toMem : memref<16xi32>
// CHECK:           aie.shim_dma_allocation @toMem(MM2S, 1, 1)

module  {
  aie.device(npu1_4col) {
    memref.global "public" @toMem : memref<16xi32>
    aiex.runtime_sequence @dma_wait_mm2s(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 16, 16][0, 0, 64, 1]) { metadata = @toMem, id = 1 : i64 } : memref<16xi32>
      aiex.npu.dma_wait {symbol = @toMem}
    }
    aie.shim_dma_allocation @toMem (MM2S, 1, 1)
  }
}

// -----

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @toMem : memref<16xi32>
// CHECK:           func.func @pretend_microkernel
// CHECK-NOT:       aiex.runtime_sequence @explicit_sym_name
// CHECK:           aie.shim_dma_allocation @toMem(MM2S, 1, 1)

module  {
  aie.device(npu1_4col) {
    memref.global "public" @toMem : memref<16xi32>
    func.func @pretend_microkernel(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
      return
    }

    aiex.runtime_sequence @explicit_sym_name(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 16, 16][0, 0, 64, 1]) { metadata = @toMem, id = 1 : i64 } : memref<16xi32>
      aiex.npu.dma_wait {symbol = @toMem}
    }
    aie.shim_dma_allocation @toMem (MM2S, 1, 1)
  } {sym_name = "explicit_sym_name_0"}
}

// -----

// Issue packet header from shim dma bd
// CHECK: memref.global "public" @toMem : memref<16xi32>
// CHECK: aie.shim_dma_allocation @toMem(S2MM, 0, 0)
// CHECK: npu_instructions = dense_resource<npu_instructions> : tensor<34xui32>, runtime_sequence_name = "packet_enable"
// CHECK: npu_instructions:
module {
  aie.device(npu1_4col) {
    memref.global "public" @toMem : memref<16xi32>
    aiex.runtime_sequence @packet_enable(%arg0: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd (0, 0, %arg0[0, 0, 0, 0][1, 1, 16, 16][0, 0, 64, 1], packet = <pkt_id = 2, pkt_type = 3>) { metadata = @toMem, id = 1 : i64 } : memref<16xi32>
    }
    aie.shim_dma_allocation @toMem (S2MM, 0, 0)
  }
}
