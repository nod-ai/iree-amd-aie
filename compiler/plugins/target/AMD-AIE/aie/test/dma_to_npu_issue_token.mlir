
// RUN: iree-opt --aie-dma-to-npu %s | FileCheck %s
// XFAIL: *
// waiting on catching up to https://github.com/Xilinx/mlir-aie/pull/1559
// i.e. we're still outputting ddr_id here

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @toMem : memref<16xi32>
// CHECK:           memref.global "public" @fromMem : memref<16xi32>
// CHECK:           func.func @test1(%[[ARG0:.*]]: memref<16xi32>, %[[ARG1:.*]]: memref<16xi32>) {
// CHECK:             aiex.npu.writebd {bd_id = 1 : i32, buffer_length = 256 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d1_size = 0 : i32, d1_stride = 63 : i32, d2_stride = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
// CHECK:             aiex.npu.address_patch {addr = 118820 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
// CHECK:             aiex.npu.write32 {address = 119300 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483649 : ui32}
// CHECK:             aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 256 : i32, buffer_offset = 64 : i32, column = 0 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d1_size = 0 : i32, d1_stride = 63 : i32, d2_stride = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
// CHECK:             aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 1 : i32, arg_plus = 64 : i32}
// CHECK:             aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 0 : ui32}
// CHECK:             return
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @fromMem(MM2S, 0, 0)
// CHECK:           aie.shim_dma_allocation @toMem(S2MM, 0, 0)
// CHECK:         }

module  {
  aie.device(npu1_4col) {
    memref.global "public" @toMem : memref<16xi32>
    memref.global "public" @fromMem : memref<16xi32>
    func.func @test1(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
        aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 16, 16][0, 0, 64]) { metadata = @toMem, id = 1 : i64, issue_token = true } : memref<16xi32>
        aiex.npu.dma_memcpy_nd(0, 1, %arg1[0, 0, 0, 16][1, 1, 16, 16][0, 0, 64]) { metadata = @fromMem, id = 0 : i64 } : memref<16xi32>
        return
    }
    aie.shim_dma_allocation @fromMem (MM2S, 0, 0)
    aie.shim_dma_allocation @toMem (S2MM, 0, 0)
  }
}
