
// RUN: iree-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @of_out_cons : memref<16xi32>
// CHECK:           memref.global "public" @of_out : memref<16xi32>
// CHECK:           memref.global "public" @of_in_cons : memref<16xi32>
// CHECK:           memref.global "public" @of_in : memref<16xi32>
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[OF_OUT_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_0]], 2) {init = 1 : i32, sym_name = "of_out_cons_prod_lock"}
// CHECK:           %[[OF_OUT_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_0]], 3) {init = 0 : i32, sym_name = "of_out_cons_cons_lock"}
// CHECK:           %[[OF_OUT_BUFF_0:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "of_out_buff_0"} : memref<16xi32>
// CHECK:           %[[OF_OUT_BUFF_1:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "of_out_buff_1"} : memref<16xi32>
// CHECK:           %[[OF_OUT_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 2) {init = 2 : i32, sym_name = "of_out_prod_lock"}
// CHECK:           %[[OF_OUT_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 3) {init = 0 : i32, sym_name = "of_out_cons_lock"}
// CHECK:           %[[OF_IN_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "of_in_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[OF_IN_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "of_in_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[OF_IN_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 0) {init = 2 : i32, sym_name = "of_in_cons_prod_lock"}
// CHECK:           %[[OF_IN_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 1) {init = 0 : i32, sym_name = "of_in_cons_cons_lock"}
// CHECK:           %[[OF_IN_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_0]], 0) {init = 1 : i32, sym_name = "of_in_prod_lock"}
// CHECK:           %[[OF_IN_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_0]], 1) {init = 0 : i32, sym_name = "of_in_cons_lock"}
// CHECK:           aie.flow(%[[TILE_2_0]], DMA : 0, %[[TILE_2_2]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_2_2]], DMA : 0, %[[TILE_2_0]], DMA : 0)
// CHECK:           %[[EXT_BUFFER_IN:.*]] = aie.external_buffer {sym_name = "ext_buffer_in"} : memref<64xi32>
// CHECK:           %[[EXT_BUFFER_OUT:.*]] = aie.external_buffer {sym_name = "ext_buffer_out"} : memref<64xi32>
// CHECK:           aie.shim_dma_allocation @of_in(MM2S, 0, 2)
// CHECK:           %[[SHIM_DMA_2_0:.*]] = aie.shim_dma(%[[TILE_2_0]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OF_IN_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[EXT_BUFFER_IN]] : memref<64xi32>, 0, 64)
// CHECK:             aie.use_lock(%[[OF_IN_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[OF_OUT_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[EXT_BUFFER_OUT]] : memref<64xi32>, 0, 64)
// CHECK:             aie.use_lock(%[[OF_OUT_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @of_out(S2MM, 0, 2)
// CHECK:           %[[MEM_2_2:.*]] = aie.mem(%[[TILE_2_2]]) {
// CHECK:             %[[VAL_2:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OF_IN_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_IN_CONS_BUFF_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OF_IN_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[OF_IN_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_IN_CONS_BUFF_1]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OF_IN_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_3:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[OF_OUT_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_OUT_BUFF_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OF_OUT_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[OF_OUT_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_OUT_BUFF_1]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OF_OUT_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @shim_AIE2 {
   aie.device(npu1_4col) {
      %tile22 = aie.tile(2, 2)
      %tile20 = aie.tile(2, 0)
      aie.objectfifo @of_in (%tile20, {%tile22}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
      aie.objectfifo @of_out (%tile22, {%tile20}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
      %ext_buffer_in  = aie.external_buffer {sym_name = "ext_buffer_in"}: memref<64xi32>
      %ext_buffer_out  = aie.external_buffer {sym_name = "ext_buffer_out"}: memref<64xi32>
      aie.objectfifo.register_external_buffers @of_in (%tile20, {%ext_buffer_in}) : (memref<64xi32>)
      aie.objectfifo.register_external_buffers @of_out (%tile20, {%ext_buffer_out}) : (memref<64xi32>)
   }
}
