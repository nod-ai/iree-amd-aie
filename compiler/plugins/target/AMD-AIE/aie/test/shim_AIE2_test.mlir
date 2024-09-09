
// RUN: iree-opt --amdaie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           memref.global "public" @of_out : memref<16xi32>
// CHECK:           memref.global "public" @of_in : memref<16xi32>
// CHECK-DAG:       %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK-DAG:       %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK-DAG:       %[[OF_OUT_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_0]]) {init = 0 : i8, sym_name = "of_out_cons_prod_lock_0"}
// CHECK-DAG:       %[[OF_OUT_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_0]]) {init = 0 : i8, sym_name = "of_out_cons_cons_lock_0"}
// CHECK-DAG:       %[[OF_OUT_BUFF_0:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "of_out_prod_buff_0_0"} : memref<16xi32>
// CHECK-DAG:       %[[OF_OUT_BUFF_1:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "of_out_prod_buff_0_1"} : memref<16xi32>
// CHECK-DAG:       %[[OF_OUT_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_2]]) {init = 2 : i8, sym_name = "of_out_prod_prod_lock_0"}
// CHECK-DAG:       %[[OF_OUT_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_2]]) {init = 0 : i8, sym_name = "of_out_prod_cons_lock_0"}
// CHECK-DAG:       %[[OF_IN_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "of_in_cons_buff_0_0"} : memref<16xi32>
// CHECK-DAG:       %[[OF_IN_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "of_in_cons_buff_0_1"} : memref<16xi32>
// CHECK-DAG:       %[[OF_IN_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_2]]) {init = 2 : i8, sym_name = "of_in_cons_prod_lock_0"}
// CHECK-DAG:       %[[OF_IN_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_2]]) {init = 0 : i8, sym_name = "of_in_cons_cons_lock_0"}
// CHECK-DAG:       %[[OF_IN_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_0]]) {init = 0 : i8, sym_name = "of_in_prod_prod_lock_0"}
// CHECK-DAG:       %[[OF_IN_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_0]]) {init = 0 : i8, sym_name = "of_in_prod_cons_lock_0"}
// CHECK-DAG:       aie.flow(%[[TILE_2_0]], DMA : 0, %[[TILE_2_2]], DMA : 0)
// CHECK-DAG:       aie.flow(%[[TILE_2_2]], DMA : 0, %[[TILE_2_0]], DMA : 0)
// CHECK-DAG:       %[[EXT_BUFFER_IN:.*]] = aie.external_buffer {sym_name = "ext_buffer_in"} : memref<64xi32>
// CHECK-DAG:       %[[EXT_BUFFER_OUT:.*]] = aie.external_buffer {sym_name = "ext_buffer_out"} : memref<64xi32>
// CHECK-DAG:       aie.shim_dma_allocation @of_in(MM2S, 0, 2)
// CHECK-DAG:       aie.shim_dma_allocation @of_out(S2MM, 0, 2)
// CHECK:           %[[MEM_2_2:.*]] = aie.mem(%[[TILE_2_2]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OF_IN_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_IN_CONS_BUFF_0]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[OF_IN_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[OF_IN_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_IN_CONS_BUFF_1]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[OF_IN_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[OF_OUT_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_OUT_BUFF_0]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[OF_OUT_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[OF_OUT_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_OUT_BUFF_1]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[OF_OUT_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }
module @shim_AIE2 {
   aie.device(xcve2302) {
      %tile22 = aie.tile(2, 2)
      %tile20 = aie.tile(2, 0)
      aie.flow(%tile20, DMA : 0, %tile22, DMA : 0) {symbol = @of_in}
      aie.flow(%tile22, DMA : 0, %tile20, DMA : 0) {symbol = @of_out}
      aie.objectfifo @of_in (%tile20, {%tile22}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
      aie.objectfifo @of_out (%tile22, {%tile20}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
      %ext_buffer_in  = aie.external_buffer {sym_name = "ext_buffer_in"}: memref<64xi32>
      %ext_buffer_out  = aie.external_buffer {sym_name = "ext_buffer_out"}: memref<64xi32>
      aie.objectfifo.register_external_buffers @of_in (%tile20, {%ext_buffer_in}) : (memref<64xi32>)
      aie.objectfifo.register_external_buffers @of_out (%tile20, {%ext_buffer_out}) : (memref<64xi32>)
   }
}
