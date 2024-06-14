
// RUN: iree-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @of_in_0_cons : memref<16xi32>
// CHECK:           memref.global "public" @of_in_1_cons : memref<16xi32>
// CHECK:           memref.global "public" @of_in_2_cons : memref<16xi32>
// CHECK:           memref.global "public" @of_in : memref<16xi32>
// CHECK:           %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[OF_IN_0_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "of_in_0_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[OF_IN_0_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "of_in_0_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[OF_IN_0_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 0) {init = 2 : i32, sym_name = "of_in_0_cons_prod_lock"}
// CHECK:           %[[OF_IN_0_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 1) {init = 0 : i32, sym_name = "of_in_0_cons_cons_lock"}
// CHECK:           %[[OF_IN_1_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "of_in_1_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[OF_IN_1_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "of_in_1_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[OF_IN_1_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_3]], 0) {init = 2 : i32, sym_name = "of_in_1_cons_prod_lock"}
// CHECK:           %[[OF_IN_1_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_3]], 1) {init = 0 : i32, sym_name = "of_in_1_cons_cons_lock"}
// CHECK:           %[[OF_IN_2_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "of_in_2_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[OF_IN_2_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "of_in_2_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[OF_IN_2_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_3_3]], 0) {init = 2 : i32, sym_name = "of_in_2_cons_prod_lock"}
// CHECK:           %[[OF_IN_2_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_3_3]], 1) {init = 0 : i32, sym_name = "of_in_2_cons_cons_lock"}
// CHECK:           %[[OF_IN_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_0]], 0) {init = 1 : i32, sym_name = "of_in_prod_lock"}
// CHECK:           %[[OF_IN_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_0]], 1) {init = 0 : i32, sym_name = "of_in_cons_lock"}
// CHECK:           aie.flow(%[[TILE_2_0]], DMA : 0, %[[TILE_3_3]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_2_0]], DMA : 0, %[[TILE_2_3]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_2_0]], DMA : 0, %[[TILE_2_2]], DMA : 0)
// CHECK:           %[[EXT_BUFFER_IN:.*]] = aie.external_buffer {sym_name = "ext_buffer_in"} : memref<64xi32>
// CHECK:           aie.shim_dma_allocation @of_in(MM2S, 0, 2)
// CHECK:           %[[SHIM_DMA_2_0:.*]] = aie.shim_dma(%[[TILE_2_0]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OF_IN_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[EXT_BUFFER_IN]] : memref<64xi32>, 0, 64)
// CHECK:             aie.use_lock(%[[OF_IN_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_2_2:.*]] = aie.mem(%[[TILE_2_2]]) {
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OF_IN_0_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_IN_0_CONS_BUFF_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OF_IN_0_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[OF_IN_0_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_IN_0_CONS_BUFF_1]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OF_IN_0_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_2_3:.*]] = aie.mem(%[[TILE_2_3]]) {
// CHECK:             %[[VAL_2:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OF_IN_1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_IN_1_CONS_BUFF_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OF_IN_1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[OF_IN_1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_IN_1_CONS_BUFF_1]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OF_IN_1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_3_3:.*]] = aie.mem(%[[TILE_3_3]]) {
// CHECK:             %[[VAL_3:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OF_IN_2_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_IN_2_CONS_BUFF_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OF_IN_2_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[OF_IN_2_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_IN_2_CONS_BUFF_1]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OF_IN_2_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @shim_broadcast {
   aie.device(npu1_4col) {
      %tile20 = aie.tile(2, 0)
      %tile22 = aie.tile(2, 2)
      %tile23 = aie.tile(2, 3)
      %tile33 = aie.tile(3, 3)
      aie.objectfifo @of_in (%tile20, {%tile22, %tile23, %tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
      %ext_buffer_in  = aie.external_buffer {sym_name = "ext_buffer_in"}: memref<64xi32>
      aie.objectfifo.register_external_buffers @of_in (%tile20, {%ext_buffer_in}) : (memref<64xi32>)
   }
}
