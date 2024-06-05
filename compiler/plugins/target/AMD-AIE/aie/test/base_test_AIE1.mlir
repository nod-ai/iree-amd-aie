
// RUN: iree-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @of1_cons : memref<16xi32>
// CHECK:           memref.global "public" @of1 : memref<16xi32>
// CHECK:           memref.global "public" @of0 : memref<16xi32>
// CHECK:           %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:           %[[TILE_1_3:.*]] = aie.tile(1, 3)
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[OF1_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "of1_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[OF1_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "of1_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[OF1_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_3_3]], 0) {init = 2 : i32, sym_name = "of1_cons_prod_lock"}
// CHECK:           %[[OF1_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_3_3]], 1) {init = 0 : i32, sym_name = "of1_cons_cons_lock"}
// CHECK:           %[[OF1_BUFF_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of1_buff_0"} : memref<16xi32>
// CHECK:           %[[OF1_BUFF_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of1_buff_1"} : memref<16xi32>
// CHECK:           %[[OF1_PROD_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 2) {init = 2 : i32, sym_name = "of1_prod_lock"}
// CHECK:           %[[OF1_CONS_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 3) {init = 0 : i32, sym_name = "of1_cons_lock"}
// CHECK:           %[[OF0_BUFF_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of0_buff_0"} : memref<16xi32>
// CHECK:           %[[OF0_BUFF_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of0_buff_1"} : memref<16xi32>
// CHECK:           %[[OF0_BUFF_2:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of0_buff_2"} : memref<16xi32>
// CHECK:           %[[OF0_BUFF_3:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of0_buff_3"} : memref<16xi32>
// CHECK:           %[[OF0_PROD_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 0) {init = 4 : i32, sym_name = "of0_prod_lock"}
// CHECK:           %[[OF0_CONS_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 1) {init = 0 : i32, sym_name = "of0_cons_lock"}
// CHECK:           aie.flow(%[[TILE_1_2]], DMA : 0, %[[TILE_3_3]], DMA : 0)
// CHECK:           %[[MEM_1_2:.*]] = aie.mem(%[[TILE_1_2]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OF1_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF1_BUFF_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OF1_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[OF1_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF1_BUFF_1]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OF1_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_3_3:.*]] = aie.mem(%[[TILE_3_3]]) {
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OF1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF1_CONS_BUFF_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OF1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[OF1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF1_CONS_BUFF_1]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OF1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @elementGenerationAIE1 {
   aie.device(npu1_4col) {
      %tile12 = aie.tile(1, 2)
      %tile13 = aie.tile(1, 3)
      %tile33 = aie.tile(3, 3)
      // In the shared memory case, the number of elements does not change.
      aie.objectfifo @of0 (%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
      // In the non-adjacent memory case, the number of elements depends on the max amount acquired by
      // the processes running on each core (here nothing is specified so it cannot be derived).
      aie.objectfifo @of1 (%tile12, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
   }
}
