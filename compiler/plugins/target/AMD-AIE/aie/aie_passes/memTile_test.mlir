
// RUN: iree-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @of_cons : memref<16xi32>
// CHECK:           memref.global "public" @of : memref<16xi32>
// CHECK:           %[[TILE_2_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[OF_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "of_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[OF_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "of_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[OF_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 0) {init = 2 : i32, sym_name = "of_cons_prod_lock"}
// CHECK:           %[[OF_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 1) {init = 0 : i32, sym_name = "of_cons_cons_lock"}
// CHECK:           %[[OF_BUFF_0:.*]] = aie.buffer(%[[TILE_2_1]]) {sym_name = "of_buff_0"} : memref<16xi32>
// CHECK:           %[[OF_BUFF_1:.*]] = aie.buffer(%[[TILE_2_1]]) {sym_name = "of_buff_1"} : memref<16xi32>
// CHECK:           %[[OF_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_1]], 0) {init = 2 : i32, sym_name = "of_prod_lock"}
// CHECK:           %[[OF_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_1]], 1) {init = 0 : i32, sym_name = "of_cons_lock"}
// CHECK:           aie.flow(%[[TILE_2_1]], DMA : 0, %[[TILE_2_2]], DMA : 0)
// CHECK:           %[[MEMTILE_DMA_2_1:.*]] = aie.memtile_dma(%[[TILE_2_1]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OF_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_BUFF_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OF_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[OF_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_BUFF_1]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OF_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_2_2:.*]] = aie.mem(%[[TILE_2_2]]) {
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OF_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_CONS_BUFF_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OF_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[OF_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_CONS_BUFF_1]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OF_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @memTile {
   aie.device(npu1_4col) {
      %tile11 = aie.tile(2, 1)
      %tile12 = aie.tile(2, 2)
      aie.objectfifo @of (%tile11, {%tile12}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
   }
}
