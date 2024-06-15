
// RUN: iree-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           memref.global "public" @of_stream_cons : memref<16xi32>
// CHECK:           memref.global "public" @of_stream : memref<16xi32>
// CHECK:           memref.global "public" @of_shared : memref<16xi32>
// CHECK:           %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:           %[[TILE_1_3:.*]] = aie.tile(1, 3)
// CHECK:           %[[OF_STREAM_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_1_3]]) {sym_name = "of_stream_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[OF_STREAM_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_1_3]]) {sym_name = "of_stream_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[OF_STREAM_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_1_3]], 0) {init = 2 : i32, sym_name = "of_stream_cons_prod_lock"}
// CHECK:           %[[OF_STREAM_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_1_3]], 1) {init = 0 : i32, sym_name = "of_stream_cons_cons_lock"}
// CHECK:           %[[OF_STREAM_BUFF_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of_stream_buff_0"} : memref<16xi32>
// CHECK:           %[[OF_STREAM_BUFF_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of_stream_buff_1"} : memref<16xi32>
// CHECK:           %[[OF_STREAM_PROD_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 2) {init = 2 : i32, sym_name = "of_stream_prod_lock"}
// CHECK:           %[[OF_STREAM_CONS_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 3) {init = 0 : i32, sym_name = "of_stream_cons_lock"}
// CHECK:           %[[OF_SHARED_BUFF_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of_shared_buff_0"} : memref<16xi32>
// CHECK:           %[[OF_SHARED_BUFF_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of_shared_buff_1"} : memref<16xi32>
// CHECK:           %[[OF_SHARED_PROD_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 0) {init = 2 : i32, sym_name = "of_shared_prod_lock"}
// CHECK:           %[[OF_SHARED_CONS_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 1) {init = 0 : i32, sym_name = "of_shared_cons_lock"}
// CHECK:           aie.flow(%[[TILE_1_2]], DMA : 0, %[[TILE_1_3]], DMA : 0)
// CHECK:           %[[MEM_1_2:.*]] = aie.mem(%[[TILE_1_2]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OF_STREAM_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_STREAM_BUFF_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OF_STREAM_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[OF_STREAM_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_STREAM_BUFF_1]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OF_STREAM_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_1_3:.*]] = aie.mem(%[[TILE_1_3]]) {
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OF_STREAM_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_STREAM_CONS_BUFF_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OF_STREAM_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[OF_STREAM_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_STREAM_CONS_BUFF_1]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OF_STREAM_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @viaDMA {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    aie.objectfifo @of_shared (%tile12, {%tile13}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_stream (%tile12, {%tile13}, 2 : i32) {via_DMA = true} : !aie.objectfifo<memref<16xi32>>
 }
}
