
// RUN: iree-opt --amdaie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           memref.global "public" @of2 : memref<128xi32>
// CHECK:           memref.global "public" @of1 : memref<128xi32>
// CHECK:           memref.global "public" @of0 : memref<256xi32>
// CHECK-DAG:       %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK-DAG:       %[[TILE_2_1:.*]] = aie.tile(2, 1)
// CHECK-DAG:       %[[TILE_3_2:.*]] = aie.tile(3, 2)
// CHECK-DAG:       %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK-DAG:       %[[OF2_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "of2_cons_buff_1_0"} : memref<128xi32>
// CHECK-DAG:       %[[OF2_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "of2_cons_buff_1_1"} : memref<128xi32>
// CHECK-DAG:       %[[OF2_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_3_3]]) {init = 2 : i8, sym_name = "of2_cons_prod_lock_1"}
// CHECK-DAG:       %[[OF2_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_3_3]]) {init = 0 : i8, sym_name = "of2_cons_cons_lock_1"}
// CHECK-DAG:       %[[OF1_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_3_2]]) {sym_name = "of1_cons_buff_0_0"} : memref<128xi32>
// CHECK-DAG:       %[[OF1_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_3_2]]) {sym_name = "of1_cons_buff_0_1"} : memref<128xi32>
// CHECK-DAG:       %[[OF1_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_3_2]]) {init = 2 : i8, sym_name = "of1_cons_prod_lock_0"}
// CHECK-DAG:       %[[OF1_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_3_2]]) {init = 0 : i8, sym_name = "of1_cons_cons_lock_0"}
// CHECK-DAG:       %[[OF0_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_2_1]]) {sym_name = "of0_link_buff_0_0"} : memref<256xi32>
// CHECK-DAG:       %[[OF0_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_2_1]]) {sym_name = "of0_link_buff_0_1"} : memref<256xi32>
// CHECK-DAG:       %[[OF0_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_1]]) {init = 4 : i8, sym_name = "of0_link_prod_lock_0"}
// CHECK-DAG:       %[[OF0_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_1]]) {init = 0 : i8, sym_name = "of0_link_cons_lock_0"}
// CHECK-DAG:       %[[OF0_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_0]]) {init = 0 : i8, sym_name = "of0_prod_prod_lock_0"}
// CHECK-DAG:       %[[OF0_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_0]]) {init = 0 : i8, sym_name = "of0_prod_cons_lock_0"}
// CHECK-DAG:       aie.flow(%[[TILE_2_0]], DMA : 0, %[[TILE_2_1]], DMA : 0)
// CHECK-DAG:       aie.flow(%[[TILE_2_1]], DMA : 0, %[[TILE_3_2]], DMA : 0)
// CHECK-DAG:       aie.flow(%[[TILE_2_1]], DMA : 1, %[[TILE_3_3]], DMA : 0)
// CHECK-DAG        aie.shim_dma_allocation @of0(MM2S, 0, 2)
// CHECK:           %[[MEMTILE_DMA_1_1:.*]] = aie.memtile_dma(%[[TILE_2_1]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OF0_CONS_PROD_LOCK]], AcquireGreaterEqual, 2)
// CHECK:             aie.dma_bd(%[[OF0_CONS_BUFF_0]] : memref<256xi32>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[OF0_CONS_CONS_LOCK]], Release, 2)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[OF0_CONS_PROD_LOCK]], AcquireGreaterEqual, 2)
// CHECK:             aie.dma_bd(%[[OF0_CONS_BUFF_1]] : memref<256xi32>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[OF0_CONS_CONS_LOCK]], Release, 2)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[OF0_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF0_CONS_BUFF_0]] : memref<256xi32>) {dimensions = #aie<bd_dim_layout_array[<size = 4, stride = 64>, <size = 2, stride = 4>, <size = 8, stride = 8>, <size = 4, stride = 1>]>, len = 128 : i32}
// CHECK:             aie.use_lock(%[[OF0_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[OF0_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF0_CONS_BUFF_1]] : memref<256xi32>) {dimensions = #aie<bd_dim_layout_array[<size = 4, stride = 64>, <size = 2, stride = 4>, <size = 8, stride = 8>, <size = 4, stride = 1>]>, len = 128 : i32}
// CHECK:             aie.use_lock(%[[OF0_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             %[[VAL_2:.*]] = aie.dma_start(MM2S, 1, ^bb7, ^bb9)
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[OF0_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF0_CONS_BUFF_0]] : memref<256xi32>) {dimensions = #aie<bd_dim_layout_array[<size = 4, stride = 64>, <size = 2, stride = 4>, <size = 8, stride = 8>, <size = 4, stride = 1>]>, len = 128 : i32, offset = 128 : i32}
// CHECK:             aie.use_lock(%[[OF0_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(%[[OF0_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF0_CONS_BUFF_1]] : memref<256xi32>) {dimensions = #aie<bd_dim_layout_array[<size = 4, stride = 64>, <size = 2, stride = 4>, <size = 8, stride = 8>, <size = 4, stride = 1>]>, len = 128 : i32, offset = 128 : i32}
// CHECK:             aie.use_lock(%[[OF0_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb9:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_2_2:.*]] = aie.mem(%[[TILE_3_2]]) {
// CHECK:             %[[VAL_3:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OF1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF1_CONS_BUFF_0]] : memref<128xi32>) {len = 128 : i32}
// CHECK:             aie.use_lock(%[[OF1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[OF1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF1_CONS_BUFF_1]] : memref<128xi32>) {len = 128 : i32}
// CHECK:             aie.use_lock(%[[OF1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_2_3:.*]] = aie.mem(%[[TILE_3_3]]) {
// CHECK:             %[[VAL_4:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OF2_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF2_CONS_BUFF_0]] : memref<128xi32>) {len = 128 : i32}
// CHECK:             aie.use_lock(%[[OF2_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[OF2_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF2_CONS_BUFF_1]] : memref<128xi32>) {len = 128 : i32}
// CHECK:             aie.use_lock(%[[OF2_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }
module @ndDMAObjFifoAIE2 {
 aie.device(xcve2302) {
    %tile10 = aie.tile(2, 0)
    %tile11 = aie.tile(2, 1)
    %tile22 = aie.tile(3, 2)
    %tile23 = aie.tile(3, 3)
    aie.flow(%tile10, DMA : 0, %tile11, DMA : 0) {symbol = @of0}
    aie.flow(%tile11, DMA : 0, %tile22, DMA : 0) {symbol = @of1}
    aie.flow(%tile11, DMA : 1, %tile23, DMA : 0) {symbol = @of2}
    aie.objectfifo @of0 (%tile10, {%tile11},
                         2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @of1 (%tile11 toStream [<size = 4, stride = 64>,
                                           <size = 2, stride = 4>,
                                           <size = 8, stride = 8>,
                                           <size = 4, stride = 1>],
                        {%tile22}, 2 : i32) : !aie.objectfifo<memref<128xi32>>
    aie.objectfifo @of2 (%tile11 toStream [<size = 4, stride = 64>,
                                           <size = 2, stride = 4>,
                                           <size = 8, stride = 8>,
                                           <size = 4, stride = 1>],
                        {%tile23}, 2 : i32) : !aie.objectfifo<memref<128xi32>>
   aie.objectfifo.link [ @of0 ] -> [ @of1, @of2 ] ([] [])
 }
}
