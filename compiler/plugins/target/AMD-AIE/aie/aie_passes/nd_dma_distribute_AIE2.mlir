
// RUN: iree-opt --aie-objectFifo-stateful-transform %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           memref.global "public" @of2_cons : memref<128xi32>
// CHECK:           memref.global "public" @of2 : memref<128xi32>
// CHECK:           memref.global "public" @of1_cons : memref<128xi32>
// CHECK:           memref.global "public" @of1 : memref<128xi32>
// CHECK:           memref.global "public" @of0_cons : memref<256xi32>
// CHECK:           memref.global "public" @of0 : memref<256xi32>
// CHECK:           %[[TILE_1_0:.*]] = aie.tile(1, 0)
// CHECK:           %[[TILE_1_1:.*]] = aie.tile(1, 1)
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK:           %[[OF2_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "of2_cons_buff_0"} : memref<128xi32>
// CHECK:           %[[OF2_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "of2_cons_buff_1"} : memref<128xi32>
// CHECK:           %[[OF2_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_3]], 0) {init = 2 : i32, sym_name = "of2_cons_prod_lock"}
// CHECK:           %[[OF2_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_3]], 1) {init = 0 : i32, sym_name = "of2_cons_cons_lock"}
// CHECK:           %[[OF1_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "of1_cons_buff_0"} : memref<128xi32>
// CHECK:           %[[OF1_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "of1_cons_buff_1"} : memref<128xi32>
// CHECK:           %[[OF1_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 0) {init = 2 : i32, sym_name = "of1_cons_prod_lock"}
// CHECK:           %[[OF1_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 1) {init = 0 : i32, sym_name = "of1_cons_cons_lock"}
// CHECK:           %[[OF0_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_1_1]]) {sym_name = "of0_cons_buff_0"} : memref<256xi32>
// CHECK:           %[[OF0_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_1_1]]) {sym_name = "of0_cons_buff_1"} : memref<256xi32>
// CHECK:           %[[OF0_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_1_1]], 0) {init = 4 : i32, sym_name = "of0_cons_prod_lock"}
// CHECK:           %[[OF0_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_1_1]], 1) {init = 0 : i32, sym_name = "of0_cons_cons_lock"}
// CHECK:           %[[OF0_PROD_LOCK:.*]] = aie.lock(%[[TILE_1_0]], 0) {init = 0 : i32, sym_name = "of0_prod_lock"}
// CHECK:           %[[OF0_CONS_LOCK:.*]] = aie.lock(%[[TILE_1_0]], 1) {init = 0 : i32, sym_name = "of0_cons_lock"}
// CHECK:           aie.flow(%[[TILE_1_0]], DMA : 0, %[[TILE_1_1]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_1_1]], DMA : 0, %[[TILE_2_2]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_1_1]], DMA : 1, %[[TILE_2_3]], DMA : 0)
// CHECK:           aie.shim_dma_allocation @of0(MM2S, 0, 1)
// CHECK:           %[[MEMTILE_DMA_1_1:.*]] = aie.memtile_dma(%[[TILE_1_1]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OF0_CONS_PROD_LOCK]], AcquireGreaterEqual, 2)
// CHECK:             aie.dma_bd(%[[OF0_CONS_BUFF_0]] : memref<256xi32>, 0, 256)
// CHECK:             aie.use_lock(%[[OF0_CONS_CONS_LOCK]], Release, 2)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[OF0_CONS_PROD_LOCK]], AcquireGreaterEqual, 2)
// CHECK:             aie.dma_bd(%[[OF0_CONS_BUFF_1]] : memref<256xi32>, 0, 256)
// CHECK:             aie.use_lock(%[[OF0_CONS_CONS_LOCK]], Release, 2)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[OF0_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF0_CONS_BUFF_0]] : memref<256xi32>, 0, 128, [<size = 4, stride = 64>, <size = 2, stride = 4>, <size = 8, stride = 8>, <size = 4, stride = 1>])
// CHECK:             aie.use_lock(%[[OF0_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[OF0_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF0_CONS_BUFF_1]] : memref<256xi32>, 0, 128, [<size = 4, stride = 64>, <size = 2, stride = 4>, <size = 8, stride = 8>, <size = 4, stride = 1>])
// CHECK:             aie.use_lock(%[[OF0_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             %[[VAL_2:.*]] = aie.dma_start(MM2S, 1, ^bb7, ^bb9)
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[OF0_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF0_CONS_BUFF_0]] : memref<256xi32>, 128, 128, [<size = 4, stride = 64>, <size = 2, stride = 4>, <size = 8, stride = 8>, <size = 4, stride = 1>])
// CHECK:             aie.use_lock(%[[OF0_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(%[[OF0_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF0_CONS_BUFF_1]] : memref<256xi32>, 128, 128, [<size = 4, stride = 64>, <size = 2, stride = 4>, <size = 8, stride = 8>, <size = 4, stride = 1>])
// CHECK:             aie.use_lock(%[[OF0_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb9:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_2_2:.*]] = aie.mem(%[[TILE_2_2]]) {
// CHECK:             %[[VAL_3:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OF1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF1_CONS_BUFF_0]] : memref<128xi32>, 0, 128)
// CHECK:             aie.use_lock(%[[OF1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[OF1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF1_CONS_BUFF_1]] : memref<128xi32>, 0, 128)
// CHECK:             aie.use_lock(%[[OF1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_2_3:.*]] = aie.mem(%[[TILE_2_3]]) {
// CHECK:             %[[VAL_4:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OF2_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF2_CONS_BUFF_0]] : memref<128xi32>, 0, 128)
// CHECK:             aie.use_lock(%[[OF2_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[OF2_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF2_CONS_BUFF_1]] : memref<128xi32>, 0, 128)
// CHECK:             aie.use_lock(%[[OF2_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @ndDMAObjFifoAIE2 {
 aie.device(xcve2302) {
    %tile10 = aie.tile(1, 0)
    %tile11 = aie.tile(1, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)
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
   // expected-error@+1 {{'aie.objectfifo.link' op currently does not support objectFifos with dimensionsFromStreamPerConsumer.}}
   aie.objectfifo.link [ @of0 ] -> [ @of1, @of2 ] ()
 }
}
