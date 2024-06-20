
// RUN: iree-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           memref.global "public" @of_out_1_cons : memref<64xi16>
// CHECK:           memref.global "public" @of_out_1 : memref<64xi16>
// CHECK:           memref.global "public" @of_in_1_cons : memref<64xi16>
// CHECK:           memref.global "public" @of_in_1 : memref<64xi16>
// CHECK:           memref.global "public" @of_out_0_cons : memref<64xi16>
// CHECK:           memref.global "public" @of_out_0 : memref<64xi16>
// CHECK:           memref.global "public" @of_in_0_cons : memref<64xi16>
// CHECK:           memref.global "public" @of_in_0 : memref<64xi16>
// CHECK:           %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK:           %[[OF_OUT_1_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_0]], 6) {init = 0 : i32, sym_name = "of_out_1_cons_prod_lock"}
// CHECK:           %[[OF_OUT_1_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_0]], 7) {init = 0 : i32, sym_name = "of_out_1_cons_cons_lock"}
// CHECK:           %[[OF_OUT_1_BUFF_0:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "of_out_1_buff_0"} : memref<64xi16>
// CHECK:           %[[OF_OUT_1_BUFF_1:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "of_out_1_buff_1"} : memref<64xi16>
// CHECK:           %[[OF_OUT_1_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_3]], 2) {init = 2 : i32, sym_name = "of_out_1_prod_lock"}
// CHECK:           %[[OF_OUT_1_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_3]], 3) {init = 0 : i32, sym_name = "of_out_1_cons_lock"}
// CHECK:           %[[OF_IN_1_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "of_in_1_cons_buff_0"} : memref<64xi16>
// CHECK:           %[[OF_IN_1_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "of_in_1_cons_buff_1"} : memref<64xi16>
// CHECK:           %[[OF_IN_1_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_3]], 0) {init = 2 : i32, sym_name = "of_in_1_cons_prod_lock"}
// CHECK:           %[[OF_IN_1_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_3]], 1) {init = 0 : i32, sym_name = "of_in_1_cons_cons_lock"}
// CHECK:           %[[OF_IN_1_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_0]], 4) {init = 0 : i32, sym_name = "of_in_1_prod_lock"}
// CHECK:           %[[OF_IN_1_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_0]], 5) {init = 0 : i32, sym_name = "of_in_1_cons_lock"}
// CHECK:           %[[OF_OUT_0_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_0]], 2) {init = 0 : i32, sym_name = "of_out_0_cons_prod_lock"}
// CHECK:           %[[OF_OUT_0_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_0]], 3) {init = 0 : i32, sym_name = "of_out_0_cons_cons_lock"}
// CHECK:           %[[OF_OUT_0_BUFF_0:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "of_out_0_buff_0"} : memref<64xi16>
// CHECK:           %[[OF_OUT_0_BUFF_1:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "of_out_0_buff_1"} : memref<64xi16>
// CHECK:           %[[OF_OUT_0_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 2) {init = 2 : i32, sym_name = "of_out_0_prod_lock"}
// CHECK:           %[[OF_OUT_0_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 3) {init = 0 : i32, sym_name = "of_out_0_cons_lock"}
// CHECK:           %[[OF_IN_0_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "of_in_0_cons_buff_0"} : memref<64xi16>
// CHECK:           %[[OF_IN_0_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "of_in_0_cons_buff_1"} : memref<64xi16>
// CHECK:           %[[OF_IN_0_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 0) {init = 2 : i32, sym_name = "of_in_0_cons_prod_lock"}
// CHECK:           %[[OF_IN_0_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 1) {init = 0 : i32, sym_name = "of_in_0_cons_cons_lock"}
// CHECK:           %[[OF_IN_0_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_0]], 0) {init = 0 : i32, sym_name = "of_in_0_prod_lock"}
// CHECK:           %[[OF_IN_0_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_0]], 1) {init = 0 : i32, sym_name = "of_in_0_cons_lock"}
// CHECK:           aie.flow(%[[TILE_2_0]], DMA : 0, %[[TILE_2_2]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_2_2]], DMA : 0, %[[TILE_2_0]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_2_0]], DMA : 1, %[[TILE_2_3]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_2_3]], DMA : 0, %[[TILE_2_0]], DMA : 1)
// CHECK:           aie.shim_dma_allocation @of_in_0(MM2S, 0, 2)
// CHECK:           aie.shim_dma_allocation @of_out_0(S2MM, 0, 2)
// CHECK:           aie.shim_dma_allocation @of_in_1(MM2S, 1, 2)
// CHECK:           %[[MEM_2_2:.*]] = aie.mem(%[[TILE_2_2]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OF_IN_0_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_IN_0_CONS_BUFF_0]] : memref<64xi16>, 0, 64)
// CHECK:             aie.use_lock(%[[OF_IN_0_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[OF_IN_0_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_IN_0_CONS_BUFF_1]] : memref<64xi16>, 0, 64)
// CHECK:             aie.use_lock(%[[OF_IN_0_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[OF_OUT_0_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_OUT_0_BUFF_0]] : memref<64xi16>, 0, 64)
// CHECK:             aie.use_lock(%[[OF_OUT_0_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[OF_OUT_0_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_OUT_0_BUFF_1]] : memref<64xi16>, 0, 64)
// CHECK:             aie.use_lock(%[[OF_OUT_0_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @of_out_1(S2MM, 1, 2)
// CHECK:           %[[MEM_2_3:.*]] = aie.mem(%[[TILE_2_3]]) {
// CHECK:             %[[VAL_2:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OF_IN_1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_IN_1_CONS_BUFF_0]] : memref<64xi16>, 0, 64)
// CHECK:             aie.use_lock(%[[OF_IN_1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[OF_IN_1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_IN_1_CONS_BUFF_1]] : memref<64xi16>, 0, 64)
// CHECK:             aie.use_lock(%[[OF_IN_1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_3:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[OF_OUT_1_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_OUT_1_BUFF_0]] : memref<64xi16>, 0, 64)
// CHECK:             aie.use_lock(%[[OF_OUT_1_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[OF_OUT_1_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_OUT_1_BUFF_1]] : memref<64xi16>, 0, 64)
// CHECK:             aie.use_lock(%[[OF_OUT_1_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @alloc {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile22 = aie.tile(2, 2)
        %tile23 = aie.tile(2, 3)
        aie.objectfifo @of_in_0 (%tile20, {%tile22}, 2 : i32) : !aie.objectfifo<memref<64xi16>>
        aie.objectfifo @of_out_0 (%tile22, {%tile20}, 2 : i32) : !aie.objectfifo<memref<64xi16>>
        aie.objectfifo @of_in_1 (%tile20, {%tile23}, 2 : i32) : !aie.objectfifo<memref<64xi16>>
        aie.objectfifo @of_out_1 (%tile23, {%tile20}, 2 : i32) : !aie.objectfifo<memref<64xi16>>
    }
}
