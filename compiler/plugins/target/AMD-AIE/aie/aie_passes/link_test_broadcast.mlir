
// RUN: iree-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @skip_connection_cons : memref<16xi32>
// CHECK:           memref.global "public" @skip_connection : memref<16xi32>
// CHECK:           memref.global "public" @link2_0_cons : memref<16xi32>
// CHECK:           memref.global "public" @link2_1_cons : memref<16xi32>
// CHECK:           memref.global "public" @link2 : memref<16xi32>
// CHECK:           memref.global "public" @link1_cons : memref<48xi32>
// CHECK:           memref.global "public" @link1 : memref<48xi32>
// CHECK:           %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[TILE_2_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[SKIP_CONNECTION_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "skip_connection_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[SKIP_CONNECTION_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "skip_connection_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[SKIP_CONNECTION_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_3_3]], 2) {init = 2 : i32, sym_name = "skip_connection_cons_prod_lock"}
// CHECK:           %[[SKIP_CONNECTION_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_3_3]], 3) {init = 0 : i32, sym_name = "skip_connection_cons_cons_lock"}
// CHECK:           %[[SKIP_CONNECTION_BUFF_0:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "skip_connection_buff_0"} : memref<16xi32>
// CHECK:           %[[SKIP_CONNECTION_BUFF_1:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "skip_connection_buff_1"} : memref<16xi32>
// CHECK:           %[[SKIP_CONNECTION_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 2) {init = 2 : i32, sym_name = "skip_connection_prod_lock"}
// CHECK:           %[[SKIP_CONNECTION_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 3) {init = 0 : i32, sym_name = "skip_connection_cons_lock"}
// CHECK:           %[[LINK2_0_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "link2_0_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[LINK2_0_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "link2_0_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[LINK2_0_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 0) {init = 2 : i32, sym_name = "link2_0_cons_prod_lock"}
// CHECK:           %[[LINK2_0_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 1) {init = 0 : i32, sym_name = "link2_0_cons_cons_lock"}
// CHECK:           %[[LINK2_1_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "link2_1_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[LINK2_1_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "link2_1_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[LINK2_1_CONS_BUFF_2:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "link2_1_cons_buff_2"} : memref<16xi32>
// CHECK:           %[[LINK2_1_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_3_3]], 0) {init = 3 : i32, sym_name = "link2_1_cons_prod_lock"}
// CHECK:           %[[LINK2_1_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_3_3]], 1) {init = 0 : i32, sym_name = "link2_1_cons_cons_lock"}
// CHECK:           %[[LINK1_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_2_1]]) {sym_name = "link1_cons_buff_0"} : memref<48xi32>
// CHECK:           %[[LINK1_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_2_1]]) {sym_name = "link1_cons_buff_1"} : memref<48xi32>
// CHECK:           %[[LINK1_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_1]], 0) {init = 2 : i32, sym_name = "link1_cons_prod_lock"}
// CHECK:           %[[LINK1_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_1]], 1) {init = 0 : i32, sym_name = "link1_cons_cons_lock"}
// CHECK:           %[[LINK1_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_0]], 0) {init = 0 : i32, sym_name = "link1_prod_lock"}
// CHECK:           %[[LINK1_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_0]], 1) {init = 0 : i32, sym_name = "link1_cons_lock"}
// CHECK:           aie.flow(%[[TILE_2_0]], DMA : 0, %[[TILE_2_1]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_2_1]], DMA : 0, %[[TILE_3_3]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_2_1]], DMA : 0, %[[TILE_2_2]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_2_2]], DMA : 0, %[[TILE_3_3]], DMA : 1)
// CHECK:           aie.shim_dma_allocation @link1(MM2S, 0, 2)
// CHECK:           %[[MEMTILE_DMA_2_1:.*]] = aie.memtile_dma(%[[TILE_2_1]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LINK1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK1_CONS_BUFF_0]] : memref<48xi32>, 0, 48)
// CHECK:             aie.use_lock(%[[LINK1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LINK1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK1_CONS_BUFF_1]] : memref<48xi32>, 0, 48)
// CHECK:             aie.use_lock(%[[LINK1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LINK1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK1_CONS_BUFF_0]] : memref<48xi32>, 0, 48)
// CHECK:             aie.use_lock(%[[LINK1_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[LINK1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK1_CONS_BUFF_1]] : memref<48xi32>, 0, 48)
// CHECK:             aie.use_lock(%[[LINK1_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_2_2:.*]] = aie.mem(%[[TILE_2_2]]) {
// CHECK:             %[[VAL_2:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LINK2_0_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK2_0_CONS_BUFF_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[LINK2_0_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LINK2_0_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK2_0_CONS_BUFF_1]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[LINK2_0_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_3:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[SKIP_CONNECTION_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[SKIP_CONNECTION_BUFF_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[SKIP_CONNECTION_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[SKIP_CONNECTION_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[SKIP_CONNECTION_BUFF_1]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[SKIP_CONNECTION_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_3_3:.*]] = aie.mem(%[[TILE_3_3]]) {
// CHECK:             %[[VAL_4:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LINK2_1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK2_1_CONS_BUFF_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[LINK2_1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LINK2_1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK2_1_CONS_BUFF_1]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[LINK2_1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LINK2_1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK2_1_CONS_BUFF_2]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[LINK2_1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             %[[VAL_5:.*]] = aie.dma_start(S2MM, 1, ^bb5, ^bb7)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[SKIP_CONNECTION_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[SKIP_CONNECTION_CONS_BUFF_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[SKIP_CONNECTION_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[SKIP_CONNECTION_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[SKIP_CONNECTION_CONS_BUFF_1]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[SKIP_CONNECTION_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb7:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @link_broadcast {
    aie.device(npu1_4col) {
        %tile20 = aie.tile(2, 0)
        %tile21 = aie.tile(2, 1)
        %tile22 = aie.tile(2, 2)
        %tile33 = aie.tile(3, 3)
        aie.objectfifo @link1 (%tile20, {%tile21}, 2 : i32) : !aie.objectfifo<memref<48xi32>>
        aie.objectfifo @link2 (%tile21, {%tile22, %tile33}, [2, 2, 3]) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @skip_connection (%tile22, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo.link [@link1] -> [@link2] ()
    }
}
