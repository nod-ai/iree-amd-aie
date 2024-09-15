
// RUN: iree-opt --amdaie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           memref.global "public" @skip_connection : memref<16xi32>
// CHECK:           memref.global "public" @link2 : memref<16xi32>
// CHECK:           memref.global "public" @link1 : memref<48xi32>
// CHECK-DAG:       %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK-DAG:       %[[TILE_2_1:.*]] = aie.tile(2, 1)
// CHECK-DAG:       %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK-DAG:       %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK-DAG:       %[[BUFFER_2_2:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "skip_connection_prod_buff_0_0"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_2_2_0:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "skip_connection_prod_buff_0_1"} : memref<16xi32>
// CHECK-DAG:       %[[LOCK_2_2:.*]] = aie.lock(%[[TILE_2_2]]) {init = 2 : i8, sym_name = "skip_connection_prod_prod_lock_0"}
// CHECK-DAG:       %[[LOCK_2_2_1:.*]] = aie.lock(%[[TILE_2_2]]) {init = 0 : i8, sym_name = "skip_connection_prod_cons_lock_0"}
// CHECK-DAG:       %[[BUFFER_3_3:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "skip_connection_cons_buff_0_0"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_3_3_2:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "skip_connection_cons_buff_0_1"} : memref<16xi32>
// CHECK-DAG:       %[[LOCK_3_3:.*]] = aie.lock(%[[TILE_3_3]]) {init = 2 : i8, sym_name = "skip_connection_cons_prod_lock_0"}
// CHECK-DAG:       %[[LOCK_3_3_3:.*]] = aie.lock(%[[TILE_3_3]]) {init = 0 : i8, sym_name = "skip_connection_cons_cons_lock_0"}
// CHECK-DAG:       %[[BUFFER_2_1:.*]] = aie.buffer(%[[TILE_2_1]]) {sym_name = "link1_link_buff_0_0"} : memref<48xi32>
// CHECK-DAG:       %[[BUFFER_2_1_4:.*]] = aie.buffer(%[[TILE_2_1]]) {sym_name = "link1_link_buff_0_1"} : memref<48xi32>
// CHECK-DAG:       %[[LOCK_2_1:.*]] = aie.lock(%[[TILE_2_1]]) {init = 2 : i8, sym_name = "link1_link_prod_lock_0"}
// CHECK-DAG:       %[[LOCK_2_1_5:.*]] = aie.lock(%[[TILE_2_1]]) {init = 0 : i8, sym_name = "link1_link_cons_lock_0"}
// CHECK-DAG:       %[[LOCK_2_0:.*]] = aie.lock(%[[TILE_2_0]]) {init = 0 : i8, sym_name = "link1_prod_prod_lock_0"}
// CHECK-DAG:       %[[LOCK_2_0_6:.*]] = aie.lock(%[[TILE_2_0]]) {init = 0 : i8, sym_name = "link1_prod_cons_lock_0"}
// CHECK-DAG:       %[[BUFFER_2_2_7:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "link2_cons_buff_0_0"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_2_2_8:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "link2_cons_buff_0_1"} : memref<16xi32>
// CHECK-DAG:       %[[LOCK_2_2_9:.*]] = aie.lock(%[[TILE_2_2]]) {init = 2 : i8, sym_name = "link2_cons_prod_lock_0"}
// CHECK-DAG:       %[[LOCK_2_2_10:.*]] = aie.lock(%[[TILE_2_2]]) {init = 0 : i8, sym_name = "link2_cons_cons_lock_0"}
// CHECK-DAG:       %[[BUFFER_3_3_11:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "link2_cons_buff_1_0"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_3_3_12:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "link2_cons_buff_1_1"} : memref<16xi32>
// CHECK-DAG:       %[[LOCK_3_3_13:.*]] = aie.lock(%[[TILE_3_3]]) {init = 2 : i8, sym_name = "link2_cons_prod_lock_1"}
// CHECK-DAG:       %[[LOCK_3_3_14:.*]] = aie.lock(%[[TILE_3_3]]) {init = 0 : i8, sym_name = "link2_cons_cons_lock_1"}
// CHECK-DAG:       aie.flow(%[[TILE_2_0]], DMA : 0, %[[TILE_2_1]], DMA : 0) {symbol = @link1}
// CHECK-DAG:       aie.flow(%[[TILE_2_1]], DMA : 0, %[[TILE_3_3]], DMA : 0) {symbol = @link2}
// CHECK-DAG:       aie.flow(%[[TILE_2_1]], DMA : 0, %[[TILE_2_2]], DMA : 0) {symbol = @link2}
// CHECK-DAG:       aie.flow(%[[TILE_2_2]], DMA : 0, %[[TILE_3_3]], DMA : 1) {symbol = @skip_connection}
// CHECK-DAG:       aie.shim_dma_allocation @link1(MM2S, 0, 2)
// CHECK:           %[[MEMTILE_DMA_2_1:.*]] = aie.memtile_dma(%[[TILE_2_1]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_2_1]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_2_1]] : memref<48xi32>) {len = 48 : i32}
// CHECK:             aie.use_lock(%[[LOCK_2_1_5]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_2_1]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_2_1_4]] : memref<48xi32>) {len = 48 : i32}
// CHECK:             aie.use_lock(%[[LOCK_2_1_5]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LOCK_2_1_5]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_2_1]] : memref<48xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_2_1]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[LOCK_2_1_5]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_2_1_4]] : memref<48xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_2_1]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_2_2:.*]] = aie.mem(%[[TILE_2_2]]) {
// CHECK:             %[[VAL_2:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_2_2_9]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_2_2_7]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_2_2_10]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_2_2_9]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_2_2_8]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_2_2_10]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_3:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LOCK_2_2_1]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_2_2]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_2_2]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[LOCK_2_2_1]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_2_2_0]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_2_2]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_3_3:.*]] = aie.mem(%[[TILE_3_3]]) {
// CHECK:             %[[VAL_4:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_3_3_13]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_3_3_11]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_3_3_14]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_3_3_13]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_3_3_12]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_3_3_14]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_5:.*]] = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LOCK_3_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_3_3]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_3_3_3]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[LOCK_3_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_3_3_2]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_3_3_3]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }
module @link_broadcast {
  aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile21 = aie.tile(2, 1)
    %tile22 = aie.tile(2, 2)
    %tile33 = aie.tile(3, 3)
    aie.flow(%tile20, DMA : 0, %tile21, DMA : 0) {symbol = @link1}
    aie.flow(%tile21, DMA : 0, %tile33, DMA : 0) {symbol = @link2}
    aie.flow(%tile21, DMA : 0, %tile22, DMA : 0) {symbol = @link2}
    aie.flow(%tile22, DMA : 0, %tile33, DMA : 1) {symbol = @skip_connection}
    aie.objectfifo @link1 (%tile20, {%tile21}, 2 : i32) : !aie.objectfifo<memref<48xi32>>
    aie.objectfifo @link2 (%tile21, {%tile22, %tile33}, [2]) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @skip_connection (%tile22, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@link1] -> [@link2] ([] [])
  }
}
