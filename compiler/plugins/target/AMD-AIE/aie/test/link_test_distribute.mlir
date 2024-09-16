
// RUN: iree-opt --amdaie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           memref.global "public" @link4 : memref<12xi32>
// CHECK:           memref.global "public" @link3 : memref<20xi32>
// CHECK:           memref.global "public" @link2 : memref<4x4xi32>
// CHECK:           memref.global "public" @link1 : memref<48xi32>
// CHECK-DAG:       %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK-DAG:       %[[TILE_2_1:.*]] = aie.tile(2, 1)
// CHECK-DAG:       %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK-DAG:       %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK-DAG:       %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK-DAG:       %[[LINK4_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "link4_cons_buff_2_0"} : memref<12xi32>
// CHECK-DAG:       %[[LINK4_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "link4_cons_buff_2_1"} : memref<12xi32>
// CHECK-DAG:       %[[LINK4_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_3_3]]) {init = 2 : i8, sym_name = "link4_cons_prod_lock_2"}
// CHECK-DAG:       %[[LINK4_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_3_3]]) {init = 0 : i8, sym_name = "link4_cons_cons_lock_2"}
// CHECK-DAG:       %[[LINK3_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "link3_cons_buff_1_0"} : memref<20xi32>
// CHECK-DAG:       %[[LINK3_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "link3_cons_buff_1_1"} : memref<20xi32>
// CHECK-DAG:       %[[LINK3_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_3]]) {init = 2 : i8, sym_name = "link3_cons_prod_lock_1"}
// CHECK-DAG:       %[[LINK3_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_3]]) {init = 0 : i8, sym_name = "link3_cons_cons_lock_1"}
// CHECK-DAG:       %[[LINK2_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "link2_cons_buff_0_0"} : memref<4x4xi32>
// CHECK-DAG:       %[[LINK2_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "link2_cons_buff_0_1"} : memref<4x4xi32>
// CHECK-DAG:       %[[LINK2_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_2]]) {init = 2 : i8, sym_name = "link2_cons_prod_lock_0"}
// CHECK-DAG:       %[[LINK2_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_2]]) {init = 0 : i8, sym_name = "link2_cons_cons_lock_0"}
// CHECK-DAG:       %[[LINK1_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_2_1]]) {sym_name = "link1_link_buff_0_0"} : memref<48xi32>
// CHECK-DAG:       %[[LINK1_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_2_1]]) {sym_name = "link1_link_buff_0_1"} : memref<48xi32>
// CHECK-DAG:       %[[LINK1_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_1]]) {init = 6 : i8, sym_name = "link1_link_prod_lock_0"}
// CHECK-DAG:       %[[LINK1_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_1]]) {init = 0 : i8, sym_name = "link1_link_cons_lock_0"}
// CHECK-DAG:       %[[LINK1_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_0]]) {init = 0 : i8, sym_name = "link1_prod_prod_lock_0"}
// CHECK-DAG:       %[[LINK1_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_0]]) {init = 0 : i8, sym_name = "link1_prod_cons_lock_0"}
// CHECK-DAG:       aie.flow(%[[TILE_2_0]], DMA : 0, %[[TILE_2_1]], DMA : 0)
// CHECK-DAG:       aie.flow(%[[TILE_2_1]], DMA : 0, %[[TILE_2_2]], DMA : 0)
// CHECK-DAG:       aie.flow(%[[TILE_2_1]], DMA : 1, %[[TILE_2_3]], DMA : 0)
// CHECK-DAG:       aie.flow(%[[TILE_2_1]], DMA : 2, %[[TILE_3_3]], DMA : 0)
// CHECK-DAG:       %[[EXT_BUFFER_IN:.*]] = aie.external_buffer {sym_name = "ext_buffer_in"} : memref<48xi32>
// CHECK-DAG:       aie.shim_dma_allocation @link1(MM2S, 0, 2)
// CHECK:           %[[MEMTILE_DMA_2_1:.*]] = aie.memtile_dma(%[[TILE_2_1]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LINK1_CONS_PROD_LOCK]], AcquireGreaterEqual, 3)
// CHECK:             aie.dma_bd(%[[LINK1_CONS_BUFF_0]] : memref<48xi32>) {len = 48 : i32}
// CHECK:             aie.use_lock(%[[LINK1_CONS_CONS_LOCK]], Release, 3)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LINK1_CONS_PROD_LOCK]], AcquireGreaterEqual, 3)
// CHECK:             aie.dma_bd(%[[LINK1_CONS_BUFF_1]] : memref<48xi32>) {len = 48 : i32}
// CHECK:             aie.use_lock(%[[LINK1_CONS_CONS_LOCK]], Release, 3)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LINK1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK1_CONS_BUFF_0]] : memref<48xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LINK1_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[LINK1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK1_CONS_BUFF_1]] : memref<48xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LINK1_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             %[[VAL_2:.*]] = aie.dma_start(MM2S, 1, ^bb7, ^bb9)
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[LINK1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK1_CONS_BUFF_0]] : memref<48xi32>) {len = 20 : i32, offset = 16 : i32}
// CHECK:             aie.use_lock(%[[LINK1_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(%[[LINK1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK1_CONS_BUFF_1]] : memref<48xi32>) {len = 20 : i32, offset = 16 : i32}
// CHECK:             aie.use_lock(%[[LINK1_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb9:
// CHECK:             %[[VAL_3:.*]] = aie.dma_start(MM2S, 2, ^bb10, ^bb12)
// CHECK:           ^bb10:
// CHECK:             aie.use_lock(%[[LINK1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK1_CONS_BUFF_0]] : memref<48xi32>) {len = 12 : i32, offset = 36 : i32}
// CHECK:             aie.use_lock(%[[LINK1_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb11
// CHECK:           ^bb11:
// CHECK:             aie.use_lock(%[[LINK1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK1_CONS_BUFF_1]] : memref<48xi32>) {len = 12 : i32, offset = 36 : i32}
// CHECK:             aie.use_lock(%[[LINK1_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb10
// CHECK:           ^bb12:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_2_2:.*]] = aie.mem(%[[TILE_2_2]]) {
// CHECK:             %[[VAL_4:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LINK2_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK2_CONS_BUFF_0]] : memref<4x4xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LINK2_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LINK2_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK2_CONS_BUFF_1]] : memref<4x4xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LINK2_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_2_3:.*]] = aie.mem(%[[TILE_2_3]]) {
// CHECK:             %[[VAL_5:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LINK3_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK3_CONS_BUFF_0]] : memref<20xi32>) {len = 20 : i32}
// CHECK:             aie.use_lock(%[[LINK3_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LINK3_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK3_CONS_BUFF_1]] : memref<20xi32>) {len = 20 : i32}
// CHECK:             aie.use_lock(%[[LINK3_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_3_3:.*]] = aie.mem(%[[TILE_3_3]]) {
// CHECK:             %[[VAL_6:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LINK4_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK4_CONS_BUFF_0]] : memref<12xi32>) {len = 12 : i32}
// CHECK:             aie.use_lock(%[[LINK4_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LINK4_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK4_CONS_BUFF_1]] : memref<12xi32>) {len = 12 : i32}
// CHECK:             aie.use_lock(%[[LINK4_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @link_distribute {
  aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile21 = aie.tile(2, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)
    %tile33 = aie.tile(3, 3)
    aie.flow(%tile20, DMA : 0, %tile21, DMA : 0) {symbol = @link1}
    aie.flow(%tile21, DMA : 0, %tile22, DMA : 0) {symbol = @link2}
    aie.flow(%tile21, DMA : 1, %tile23, DMA : 0) {symbol = @link3}
    aie.flow(%tile21, DMA : 2, %tile33, DMA : 0) {symbol = @link4}
    aie.objectfifo @link1 (%tile20, {%tile21}, 2 : i32) : !aie.objectfifo<memref<48xi32>>
    aie.objectfifo @link2 (%tile21, {%tile22}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
    aie.objectfifo @link3 (%tile21, {%tile23}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo @link4 (%tile21, {%tile33}, 2 : i32) : !aie.objectfifo<memref<12xi32>>
    %ext_buffer_in  = aie.external_buffer {sym_name = "ext_buffer_in"}: memref<48xi32>
    aie.objectfifo.register_external_buffers @link1 (%tile20, {%ext_buffer_in}) : (memref<48xi32>)
    aie.objectfifo.link [@link1] -> [@link2, @link3, @link4] ([] [])
  }
}
