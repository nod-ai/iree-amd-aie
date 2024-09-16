
// RUN: iree-opt --amdaie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           memref.global "public" @from_memTile : memref<16xi32>
// CHECK:           memref.global "public" @to_memTile : memref<16xi32>
// CHECK-DAG:       %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK-DAG:       %[[TILE_2_1:.*]] = aie.tile(2, 1)
// CHECK-DAG:       %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK-DAG:       %[[FROM_MEMTILE_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "from_memTile_cons_buff_0_0"} : memref<16xi32>
// CHECK-DAG:       %[[FROM_MEMTILE_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "from_memTile_cons_buff_0_1"} : memref<16xi32>
// CHECK-DAG:       %[[FROM_MEMTILE_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_2]]) {init = 2 : i8, sym_name = "from_memTile_cons_prod_lock_0"}
// CHECK-DAG:       %[[FROM_MEMTILE_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_2]]) {init = 0 : i8, sym_name = "from_memTile_cons_cons_lock_0"}
// CHECK-DAG:       %[[TO_MEMTILE_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_2_1]]) {sym_name = "to_memTile_link_buff_0_0"} : memref<16xi32>
// CHECK-DAG:       %[[TO_MEMTILE_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_2_1]]) {sym_name = "to_memTile_link_buff_0_1"} : memref<16xi32>
// CHECK-DAG:       %[[TO_MEMTILE_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_1]]) {init = 2 : i8, sym_name = "to_memTile_link_prod_lock_0"}
// CHECK-DAG:       %[[TO_MEMTILE_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_1]]) {init = 0 : i8, sym_name = "to_memTile_link_cons_lock_0"}
// CHECK-DAG:       %[[TO_MEMTILE_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_0]]) {init = 0 : i8, sym_name = "to_memTile_prod_prod_lock_0"}
// CHECK-DAG:       %[[TO_MEMTILE_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_0]]) {init = 0 : i8, sym_name = "to_memTile_prod_cons_lock_0"}
// CHECK-DAG:       aie.flow(%[[TILE_2_0]], DMA : 0, %[[TILE_2_1]], DMA : 0)
// CHECK-DAG:       aie.flow(%[[TILE_2_1]], DMA : 0, %[[TILE_2_2]], DMA : 0)
// CHECK-DAG:       %[[EXT_BUFF_IN:.*]] = aie.external_buffer {sym_name = "ext_buff_in"} : memref<16xi32>
// CHECK-DAG:       aie.shim_dma_allocation @to_memTile(MM2S, 0, 2)
// CHECK:           %[[MEMTILE_DMA_2_1:.*]] = aie.memtile_dma(%[[TILE_2_1]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[TO_MEMTILE_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[TO_MEMTILE_CONS_BUFF_0]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[TO_MEMTILE_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[TO_MEMTILE_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[TO_MEMTILE_CONS_BUFF_1]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[TO_MEMTILE_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[TO_MEMTILE_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[TO_MEMTILE_CONS_BUFF_0]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[TO_MEMTILE_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[TO_MEMTILE_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[TO_MEMTILE_CONS_BUFF_1]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[TO_MEMTILE_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_2_2:.*]] = aie.mem(%[[TILE_2_2]]) {
// CHECK:             %[[VAL_2:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[FROM_MEMTILE_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[FROM_MEMTILE_CONS_BUFF_0]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[FROM_MEMTILE_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[FROM_MEMTILE_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[FROM_MEMTILE_CONS_BUFF_1]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[FROM_MEMTILE_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }
module @link_DDR_L1 {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile21 = aie.tile(2, 1)
        %tile22 = aie.tile(2, 2)
        aie.flow(%tile20, DMA : 0, %tile21, DMA : 0) {symbol = @to_memTile}
        aie.flow(%tile21, DMA : 0, %tile22, DMA : 0) {symbol = @from_memTile}
        aie.objectfifo @to_memTile (%tile20, {%tile21}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @from_memTile (%tile21, {%tile22}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo.link [@to_memTile] -> [@from_memTile] ([] [])
        %ext_buff_in = aie.external_buffer {sym_name = "ext_buff_in"}: memref<16xi32>
        aie.objectfifo.register_external_buffers @to_memTile (%tile20, {%ext_buff_in}) : (memref<16xi32>)
    }
}
