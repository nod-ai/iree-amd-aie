
// RUN: iree-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @from_memTile_cons : memref<48xi32>
// CHECK:           memref.global "public" @from_memTile : memref<48xi32>
// CHECK:           memref.global "public" @to_memTile_cons : memref<16xi32>
// CHECK:           memref.global "public" @to_memTile : memref<16xi32>
// CHECK:           %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[TILE_2_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[FROM_MEMTILE_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_0]], 0) {init = 1 : i32, sym_name = "from_memTile_cons_prod_lock"}
// CHECK:           %[[FROM_MEMTILE_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_0]], 1) {init = 0 : i32, sym_name = "from_memTile_cons_cons_lock"}
// CHECK:           %[[FROM_MEMTILE_BUFF_0:.*]] = aie.buffer(%[[TILE_2_1]]) {sym_name = "from_memTile_buff_0"} : memref<48xi32>
// CHECK:           %[[FROM_MEMTILE_BUFF_1:.*]] = aie.buffer(%[[TILE_2_1]]) {sym_name = "from_memTile_buff_1"} : memref<48xi32>
// CHECK:           %[[FROM_MEMTILE_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_1]], 0) {init = 2 : i32, sym_name = "from_memTile_prod_lock"}
// CHECK:           %[[FROM_MEMTILE_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_1]], 1) {init = 0 : i32, sym_name = "from_memTile_cons_lock"}
// CHECK:           %[[TO_MEMTILE_BUFF_0:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "to_memTile_buff_0"} : memref<16xi32>
// CHECK:           %[[TO_MEMTILE_BUFF_1:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "to_memTile_buff_1"} : memref<16xi32>
// CHECK:           %[[TO_MEMTILE_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 0) {init = 2 : i32, sym_name = "to_memTile_prod_lock"}
// CHECK:           %[[TO_MEMTILE_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 1) {init = 0 : i32, sym_name = "to_memTile_cons_lock"}
// CHECK:           aie.flow(%[[TILE_2_2]], DMA : 0, %[[TILE_2_1]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_2_1]], DMA : 0, %[[TILE_2_0]], DMA : 0)
// CHECK:           %[[EXT_BUFF_IN:.*]] = aie.external_buffer {sym_name = "ext_buff_in"} : memref<48xi32>
// CHECK:           %[[MEM_2_2:.*]] = aie.mem(%[[TILE_2_2]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[TO_MEMTILE_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[TO_MEMTILE_BUFF_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[TO_MEMTILE_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[TO_MEMTILE_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[TO_MEMTILE_BUFF_1]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[TO_MEMTILE_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEMTILE_DMA_2_1:.*]] = aie.memtile_dma(%[[TILE_2_1]]) {
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[FROM_MEMTILE_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[FROM_MEMTILE_BUFF_0]] : memref<48xi32>, 0, 48)
// CHECK:             aie.use_lock(%[[FROM_MEMTILE_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[FROM_MEMTILE_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[FROM_MEMTILE_BUFF_1]] : memref<48xi32>, 0, 48)
// CHECK:             aie.use_lock(%[[FROM_MEMTILE_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_2:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[FROM_MEMTILE_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[FROM_MEMTILE_BUFF_0]] : memref<48xi32>, 0, 48)
// CHECK:             aie.use_lock(%[[FROM_MEMTILE_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[FROM_MEMTILE_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[FROM_MEMTILE_BUFF_1]] : memref<48xi32>, 0, 48)
// CHECK:             aie.use_lock(%[[FROM_MEMTILE_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @from_memTile(S2MM, 0, 2)
// CHECK:           %[[SHIM_DMA_2_0:.*]] = aie.shim_dma(%[[TILE_2_0]]) {
// CHECK:             %[[VAL_3:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[FROM_MEMTILE_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[EXT_BUFF_IN]] : memref<48xi32>, 0, 48)
// CHECK:             aie.use_lock(%[[FROM_MEMTILE_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @link_L1_DDR {
    aie.device(npu1_4col) {
        %tile20 = aie.tile(2, 0)
        %tile21 = aie.tile(2, 1)
        %tile22 = aie.tile(2, 2)
        aie.objectfifo @to_memTile (%tile22, {%tile21}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @from_memTile (%tile21, {%tile20}, 2 : i32) : !aie.objectfifo<memref<48xi32>>
        aie.objectfifo.link [@to_memTile] -> [@from_memTile] ()
        %ext_buff_in = aie.external_buffer {sym_name = "ext_buff_in"}: memref<48xi32>
        aie.objectfifo.register_external_buffers @from_memTile (%tile20, {%ext_buff_in}) : (memref<48xi32>)
    }
}
