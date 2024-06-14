
// RUN: iree-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @link4_cons : memref<12xi32>
// CHECK:           memref.global "public" @link4 : memref<12xi32>
// CHECK:           memref.global "public" @link3_cons : memref<20xi32>
// CHECK:           memref.global "public" @link3 : memref<20xi32>
// CHECK:           memref.global "public" @link2_cons : memref<4x4xi32>
// CHECK:           memref.global "public" @link2 : memref<4x4xi32>
// CHECK:           memref.global "public" @link1_cons : memref<48xi32>
// CHECK:           memref.global "public" @link1 : memref<48xi32>
// CHECK:           %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[TILE_2_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[LINK4_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "link4_cons_buff_0"} : memref<12xi32>
// CHECK:           %[[LINK4_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "link4_cons_buff_1"} : memref<12xi32>
// CHECK:           %[[LINK4_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_3_3]], 0) {init = 2 : i32, sym_name = "link4_cons_prod_lock"}
// CHECK:           %[[LINK4_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_3_3]], 1) {init = 0 : i32, sym_name = "link4_cons_cons_lock"}
// CHECK:           %[[LINK3_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "link3_cons_buff_0"} : memref<20xi32>
// CHECK:           %[[LINK3_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "link3_cons_buff_1"} : memref<20xi32>
// CHECK:           %[[LINK3_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_3]], 0) {init = 2 : i32, sym_name = "link3_cons_prod_lock"}
// CHECK:           %[[LINK3_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_3]], 1) {init = 0 : i32, sym_name = "link3_cons_cons_lock"}
// CHECK:           %[[LINK2_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "link2_cons_buff_0"} : memref<4x4xi32>
// CHECK:           %[[LINK2_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "link2_cons_buff_1"} : memref<4x4xi32>
// CHECK:           %[[LINK2_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 0) {init = 2 : i32, sym_name = "link2_cons_prod_lock"}
// CHECK:           %[[LINK2_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 1) {init = 0 : i32, sym_name = "link2_cons_cons_lock"}
// CHECK:           %[[LINK1_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_2_1]]) {sym_name = "link1_cons_buff_0"} : memref<48xi32>
// CHECK:           %[[LINK1_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_2_1]]) {sym_name = "link1_cons_buff_1"} : memref<48xi32>
// CHECK:           %[[LINK1_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_1]], 0) {init = 6 : i32, sym_name = "link1_cons_prod_lock"}
// CHECK:           %[[LINK1_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_1]], 1) {init = 0 : i32, sym_name = "link1_cons_cons_lock"}
// CHECK:           %[[LINK1_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_0]], 0) {init = 1 : i32, sym_name = "link1_prod_lock"}
// CHECK:           %[[LINK1_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_0]], 1) {init = 0 : i32, sym_name = "link1_cons_lock"}
// CHECK:           aie.flow(%[[TILE_2_0]], DMA : 0, %[[TILE_2_1]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_2_1]], DMA : 0, %[[TILE_2_2]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_2_1]], DMA : 1, %[[TILE_2_3]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_2_1]], DMA : 2, %[[TILE_3_3]], DMA : 0)
// CHECK:           %[[EXT_BUFFER_IN:.*]] = aie.external_buffer {sym_name = "ext_buffer_in"} : memref<48xi32>
// CHECK:           aie.shim_dma_allocation @link1(MM2S, 0, 2)
// CHECK:           %[[SHIM_DMA_2_0:.*]] = aie.shim_dma(%[[TILE_2_0]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LINK1_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[EXT_BUFFER_IN]] : memref<48xi32>, 0, 48)
// CHECK:             aie.use_lock(%[[LINK1_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEMTILE_DMA_2_1:.*]] = aie.memtile_dma(%[[TILE_2_1]]) {
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LINK1_CONS_PROD_LOCK]], AcquireGreaterEqual, 3)
// CHECK:             aie.dma_bd(%[[LINK1_CONS_BUFF_0]] : memref<48xi32>, 0, 48)
// CHECK:             aie.use_lock(%[[LINK1_CONS_CONS_LOCK]], Release, 3)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LINK1_CONS_PROD_LOCK]], AcquireGreaterEqual, 3)
// CHECK:             aie.dma_bd(%[[LINK1_CONS_BUFF_1]] : memref<48xi32>, 0, 48)
// CHECK:             aie.use_lock(%[[LINK1_CONS_CONS_LOCK]], Release, 3)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_2:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LINK1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK1_CONS_BUFF_0]] : memref<48xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[LINK1_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[LINK1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK1_CONS_BUFF_1]] : memref<48xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[LINK1_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             %[[VAL_3:.*]] = aie.dma_start(MM2S, 1, ^bb7, ^bb9)
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[LINK1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK1_CONS_BUFF_0]] : memref<48xi32>, 16, 20)
// CHECK:             aie.use_lock(%[[LINK1_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(%[[LINK1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK1_CONS_BUFF_1]] : memref<48xi32>, 16, 20)
// CHECK:             aie.use_lock(%[[LINK1_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb9:
// CHECK:             %[[VAL_4:.*]] = aie.dma_start(MM2S, 2, ^bb10, ^bb12)
// CHECK:           ^bb10:
// CHECK:             aie.use_lock(%[[LINK1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK1_CONS_BUFF_0]] : memref<48xi32>, 36, 12)
// CHECK:             aie.use_lock(%[[LINK1_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb11
// CHECK:           ^bb11:
// CHECK:             aie.use_lock(%[[LINK1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK1_CONS_BUFF_1]] : memref<48xi32>, 36, 12)
// CHECK:             aie.use_lock(%[[LINK1_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb10
// CHECK:           ^bb12:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_2_2:.*]] = aie.mem(%[[TILE_2_2]]) {
// CHECK:             %[[VAL_5:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LINK2_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK2_CONS_BUFF_0]] : memref<4x4xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[LINK2_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LINK2_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK2_CONS_BUFF_1]] : memref<4x4xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[LINK2_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_2_3:.*]] = aie.mem(%[[TILE_2_3]]) {
// CHECK:             %[[VAL_6:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LINK3_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK3_CONS_BUFF_0]] : memref<20xi32>, 0, 20)
// CHECK:             aie.use_lock(%[[LINK3_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LINK3_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK3_CONS_BUFF_1]] : memref<20xi32>, 0, 20)
// CHECK:             aie.use_lock(%[[LINK3_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_3_3:.*]] = aie.mem(%[[TILE_3_3]]) {
// CHECK:             %[[VAL_7:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LINK4_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK4_CONS_BUFF_0]] : memref<12xi32>, 0, 12)
// CHECK:             aie.use_lock(%[[LINK4_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LINK4_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK4_CONS_BUFF_1]] : memref<12xi32>, 0, 12)
// CHECK:             aie.use_lock(%[[LINK4_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @link_distribute {
    aie.device(npu1_4col) {
        %tile20 = aie.tile(2, 0)
        %tile21 = aie.tile(2, 1)
        %tile22 = aie.tile(2, 2)
        %tile23 = aie.tile(2, 3)
        %tile33 = aie.tile(3, 3)
        aie.objectfifo @link1 (%tile20, {%tile21}, 2 : i32) : !aie.objectfifo<memref<48xi32>>
        aie.objectfifo @link2 (%tile21, {%tile22}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
        aie.objectfifo @link3 (%tile21, {%tile23}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
        aie.objectfifo @link4 (%tile21, {%tile33}, 2 : i32) : !aie.objectfifo<memref<12xi32>>
        %ext_buffer_in  = aie.external_buffer {sym_name = "ext_buffer_in"}: memref<48xi32>
        aie.objectfifo.register_external_buffers @link1 (%tile20, {%ext_buffer_in}) : (memref<48xi32>)
        aie.objectfifo.link [@link1] -> [@link2, @link3, @link4] ()
    }
}
