
// RUN: iree-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @link5_cons : memref<512xi8>
// CHECK:           memref.global "public" @link5 : memref<512xi8>
// CHECK:           memref.global "public" @link4_cons : memref<128xi8>
// CHECK:           memref.global "public" @link4 : memref<128xi8>
// CHECK:           memref.global "public" @link3_cons : memref<128xi8>
// CHECK:           memref.global "public" @link3 : memref<128xi8>
// CHECK:           memref.global "public" @link2_cons : memref<128xi8>
// CHECK:           memref.global "public" @link2 : memref<128xi8>
// CHECK:           memref.global "public" @link1_cons : memref<128xi8>
// CHECK:           memref.global "public" @link1 : memref<128xi8>
// CHECK:           %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[TILE_2_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[LINK5_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_0]], 0) {init = 1 : i32, sym_name = "link5_cons_prod_lock"}
// CHECK:           %[[LINK5_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_0]], 1) {init = 0 : i32, sym_name = "link5_cons_cons_lock"}
// CHECK:           %[[LINK5_BUFF_0:.*]] = aie.buffer(%[[TILE_2_1]]) {sym_name = "link5_buff_0"} : memref<512xi8>
// CHECK:           %[[LINK5_BUFF_1:.*]] = aie.buffer(%[[TILE_2_1]]) {sym_name = "link5_buff_1"} : memref<512xi8>
// CHECK:           %[[LINK5_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_1]], 0) {init = 8 : i32, sym_name = "link5_prod_lock"}
// CHECK:           %[[LINK5_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_1]], 1) {init = 0 : i32, sym_name = "link5_cons_lock"}
// CHECK:           %[[LINK4_BUFF_0:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "link4_buff_0"} : memref<128xi8>
// CHECK:           %[[LINK4_BUFF_1:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "link4_buff_1"} : memref<128xi8>
// CHECK:           %[[LINK4_PROD_LOCK:.*]] = aie.lock(%[[TILE_3_3]], 0) {init = 2 : i32, sym_name = "link4_prod_lock"}
// CHECK:           %[[LINK4_CONS_LOCK:.*]] = aie.lock(%[[TILE_3_3]], 1) {init = 0 : i32, sym_name = "link4_cons_lock"}
// CHECK:           %[[LINK3_BUFF_0:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "link3_buff_0"} : memref<128xi8>
// CHECK:           %[[LINK3_BUFF_1:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "link3_buff_1"} : memref<128xi8>
// CHECK:           %[[LINK3_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_3]], 0) {init = 2 : i32, sym_name = "link3_prod_lock"}
// CHECK:           %[[LINK3_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_3]], 1) {init = 0 : i32, sym_name = "link3_cons_lock"}
// CHECK:           %[[LINK2_BUFF_0:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "link2_buff_0"} : memref<128xi8>
// CHECK:           %[[LINK2_BUFF_1:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "link2_buff_1"} : memref<128xi8>
// CHECK:           %[[LINK2_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 0) {init = 2 : i32, sym_name = "link2_prod_lock"}
// CHECK:           %[[LINK2_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 1) {init = 0 : i32, sym_name = "link2_cons_lock"}
// CHECK:           %[[LINK1_BUFF_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "link1_buff_0"} : memref<128xi8>
// CHECK:           %[[LINK1_BUFF_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "link1_buff_1"} : memref<128xi8>
// CHECK:           %[[LINK1_PROD_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 0) {init = 2 : i32, sym_name = "link1_prod_lock"}
// CHECK:           %[[LINK1_CONS_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 1) {init = 0 : i32, sym_name = "link1_cons_lock"}
// CHECK:           aie.flow(%[[TILE_1_2]], DMA : 0, %[[TILE_2_1]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_2_2]], DMA : 0, %[[TILE_2_1]], DMA : 1)
// CHECK:           aie.flow(%[[TILE_2_3]], DMA : 0, %[[TILE_2_1]], DMA : 2)
// CHECK:           aie.flow(%[[TILE_3_3]], DMA : 0, %[[TILE_2_1]], DMA : 3)
// CHECK:           aie.flow(%[[TILE_2_1]], DMA : 0, %[[TILE_2_0]], DMA : 0)
// CHECK:           %[[EXT_BUFFER_IN:.*]] = aie.external_buffer {sym_name = "ext_buffer_in"} : memref<512xi8>
// CHECK:           %[[MEM_1_2:.*]] = aie.mem(%[[TILE_1_2]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LINK1_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK1_BUFF_0]] : memref<128xi8>, 0, 128)
// CHECK:             aie.use_lock(%[[LINK1_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LINK1_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK1_BUFF_1]] : memref<128xi8>, 0, 128)
// CHECK:             aie.use_lock(%[[LINK1_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEMTILE_DMA_2_1:.*]] = aie.memtile_dma(%[[TILE_2_1]]) {
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LINK5_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK5_BUFF_0]] : memref<512xi8>, 0, 128)
// CHECK:             aie.use_lock(%[[LINK5_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LINK5_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK5_BUFF_1]] : memref<512xi8>, 0, 128)
// CHECK:             aie.use_lock(%[[LINK5_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_2:.*]] = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LINK5_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK5_BUFF_0]] : memref<512xi8>, 128, 128)
// CHECK:             aie.use_lock(%[[LINK5_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[LINK5_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK5_BUFF_1]] : memref<512xi8>, 128, 128)
// CHECK:             aie.use_lock(%[[LINK5_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             %[[VAL_3:.*]] = aie.dma_start(S2MM, 2, ^bb7, ^bb9)
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[LINK5_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK5_BUFF_0]] : memref<512xi8>, 256, 128)
// CHECK:             aie.use_lock(%[[LINK5_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(%[[LINK5_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK5_BUFF_1]] : memref<512xi8>, 256, 128)
// CHECK:             aie.use_lock(%[[LINK5_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb9:
// CHECK:             %[[VAL_4:.*]] = aie.dma_start(S2MM, 3, ^bb10, ^bb12)
// CHECK:           ^bb10:
// CHECK:             aie.use_lock(%[[LINK5_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK5_BUFF_0]] : memref<512xi8>, 384, 128)
// CHECK:             aie.use_lock(%[[LINK5_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb11
// CHECK:           ^bb11:
// CHECK:             aie.use_lock(%[[LINK5_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK5_BUFF_1]] : memref<512xi8>, 384, 128)
// CHECK:             aie.use_lock(%[[LINK5_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb10
// CHECK:           ^bb12:
// CHECK:             %[[VAL_5:.*]] = aie.dma_start(MM2S, 0, ^bb13, ^bb15)
// CHECK:           ^bb13:
// CHECK:             aie.use_lock(%[[LINK5_CONS_LOCK]], AcquireGreaterEqual, 4)
// CHECK:             aie.dma_bd(%[[LINK5_BUFF_0]] : memref<512xi8>, 0, 512)
// CHECK:             aie.use_lock(%[[LINK5_PROD_LOCK]], Release, 4)
// CHECK:             aie.next_bd ^bb14
// CHECK:           ^bb14:
// CHECK:             aie.use_lock(%[[LINK5_CONS_LOCK]], AcquireGreaterEqual, 4)
// CHECK:             aie.dma_bd(%[[LINK5_BUFF_1]] : memref<512xi8>, 0, 512)
// CHECK:             aie.use_lock(%[[LINK5_PROD_LOCK]], Release, 4)
// CHECK:             aie.next_bd ^bb13
// CHECK:           ^bb15:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_2_2:.*]] = aie.mem(%[[TILE_2_2]]) {
// CHECK:             %[[VAL_6:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LINK2_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK2_BUFF_0]] : memref<128xi8>, 0, 128)
// CHECK:             aie.use_lock(%[[LINK2_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LINK2_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK2_BUFF_1]] : memref<128xi8>, 0, 128)
// CHECK:             aie.use_lock(%[[LINK2_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_2_3:.*]] = aie.mem(%[[TILE_2_3]]) {
// CHECK:             %[[VAL_7:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LINK3_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK3_BUFF_0]] : memref<128xi8>, 0, 128)
// CHECK:             aie.use_lock(%[[LINK3_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LINK3_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK3_BUFF_1]] : memref<128xi8>, 0, 128)
// CHECK:             aie.use_lock(%[[LINK3_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_3_3:.*]] = aie.mem(%[[TILE_3_3]]) {
// CHECK:             %[[VAL_8:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LINK4_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK4_BUFF_0]] : memref<128xi8>, 0, 128)
// CHECK:             aie.use_lock(%[[LINK4_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LINK4_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK4_BUFF_1]] : memref<128xi8>, 0, 128)
// CHECK:             aie.use_lock(%[[LINK4_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @link5(S2MM, 0, 2)
// CHECK:           %[[SHIM_DMA_2_0:.*]] = aie.shim_dma(%[[TILE_2_0]]) {
// CHECK:             %[[VAL_9:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LINK5_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[EXT_BUFFER_IN]] : memref<512xi8>, 0, 512)
// CHECK:             aie.use_lock(%[[LINK5_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @link_join {
    aie.device(npu1_4col) {
        %tile20 = aie.tile(2, 0)
        %tile21 = aie.tile(2, 1)
        %tile12 = aie.tile(1, 2)
        %tile22 = aie.tile(2, 2)
        %tile23 = aie.tile(2, 3)
        %tile33 = aie.tile(3, 3)
        aie.objectfifo @link1 (%tile12, {%tile21}, 2 : i32) : !aie.objectfifo<memref<128xi8>>
        aie.objectfifo @link2 (%tile22, {%tile21}, 2 : i32) : !aie.objectfifo<memref<128xi8>>
        aie.objectfifo @link3 (%tile23, {%tile21}, 2 : i32) : !aie.objectfifo<memref<128xi8>>
        aie.objectfifo @link4 (%tile33, {%tile21}, 2 : i32) : !aie.objectfifo<memref<128xi8>>
        aie.objectfifo @link5 (%tile21, {%tile20}, 2 : i32) : !aie.objectfifo<memref<512xi8>>
        %ext_buffer_in  = aie.external_buffer {sym_name = "ext_buffer_in"}: memref<512xi8>
        aie.objectfifo.register_external_buffers @link5 (%tile20, {%ext_buffer_in}) : (memref<512xi8>)
        aie.objectfifo.link [@link1, @link2, @link3, @link4] -> [@link5] ()
    }
}
