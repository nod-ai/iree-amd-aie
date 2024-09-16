
// RUN: iree-opt --amdaie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK-DAG:       memref.global "public" @link5 : memref<512xi8>
// CHECK-DAG:       memref.global "public" @link4 : memref<128xi8>
// CHECK-DAG:       memref.global "public" @link3 : memref<128xi8>
// CHECK-DAG:       memref.global "public" @link2 : memref<128xi8>
// CHECK-DAG:       memref.global "public" @link1 : memref<128xi8>
// CHECK-DAG:       %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK-DAG:       %[[TILE_2_1:.*]] = aie.tile(2, 1)
// CHECK-DAG:       %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK-DAG:       %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK-DAG:       %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK-DAG:       %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK-DAG:       %[[LINK5_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_0]]) {init = 0 : i8, sym_name = "link5_cons_prod_lock_0"}
// CHECK-DAG:       %[[LINK5_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_0]]) {init = 0 : i8, sym_name = "link5_cons_cons_lock_0"}
// CHECK-DAG:       %[[LINK5_BUFF_0:.*]] = aie.buffer(%[[TILE_2_1]]) {sym_name = "link5_link_buff_0_0"} : memref<512xi8>
// CHECK-DAG:       %[[LINK5_BUFF_1:.*]] = aie.buffer(%[[TILE_2_1]]) {sym_name = "link5_link_buff_0_1"} : memref<512xi8>
// CHECK-DAG:       %[[LINK5_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_1]]) {init = 8 : i8, sym_name = "link5_link_prod_lock_0"}
// CHECK-DAG:       %[[LINK5_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_1]]) {init = 0 : i8, sym_name = "link5_link_cons_lock_0"}
// CHECK-DAG:       %[[LINK4_BUFF_0:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "link4_prod_buff_3_0"} : memref<128xi8>
// CHECK-DAG:       %[[LINK4_BUFF_1:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "link4_prod_buff_3_1"} : memref<128xi8>
// CHECK-DAG:       %[[LINK4_PROD_LOCK:.*]] = aie.lock(%[[TILE_3_3]]) {init = 2 : i8, sym_name = "link4_prod_prod_lock_3"}
// CHECK-DAG:       %[[LINK4_CONS_LOCK:.*]] = aie.lock(%[[TILE_3_3]]) {init = 0 : i8, sym_name = "link4_prod_cons_lock_3"}
// CHECK-DAG:       %[[LINK3_BUFF_0:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "link3_prod_buff_2_0"} : memref<128xi8>
// CHECK-DAG:       %[[LINK3_BUFF_1:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "link3_prod_buff_2_1"} : memref<128xi8>
// CHECK-DAG:       %[[LINK3_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_3]]) {init = 2 : i8, sym_name = "link3_prod_prod_lock_2"}
// CHECK-DAG:       %[[LINK3_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_3]]) {init = 0 : i8, sym_name = "link3_prod_cons_lock_2"}
// CHECK-DAG:       %[[LINK2_BUFF_0:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "link2_prod_buff_1_0"} : memref<128xi8>
// CHECK-DAG:       %[[LINK2_BUFF_1:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "link2_prod_buff_1_1"} : memref<128xi8>
// CHECK-DAG:       %[[LINK2_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_2]]) {init = 2 : i8, sym_name = "link2_prod_prod_lock_1"}
// CHECK-DAG:       %[[LINK2_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_2]]) {init = 0 : i8, sym_name = "link2_prod_cons_lock_1"}
// CHECK-DAG:       %[[LINK1_BUFF_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "link1_prod_buff_0_0"} : memref<128xi8>
// CHECK-DAG:       %[[LINK1_BUFF_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "link1_prod_buff_0_1"} : memref<128xi8>
// CHECK-DAG:       %[[LINK1_PROD_LOCK:.*]] = aie.lock(%[[TILE_1_2]]) {init = 2 : i8, sym_name = "link1_prod_prod_lock_0"}
// CHECK-DAG:       %[[LINK1_CONS_LOCK:.*]] = aie.lock(%[[TILE_1_2]]) {init = 0 : i8, sym_name = "link1_prod_cons_lock_0"}
// CHECK-DAG:       aie.flow(%[[TILE_1_2]], DMA : 0, %[[TILE_2_1]], DMA : 0)
// CHECK-DAG:       aie.flow(%[[TILE_2_2]], DMA : 0, %[[TILE_2_1]], DMA : 1)
// CHECK-DAG:       aie.flow(%[[TILE_2_3]], DMA : 0, %[[TILE_2_1]], DMA : 2)
// CHECK-DAG:       aie.flow(%[[TILE_3_3]], DMA : 0, %[[TILE_2_1]], DMA : 3)
// CHECK-DAG:       aie.flow(%[[TILE_2_1]], DMA : 0, %[[TILE_2_0]], DMA : 0)
// CHECK-DAG:       %[[EXT_BUFFER_IN:.*]] = aie.external_buffer {sym_name = "ext_buffer_in"} : memref<512xi8>
// CHECK:           %[[MEM_1_2:.*]] = aie.mem(%[[TILE_1_2]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LINK1_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK1_BUFF_0]] : memref<128xi8>) {len = 128 : i32}
// CHECK:             aie.use_lock(%[[LINK1_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LINK1_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK1_BUFF_1]] : memref<128xi8>) {len = 128 : i32}
// CHECK:             aie.use_lock(%[[LINK1_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEMTILE_DMA_2_1:.*]] = aie.memtile_dma(%[[TILE_2_1]]) {
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LINK5_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK5_BUFF_0]] : memref<512xi8>) {len = 128 : i32}
// CHECK:             aie.use_lock(%[[LINK5_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LINK5_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK5_BUFF_1]] : memref<512xi8>) {len = 128 : i32}
// CHECK:             aie.use_lock(%[[LINK5_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_2:.*]] = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LINK5_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK5_BUFF_0]] : memref<512xi8>) {len = 128 : i32, offset = 128 : i32}
// CHECK:             aie.use_lock(%[[LINK5_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[LINK5_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK5_BUFF_1]] : memref<512xi8>) {len = 128 : i32, offset = 128 : i32}
// CHECK:             aie.use_lock(%[[LINK5_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             %[[VAL_3:.*]] = aie.dma_start(S2MM, 2, ^bb7, ^bb9)
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[LINK5_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK5_BUFF_0]] : memref<512xi8>) {len = 128 : i32, offset = 256 : i32}
// CHECK:             aie.use_lock(%[[LINK5_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(%[[LINK5_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK5_BUFF_1]] : memref<512xi8>) {len = 128 : i32, offset = 256 : i32}
// CHECK:             aie.use_lock(%[[LINK5_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb9:
// CHECK:             %[[VAL_4:.*]] = aie.dma_start(S2MM, 3, ^bb10, ^bb12)
// CHECK:           ^bb10:
// CHECK:             aie.use_lock(%[[LINK5_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK5_BUFF_0]] : memref<512xi8>) {len = 128 : i32, offset = 384 : i32}
// CHECK:             aie.use_lock(%[[LINK5_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb11
// CHECK:           ^bb11:
// CHECK:             aie.use_lock(%[[LINK5_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK5_BUFF_1]] : memref<512xi8>) {len = 128 : i32, offset = 384 : i32}
// CHECK:             aie.use_lock(%[[LINK5_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb10
// CHECK:           ^bb12:
// CHECK:             %[[VAL_5:.*]] = aie.dma_start(MM2S, 0, ^bb13, ^bb15)
// CHECK:           ^bb13:
// CHECK:             aie.use_lock(%[[LINK5_CONS_LOCK]], AcquireGreaterEqual, 4)
// CHECK:             aie.dma_bd(%[[LINK5_BUFF_0]] : memref<512xi8>) {len = 512 : i32}
// CHECK:             aie.use_lock(%[[LINK5_PROD_LOCK]], Release, 4)
// CHECK:             aie.next_bd ^bb14
// CHECK:           ^bb14:
// CHECK:             aie.use_lock(%[[LINK5_CONS_LOCK]], AcquireGreaterEqual, 4)
// CHECK:             aie.dma_bd(%[[LINK5_BUFF_1]] : memref<512xi8>) {len = 512 : i32}
// CHECK:             aie.use_lock(%[[LINK5_PROD_LOCK]], Release, 4)
// CHECK:             aie.next_bd ^bb13
// CHECK:           ^bb15:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_2_2:.*]] = aie.mem(%[[TILE_2_2]]) {
// CHECK:             %[[VAL_6:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LINK2_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK2_BUFF_0]] : memref<128xi8>) {len = 128 : i32}
// CHECK:             aie.use_lock(%[[LINK2_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LINK2_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK2_BUFF_1]] : memref<128xi8>) {len = 128 : i32}
// CHECK:             aie.use_lock(%[[LINK2_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_2_3:.*]] = aie.mem(%[[TILE_2_3]]) {
// CHECK:             %[[VAL_7:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LINK3_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK3_BUFF_0]] : memref<128xi8>) {len = 128 : i32}
// CHECK:             aie.use_lock(%[[LINK3_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LINK3_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK3_BUFF_1]] : memref<128xi8>) {len = 128 : i32}
// CHECK:             aie.use_lock(%[[LINK3_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @link5(S2MM, 0, 2)
// CHECK:           %[[MEM_3_3:.*]] = aie.mem(%[[TILE_3_3]]) {
// CHECK:             %[[VAL_8:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LINK4_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK4_BUFF_0]] : memref<128xi8>) {len = 128 : i32}
// CHECK:             aie.use_lock(%[[LINK4_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LINK4_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[LINK4_BUFF_1]] : memref<128xi8>) {len = 128 : i32}
// CHECK:             aie.use_lock(%[[LINK4_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }
module @link_join {
  aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile21 = aie.tile(2, 1)
    %tile12 = aie.tile(1, 2)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)
    %tile33 = aie.tile(3, 3)
    aie.flow(%tile12, DMA : 0, %tile21, DMA : 0) {symbol = @link1}
    aie.flow(%tile22, DMA : 0, %tile21, DMA : 1) {symbol = @link2}
    aie.flow(%tile23, DMA : 0, %tile21, DMA : 2) {symbol = @link3}
    aie.flow(%tile33, DMA : 0, %tile21, DMA : 3) {symbol = @link4}
    aie.flow(%tile21, DMA : 0, %tile20, DMA : 0) {symbol = @link5}
    aie.objectfifo @link1 (%tile12, {%tile21}, 2 : i32) : !aie.objectfifo<memref<128xi8>>
    aie.objectfifo @link2 (%tile22, {%tile21}, 2 : i32) : !aie.objectfifo<memref<128xi8>>
    aie.objectfifo @link3 (%tile23, {%tile21}, 2 : i32) : !aie.objectfifo<memref<128xi8>>
    aie.objectfifo @link4 (%tile33, {%tile21}, 2 : i32) : !aie.objectfifo<memref<128xi8>>
    aie.objectfifo @link5 (%tile21, {%tile20}, 2 : i32) : !aie.objectfifo<memref<512xi8>>
    %ext_buffer_in  = aie.external_buffer {sym_name = "ext_buffer_in"}: memref<512xi8>
    aie.objectfifo.register_external_buffers @link5 (%tile20, {%ext_buffer_in}) : (memref<512xi8>)
    aie.objectfifo.link [@link1, @link2, @link3, @link4] -> [@link5] ([] [])
  }
}
