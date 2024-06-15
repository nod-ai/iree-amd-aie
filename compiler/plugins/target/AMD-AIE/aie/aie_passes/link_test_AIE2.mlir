
// RUN: iree-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           memref.global "public" @mem_out_cons : memref<3000xi32>
// CHECK:           memref.global "public" @mem_out : memref<3000xi32>
// CHECK:           memref.global "public" @mem_in_0_cons : memref<3000xi32>
// CHECK:           memref.global "public" @mem_in_1_cons : memref<3000xi32>
// CHECK:           memref.global "public" @mem_in : memref<3000xi32>
// CHECK:           %[[TILE_0_0:.*]] = aie.tile(0, 0)
// CHECK:           %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[TILE_0_3:.*]] = aie.tile(0, 3)
// CHECK:           %[[MEM_OUT_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_0_3]]) {sym_name = "mem_out_cons_buff_0"} : memref<3000xi32>
// CHECK:           %[[MEM_OUT_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_0_3]]) {sym_name = "mem_out_cons_buff_1"} : memref<3000xi32>
// CHECK:           %[[MEM_OUT_CONS_BUFF_2:.*]] = aie.buffer(%[[TILE_0_3]]) {sym_name = "mem_out_cons_buff_2"} : memref<3000xi32>
// CHECK:           %[[MEM_OUT_CONS_BUFF_3:.*]] = aie.buffer(%[[TILE_0_3]]) {sym_name = "mem_out_cons_buff_3"} : memref<3000xi32>
// CHECK:           %[[MEM_OUT_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_0_3]], 0) {init = 4 : i32, sym_name = "mem_out_cons_prod_lock"}
// CHECK:           %[[MEM_OUT_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_0_3]], 1) {init = 0 : i32, sym_name = "mem_out_cons_cons_lock"}
// CHECK:           %[[MEM_IN_0_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "mem_in_0_cons_buff_0"} : memref<3000xi32>
// CHECK:           %[[MEM_IN_0_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "mem_in_0_cons_buff_1"} : memref<3000xi32>
// CHECK:           %[[MEM_IN_0_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_0_2]], 0) {init = 2 : i32, sym_name = "mem_in_0_cons_prod_lock"}
// CHECK:           %[[MEM_IN_0_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_0_2]], 1) {init = 0 : i32, sym_name = "mem_in_0_cons_cons_lock"}
// CHECK:           %[[MEM_IN_1_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "mem_in_1_cons_buff_0"} : memref<3000xi32>
// CHECK:           %[[MEM_IN_1_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "mem_in_1_cons_buff_1"} : memref<3000xi32>
// CHECK:           %[[MEM_IN_1_CONS_BUFF_2:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "mem_in_1_cons_buff_2"} : memref<3000xi32>
// CHECK:           %[[MEM_IN_1_CONS_BUFF_3:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "mem_in_1_cons_buff_3"} : memref<3000xi32>
// CHECK:           %[[MEM_IN_1_CONS_BUFF_4:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "mem_in_1_cons_buff_4"} : memref<3000xi32>
// CHECK:           %[[MEM_IN_1_CONS_BUFF_5:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "mem_in_1_cons_buff_5"} : memref<3000xi32>
// CHECK:           %[[MEM_IN_1_CONS_BUFF_6:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "mem_in_1_cons_buff_6"} : memref<3000xi32>
// CHECK:           %[[MEM_IN_1_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_0_1]], 0) {init = 7 : i32, sym_name = "mem_in_1_cons_prod_lock"}
// CHECK:           %[[MEM_IN_1_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_0_1]], 1) {init = 0 : i32, sym_name = "mem_in_1_cons_cons_lock"}
// CHECK:           %[[MEM_IN_PROD_LOCK:.*]] = aie.lock(%[[TILE_0_0]], 0) {init = 0 : i32, sym_name = "mem_in_prod_lock"}
// CHECK:           %[[MEM_IN_CONS_LOCK:.*]] = aie.lock(%[[TILE_0_0]], 1) {init = 0 : i32, sym_name = "mem_in_cons_lock"}
// CHECK:           aie.flow(%[[TILE_0_0]], DMA : 0, %[[TILE_0_1]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_0_0]], DMA : 0, %[[TILE_0_2]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_0_1]], DMA : 0, %[[TILE_0_3]], DMA : 0)
// CHECK:           %[[CORE_0_2:.*]] = aie.core(%[[TILE_0_2]]) {
// CHECK:             %[[C11_I32:.*]] = arith.constant 11 : i32
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             aie.use_lock(%[[MEM_IN_0_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             memref.store %[[C11_I32]], %[[MEM_IN_0_CONS_BUFF_0]]{{\[}}%[[C0]]] : memref<3000xi32>
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @mem_in(MM2S, 0, 0)
// CHECK:           %[[CORE_0_3:.*]] = aie.core(%[[TILE_0_3]]) {
// CHECK:             %[[C11_I32:.*]] = arith.constant 11 : i32
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             aie.use_lock(%[[MEM_OUT_CONS_CONS_LOCK]], AcquireGreaterEqual, 3)
// CHECK:             memref.store %[[C11_I32]], %[[MEM_OUT_CONS_BUFF_0]]{{\[}}%[[C0]]] : memref<3000xi32>
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_0_2:.*]] = aie.mem(%[[TILE_0_2]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[MEM_IN_0_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[MEM_IN_0_CONS_BUFF_0]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[MEM_IN_0_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[MEM_IN_0_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[MEM_IN_0_CONS_BUFF_1]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[MEM_IN_0_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEMTILE_DMA_0_1:.*]] = aie.memtile_dma(%[[TILE_0_1]]) {
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb8)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[MEM_IN_1_CONS_BUFF_0]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[MEM_IN_1_CONS_BUFF_1]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[MEM_IN_1_CONS_BUFF_2]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[MEM_IN_1_CONS_BUFF_3]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[MEM_IN_1_CONS_BUFF_4]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[MEM_IN_1_CONS_BUFF_5]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[MEM_IN_1_CONS_BUFF_6]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb8:
// CHECK:             %[[VAL_2:.*]] = aie.dma_start(MM2S, 0, ^bb9, ^bb16)
// CHECK:           ^bb9:
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[MEM_IN_1_CONS_BUFF_0]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb10
// CHECK:           ^bb10:
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[MEM_IN_1_CONS_BUFF_1]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb11
// CHECK:           ^bb11:
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[MEM_IN_1_CONS_BUFF_2]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb12
// CHECK:           ^bb12:
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[MEM_IN_1_CONS_BUFF_3]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb13
// CHECK:           ^bb13:
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[MEM_IN_1_CONS_BUFF_4]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb14
// CHECK:           ^bb14:
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[MEM_IN_1_CONS_BUFF_5]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb15
// CHECK:           ^bb15:
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[MEM_IN_1_CONS_BUFF_6]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[MEM_IN_1_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb9
// CHECK:           ^bb16:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_0_3:.*]] = aie.mem(%[[TILE_0_3]]) {
// CHECK:             %[[VAL_3:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[MEM_OUT_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[MEM_OUT_CONS_BUFF_0]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[MEM_OUT_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[MEM_OUT_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[MEM_OUT_CONS_BUFF_1]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[MEM_OUT_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[MEM_OUT_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[MEM_OUT_CONS_BUFF_2]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[MEM_OUT_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[MEM_OUT_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[MEM_OUT_CONS_BUFF_3]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[MEM_OUT_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @link_AIE2 {
    aie.device(xcve2302) {
        %tile00 = aie.tile(0, 0)
        %tile01 = aie.tile(0, 1)
        %tile02 = aie.tile(0, 2)
        %tile03 = aie.tile(0, 3)
        aie.objectfifo @mem_in (%tile00, {%tile02, %tile01}, [2,2,7]) : !aie.objectfifo<memref<3000xi32>>
        aie.objectfifo @mem_out (%tile01, {%tile03}, 7 : i32) : !aie.objectfifo<memref<3000xi32>>
        aie.objectfifo.link [@mem_in] -> [@mem_out] ()
        %core02 = aie.core(%tile02) {
            %v11 = arith.constant 11 : i32
            %c0 = arith.constant 0 : index
            %subview = aie.objectfifo.acquire @mem_in (Consume, 1) : !aie.objectfifosubview<memref<3000xi32>>
            %subview_obj = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<3000xi32>> -> memref<3000xi32>
            memref.store %v11, %subview_obj[%c0] : memref<3000xi32>
            aie.end
        }
        %core03 = aie.core(%tile03) {
            %v11 = arith.constant 11 : i32
            %c0 = arith.constant 0 : index
            %subview = aie.objectfifo.acquire @mem_out (Consume, 3) : !aie.objectfifosubview<memref<3000xi32>>
            %subview_obj = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<3000xi32>> -> memref<3000xi32>
            memref.store %v11, %subview_obj[%c0] : memref<3000xi32>
            aie.end
        }
    }
}
