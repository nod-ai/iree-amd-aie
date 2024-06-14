
// RUN: iree-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @fifo1_cons : memref<1xi32>
// CHECK:           memref.global "public" @fifo1 : memref<1xi32>
// CHECK:           memref.global "public" @fifo0_cons : memref<1xi32>
// CHECK:           memref.global "public" @fifo0 : memref<1xi32>
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[TILE_2_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[FIFO1_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "fifo1_cons_buff_0"} : memref<1xi32>
// CHECK:           %[[FIFO1_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "fifo1_cons_buff_1"} : memref<1xi32>
// CHECK:           %[[FIFO1_CONS_BUFF_2:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "fifo1_cons_buff_2"} : memref<1xi32>
// CHECK:           %[[FIFO1_CONS_BUFF_3:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "fifo1_cons_buff_3"} : memref<1xi32>
// CHECK:           %[[FIFO1_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_3_3]], 0) {init = 4 : i32, sym_name = "fifo1_cons_prod_lock"}
// CHECK:           %[[FIFO1_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_3_3]], 1) {init = 0 : i32, sym_name = "fifo1_cons_cons_lock"}
// CHECK:           %[[FIFO0_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_2_1]]) {sym_name = "fifo0_cons_buff_0"} : memref<1xi32>
// CHECK:           %[[FIFO0_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_2_1]]) {sym_name = "fifo0_cons_buff_1"} : memref<1xi32>
// CHECK:           %[[FIFO0_CONS_BUFF_2:.*]] = aie.buffer(%[[TILE_2_1]]) {sym_name = "fifo0_cons_buff_2"} : memref<1xi32>
// CHECK:           %[[FIFO0_CONS_BUFF_3:.*]] = aie.buffer(%[[TILE_2_1]]) {sym_name = "fifo0_cons_buff_3"} : memref<1xi32>
// CHECK:           %[[FIFO0_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_1]], 0) {init = 4 : i32, sym_name = "fifo0_cons_prod_lock"}
// CHECK:           %[[FIFO0_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_1]], 1) {init = 0 : i32, sym_name = "fifo0_cons_cons_lock"}
// CHECK:           %[[FIFO0_BUFF_0:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "fifo0_buff_0"} : memref<1xi32>
// CHECK:           %[[FIFO0_BUFF_1:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "fifo0_buff_1"} : memref<1xi32>
// CHECK:           %[[FIFO0_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 0) {init = 2 : i32, sym_name = "fifo0_prod_lock"}
// CHECK:           %[[FIFO0_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 1) {init = 0 : i32, sym_name = "fifo0_cons_lock"}
// CHECK:           %[[BUF33:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "buf33"} : memref<1xi32>
// CHECK:           aie.flow(%[[TILE_2_2]], DMA : 0, %[[TILE_2_1]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_2_1]], DMA : 0, %[[TILE_3_3]], DMA : 0)
// CHECK:           %[[CORE_2_2:.*]] = aie.core(%[[TILE_2_2]]) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C55_I32:.*]] = arith.constant 55 : i32
// CHECK:             %[[C66_I32:.*]] = arith.constant 66 : i32
// CHECK:             %[[C77_I32:.*]] = arith.constant 77 : i32
// CHECK:             %[[C88_I32:.*]] = arith.constant 88 : i32
// CHECK:             aie.use_lock(%[[FIFO0_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             memref.store %[[C55_I32]], %[[FIFO0_BUFF_0]]{{\[}}%[[C0]]] : memref<1xi32>
// CHECK:             aie.use_lock(%[[FIFO0_CONS_LOCK]], Release, 1)
// CHECK:             aie.use_lock(%[[FIFO0_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             memref.store %[[C66_I32]], %[[FIFO0_BUFF_1]]{{\[}}%[[C0]]] : memref<1xi32>
// CHECK:             aie.use_lock(%[[FIFO0_CONS_LOCK]], Release, 1)
// CHECK:             aie.use_lock(%[[FIFO0_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             memref.store %[[C77_I32]], %[[FIFO0_BUFF_0]]{{\[}}%[[C0]]] : memref<1xi32>
// CHECK:             aie.use_lock(%[[FIFO0_CONS_LOCK]], Release, 1)
// CHECK:             aie.use_lock(%[[FIFO0_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             memref.store %[[C88_I32]], %[[FIFO0_BUFF_1]]{{\[}}%[[C0]]] : memref<1xi32>
// CHECK:             aie.use_lock(%[[FIFO0_CONS_LOCK]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_3_3:.*]] = aie.core(%[[TILE_3_3]]) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C2:.*]] = arith.constant 2 : index
// CHECK:             %[[C3:.*]] = arith.constant 3 : index
// CHECK:             aie.use_lock(%[[FIFO1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             %[[VAL_0:.*]] = memref.load %[[FIFO1_CONS_BUFF_0]]{{\[}}%[[C0]]] : memref<1xi32>
// CHECK:             memref.store %[[VAL_0]], %[[BUF33]]{{\[}}%[[C0]]] : memref<1xi32>
// CHECK:             aie.use_lock(%[[FIFO1_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.use_lock(%[[FIFO1_CONS_CONS_LOCK]], AcquireGreaterEqual, 2)
// CHECK:             %[[VAL_1:.*]] = memref.load %[[FIFO1_CONS_BUFF_1]]{{\[}}%[[C0]]] : memref<1xi32>
// CHECK:             %[[VAL_2:.*]] = memref.load %[[FIFO1_CONS_BUFF_2]]{{\[}}%[[C0]]] : memref<1xi32>
// CHECK:             memref.store %[[VAL_1]], %[[BUF33]]{{\[}}%[[C1]]] : memref<1xi32>
// CHECK:             memref.store %[[VAL_2]], %[[BUF33]]{{\[}}%[[C2]]] : memref<1xi32>
// CHECK:             aie.use_lock(%[[FIFO1_CONS_PROD_LOCK]], Release, 2)
// CHECK:             aie.use_lock(%[[FIFO1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             %[[VAL_3:.*]] = memref.load %[[FIFO1_CONS_BUFF_3]]{{\[}}%[[C0]]] : memref<1xi32>
// CHECK:             memref.store %[[VAL_3]], %[[BUF33]]{{\[}}%[[C3]]] : memref<1xi32>
// CHECK:             aie.use_lock(%[[FIFO1_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_2_2:.*]] = aie.mem(%[[TILE_2_2]]) {
// CHECK:             %[[VAL_4:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[FIFO0_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[FIFO0_BUFF_0]] : memref<1xi32>, 0, 1)
// CHECK:             aie.use_lock(%[[FIFO0_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[FIFO0_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[FIFO0_BUFF_1]] : memref<1xi32>, 0, 1)
// CHECK:             aie.use_lock(%[[FIFO0_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEMTILE_DMA_2_1:.*]] = aie.memtile_dma(%[[TILE_2_1]]) {
// CHECK:             %[[VAL_5:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[FIFO0_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[FIFO0_CONS_BUFF_0]] : memref<1xi32>, 0, 1)
// CHECK:             aie.use_lock(%[[FIFO0_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[FIFO0_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[FIFO0_CONS_BUFF_1]] : memref<1xi32>, 0, 1)
// CHECK:             aie.use_lock(%[[FIFO0_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[FIFO0_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[FIFO0_CONS_BUFF_2]] : memref<1xi32>, 0, 1)
// CHECK:             aie.use_lock(%[[FIFO0_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[FIFO0_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[FIFO0_CONS_BUFF_3]] : memref<1xi32>, 0, 1)
// CHECK:             aie.use_lock(%[[FIFO0_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             %[[VAL_6:.*]] = aie.dma_start(MM2S, 0, ^bb6, ^bb10)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[FIFO0_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[FIFO0_CONS_BUFF_0]] : memref<1xi32>, 0, 1)
// CHECK:             aie.use_lock(%[[FIFO0_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[FIFO0_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[FIFO0_CONS_BUFF_1]] : memref<1xi32>, 0, 1)
// CHECK:             aie.use_lock(%[[FIFO0_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(%[[FIFO0_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[FIFO0_CONS_BUFF_2]] : memref<1xi32>, 0, 1)
// CHECK:             aie.use_lock(%[[FIFO0_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb9
// CHECK:           ^bb9:
// CHECK:             aie.use_lock(%[[FIFO0_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[FIFO0_CONS_BUFF_3]] : memref<1xi32>, 0, 1)
// CHECK:             aie.use_lock(%[[FIFO0_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb10:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_3_3:.*]] = aie.mem(%[[TILE_3_3]]) {
// CHECK:             %[[VAL_7:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[FIFO1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[FIFO1_CONS_BUFF_0]] : memref<1xi32>, 0, 1)
// CHECK:             aie.use_lock(%[[FIFO1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[FIFO1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[FIFO1_CONS_BUFF_1]] : memref<1xi32>, 0, 1)
// CHECK:             aie.use_lock(%[[FIFO1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[FIFO1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[FIFO1_CONS_BUFF_2]] : memref<1xi32>, 0, 1)
// CHECK:             aie.use_lock(%[[FIFO1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[FIFO1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[FIFO1_CONS_BUFF_3]] : memref<1xi32>, 0, 1)
// CHECK:             aie.use_lock(%[[FIFO1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @aie2_cyclostatic_l2 {
    aie.device(npu1_4col) {
        %tile22 = aie.tile(2, 2)  // producer tile
        %memtile = aie.tile(2, 1) // mem tile
        %tile33 = aie.tile(3, 3)  // consumer tile
        %buf33  = aie.buffer(%tile33) {sym_name = "buf33"} : memref<1xi32>
        // ObjectFifo that can hold 4 memref<1xi32>s, populated by tile22 and
        // consumed by tile23
        aie.objectfifo @fifo0 (%tile22, {%memtile}, 4 : i32) : !aie.objectfifo<memref<1xi32>>
        aie.objectfifo @fifo1 (%memtile, {%tile33}, [4, 4]) : !aie.objectfifo<memref<1xi32>>
        aie.objectfifo.link [@fifo0] -> [@fifo1] ()
        // Producer core
        %core22 = aie.core(%tile22) {
            %i0 = arith.constant 0 : index
            %c55 = arith.constant 55 : i32
            %c66 = arith.constant 66 : i32
            %c77 = arith.constant 77 : i32
            %c88 = arith.constant 88 : i32
            // Push 55
            %subview0 = aie.objectfifo.acquire @fifo0 (Produce, 1) : !aie.objectfifosubview<memref<1xi32>>
            %subview0_obj = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
            memref.store %c55, %subview0_obj[%i0] : memref<1xi32>
            aie.objectfifo.release @fifo0 (Produce, 1)
            // Push 66
            %subview1 = aie.objectfifo.acquire @fifo0 (Produce, 1) : !aie.objectfifosubview<memref<1xi32>>
            %subview1_obj = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
            memref.store %c66, %subview1_obj[%i0] : memref<1xi32>
            aie.objectfifo.release @fifo0 (Produce, 1)
            // Push 77
            %subview2 = aie.objectfifo.acquire @fifo0 (Produce, 1) : !aie.objectfifosubview<memref<1xi32>>
            %subview2_obj = aie.objectfifo.subview.access %subview2[0] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
            memref.store %c77, %subview2_obj[%i0] : memref<1xi32>
            aie.objectfifo.release @fifo0 (Produce, 1)
            // Push 88
            %subview3 = aie.objectfifo.acquire @fifo0 (Produce, 1) : !aie.objectfifosubview<memref<1xi32>>
            %subview3_obj = aie.objectfifo.subview.access %subview3[0] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
            memref.store %c88, %subview3_obj[%i0] : memref<1xi32>
            aie.objectfifo.release @fifo0 (Produce, 1)
            aie.end
        }
        // Consumer core
        %core28 = aie.core(%tile33) {
            // Consumer pattern: {1, 2, 1}
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i2 = arith.constant 2 : index
            %i3 = arith.constant 3 : index
            // Pop 1 object off queue
            %subview0 = aie.objectfifo.acquire @fifo1 (Consume, 1) : !aie.objectfifosubview<memref<1xi32>>
            %subview0_obj = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
            %v55 = memref.load %subview0_obj[%i0] : memref<1xi32>
            memref.store %v55, %buf33[%i0] : memref<1xi32>
            aie.objectfifo.release @fifo1 (Consume, 1)
            // Pop 2 objects off queue
            %subview1 = aie.objectfifo.acquire @fifo1 (Consume, 2) : !aie.objectfifosubview<memref<1xi32>>
            %subview1_obj0 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
            %subview1_obj1 = aie.objectfifo.subview.access %subview1[1] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
            %v66 = memref.load %subview1_obj0[%i0] : memref<1xi32>
            %v77 = memref.load %subview1_obj1[%i0] : memref<1xi32>
            memref.store %v66, %buf33[%i1] : memref<1xi32>
            memref.store %v77, %buf33[%i2] : memref<1xi32>
            aie.objectfifo.release @fifo1 (Consume, 2)
            // Pop 1 object off queue
            %subview2 = aie.objectfifo.acquire @fifo1 (Consume, 1) : !aie.objectfifosubview<memref<1xi32>>
            %subview2_obj = aie.objectfifo.subview.access %subview2[0] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
            %v88 = memref.load %subview2_obj[%i0] : memref<1xi32>
            memref.store %v88, %buf33[%i3] : memref<1xi32>
            aie.objectfifo.release @fifo1 (Consume, 1)
            aie.end
        }
    }
}
