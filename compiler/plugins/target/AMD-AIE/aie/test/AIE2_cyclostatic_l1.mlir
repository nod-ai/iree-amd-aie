
// RUN: iree-opt --amdaie-objectFifo-stateful-transform %s | FileCheck %s


// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           memref.global "public" @fifo : memref<i32>
// CHECK-DAG:       %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK-DAG:       %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK-DAG:       %[[BUFFER_2_2:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "fifo_prod_buff_0_0"} : memref<i32>
// CHECK-DAG:       %[[BUFFER_2_2_0:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "fifo_prod_buff_0_1"} : memref<i32>
// CHECK-DAG:       %[[BUFFER_2_2_1:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "fifo_prod_buff_0_2"} : memref<i32>
// CHECK-DAG:       %[[BUFFER_2_2_2:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "fifo_prod_buff_0_3"} : memref<i32>
// CHECK-DAG:       %[[LOCK_2_2:.*]] = aie.lock(%[[TILE_2_2]]) {init = 4 : i8, sym_name = "fifo_prod_prod_lock_0"}
// CHECK-DAG:       %[[LOCK_2_2_3:.*]] = aie.lock(%[[TILE_2_2]]) {init = 0 : i8, sym_name = "fifo_prod_cons_lock_0"}
// CHECK-DAG:       %[[BUFFER_2_3:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "fifo_cons_buff_0_0"} : memref<i32>
// CHECK-DAG:       %[[BUFFER_2_3_4:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "fifo_cons_buff_0_1"} : memref<i32>
// CHECK-DAG:       %[[BUFFER_2_3_5:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "fifo_cons_buff_0_2"} : memref<i32>
// CHECK-DAG:       %[[BUFFER_2_3_6:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "fifo_cons_buff_0_3"} : memref<i32>
// CHECK-DAG:       %[[LOCK_2_3:.*]] = aie.lock(%[[TILE_2_3]]) {init = 4 : i8, sym_name = "fifo_cons_prod_lock_0"}
// CHECK-DAG:       %[[LOCK_2_3_7:.*]] = aie.lock(%[[TILE_2_3]]) {init = 0 : i8, sym_name = "fifo_cons_cons_lock_0"}
// CHECK-DAG:       %[[BUFFER_2_3_8:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "buf23"} : memref<4xi32>
// CHECK-DAG:       aie.flow(%[[TILE_2_2]], DMA : 0, %[[TILE_2_3]], DMA : 0) {symbol = @fifo}
// CHECK:           %[[CORE_2_2:.*]] = aie.core(%[[TILE_2_2]]) {
// CHECK:             %[[C55_I32:.*]] = arith.constant 55 : i32
// CHECK:             %[[C66_I32:.*]] = arith.constant 66 : i32
// CHECK:             %[[C77_I32:.*]] = arith.constant 77 : i32
// CHECK:             %[[C88_I32:.*]] = arith.constant 88 : i32
// CHECK:             aie.use_lock(%[[LOCK_2_2]], AcquireGreaterEqual, 1)
// CHECK:             memref.store %[[C55_I32]], %[[BUFFER_2_2]][] : memref<i32>
// CHECK:             aie.use_lock(%[[LOCK_2_2_3]], Release, 1)
// CHECK:             aie.use_lock(%[[LOCK_2_2]], AcquireGreaterEqual, 1)
// CHECK:             memref.store %[[C66_I32]], %[[BUFFER_2_2_0]][] : memref<i32>
// CHECK:             aie.use_lock(%[[LOCK_2_2_3]], Release, 1)
// CHECK:             aie.use_lock(%[[LOCK_2_2]], AcquireGreaterEqual, 1)
// CHECK:             memref.store %[[C77_I32]], %[[BUFFER_2_2_1]][] : memref<i32>
// CHECK:             aie.use_lock(%[[LOCK_2_2_3]], Release, 1)
// CHECK:             aie.use_lock(%[[LOCK_2_2]], AcquireGreaterEqual, 1)
// CHECK:             memref.store %[[C88_I32]], %[[BUFFER_2_2_2]][] : memref<i32>
// CHECK:             aie.use_lock(%[[LOCK_2_2_3]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_2_3:.*]] = aie.core(%[[TILE_2_3]]) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C2:.*]] = arith.constant 2 : index
// CHECK:             %[[C3:.*]] = arith.constant 3 : index
// CHECK:             aie.use_lock(%[[LOCK_2_3_7]], AcquireGreaterEqual, 1)
// CHECK:             %[[VAL_0:.*]] = memref.load %[[BUFFER_2_3]][] : memref<i32>
// CHECK:             memref.store %[[VAL_0]], %[[BUFFER_2_3_8]]{{\[}}%[[C0]]] : memref<4xi32>
// CHECK:             aie.use_lock(%[[LOCK_2_3]], Release, 1)
// CHECK:             aie.use_lock(%[[LOCK_2_3_7]], AcquireGreaterEqual, 2)
// CHECK:             %[[VAL_1:.*]] = memref.load %[[BUFFER_2_3_4]][] : memref<i32>
// CHECK:             %[[VAL_2:.*]] = memref.load %[[BUFFER_2_3_5]][] : memref<i32>
// CHECK:             memref.store %[[VAL_1]], %[[BUFFER_2_3_8]]{{\[}}%[[C1]]] : memref<4xi32>
// CHECK:             memref.store %[[VAL_2]], %[[BUFFER_2_3_8]]{{\[}}%[[C2]]] : memref<4xi32>
// CHECK:             aie.use_lock(%[[LOCK_2_3]], Release, 2)
// CHECK:             aie.use_lock(%[[LOCK_2_3_7]], AcquireGreaterEqual, 1)
// CHECK:             %[[VAL_3:.*]] = memref.load %[[BUFFER_2_3_6]][] : memref<i32>
// CHECK:             memref.store %[[VAL_3]], %[[BUFFER_2_3_8]]{{\[}}%[[C3]]] : memref<4xi32>
// CHECK:             aie.use_lock(%[[LOCK_2_3]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_2_2:.*]] = aie.mem(%[[TILE_2_2]]) {
// CHECK:             %[[VAL_4:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_2_2_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_2_2]] : memref<i32>) {len = 1 : i32}
// CHECK:             aie.use_lock(%[[LOCK_2_2]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_2_2_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_2_2_0]] : memref<i32>) {len = 1 : i32}
// CHECK:             aie.use_lock(%[[LOCK_2_2]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LOCK_2_2_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_2_2_1]] : memref<i32>) {len = 1 : i32}
// CHECK:             aie.use_lock(%[[LOCK_2_2]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LOCK_2_2_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_2_2_2]] : memref<i32>) {len = 1 : i32}
// CHECK:             aie.use_lock(%[[LOCK_2_2]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_2_3:.*]] = aie.mem(%[[TILE_2_3]]) {
// CHECK:             %[[VAL_5:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_2_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_2_3]] : memref<i32>) {len = 1 : i32}
// CHECK:             aie.use_lock(%[[LOCK_2_3_7]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_2_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_2_3_4]] : memref<i32>) {len = 1 : i32}
// CHECK:             aie.use_lock(%[[LOCK_2_3_7]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LOCK_2_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_2_3_5]] : memref<i32>) {len = 1 : i32}
// CHECK:             aie.use_lock(%[[LOCK_2_3_7]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LOCK_2_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_2_3_6]] : memref<i32>) {len = 1 : i32}
// CHECK:             aie.use_lock(%[[LOCK_2_3_7]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }
module @aie2_cyclostatic_l1 {
    aie.device(xcve2302) {
        %tile22 = aie.tile(2, 2)  // producer tile
        %tile23 = aie.tile(2, 3)  // consumer tile
        %buf23  = aie.buffer(%tile23) {sym_name = "buf23"} : memref<4xi32>
        aie.flow(%tile22, DMA : 0, %tile23, DMA : 0) {symbol = @fifo}
        // ObjectFifo that can hold 4 memref<i32>s, populated by tile22 and
        // consumed by tile23
        aie.objectfifo @fifo (%tile22, {%tile23}, 4 : i32) : !aie.objectfifo<memref<i32>>
        // Producer core
        %core22 = aie.core(%tile22) {
            %c55 = arith.constant 55 : i32
            %c66 = arith.constant 66 : i32
            %c77 = arith.constant 77 : i32
            %c88 = arith.constant 88 : i32
            // Push 55
            %subview0 = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i32>>
            %subview0_obj = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            memref.store %c55, %subview0_obj[] : memref<i32>
            aie.objectfifo.release @fifo (Produce, 1)
            // Push 66
            %subview1 = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i32>>
            %subview1_obj = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            memref.store %c66, %subview1_obj[] : memref<i32>
            aie.objectfifo.release @fifo (Produce, 1)
            // Push 77
            %subview2 = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i32>>
            %subview2_obj = aie.objectfifo.subview.access %subview2[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            memref.store %c77, %subview2_obj[] : memref<i32>
            aie.objectfifo.release @fifo (Produce, 1)
            // Push 88
            %subview3 = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i32>>
            %subview3_obj = aie.objectfifo.subview.access %subview3[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            memref.store %c88, %subview3_obj[] : memref<i32>
            aie.objectfifo.release @fifo (Produce, 1)
            aie.end
        }
        // Consumer core
        %core23 = aie.core(%tile23) {
            // Consumer pattern: {1, 2, 1}
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i2 = arith.constant 2 : index
            %i3 = arith.constant 3 : index
            // Pop 1 object off queue
            %subview0 = aie.objectfifo.acquire @fifo (Consume, 1) : !aie.objectfifosubview<memref<i32>>
            %subview0_obj = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            %v55 = memref.load %subview0_obj[] : memref<i32>
            memref.store %v55, %buf23[%i0] : memref<4xi32>
            aie.objectfifo.release @fifo (Consume, 1)
            // Pop 2 objects off queue
            %subview1 = aie.objectfifo.acquire @fifo (Consume, 2) : !aie.objectfifosubview<memref<i32>>
            %subview1_obj0 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            %subview1_obj1 = aie.objectfifo.subview.access %subview1[1] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            %v66 = memref.load %subview1_obj0[] : memref<i32>
            %v77 = memref.load %subview1_obj1[] : memref<i32>
            memref.store %v66, %buf23[%i1] : memref<4xi32>
            memref.store %v77, %buf23[%i2] : memref<4xi32>
            aie.objectfifo.release @fifo (Consume, 2)
            // Pop 1 object off queue
            %subview2 = aie.objectfifo.acquire @fifo (Consume, 1) : !aie.objectfifosubview<memref<i32>>
            %subview2_obj = aie.objectfifo.subview.access %subview2[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            %v88 = memref.load %subview2_obj[] : memref<i32>
            memref.store %v88, %buf23[%i3] : memref<4xi32>
            aie.objectfifo.release @fifo (Consume, 1)
            aie.end
        }
    }
}
