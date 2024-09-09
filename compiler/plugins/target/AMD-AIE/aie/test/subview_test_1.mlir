
// RUN: iree-opt --amdaie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @objfifo : memref<16xi32>
// CHECK-DAG:       %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK-DAG:       %[[TILE_1_3:.*]] = aie.tile(1, 3)
// CHECK-DAG:       %[[BUFFER_1_2:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "objfifo_prod_buff_0_0"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_1_2_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "objfifo_prod_buff_0_1"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_1_2_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "objfifo_prod_buff_0_2"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_1_2_2:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "objfifo_prod_buff_0_3"} : memref<16xi32>
// CHECK-DAG:       %[[LOCK_1_2:.*]] = aie.lock(%[[TILE_1_2]]) {init = 4 : i8, sym_name = "objfifo_prod_prod_lock_0"}
// CHECK-DAG:       %[[LOCK_1_2_3:.*]] = aie.lock(%[[TILE_1_2]]) {init = 0 : i8, sym_name = "objfifo_prod_cons_lock_0"}
// CHECK-DAG:       %[[BUFFER_1_3:.*]] = aie.buffer(%[[TILE_1_3]]) {sym_name = "objfifo_cons_buff_0_0"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_1_3_4:.*]] = aie.buffer(%[[TILE_1_3]]) {sym_name = "objfifo_cons_buff_0_1"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_1_3_5:.*]] = aie.buffer(%[[TILE_1_3]]) {sym_name = "objfifo_cons_buff_0_2"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_1_3_6:.*]] = aie.buffer(%[[TILE_1_3]]) {sym_name = "objfifo_cons_buff_0_3"} : memref<16xi32>
// CHECK-DAG:       %[[LOCK_1_3:.*]] = aie.lock(%[[TILE_1_3]]) {init = 4 : i8, sym_name = "objfifo_cons_prod_lock_0"}
// CHECK-DAG:       %[[LOCK_1_3_7:.*]] = aie.lock(%[[TILE_1_3]]) {init = 0 : i8, sym_name = "objfifo_cons_cons_lock_0"}
// CHECK-DAG:       aie.flow(%[[TILE_1_2]], DMA : 0, %[[TILE_1_3]], DMA : 0) {symbol = @objfifo}
// CHECK:           func.func @some_work(%[[ARG0:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[CORE_1_2:.*]] = aie.core(%[[TILE_1_2]]) {
// CHECK:             aie.use_lock(%[[LOCK_1_2]], AcquireGreaterEqual, 3)
// CHECK:             func.call @some_work(%[[BUFFER_1_2]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[BUFFER_1_2_0]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[BUFFER_1_2_1]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[LOCK_1_2]], AcquireGreaterEqual, 1)
// CHECK:             func.call @some_work(%[[BUFFER_1_2_2]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[LOCK_1_2_3]], Release, 3)
// CHECK:             aie.use_lock(%[[LOCK_1_2_3]], Release, 1)
// CHECK:             aie.use_lock(%[[LOCK_1_2]], AcquireGreaterEqual, 2)
// CHECK:             func.call @some_work(%[[BUFFER_1_2]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[BUFFER_1_2_0]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[LOCK_1_2]], AcquireGreaterEqual, 2)
// CHECK:             func.call @some_work(%[[BUFFER_1_2_1]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[BUFFER_1_2_2]]) : (memref<16xi32>) -> ()
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_1_2:.*]] = aie.mem(%[[TILE_1_2]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_1_2_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_1_2_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_0]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LOCK_1_2_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_1]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LOCK_1_2_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_2]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_1_3:.*]] = aie.mem(%[[TILE_1_3]]) {
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_1_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_3]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_3_7]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_1_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_3_4]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_3_7]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LOCK_1_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_3_5]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_3_7]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LOCK_1_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_3_6]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_3_7]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }
module @singleFifo {
    aie.device(npu1_4col) {
        %tile12 = aie.tile(1, 2)
        %tile13 = aie.tile(1, 3)
        aie.flow(%tile12, DMA : 0, %tile13, DMA : 0) {symbol = @objfifo}
        aie.objectfifo @objfifo (%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
        func.func @some_work(%line_in:memref<16xi32>) -> () {
            return
        }
        %core12 = aie.core(%tile12) {
            // this acquires 2 elements
            %subview0 = aie.objectfifo.acquire @objfifo (Produce, 3) : !aie.objectfifosubview<memref<16xi32>>
            %elem00 = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem01 = aie.objectfifo.subview.access %subview0[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem02 = aie.objectfifo.subview.access %subview0[2] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem00) : (memref<16xi32>) -> ()
            func.call @some_work(%elem01) : (memref<16xi32>) -> ()
            func.call @some_work(%elem02) : (memref<16xi32>) -> ()
            // this should only acquire one new element, previous two are still acquired
            %subview1 = aie.objectfifo.acquire @objfifo (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
            %elem10 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem10) : (memref<16xi32>) -> ()
            // one new acquire should take place
            aie.objectfifo.release @objfifo (Produce, 3)
            aie.objectfifo.release @objfifo (Produce, 1)
            %subview2 = aie.objectfifo.acquire @objfifo (Produce, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elem20 = aie.objectfifo.subview.access %subview2[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem21 = aie.objectfifo.subview.access %subview2[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem20) : (memref<16xi32>) -> ()
            func.call @some_work(%elem21) : (memref<16xi32>) -> ()
            // no new acquires should take place, elem30 should be third element of objFifo (with index 2)
            %subview3 = aie.objectfifo.acquire @objfifo (Produce, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elem30 = aie.objectfifo.subview.access %subview3[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem31 = aie.objectfifo.subview.access %subview3[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            //%elem32 = aie.subview.access %subview3[2] : !aie.subview<memref<16xi32>> -> memref<16xi32> // expected to fail if this line is uncommented
            func.call @some_work(%elem30) : (memref<16xi32>) -> ()
            func.call @some_work(%elem31) : (memref<16xi32>) -> ()
            aie.end
        }
    }
}
