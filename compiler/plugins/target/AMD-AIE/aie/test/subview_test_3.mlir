
// RUN: iree-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @of2 : memref<16xi32>
// CHECK:           memref.global "public" @of : memref<16xi32>
// CHECK:           %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:           %[[TILE_1_3:.*]] = aie.tile(1, 3)
// CHECK:           %[[OF2_BUFF_0:.*]] = aie.buffer(%[[TILE_1_3]]) {sym_name = "of2_buff_0"} : memref<16xi32>
// CHECK:           %[[OF2_BUFF_1:.*]] = aie.buffer(%[[TILE_1_3]]) {sym_name = "of2_buff_1"} : memref<16xi32>
// CHECK:           %[[OF2_BUFF_2:.*]] = aie.buffer(%[[TILE_1_3]]) {sym_name = "of2_buff_2"} : memref<16xi32>
// CHECK:           %[[OF2_PROD_LOCK:.*]] = aie.lock(%[[TILE_1_3]], 0) {init = 3 : i32, sym_name = "of2_prod_lock"}
// CHECK:           %[[OF2_CONS_LOCK:.*]] = aie.lock(%[[TILE_1_3]], 1) {init = 0 : i32, sym_name = "of2_cons_lock"}
// CHECK:           %[[OF_BUFF_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of_buff_0"} : memref<16xi32>
// CHECK:           %[[OF_BUFF_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of_buff_1"} : memref<16xi32>
// CHECK:           %[[OF_BUFF_2:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of_buff_2"} : memref<16xi32>
// CHECK:           %[[OF_BUFF_3:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of_buff_3"} : memref<16xi32>
// CHECK:           %[[OF_PROD_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 0) {init = 4 : i32, sym_name = "of_prod_lock"}
// CHECK:           %[[OF_CONS_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 1) {init = 0 : i32, sym_name = "of_cons_lock"}
// CHECK:           func.func @some_work(%[[ARG0:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[CORE_1_2:.*]] = aie.core(%[[TILE_1_2]]) {
// CHECK:             aie.use_lock(%[[OF_PROD_LOCK]], AcquireGreaterEqual, 2)
// CHECK:             func.call @some_work(%[[OF_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[OF_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[OF2_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             func.call @some_work(%[[OF2_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[OF_CONS_LOCK]], Release, 1)
// CHECK:             aie.use_lock(%[[OF_PROD_LOCK]], AcquireGreaterEqual, 2)
// CHECK:             func.call @some_work(%[[OF_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[OF_BUFF_2]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[OF_BUFF_3]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[OF_CONS_LOCK]], Release, 3)
// CHECK:             aie.use_lock(%[[OF2_PROD_LOCK]], Release, 1)
// CHECK:             aie.use_lock(%[[OF2_CONS_LOCK]], AcquireGreaterEqual, 2)
// CHECK:             func.call @some_work(%[[OF2_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[OF2_BUFF_2]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[OF2_PROD_LOCK]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_1_3:.*]] = aie.core(%[[TILE_1_3]]) {
// CHECK:             aie.use_lock(%[[OF_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             func.call @some_work(%[[OF_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[OF2_PROD_LOCK]], AcquireGreaterEqual, 2)
// CHECK:             func.call @some_work(%[[OF2_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[OF2_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[OF2_CONS_LOCK]], Release, 2)
// CHECK:             aie.use_lock(%[[OF_PROD_LOCK]], Release, 1)
// CHECK:             aie.use_lock(%[[OF_CONS_LOCK]], AcquireGreaterEqual, 2)
// CHECK:             func.call @some_work(%[[OF_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[OF_BUFF_2]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[OF_PROD_LOCK]], Release, 2)
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @multiCoreMixedFifo {
    aie.device(npu1_4col) {
        %tile12 = aie.tile(1, 2)
        %tile13 = aie.tile(1, 3)
        aie.objectfifo @of (%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @of2 (%tile13, {%tile12}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
        func.func @some_work(%line_in:memref<16xi32>) -> () {
            return
        }
        %core11 = aie.core(%tile12) {
            %subview0 = aie.objectfifo.acquire @of (Produce, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elem00 = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem01 = aie.objectfifo.subview.access %subview0[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem00) : (memref<16xi32>) -> ()
            func.call @some_work(%elem01) : (memref<16xi32>) -> ()
            %subview02 = aie.objectfifo.acquire @of2 (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
            %elem002 = aie.objectfifo.subview.access %subview02[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem002) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of (Produce, 1)
            %subview1 = aie.objectfifo.acquire @of (Produce, 3) : !aie.objectfifosubview<memref<16xi32>>
            %elem10 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem11 = aie.objectfifo.subview.access %subview1[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem12 = aie.objectfifo.subview.access %subview1[2] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem10) : (memref<16xi32>) -> ()
            func.call @some_work(%elem11) : (memref<16xi32>) -> ()
            func.call @some_work(%elem12) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of (Produce, 3)
            aie.objectfifo.release @of2 (Consume, 1)
            %subview12 = aie.objectfifo.acquire @of2 (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elem102 = aie.objectfifo.subview.access %subview12[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem112 = aie.objectfifo.subview.access %subview12[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem102) : (memref<16xi32>) -> ()
            func.call @some_work(%elem112) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of2 (Consume, 1)
            aie.end
        }
        %core12 = aie.core(%tile13) {
            %subview0 = aie.objectfifo.acquire @of (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
            %elem00 = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem00) : (memref<16xi32>) -> ()
            %subview02 = aie.objectfifo.acquire @of2 (Produce, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elem002 = aie.objectfifo.subview.access %subview02[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem012 = aie.objectfifo.subview.access %subview02[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem002) : (memref<16xi32>) -> ()
            func.call @some_work(%elem012) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of2 (Produce, 2)
            aie.objectfifo.release @of (Consume, 1)
            %subview1 = aie.objectfifo.acquire @of (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elem10 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem11 = aie.objectfifo.subview.access %subview1[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem10) : (memref<16xi32>) -> ()
            func.call @some_work(%elem11) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of (Consume, 2)
            aie.end
        }
    }
}
