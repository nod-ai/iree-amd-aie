
// RUN: iree-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           memref.global "public" @of : memref<16xi32>
// CHECK:           %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:           %[[OF_BUFF_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of_buff_0"} : memref<16xi32>
// CHECK:           %[[OF_BUFF_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of_buff_1"} : memref<16xi32>
// CHECK:           %[[OF_BUFF_2:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of_buff_2"} : memref<16xi32>
// CHECK:           %[[OF_PROD_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 0) {init = 3 : i32, sym_name = "of_prod_lock"}
// CHECK:           %[[OF_CONS_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 1) {init = 0 : i32, sym_name = "of_cons_lock"}
// CHECK:           func.func @some_work(%[[ARG0:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[CORE_1_2:.*]] = aie.core(%[[TILE_1_2]]) {
// CHECK:             aie.use_lock(%[[OF_PROD_LOCK]], AcquireGreaterEqual, 2)
// CHECK:             func.call @some_work(%[[OF_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[OF_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[OF_CONS_LOCK]], Release, 1)
// CHECK:             aie.use_lock(%[[OF_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             func.call @some_work(%[[OF_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[OF_PROD_LOCK]], Release, 1)
// CHECK:             func.call @some_work(%[[OF_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[OF_CONS_LOCK]], Release, 1)
// CHECK:             aie.use_lock(%[[OF_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             func.call @some_work(%[[OF_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[OF_PROD_LOCK]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @same_core {
    aie.device(xcve2302) {
        %tile12 = aie.tile(1, 2)
        aie.objectfifo @of (%tile12, {%tile12}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
        func.func @some_work(%line_in:memref<16xi32>) -> () {
            return
        }
        %core12 = aie.core(%tile12) {
            // this acquires 2 elements
            %subview0 = aie.objectfifo.acquire @of (Produce, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elem00 = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem01 = aie.objectfifo.subview.access %subview0[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem00) : (memref<16xi32>) -> ()
            func.call @some_work(%elem01) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of (Produce, 1)
            %subview1 = aie.objectfifo.acquire @of (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
            %elem10 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem10) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of (Consume, 1)
            %subview2 = aie.objectfifo.acquire @of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
            %elem20 = aie.objectfifo.subview.access %subview2[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem20) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of (Produce, 1)
            %subview3 = aie.objectfifo.acquire @of (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
            %elem30 = aie.objectfifo.subview.access %subview3[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem30) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of (Consume, 1)
            aie.end
        }
    }
}
