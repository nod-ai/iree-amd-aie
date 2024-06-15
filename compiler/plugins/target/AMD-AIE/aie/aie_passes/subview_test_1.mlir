
// RUN: iree-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           memref.global "public" @objfifo : memref<16xi32>
// CHECK:           %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:           %[[TILE_1_3:.*]] = aie.tile(1, 3)
// CHECK:           %[[OBJFIFO_BUFF_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "objfifo_buff_0"} : memref<16xi32>
// CHECK:           %[[OBJFIFO_BUFF_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "objfifo_buff_1"} : memref<16xi32>
// CHECK:           %[[OBJFIFO_BUFF_2:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "objfifo_buff_2"} : memref<16xi32>
// CHECK:           %[[OBJFIFO_BUFF_3:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "objfifo_buff_3"} : memref<16xi32>
// CHECK:           %[[OBJFIFO_LOCK_0:.*]] = aie.lock(%[[TILE_1_2]], 0) {init = 0 : i32, sym_name = "objfifo_lock_0"}
// CHECK:           %[[OBJFIFO_LOCK_1:.*]] = aie.lock(%[[TILE_1_2]], 1) {init = 0 : i32, sym_name = "objfifo_lock_1"}
// CHECK:           %[[OBJFIFO_LOCK_2:.*]] = aie.lock(%[[TILE_1_2]], 2) {init = 0 : i32, sym_name = "objfifo_lock_2"}
// CHECK:           %[[OBJFIFO_LOCK_3:.*]] = aie.lock(%[[TILE_1_2]], 3) {init = 0 : i32, sym_name = "objfifo_lock_3"}
// CHECK:           func.func @some_work(%[[ARG0:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[CORE_1_2:.*]] = aie.core(%[[TILE_1_2]]) {
// CHECK:             aie.use_lock(%[[OBJFIFO_LOCK_0]], Acquire, 0)
// CHECK:             aie.use_lock(%[[OBJFIFO_LOCK_1]], Acquire, 0)
// CHECK:             func.call @some_work(%[[OBJFIFO_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[OBJFIFO_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[OBJFIFO_LOCK_2]], Acquire, 0)
// CHECK:             func.call @some_work(%[[OBJFIFO_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[OBJFIFO_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[OBJFIFO_BUFF_2]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[OBJFIFO_LOCK_0]], Release, 1)
// CHECK:             aie.use_lock(%[[OBJFIFO_LOCK_1]], Release, 1)
// CHECK:             aie.use_lock(%[[OBJFIFO_LOCK_3]], Acquire, 0)
// CHECK:             func.call @some_work(%[[OBJFIFO_BUFF_2]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[OBJFIFO_BUFF_3]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[OBJFIFO_BUFF_2]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[OBJFIFO_BUFF_3]]) : (memref<16xi32>) -> ()
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @singleFifo {
    aie.device(xcvc1902) {
        %tile12 = aie.tile(1, 2)
        %tile13 = aie.tile(1, 3)
        aie.objectfifo @objfifo (%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
        func.func @some_work(%line_in:memref<16xi32>) -> () {
            return
        }
        %core12 = aie.core(%tile12) {
            // this acquires 2 elements
            %subview0 = aie.objectfifo.acquire @objfifo (Produce, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elem00 = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem01 = aie.objectfifo.subview.access %subview0[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem00) : (memref<16xi32>) -> ()
            func.call @some_work(%elem01) : (memref<16xi32>) -> ()
            // this should only acquire one new element, previous two are still acquired
            %subview1 = aie.objectfifo.acquire @objfifo (Produce, 3) : !aie.objectfifosubview<memref<16xi32>>
            %elem10 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem11 = aie.objectfifo.subview.access %subview1[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem12 = aie.objectfifo.subview.access %subview1[2] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem10) : (memref<16xi32>) -> ()
            func.call @some_work(%elem11) : (memref<16xi32>) -> ()
            func.call @some_work(%elem12) : (memref<16xi32>) -> ()
            // one new acquire should take place
            aie.objectfifo.release @objfifo (Produce, 1)
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
