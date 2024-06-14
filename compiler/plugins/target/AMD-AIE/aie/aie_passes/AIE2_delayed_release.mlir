
// RUN: iree-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @fifo : memref<i32>
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK:           %[[FIFO_BUFF_0:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "fifo_buff_0"} : memref<i32>
// CHECK:           %[[FIFO_BUFF_1:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "fifo_buff_1"} : memref<i32>
// CHECK:           %[[FIFO_BUFF_2:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "fifo_buff_2"} : memref<i32>
// CHECK:           %[[FIFO_BUFF_3:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "fifo_buff_3"} : memref<i32>
// CHECK:           %[[FIFO_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 0) {init = 4 : i32, sym_name = "fifo_prod_lock"}
// CHECK:           %[[FIFO_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 1) {init = 0 : i32, sym_name = "fifo_cons_lock"}
// CHECK:           %[[BUF23:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "buf23"} : memref<4xi32>
// CHECK:           %[[CORE_2_2:.*]] = aie.core(%[[TILE_2_2]]) {
// CHECK:             %[[C99_I32:.*]] = arith.constant 99 : i32
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C4:.*]] = arith.constant 4 : index
// CHECK:             %[[C4_0:.*]] = arith.constant 4 : index
// CHECK:             aie.use_lock(%[[FIFO_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             memref.store %[[C99_I32]], %[[FIFO_BUFF_0]][] : memref<i32>
// CHECK:             aie.use_lock(%[[FIFO_CONS_LOCK]], Release, 1)
// CHECK:             aie.use_lock(%[[FIFO_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             memref.store %[[C99_I32]], %[[FIFO_BUFF_1]][] : memref<i32>
// CHECK:             aie.use_lock(%[[FIFO_CONS_LOCK]], Release, 1)
// CHECK:             aie.use_lock(%[[FIFO_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             memref.store %[[C99_I32]], %[[FIFO_BUFF_2]][] : memref<i32>
// CHECK:             aie.use_lock(%[[FIFO_CONS_LOCK]], Release, 1)
// CHECK:             aie.use_lock(%[[FIFO_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             memref.store %[[C99_I32]], %[[FIFO_BUFF_3]][] : memref<i32>
// CHECK:             aie.use_lock(%[[FIFO_CONS_LOCK]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_2_3:.*]] = aie.core(%[[TILE_2_3]]) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C2:.*]] = arith.constant 2 : index
// CHECK:             %[[C3:.*]] = arith.constant 3 : index
// CHECK:             aie.use_lock(%[[FIFO_CONS_LOCK]], AcquireGreaterEqual, 2)
// CHECK:             %[[VAL_0:.*]] = memref.load %[[FIFO_BUFF_0]][] : memref<i32>
// CHECK:             memref.store %[[VAL_0]], %[[BUF23]]{{\[}}%[[C0]]] : memref<4xi32>
// CHECK:             %[[VAL_1:.*]] = memref.load %[[FIFO_BUFF_0]][] : memref<i32>
// CHECK:             memref.store %[[VAL_1]], %[[BUF23]]{{\[}}%[[C1]]] : memref<4xi32>
// CHECK:             aie.use_lock(%[[FIFO_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             %[[VAL_2:.*]] = memref.load %[[FIFO_BUFF_0]][] : memref<i32>
// CHECK:             memref.store %[[VAL_2]], %[[BUF23]]{{\[}}%[[C2]]] : memref<4xi32>
// CHECK:             %[[VAL_3:.*]] = memref.load %[[FIFO_BUFF_0]][] : memref<i32>
// CHECK:             memref.store %[[VAL_3]], %[[BUF23]]{{\[}}%[[C3]]] : memref<4xi32>
// CHECK:             aie.use_lock(%[[FIFO_PROD_LOCK]], Release, 3)
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @AIE2_delayed_release {
    aie.device(npu1_4col) {
        %tile22 = aie.tile(2, 2)
        %tile23 = aie.tile(2, 3)
        %buf23 = aie.buffer(%tile23) {sym_name = "buf23"} : memref<4xi32>
        aie.objectfifo @fifo (%tile22, {%tile23}, 4 : i32) : !aie.objectfifo<memref<i32>>
        // Producer -- produces one element at a time
        %core22 = aie.core(%tile22) {
            %c99 = arith.constant 99 : i32
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i4 = arith.constant 4 : index
            scf.for %it = %i0 to %i4 step %i1 {
                // Produce one 1 element (acquire producer lock) ...
                %subview = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i32>>
                %subview_obj = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                memref.store %c99, %subview_obj[] : memref<i32>
                aie.objectfifo.release @fifo (Produce, 1)
                // ... done producing (release consumer lock)
            }
            aie.end
        }
        // Consumer -- consumes {2, 1, 3, 1}; releases {0, 0, 0, 2}
        %core23 = aie.core(%tile23) {
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i2 = arith.constant 2 : index
            %i3 = arith.constant 3 : index
            // Begin consuming 2 elements (acquire consumer lock with value 2)
            %subview0 = aie.objectfifo.acquire @fifo (Consume, 2) : !aie.objectfifosubview<memref<i32>>
            %subview0_obj = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            %v0 = memref.load %subview0_obj[] : memref<i32>
            memref.store %v0, %buf23[%i0] : memref<4xi32>
            // For the next step, we only need one element (this could be a subroutine that acquires 1, not knowing that we already acquired 2)
            %subview1 = aie.objectfifo.acquire @fifo (Consume, 1) : !aie.objectfifosubview<memref<i32>>
            %subview1_obj = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            %v1 = memref.load %subview1_obj[] : memref<i32>
            memref.store %v1, %buf23[%i1] : memref<4xi32>
            // Actually, give us the two from before and one more for three objects total (consumer lock should increase by one)
            %subview2 = aie.objectfifo.acquire @fifo (Consume, 3) : !aie.objectfifosubview<memref<i32>>
            %subview2_obj = aie.objectfifo.subview.access %subview2[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            %v2 = memref.load %subview2_obj[] : memref<i32>
            memref.store %v2, %buf23[%i2] : memref<4xi32>
            // Now let's just work on one element (consumer lock should not change value)
            %subview3 = aie.objectfifo.acquire @fifo (Consume, 1) : !aie.objectfifosubview<memref<i32>>
            %subview3_obj = aie.objectfifo.subview.access %subview3[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            %v3 = memref.load %subview3_obj[] : memref<i32>
            memref.store %v3, %buf23[%i3] : memref<4xi32>
            // Done, let's release everything we hold (we hold 3 objects from our max acquire)
            aie.objectfifo.release @fifo (Consume, 3)
            aie.end
        }
    }
}
