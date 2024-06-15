
// RUN: iree-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           memref.global "public" @fifo0 : memref<16xi32>
// CHECK:           %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[FIFO0_BUFF_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "fifo0_buff_0"} : memref<16xi32>
// CHECK:           %[[FIFO0_BUFF_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "fifo0_buff_1"} : memref<16xi32>
// CHECK:           %[[FIFO0_BUFF_2:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "fifo0_buff_2"} : memref<16xi32>
// CHECK:           %[[FIFO0_BUFF_3:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "fifo0_buff_3"} : memref<16xi32>
// CHECK:           %[[FIFO0_PROD_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 0) {init = 4 : i32, sym_name = "fifo0_prod_lock"}
// CHECK:           %[[FIFO0_CONS_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 1) {init = 0 : i32, sym_name = "fifo0_cons_lock"}
// CHECK:           %[[CORE_1_2:.*]] = aie.core(%[[TILE_1_2]]) {
// CHECK:             %[[C11_I32:.*]] = arith.constant 11 : i32
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C9:.*]] = arith.constant 9 : index
// CHECK:             %[[C8:.*]] = arith.constant 8 : index
// CHECK:             %[[C4:.*]] = arith.constant 4 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C8]] step %[[C4]] {
// CHECK:               aie.use_lock(%[[FIFO0_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               memref.store %[[C11_I32]], %[[FIFO0_BUFF_0]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:               aie.use_lock(%[[FIFO0_CONS_LOCK]], Release, 1)
// CHECK:               aie.use_lock(%[[FIFO0_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               memref.store %[[C11_I32]], %[[FIFO0_BUFF_1]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:               aie.use_lock(%[[FIFO0_CONS_LOCK]], Release, 1)
// CHECK:               aie.use_lock(%[[FIFO0_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               memref.store %[[C11_I32]], %[[FIFO0_BUFF_2]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:               aie.use_lock(%[[FIFO0_CONS_LOCK]], Release, 1)
// CHECK:               aie.use_lock(%[[FIFO0_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               memref.store %[[C11_I32]], %[[FIFO0_BUFF_3]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:               aie.use_lock(%[[FIFO0_CONS_LOCK]], Release, 1)
// CHECK:             }
// CHECK:             aie.use_lock(%[[FIFO0_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             memref.store %[[C11_I32]], %[[FIFO0_BUFF_0]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:             aie.use_lock(%[[FIFO0_CONS_LOCK]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_2_2:.*]] = aie.core(%[[TILE_2_2]]) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C9:.*]] = arith.constant 9 : index
// CHECK:             aie.use_lock(%[[FIFO0_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             %[[VAL_0:.*]] = memref.load %[[FIFO0_BUFF_0]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:             aie.use_lock(%[[FIFO0_PROD_LOCK]], Release, 1)
// CHECK:             %[[C8:.*]] = arith.constant 8 : index
// CHECK:             %[[C4:.*]] = arith.constant 4 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C8]] step %[[C4]] {
// CHECK:               aie.use_lock(%[[FIFO0_CONS_LOCK]], AcquireGreaterEqual, 3)
// CHECK:               %[[VAL_1:.*]] = memref.load %[[FIFO0_BUFF_1]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:               %[[VAL_2:.*]] = memref.load %[[FIFO0_BUFF_2]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:               %[[VAL_3:.*]] = memref.load %[[FIFO0_BUFF_3]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:               aie.use_lock(%[[FIFO0_PROD_LOCK]], Release, 1)
// CHECK:               aie.use_lock(%[[FIFO0_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               %[[VAL_4:.*]] = memref.load %[[FIFO0_BUFF_2]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:               %[[VAL_5:.*]] = memref.load %[[FIFO0_BUFF_3]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:               %[[VAL_6:.*]] = memref.load %[[FIFO0_BUFF_0]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:               aie.use_lock(%[[FIFO0_PROD_LOCK]], Release, 1)
// CHECK:               aie.use_lock(%[[FIFO0_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               %[[VAL_7:.*]] = memref.load %[[FIFO0_BUFF_3]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:               %[[VAL_8:.*]] = memref.load %[[FIFO0_BUFF_0]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:               %[[VAL_9:.*]] = memref.load %[[FIFO0_BUFF_1]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:               aie.use_lock(%[[FIFO0_PROD_LOCK]], Release, 1)
// CHECK:               aie.use_lock(%[[FIFO0_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               %[[VAL_10:.*]] = memref.load %[[FIFO0_BUFF_0]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:               %[[VAL_11:.*]] = memref.load %[[FIFO0_BUFF_1]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:               %[[VAL_12:.*]] = memref.load %[[FIFO0_BUFF_2]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:               aie.use_lock(%[[FIFO0_PROD_LOCK]], Release, 1)
// CHECK:             }
// CHECK:             aie.use_lock(%[[FIFO0_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             %[[VAL_13:.*]] = memref.load %[[FIFO0_BUFF_1]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:             %[[VAL_14:.*]] = memref.load %[[FIFO0_BUFF_2]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:             %[[VAL_15:.*]] = memref.load %[[FIFO0_BUFF_3]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:             aie.use_lock(%[[FIFO0_PROD_LOCK]], Release, 1)
// CHECK:             %[[VAL_16:.*]] = memref.load %[[FIFO0_BUFF_2]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:             %[[VAL_17:.*]] = memref.load %[[FIFO0_BUFF_3]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:             aie.use_lock(%[[FIFO0_PROD_LOCK]], Release, 2)
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @cyclostatic {
    aie.device(xcve2302) {
        %tile12 = aie.tile(1, 2)
        %tile23 = aie.tile(2, 2)
        aie.objectfifo @fifo0 (%tile12, {%tile23}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
        %core12 = aie.core(%tile12) {
            %v11 = arith.constant 11 : i32
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c9 = arith.constant 9 : index
            scf.for %indexInHeight = %c0 to %c9 step %c1 {
                %subview1 = aie.objectfifo.acquire @fifo0 (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
                %subview1_obj = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                memref.store %v11, %subview1_obj[%c0] : memref<16xi32>
                aie.objectfifo.release @fifo0 (Produce, 1)
            }
            aie.end
        }
        %core23 = aie.core(%tile23) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c9 = arith.constant 9 : index
            %subview0 = aie.objectfifo.acquire @fifo0 (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
            %subview0_obj = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %v0 = memref.load %subview0_obj[%c0] : memref<16xi32>
            aie.objectfifo.release @fifo0 (Consume, 1)
            scf.for %indexInHeight = %c0 to %c9 step %c1 {
                %subview1 = aie.objectfifo.acquire @fifo0 (Consume, 3) : !aie.objectfifosubview<memref<16xi32>>
                %subview1_obj = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %subview1_obj1 = aie.objectfifo.subview.access %subview1[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %subview1_obj2 = aie.objectfifo.subview.access %subview1[2] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %v1 = memref.load %subview1_obj[%c0] : memref<16xi32>
                %v2 = memref.load %subview1_obj1[%c0] : memref<16xi32>
                %v3 = memref.load %subview1_obj2[%c0] : memref<16xi32>
                aie.objectfifo.release @fifo0 (Consume, 1)
            }
            %subview2 = aie.objectfifo.acquire @fifo0 (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
            %subview2_obj = aie.objectfifo.subview.access %subview2[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %subview2_obj1 = aie.objectfifo.subview.access %subview2[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %v4 = memref.load %subview2_obj[%c0] : memref<16xi32>
            %v5 = memref.load %subview2_obj1[%c0] : memref<16xi32>
            aie.objectfifo.release @fifo0 (Consume, 2)
            aie.end
        }
    }
}
