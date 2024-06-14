
// RUN: iree-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @fifo : memref<i32>
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[C16:.*]] = arith.constant 16 : index
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[SRCBUF22:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "srcbuf22"} : memref<i32>
// CHECK:           %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK:           %[[FIFO_BUFF_0:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "fifo_buff_0"} : memref<i32>
// CHECK:           %[[FIFO_BUFF_1:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "fifo_buff_1"} : memref<i32>
// CHECK:           %[[FIFO_BUFF_2:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "fifo_buff_2"} : memref<i32>
// CHECK:           %[[FIFO_BUFF_3:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "fifo_buff_3"} : memref<i32>
// CHECK:           %[[FIFO_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 0) {init = 4 : i32, sym_name = "fifo_prod_lock"}
// CHECK:           %[[FIFO_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 1) {init = 0 : i32, sym_name = "fifo_cons_lock"}
// CHECK:           %[[DSTBUF22:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "dstbuf22"} : memref<16xi32>
// CHECK:           %[[CORE_2_2:.*]] = aie.core(%[[TILE_2_2]]) {
// CHECK:             %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:             %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:             memref.store %[[C0_I32]], %[[SRCBUF22]][] : memref<i32>
// CHECK:             %[[C4:.*]] = arith.constant 4 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C16]] step %[[C4]] {
// CHECK:               %[[VAL_0:.*]] = memref.load %[[SRCBUF22]][] : memref<i32>
// CHECK:               aie.use_lock(%[[FIFO_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               memref.store %[[VAL_0]], %[[FIFO_BUFF_0]][] : memref<i32>
// CHECK:               aie.use_lock(%[[FIFO_CONS_LOCK]], Release, 1)
// CHECK:               %[[VAL_1:.*]] = arith.addi %[[C1_I32]], %[[VAL_0]] : i32
// CHECK:               memref.store %[[VAL_1]], %[[SRCBUF22]][] : memref<i32>
// CHECK:               %[[VAL_2:.*]] = memref.load %[[SRCBUF22]][] : memref<i32>
// CHECK:               aie.use_lock(%[[FIFO_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               memref.store %[[VAL_2]], %[[FIFO_BUFF_1]][] : memref<i32>
// CHECK:               aie.use_lock(%[[FIFO_CONS_LOCK]], Release, 1)
// CHECK:               %[[VAL_3:.*]] = arith.addi %[[C1_I32]], %[[VAL_2]] : i32
// CHECK:               memref.store %[[VAL_3]], %[[SRCBUF22]][] : memref<i32>
// CHECK:               %[[VAL_4:.*]] = memref.load %[[SRCBUF22]][] : memref<i32>
// CHECK:               aie.use_lock(%[[FIFO_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               memref.store %[[VAL_4]], %[[FIFO_BUFF_2]][] : memref<i32>
// CHECK:               aie.use_lock(%[[FIFO_CONS_LOCK]], Release, 1)
// CHECK:               %[[VAL_5:.*]] = arith.addi %[[C1_I32]], %[[VAL_4]] : i32
// CHECK:               memref.store %[[VAL_5]], %[[SRCBUF22]][] : memref<i32>
// CHECK:               %[[VAL_6:.*]] = memref.load %[[SRCBUF22]][] : memref<i32>
// CHECK:               aie.use_lock(%[[FIFO_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               memref.store %[[VAL_6]], %[[FIFO_BUFF_3]][] : memref<i32>
// CHECK:               aie.use_lock(%[[FIFO_CONS_LOCK]], Release, 1)
// CHECK:               %[[VAL_7:.*]] = arith.addi %[[C1_I32]], %[[VAL_6]] : i32
// CHECK:               memref.store %[[VAL_7]], %[[SRCBUF22]][] : memref<i32>
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_2_3:.*]] = aie.core(%[[TILE_2_3]]) {
// CHECK:             %[[C8:.*]] = arith.constant 8 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C16]] step %[[C8]] {
// CHECK:               aie.use_lock(%[[FIFO_CONS_LOCK]], AcquireGreaterEqual, 2)
// CHECK:               %[[VAL_8:.*]] = memref.load %[[FIFO_BUFF_0]][] : memref<i32>
// CHECK:               %[[VAL_9:.*]] = memref.load %[[FIFO_BUFF_1]][] : memref<i32>
// CHECK:               func.call @store2(%[[VAL_8]], %[[VAL_9]], %[[ARG0]], %[[DSTBUF22]]) : (i32, i32, index, memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[FIFO_PROD_LOCK]], Release, 2)
// CHECK:               %[[C1_0:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_10:.*]] = arith.muli %[[C2]], %[[C1_0]] : index
// CHECK:               %[[VAL_11:.*]] = arith.addi %[[ARG0]], %[[VAL_10]] : index
// CHECK:               aie.use_lock(%[[FIFO_CONS_LOCK]], AcquireGreaterEqual, 2)
// CHECK:               %[[VAL_12:.*]] = memref.load %[[FIFO_BUFF_2]][] : memref<i32>
// CHECK:               %[[VAL_13:.*]] = memref.load %[[FIFO_BUFF_3]][] : memref<i32>
// CHECK:               func.call @store2(%[[VAL_12]], %[[VAL_13]], %[[VAL_11]], %[[DSTBUF22]]) : (i32, i32, index, memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[FIFO_PROD_LOCK]], Release, 2)
// CHECK:               %[[C2_1:.*]] = arith.constant 2 : index
// CHECK:               %[[VAL_14:.*]] = arith.muli %[[C2]], %[[C2_1]] : index
// CHECK:               %[[VAL_15:.*]] = arith.addi %[[ARG0]], %[[VAL_14]] : index
// CHECK:               aie.use_lock(%[[FIFO_CONS_LOCK]], AcquireGreaterEqual, 2)
// CHECK:               %[[VAL_16:.*]] = memref.load %[[FIFO_BUFF_0]][] : memref<i32>
// CHECK:               %[[VAL_17:.*]] = memref.load %[[FIFO_BUFF_1]][] : memref<i32>
// CHECK:               func.call @store2(%[[VAL_16]], %[[VAL_17]], %[[VAL_15]], %[[DSTBUF22]]) : (i32, i32, index, memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[FIFO_PROD_LOCK]], Release, 2)
// CHECK:               %[[C3:.*]] = arith.constant 3 : index
// CHECK:               %[[VAL_18:.*]] = arith.muli %[[C2]], %[[C3]] : index
// CHECK:               %[[VAL_19:.*]] = arith.addi %[[ARG0]], %[[VAL_18]] : index
// CHECK:               aie.use_lock(%[[FIFO_CONS_LOCK]], AcquireGreaterEqual, 2)
// CHECK:               %[[VAL_20:.*]] = memref.load %[[FIFO_BUFF_2]][] : memref<i32>
// CHECK:               %[[VAL_21:.*]] = memref.load %[[FIFO_BUFF_3]][] : memref<i32>
// CHECK:               func.call @store2(%[[VAL_20]], %[[VAL_21]], %[[VAL_19]], %[[DSTBUF22]]) : (i32, i32, index, memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[FIFO_PROD_LOCK]], Release, 2)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           func.func @store2(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: index, %[[ARG3:.*]]: memref<16xi32>) {
// CHECK:             %[[C0_0:.*]] = arith.constant 0 : index
// CHECK:             %[[C1_1:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_22:.*]] = arith.addi %[[C0_0]], %[[ARG2]] : index
// CHECK:             %[[VAL_23:.*]] = arith.addi %[[C1_1]], %[[ARG2]] : index
// CHECK:             memref.store %[[ARG0]], %[[ARG3]]{{\[}}%[[VAL_22]]] : memref<16xi32>
// CHECK:             memref.store %[[ARG1]], %[[ARG3]]{{\[}}%[[VAL_23]]] : memref<16xi32>
// CHECK:             return
// CHECK:           }
// CHECK:         }

module @aie2_static_l1 {
    aie.device(npu1_4col) {
        %i_c0 = arith.constant 0 : index
        %i_c1 = arith.constant 1 : index
        %i_c2 = arith.constant 2 : index
        %i_c16 = arith.constant 16 : index
        %tile22 = aie.tile(2, 2)  // producer tile
        %srcbuf22 = aie.buffer(%tile22) { sym_name = "srcbuf22" } : memref<i32>
        %tile23 = aie.tile(2, 3)  // consumer tile
        %dstbuf23 = aie.buffer(%tile23) { sym_name = "dstbuf22" } : memref<16xi32>
        // ObjectFifo that can hold 4 memref<i32>s, populated by tile22 and
        // consumed by tile23
        aie.objectfifo @fifo (%tile22, {%tile23}, 4 : i32) : !aie.objectfifo<memref<i32>>
        // Producer core
        %core22 = aie.core(%tile22) {
            // Initialize value to zero
            %c0 = arith.constant 0 : i32
            %c1 = arith.constant 1 : i32
            memref.store %c0, %srcbuf22[] : memref<i32>
            // Count up, with each iteration pushing a new element on to the fifo
            scf.for %idx = %i_c0 to %i_c16 step %i_c1 {
                %val0 = memref.load %srcbuf22[] : memref<i32>
                // Produce 1 elements, so acquire 1 element
                %subview = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i32>>
                %elem = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                memref.store %val0, %elem[] : memref<i32>
                aie.objectfifo.release @fifo (Produce, 1)
                // Increment
                %val1 = arith.addi %c1, %val0 : i32
                memref.store %val1, %srcbuf22[] : memref<i32>
            }
            aie.end
        }
        // Consumer core
        %core23 = aie.core(%tile23) {
            scf.for %idx = %i_c0 to %i_c16 step %i_c2 {
                // Consume _two_ elements at once (cyclo static)
                %subview = aie.objectfifo.acquire @fifo (Consume, 2) : !aie.objectfifosubview<memref<i32>>
                %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                %elem1 = aie.objectfifo.subview.access %subview[1] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                %val0 = memref.load %elem0[] : memref<i32>
                %val1 = memref.load %elem1[] : memref<i32>
                // Pass through to destination buffer
                func.call @store2(%val0, %val1, %idx, %dstbuf23) : (i32, i32, index, memref<16xi32>) -> ()
                // Release consumed objects
                aie.objectfifo.release @fifo (Consume, 2)
            }
            aie.end
        }
        func.func @store2(%val0: i32, %val1: i32, %base : index, %buf : memref<16xi32>) -> () {
                %ic0 = arith.constant 0 : index
                %ic1 = arith.constant 1 : index
                %idx0 = arith.addi %ic0, %base : index
                %idx1 = arith.addi %ic1, %base : index
                memref.store %val0, %buf[%idx0] : memref<16xi32>
                memref.store %val1, %buf[%idx1] : memref<16xi32>
                return
        }
    }
}
