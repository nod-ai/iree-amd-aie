
// RUN: iree-opt --amdaie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK-DAG:       memref.global "public" @fifo : memref<i32>
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:       %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:       %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG:       %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK-DAG:       %[[SRCBUF22:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "srcbuf22"} : memref<i32>
// CHECK-DAG:       %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK-DAG:       %[[FIFO_BUFF_0:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "fifo_buff_0"} : memref<i32>
// CHECK-DAG:       %[[FIFO_BUFF_1:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "fifo_buff_1"} : memref<i32>
// CHECK-DAG:       %[[FIFO_BUFF_2:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "fifo_buff_2"} : memref<i32>
// CHECK-DAG:       %[[FIFO_BUFF_3:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "fifo_buff_3"} : memref<i32>
// CHECK-DAG:       %[[FIFO_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_2]]) {init = 4 : i8, sym_name = "fifo_prod_lock"}
// CHECK-DAG:       %[[FIFO_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_2]]) {init = 0 : i8, sym_name = "fifo_cons_lock"}
// CHECK-DAG:       %[[DSTBUF22:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "dstbuf22"} : memref<16xi32>
// CHECK:           %[[CORE_2_2:.*]] = aie.core(%[[TILE_2_2]]) {
// CHECK:             %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:             %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:             memref.store %[[C0_I32]], %[[SRCBUF22]][] : memref<i32>
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
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C16]] step %[[C8]] {
// CHECK:               aie.use_lock(%[[FIFO_CONS_LOCK]], AcquireGreaterEqual, 2)
// CHECK:               %[[VAL_8:.*]] = memref.load %[[FIFO_BUFF_0]][] : memref<i32>
// CHECK:               %[[VAL_9:.*]] = memref.load %[[FIFO_BUFF_1]][] : memref<i32>
// CHECK:               func.call @store2(%[[VAL_8]], %[[VAL_9]], %[[ARG0]], %[[DSTBUF22]]) : (i32, i32, index, memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[FIFO_PROD_LOCK]], Release, 2)
// CHECK:               aie.use_lock(%[[FIFO_CONS_LOCK]], AcquireGreaterEqual, 2)
// CHECK:               %[[VAL_10:.*]] = memref.load %[[FIFO_BUFF_2]][] : memref<i32>
// CHECK:               %[[VAL_11:.*]] = memref.load %[[FIFO_BUFF_3]][] : memref<i32>
// CHECK:               %[[VAL_12:.*]] = arith.addi %[[ARG0]], %[[C2]] : index
// CHECK:               func.call @store2(%[[VAL_10]], %[[VAL_11]], %[[VAL_12]], %[[DSTBUF22]]) : (i32, i32, index, memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[FIFO_PROD_LOCK]], Release, 2)
// CHECK:               aie.use_lock(%[[FIFO_CONS_LOCK]], AcquireGreaterEqual, 2)
// CHECK:               %[[VAL_13:.*]] = memref.load %[[FIFO_BUFF_0]][] : memref<i32>
// CHECK:               %[[VAL_14:.*]] = memref.load %[[FIFO_BUFF_1]][] : memref<i32>
// CHECK:               %[[VAL_15:.*]] = arith.addi %[[ARG0]], %[[C4]] : index
// CHECK:               func.call @store2(%[[VAL_13]], %[[VAL_14]], %[[VAL_15]], %[[DSTBUF22]]) : (i32, i32, index, memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[FIFO_PROD_LOCK]], Release, 2)
// CHECK:               aie.use_lock(%[[FIFO_CONS_LOCK]], AcquireGreaterEqual, 2)
// CHECK:               %[[VAL_16:.*]] = memref.load %[[FIFO_BUFF_2]][] : memref<i32>
// CHECK:               %[[VAL_17:.*]] = memref.load %[[FIFO_BUFF_3]][] : memref<i32>
// CHECK:               %[[VAL_18:.*]] = arith.addi %[[ARG0]], %[[C6]] : index
// CHECK:               func.call @store2(%[[VAL_16]], %[[VAL_17]], %[[VAL_18]], %[[DSTBUF22]]) : (i32, i32, index, memref<16xi32>) -> ()
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
  aie.device(xcve2302) {
    %i_c0 = arith.constant 0 : index
    %i_c1 = arith.constant 1 : index
    %i_c2 = arith.constant 2 : index
    %i_c4 = arith.constant 4 : index
    %i_c6 = arith.constant 6 : index
    %i_c8 = arith.constant 8 : index
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
      scf.for %idx = %i_c0 to %i_c16 step %i_c4 {
        %val0 = memref.load %srcbuf22[] : memref<i32>
        // Produce 1 elements, so acquire 1 element
        %subview = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i32>>
        %elem = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
        memref.store %val0, %elem[] : memref<i32>
        aie.objectfifo.release @fifo (Produce, 1)
        // Increment
        %val1 = arith.addi %c1, %val0 : i32
        memref.store %val1, %srcbuf22[] : memref<i32>
        
        %val2 = memref.load %srcbuf22[] : memref<i32>
        // Produce 1 elements, so acquire 1 element
        %subview1 = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i32>>
        %elem1 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
        memref.store %val2, %elem1[] : memref<i32>
        aie.objectfifo.release @fifo (Produce, 1)
        // Increment
        %val3 = arith.addi %c1, %val2 : i32
        memref.store %val3, %srcbuf22[] : memref<i32>

        %val4 = memref.load %srcbuf22[] : memref<i32>
        // Produce 1 elements, so acquire 1 element
        %subview2 = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i32>>
        %elem2 = aie.objectfifo.subview.access %subview2[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
        memref.store %val4, %elem2[] : memref<i32>
        aie.objectfifo.release @fifo (Produce, 1)
        // Increment
        %val5 = arith.addi %c1, %val4 : i32
        memref.store %val5, %srcbuf22[] : memref<i32>

        %val6 = memref.load %srcbuf22[] : memref<i32>
        // Produce 1 elements, so acquire 1 element
        %subview3 = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i32>>
        %elem3 = aie.objectfifo.subview.access %subview3[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
        memref.store %val6, %elem3[] : memref<i32>
        aie.objectfifo.release @fifo (Produce, 1)
        // Increment
        %val7 = arith.addi %c1, %val6 : i32
        memref.store %val7, %srcbuf22[] : memref<i32>
      }
      aie.end
    }
    // Consumer core
    %core23 = aie.core(%tile23) {
      scf.for %idx = %i_c0 to %i_c16 step %i_c8 {
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

        // Consume _two_ elements at once (cyclo static)
        %subview1 = aie.objectfifo.acquire @fifo (Consume, 2) : !aie.objectfifosubview<memref<i32>>
        %elem2 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
        %elem3 = aie.objectfifo.subview.access %subview1[1] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
        %val2 = memref.load %elem2[] : memref<i32>
        %val3 = memref.load %elem3[] : memref<i32>
        // Pass through to destination buffer
        %add1 = arith.addi %idx, %i_c2 : index
        func.call @store2(%val2, %val3, %add1, %dstbuf23) : (i32, i32, index, memref<16xi32>) -> ()
        // Release consumed objects
        aie.objectfifo.release @fifo (Consume, 2)

        // Consume _two_ elements at once (cyclo static)
        %subview2 = aie.objectfifo.acquire @fifo (Consume, 2) : !aie.objectfifosubview<memref<i32>>
        %elem4 = aie.objectfifo.subview.access %subview2[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
        %elem5 = aie.objectfifo.subview.access %subview2[1] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
        %val4 = memref.load %elem4[] : memref<i32>
        %val5 = memref.load %elem5[] : memref<i32>
        // Pass through to destination buffer
        %add2 = arith.addi %idx, %i_c4 : index
        func.call @store2(%val4, %val5, %add2, %dstbuf23) : (i32, i32, index, memref<16xi32>) -> ()
        // Release consumed objects
        aie.objectfifo.release @fifo (Consume, 2)

        // Consume _two_ elements at once (cyclo static)
        %subview3 = aie.objectfifo.acquire @fifo (Consume, 2) : !aie.objectfifosubview<memref<i32>>
        %elem6 = aie.objectfifo.subview.access %subview3[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
        %elem7 = aie.objectfifo.subview.access %subview3[1] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
        %val6 = memref.load %elem6[] : memref<i32>
        %val7 = memref.load %elem7[] : memref<i32>
        // Pass through to destination buffer
        %add3 = arith.addi %idx, %i_c6 : index
        func.call @store2(%val6, %val7, %add3, %dstbuf23) : (i32, i32, index, memref<16xi32>) -> ()
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
