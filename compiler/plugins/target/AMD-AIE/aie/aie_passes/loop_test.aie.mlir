
// RUN: iree-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           memref.global "public" @loop_of : memref<16xi32>
// CHECK:           %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:           %[[TILE_1_3:.*]] = aie.tile(1, 3)
// CHECK:           %[[LOOP_OF_BUFF_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "loop_of_buff_0"} : memref<16xi32>
// CHECK:           %[[LOOP_OF_BUFF_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "loop_of_buff_1"} : memref<16xi32>
// CHECK:           %[[LOOP_OF_BUFF_2:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "loop_of_buff_2"} : memref<16xi32>
// CHECK:           %[[LOOP_OF_BUFF_3:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "loop_of_buff_3"} : memref<16xi32>
// CHECK:           %[[LOOP_OF_LOCK_0:.*]] = aie.lock(%[[TILE_1_2]], 0) {init = 0 : i32, sym_name = "loop_of_lock_0"}
// CHECK:           %[[LOOP_OF_LOCK_1:.*]] = aie.lock(%[[TILE_1_2]], 1) {init = 0 : i32, sym_name = "loop_of_lock_1"}
// CHECK:           %[[LOOP_OF_LOCK_2:.*]] = aie.lock(%[[TILE_1_2]], 2) {init = 0 : i32, sym_name = "loop_of_lock_2"}
// CHECK:           %[[LOOP_OF_LOCK_3:.*]] = aie.lock(%[[TILE_1_2]], 3) {init = 0 : i32, sym_name = "loop_of_lock_3"}
// CHECK:           func.func @some_work(%[[ARG0:.*]]: memref<16xi32>, %[[ARG1:.*]]: index) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[CORE_1_2:.*]] = aie.core(%[[TILE_1_2]]) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C2:.*]] = arith.constant 2 : index
// CHECK:             %[[C4:.*]] = arith.constant 4 : index
// CHECK:             %[[C21:.*]] = arith.constant 21 : index
// CHECK:             aie.use_lock(%[[LOOP_OF_LOCK_0]], Acquire, 0)
// CHECK:             func.call @some_work(%[[LOOP_OF_BUFF_0]], %[[C0]]) : (memref<16xi32>, index) -> ()
// CHECK:             aie.use_lock(%[[LOOP_OF_LOCK_0]], Release, 1)
// CHECK:             %[[C17:.*]] = arith.constant 17 : index
// CHECK:             %[[C8:.*]] = arith.constant 8 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C1]] to %[[C17]] step %[[C8]] {
// CHECK:               aie.use_lock(%[[LOOP_OF_LOCK_1]], Acquire, 0)
// CHECK:               func.call @some_work(%[[LOOP_OF_BUFF_1]], %[[ARG0]]) : (memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[LOOP_OF_LOCK_1]], Release, 1)
// CHECK:               %[[C1_2:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_0:.*]] = arith.muli %[[C2]], %[[C1_2]] : index
// CHECK:               %[[VAL_1:.*]] = arith.addi %[[ARG0]], %[[VAL_0]] : index
// CHECK:               aie.use_lock(%[[LOOP_OF_LOCK_2]], Acquire, 0)
// CHECK:               func.call @some_work(%[[LOOP_OF_BUFF_2]], %[[VAL_1]]) : (memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[LOOP_OF_LOCK_2]], Release, 1)
// CHECK:               %[[C2_3:.*]] = arith.constant 2 : index
// CHECK:               %[[VAL_2:.*]] = arith.muli %[[C2]], %[[C2_3]] : index
// CHECK:               %[[VAL_3:.*]] = arith.addi %[[ARG0]], %[[VAL_2]] : index
// CHECK:               aie.use_lock(%[[LOOP_OF_LOCK_3]], Acquire, 0)
// CHECK:               func.call @some_work(%[[LOOP_OF_BUFF_3]], %[[VAL_3]]) : (memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[LOOP_OF_LOCK_3]], Release, 1)
// CHECK:               %[[C3:.*]] = arith.constant 3 : index
// CHECK:               %[[VAL_4:.*]] = arith.muli %[[C2]], %[[C3]] : index
// CHECK:               %[[VAL_5:.*]] = arith.addi %[[ARG0]], %[[VAL_4]] : index
// CHECK:               aie.use_lock(%[[LOOP_OF_LOCK_0]], Acquire, 0)
// CHECK:               func.call @some_work(%[[LOOP_OF_BUFF_0]], %[[VAL_5]]) : (memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[LOOP_OF_LOCK_0]], Release, 1)
// CHECK:             }
// CHECK:             scf.for %[[ARG0:.*]] = %[[C17]] to %[[C21]] step %[[C2]] {
// CHECK:               aie.use_lock(%[[LOOP_OF_LOCK_1]], Acquire, 0)
// CHECK:               func.call @some_work(%[[LOOP_OF_BUFF_1]], %[[ARG0]]) : (memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[LOOP_OF_LOCK_1]], Release, 1)
// CHECK:             }
// CHECK:             %[[C1_0:.*]] = arith.constant 1 : index
// CHECK:             %[[C4_1:.*]] = arith.constant 4 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C1]] to %[[C1_0]] step %[[C4_1]] {
// CHECK:               aie.use_lock(%[[LOOP_OF_LOCK_2]], Acquire, 0)
// CHECK:               func.call @some_work(%[[LOOP_OF_BUFF_2]], %[[ARG0]]) : (memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[LOOP_OF_LOCK_2]], Release, 1)
// CHECK:               %[[C1_2:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_6:.*]] = arith.muli %[[C1]], %[[C1_2]] : index
// CHECK:               %[[VAL_7:.*]] = arith.addi %[[ARG0]], %[[VAL_6]] : index
// CHECK:               aie.use_lock(%[[LOOP_OF_LOCK_3]], Acquire, 0)
// CHECK:               func.call @some_work(%[[LOOP_OF_BUFF_3]], %[[VAL_7]]) : (memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[LOOP_OF_LOCK_3]], Release, 1)
// CHECK:               %[[C2_3:.*]] = arith.constant 2 : index
// CHECK:               %[[VAL_8:.*]] = arith.muli %[[C1]], %[[C2_3]] : index
// CHECK:               %[[VAL_9:.*]] = arith.addi %[[ARG0]], %[[VAL_8]] : index
// CHECK:               aie.use_lock(%[[LOOP_OF_LOCK_0]], Acquire, 0)
// CHECK:               func.call @some_work(%[[LOOP_OF_BUFF_0]], %[[VAL_9]]) : (memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[LOOP_OF_LOCK_0]], Release, 1)
// CHECK:               %[[C3:.*]] = arith.constant 3 : index
// CHECK:               %[[VAL_10:.*]] = arith.muli %[[C1]], %[[C3]] : index
// CHECK:               %[[VAL_11:.*]] = arith.addi %[[ARG0]], %[[VAL_10]] : index
// CHECK:               aie.use_lock(%[[LOOP_OF_LOCK_1]], Acquire, 0)
// CHECK:               func.call @some_work(%[[LOOP_OF_BUFF_1]], %[[VAL_11]]) : (memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[LOOP_OF_LOCK_1]], Release, 1)
// CHECK:             }
// CHECK:             scf.for %[[ARG0:.*]] = %[[C1_0]] to %[[C4]] step %[[C1]] {
// CHECK:               aie.use_lock(%[[LOOP_OF_LOCK_2]], Acquire, 0)
// CHECK:               func.call @some_work(%[[LOOP_OF_BUFF_2]], %[[ARG0]]) : (memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[LOOP_OF_LOCK_2]], Release, 1)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module {
  aie.device(xcvc1902) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    aie.objectfifo @loop_of (%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    func.func @some_work(%line_in:memref<16xi32>, %index:index) -> () {
      return
    }
    %core12 = aie.core(%tile12) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c21 = arith.constant 21 : index
      %subviewTop0 = aie.objectfifo.acquire @loop_of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elemTop0 = aie.objectfifo.subview.access %subviewTop0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      func.call @some_work(%elemTop0, %c0) : (memref<16xi32>,index) -> ()
      aie.objectfifo.release @loop_of (Produce, 1)
      scf.for %indexInHeight = %c1 to %c21 step %c2 {
        %subview = aie.objectfifo.acquire @loop_of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @some_work(%elem0,%indexInHeight) : (memref<16xi32>,index) -> ()
        aie.objectfifo.release @loop_of (Produce, 1)
      }
      scf.for %indexInHeight = %c1 to %c4 step %c1 {
        %subview = aie.objectfifo.acquire @loop_of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @some_work(%elem0,%indexInHeight) : (memref<16xi32>,index) -> ()
        aie.objectfifo.release @loop_of (Produce, 1)
      }
      aie.end
    }
  }
}
