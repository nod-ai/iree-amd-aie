
// RUN: iree-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @loop_of : memref<16xi32>
// CHECK:           %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:           %[[TILE_1_3:.*]] = aie.tile(1, 3)
// CHECK:           %[[LOOP_OF_BUFF_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "loop_of_buff_0"} : memref<16xi32>
// CHECK:           %[[LOOP_OF_BUFF_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "loop_of_buff_1"} : memref<16xi32>
// CHECK:           %[[LOOP_OF_PROD_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 0) {init = 2 : i32, sym_name = "loop_of_prod_lock"}
// CHECK:           %[[LOOP_OF_CONS_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 1) {init = 0 : i32, sym_name = "loop_of_cons_lock"}
// CHECK:           func.func @some_work(%[[ARG0:.*]]: memref<4x4xi32>, %[[ARG1:.*]]: index) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[CORE_1_2:.*]] = aie.core(%[[TILE_1_2]]) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C2:.*]] = arith.constant 2 : index
// CHECK:             %[[C4:.*]] = arith.constant 4 : index
// CHECK:             %[[C21:.*]] = arith.constant 21 : index
// CHECK:             %[[C4294967295:.*]] = arith.constant 4294967295 : index
// CHECK:             %[[C4294967294:.*]] = arith.constant 4294967294 : index
// CHECK:             %[[C2_0:.*]] = arith.constant 2 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C4294967294]] step %[[C2_0]] {
// CHECK:               aie.use_lock(%[[LOOP_OF_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               %[[REINTERPRET_CAST_3:.*]] = memref.reinterpret_cast %[[LOOP_OF_BUFF_0]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK:               func.call @some_work(%[[REINTERPRET_CAST_3]], %[[C0]]) : (memref<4x4xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[LOOP_OF_CONS_LOCK]], Release, 1)
// CHECK:               %[[C2_4:.*]] = arith.constant 2 : index
// CHECK:               scf.for %[[ARG1:.*]] = %[[C1]] to %[[C21]] step %[[C2_4]] {
// CHECK:                 aie.use_lock(%[[LOOP_OF_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:                 %[[REINTERPRET_CAST_9:.*]] = memref.reinterpret_cast %[[LOOP_OF_BUFF_1]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK:                 func.call @some_work(%[[REINTERPRET_CAST_9]], %[[ARG1]]) : (memref<4x4xi32>, index) -> ()
// CHECK:                 aie.use_lock(%[[LOOP_OF_CONS_LOCK]], Release, 1)
// CHECK:                 %[[C1_10:.*]] = arith.constant 1 : index
// CHECK:                 %[[VAL_0:.*]] = arith.muli %[[C1]], %[[C1_10]] : index
// CHECK:                 %[[VAL_1:.*]] = arith.addi %[[ARG1]], %[[VAL_0]] : index
// CHECK:                 aie.use_lock(%[[LOOP_OF_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:                 %[[REINTERPRET_CAST_11:.*]] = memref.reinterpret_cast %[[LOOP_OF_BUFF_0]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK:                 func.call @some_work(%[[REINTERPRET_CAST_11]], %[[VAL_1]]) : (memref<4x4xi32>, index) -> ()
// CHECK:                 aie.use_lock(%[[LOOP_OF_CONS_LOCK]], Release, 1)
// CHECK:               }
// CHECK:               aie.use_lock(%[[LOOP_OF_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               %[[REINTERPRET_CAST_5:.*]] = memref.reinterpret_cast %[[LOOP_OF_BUFF_1]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK:               func.call @some_work(%[[REINTERPRET_CAST_5]], %[[C0]]) : (memref<4x4xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[LOOP_OF_CONS_LOCK]], Release, 1)
// CHECK:               aie.use_lock(%[[LOOP_OF_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               %[[REINTERPRET_CAST_6:.*]] = memref.reinterpret_cast %[[LOOP_OF_BUFF_0]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK:               func.call @some_work(%[[REINTERPRET_CAST_6]], %[[C0]]) : (memref<4x4xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[LOOP_OF_CONS_LOCK]], Release, 1)
// CHECK:               %[[C2_7:.*]] = arith.constant 2 : index
// CHECK:               scf.for %[[ARG1:.*]] = %[[C1]] to %[[C21]] step %[[C2_7]] {
// CHECK:                 aie.use_lock(%[[LOOP_OF_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:                 %[[REINTERPRET_CAST_9:.*]] = memref.reinterpret_cast %[[LOOP_OF_BUFF_1]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK:                 func.call @some_work(%[[REINTERPRET_CAST_9]], %[[ARG1]]) : (memref<4x4xi32>, index) -> ()
// CHECK:                 aie.use_lock(%[[LOOP_OF_CONS_LOCK]], Release, 1)
// CHECK:                 %[[C1_10:.*]] = arith.constant 1 : index
// CHECK:                 %[[VAL_2:.*]] = arith.muli %[[C1]], %[[C1_10]] : index
// CHECK:                 %[[VAL_3:.*]] = arith.addi %[[ARG1]], %[[VAL_2]] : index
// CHECK:                 aie.use_lock(%[[LOOP_OF_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:                 %[[REINTERPRET_CAST_11:.*]] = memref.reinterpret_cast %[[LOOP_OF_BUFF_0]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK:                 func.call @some_work(%[[REINTERPRET_CAST_11]], %[[VAL_3]]) : (memref<4x4xi32>, index) -> ()
// CHECK:                 aie.use_lock(%[[LOOP_OF_CONS_LOCK]], Release, 1)
// CHECK:               }
// CHECK:               aie.use_lock(%[[LOOP_OF_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               %[[REINTERPRET_CAST_8:.*]] = memref.reinterpret_cast %[[LOOP_OF_BUFF_1]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK:               func.call @some_work(%[[REINTERPRET_CAST_8]], %[[C0]]) : (memref<4x4xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[LOOP_OF_CONS_LOCK]], Release, 1)
// CHECK:             }
// CHECK:             aie.use_lock(%[[LOOP_OF_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             %[[REINTERPRET_CAST:.*]] = memref.reinterpret_cast %[[LOOP_OF_BUFF_0]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK:             func.call @some_work(%[[REINTERPRET_CAST]], %[[C0]]) : (memref<4x4xi32>, index) -> ()
// CHECK:             aie.use_lock(%[[LOOP_OF_CONS_LOCK]], Release, 1)
// CHECK:             %[[C2_1:.*]] = arith.constant 2 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C1]] to %[[C21]] step %[[C2_1]] {
// CHECK:               aie.use_lock(%[[LOOP_OF_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               %[[REINTERPRET_CAST_3:.*]] = memref.reinterpret_cast %[[LOOP_OF_BUFF_1]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK:               func.call @some_work(%[[REINTERPRET_CAST_3]], %[[ARG0]]) : (memref<4x4xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[LOOP_OF_CONS_LOCK]], Release, 1)
// CHECK:               %[[C1_4:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_4:.*]] = arith.muli %[[C1]], %[[C1_4]] : index
// CHECK:               %[[VAL_5:.*]] = arith.addi %[[ARG0]], %[[VAL_4]] : index
// CHECK:               aie.use_lock(%[[LOOP_OF_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               %[[REINTERPRET_CAST_5:.*]] = memref.reinterpret_cast %[[LOOP_OF_BUFF_0]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK:               func.call @some_work(%[[REINTERPRET_CAST_5]], %[[VAL_5]]) : (memref<4x4xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[LOOP_OF_CONS_LOCK]], Release, 1)
// CHECK:             }
// CHECK:             aie.use_lock(%[[LOOP_OF_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             %[[REINTERPRET_CAST_2:.*]] = memref.reinterpret_cast %[[LOOP_OF_BUFF_1]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK:             func.call @some_work(%[[REINTERPRET_CAST_2]], %[[C0]]) : (memref<4x4xi32>, index) -> ()
// CHECK:             aie.use_lock(%[[LOOP_OF_CONS_LOCK]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module {
  aie.device(npu1_4col) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    aie.objectfifo @loop_of (%tile12, {%tile13}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    func.func @some_work(%line_in: memref<4x4xi32>, %index: index) -> () {
      return
    }
    %core12 = aie.core(%tile12) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c21 = arith.constant 21 : index
      %cmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg0 = %c0 to %cmax step %c1 {
        %subviewTop0 = aie.objectfifo.acquire @loop_of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %elemTop0 = aie.objectfifo.subview.access %subviewTop0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        %reinterpret_cast_0 = memref.reinterpret_cast %elemTop0 to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
        func.call @some_work(%reinterpret_cast_0, %c0) : (memref<4x4xi32>, index) -> ()
        aie.objectfifo.release @loop_of (Produce, 1)
        scf.for %indexInHeight = %c1 to %c21 step %c1 {
          %subview = aie.objectfifo.acquire @loop_of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
          %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
          %reinterpret_cast_1 = memref.reinterpret_cast %elem0 to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
          func.call @some_work(%reinterpret_cast_1, %indexInHeight) : (memref<4x4xi32>, index) -> ()
          aie.objectfifo.release @loop_of (Produce, 1)
        }
        %subviewTop1 = aie.objectfifo.acquire @loop_of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %elemTop1 = aie.objectfifo.subview.access %subviewTop1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        %reinterpret_cast_2 = memref.reinterpret_cast %elemTop1 to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
        func.call @some_work(%reinterpret_cast_2, %c0) : (memref<4x4xi32>, index) -> ()
        aie.objectfifo.release @loop_of (Produce, 1)
      }
      aie.end
    }
  }
}
