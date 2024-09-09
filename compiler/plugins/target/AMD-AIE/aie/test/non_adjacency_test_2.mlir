
// RUN: iree-opt --amdaie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @objfifo : memref<16xi32>
// CHECK-DAG:       %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK-DAG:       %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK-DAG:       %[[BUFFER_1_2:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "objfifo_prod_buff_0_0"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_1_2_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "objfifo_prod_buff_0_1"} : memref<16xi32>
// CHECK-DAG:       %[[LOCK_1_2:.*]] = aie.lock(%[[TILE_1_2]]) {init = 2 : i8, sym_name = "objfifo_prod_prod_lock_0"}
// CHECK-DAG:       %[[LOCK_1_2_1:.*]] = aie.lock(%[[TILE_1_2]]) {init = 0 : i8, sym_name = "objfifo_prod_cons_lock_0"}
// CHECK-DAG:       %[[BUFFER_3_3:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "objfifo_cons_buff_0_0"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_3_3_2:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "objfifo_cons_buff_0_1"} : memref<16xi32>
// CHECK-DAG:       %[[LOCK_3_3:.*]] = aie.lock(%[[TILE_3_3]]) {init = 2 : i8, sym_name = "objfifo_cons_prod_lock_0"}
// CHECK-DAG:       %[[LOCK_3_3_3:.*]] = aie.lock(%[[TILE_3_3]]) {init = 0 : i8, sym_name = "objfifo_cons_cons_lock_0"}
// CHECK-DAG:       aie.flow(%[[TILE_1_2]], DMA : 0, %[[TILE_3_3]], DMA : 0) {symbol = @objfifo}
// CHECK:           func.func @some_work(%[[ARG0:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[CORE_1_2:.*]] = aie.core(%[[TILE_1_2]]) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C2:.*]] = arith.constant 2 : index
// CHECK:             %[[C12:.*]] = arith.constant 12 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C12]] step %[[C2]] {
// CHECK:               aie.use_lock(%[[LOCK_1_2]], AcquireGreaterEqual, 1)
// CHECK:               func.call @some_work(%[[BUFFER_1_2]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[LOCK_1_2_1]], Release, 1)
// CHECK:               aie.use_lock(%[[LOCK_1_2]], AcquireGreaterEqual, 1)
// CHECK:               func.call @some_work(%[[BUFFER_1_2_0]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[LOCK_1_2_1]], Release, 1)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_3_3:.*]] = aie.core(%[[TILE_3_3]]) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C4:.*]] = arith.constant 4 : index
// CHECK:             %[[C12:.*]] = arith.constant 12 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C12]] step %[[C4]] {
// CHECK:               aie.use_lock(%[[LOCK_3_3_3]], AcquireGreaterEqual, 2)
// CHECK:               func.call @some_work(%[[BUFFER_3_3]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[LOCK_3_3]], Release, 2)
// CHECK:               aie.use_lock(%[[LOCK_3_3_3]], AcquireGreaterEqual, 2)
// CHECK:               func.call @some_work(%[[BUFFER_3_3]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[LOCK_3_3]], Release, 2)
// CHECK:               aie.use_lock(%[[LOCK_3_3_3]], AcquireGreaterEqual, 2)
// CHECK:               func.call @some_work(%[[BUFFER_3_3]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[LOCK_3_3]], Release, 2)
// CHECK:               aie.use_lock(%[[LOCK_3_3_3]], AcquireGreaterEqual, 2)
// CHECK:               func.call @some_work(%[[BUFFER_3_3]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[LOCK_3_3]], Release, 2)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_1_2:.*]] = aie.mem(%[[TILE_1_2]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_1_2_1]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_1_2_1]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_0]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_3_3:.*]] = aie.mem(%[[TILE_3_3]]) {
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_3_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_3_3]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_3_3_3]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_3_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_3_3_2]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_3_3_3]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }
module @non_adjacency {
  aie.device(npu1_4col) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)
    aie.flow(%tile12, DMA : 0, %tile33, DMA : 0) {symbol = @objfifo}
    aie.objectfifo @objfifo (%tile12, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    func.func @some_work(%lineOut : memref<16xi32>) -> () {
      return
    }
    %core12 = aie.core(%tile12) {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %height = arith.constant 12 : index
      scf.for %indexInHeight = %c0 to %height step %c2 {
        %subview = aie.objectfifo.acquire @objfifo (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @some_work(%elem0) : (memref<16xi32>) -> ()
        aie.objectfifo.release @objfifo (Produce, 1)
        %subview1 = aie.objectfifo.acquire @objfifo (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %elem1 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @some_work(%elem1) : (memref<16xi32>) -> ()
        aie.objectfifo.release @objfifo (Produce, 1)
      }
      aie.end
    }
    %core33 = aie.core(%tile33) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %height = arith.constant 12 : index
      scf.for %indexInHeight = %c0 to %height step %c4 {
        %subview = aie.objectfifo.acquire @objfifo (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
        %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        %elem1 = aie.objectfifo.subview.access %subview[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @some_work(%elem0) : (memref<16xi32>) -> ()
        aie.objectfifo.release @objfifo (Consume, 2)
        %subview1 = aie.objectfifo.acquire @objfifo (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
        %elem3 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        %elem4 = aie.objectfifo.subview.access %subview1[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @some_work(%elem3) : (memref<16xi32>) -> ()
        aie.objectfifo.release @objfifo (Consume, 2)
        %subview2 = aie.objectfifo.acquire @objfifo (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
        %elem6 = aie.objectfifo.subview.access %subview2[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        %elem7 = aie.objectfifo.subview.access %subview2[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @some_work(%elem6) : (memref<16xi32>) -> ()
        aie.objectfifo.release @objfifo (Consume, 2)
        %subview3 = aie.objectfifo.acquire @objfifo (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
        %elem9 = aie.objectfifo.subview.access %subview3[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        %elem10 = aie.objectfifo.subview.access %subview3[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @some_work(%elem9) : (memref<16xi32>) -> ()
        aie.objectfifo.release @objfifo (Consume, 2)
      }
      aie.end
    }
  }
}
