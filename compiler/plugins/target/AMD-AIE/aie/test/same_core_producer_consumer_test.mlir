
// RUN: iree-opt --amdaie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           memref.global "public" @of : memref<16xi32>
// CHECK-DAG:       %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK-DAG:       %[[BUFFER_1_2:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of_prod_buff_0_0"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_1_2_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of_prod_buff_0_1"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_1_2_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of_prod_buff_0_2"} : memref<16xi32>
// CHECK-DAG:       %[[LOCK_1_2:.*]] = aie.lock(%[[TILE_1_2]]) {init = 3 : i8, sym_name = "of_prod_prod_lock_0"}
// CHECK-DAG:       %[[LOCK_1_2_2:.*]] = aie.lock(%[[TILE_1_2]]) {init = 0 : i8, sym_name = "of_prod_cons_lock_0"}
// CHECK-DAG:       %[[BUFFER_1_2_3:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of_cons_buff_0_0"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_1_2_4:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of_cons_buff_0_1"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_1_2_5:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of_cons_buff_0_2"} : memref<16xi32>
// CHECK-DAG:       %[[LOCK_1_2_6:.*]] = aie.lock(%[[TILE_1_2]]) {init = 3 : i8, sym_name = "of_cons_prod_lock_0"}
// CHECK-DAG:       %[[LOCK_1_2_7:.*]] = aie.lock(%[[TILE_1_2]]) {init = 0 : i8, sym_name = "of_cons_cons_lock_0"}
// CHECK-DAG:       aie.flow(%[[TILE_1_2]], DMA : 0, %[[TILE_1_2]], DMA : 0) {symbol = @of}
// CHECK:           func.func @some_work(%[[ARG0:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[CORE_1_2:.*]] = aie.core(%[[TILE_1_2]]) {
// CHECK:             aie.use_lock(%[[LOCK_1_2]], AcquireGreaterEqual, 1)
// CHECK:             func.call @some_work(%[[BUFFER_1_2]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[LOCK_1_2_2]], Release, 1)
// CHECK:             aie.use_lock(%[[LOCK_1_2_7]], AcquireGreaterEqual, 1)
// CHECK:             func.call @some_work(%[[BUFFER_1_2_4]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[LOCK_1_2_6]], Release, 1)
// CHECK:             aie.use_lock(%[[LOCK_1_2]], AcquireGreaterEqual, 1)
// CHECK:             func.call @some_work(%[[BUFFER_1_2_1]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[LOCK_1_2_2]], Release, 1)
// CHECK:             aie.use_lock(%[[LOCK_1_2_7]], AcquireGreaterEqual, 1)
// CHECK:             func.call @some_work(%[[BUFFER_1_2_3]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[LOCK_1_2_6]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_1_2:.*]] = aie.mem(%[[TILE_1_2]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_1_2_2]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_1_2_2]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_0]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LOCK_1_2_2]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_1]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(S2MM, 0, ^bb5, ^bb8)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[LOCK_1_2_6]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_3]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2_7]], Release, 1)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[LOCK_1_2_6]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_4]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2_7]], Release, 1)
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[LOCK_1_2_6]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_5]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2_7]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb8:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }
module @same_core {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    aie.flow(%tile12, DMA : 0, %tile12, DMA : 0) {symbol = @of}
    aie.objectfifo @of (%tile12, {%tile12}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    func.func @some_work(%line_in:memref<16xi32>) -> () {
      return
    }
    %core12 = aie.core(%tile12) {
      // this acquires 2 elements
      %subview0 = aie.objectfifo.acquire @of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elem00 = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      func.call @some_work(%elem00) : (memref<16xi32>) -> ()
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
