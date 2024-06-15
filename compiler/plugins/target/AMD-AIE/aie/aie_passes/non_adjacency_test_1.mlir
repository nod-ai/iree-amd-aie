
// RUN: iree-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           memref.global "public" @objfifo_cons : memref<16xi32>
// CHECK:           memref.global "public" @objfifo : memref<16xi32>
// CHECK:           %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[OBJFIFO_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "objfifo_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[OBJFIFO_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "objfifo_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[OBJFIFO_CONS_LOCK_0:.*]] = aie.lock(%[[TILE_3_3]], 0) {init = 0 : i32, sym_name = "objfifo_cons_lock_0"}
// CHECK:           %[[OBJFIFO_CONS_LOCK_1:.*]] = aie.lock(%[[TILE_3_3]], 1) {init = 0 : i32, sym_name = "objfifo_cons_lock_1"}
// CHECK:           %[[OBJFIFO_BUFF_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "objfifo_buff_0"} : memref<16xi32>
// CHECK:           %[[OBJFIFO_BUFF_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "objfifo_buff_1"} : memref<16xi32>
// CHECK:           %[[OBJFIFO_LOCK_0:.*]] = aie.lock(%[[TILE_1_2]], 0) {init = 0 : i32, sym_name = "objfifo_lock_0"}
// CHECK:           %[[OBJFIFO_LOCK_1:.*]] = aie.lock(%[[TILE_1_2]], 1) {init = 0 : i32, sym_name = "objfifo_lock_1"}
// CHECK:           aie.flow(%[[TILE_1_2]], DMA : 0, %[[TILE_3_3]], DMA : 0)
// CHECK:           func.func @some_work(%[[ARG0:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[CORE_1_2:.*]] = aie.core(%[[TILE_1_2]]) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C12:.*]] = arith.constant 12 : index
// CHECK:             %[[C2:.*]] = arith.constant 2 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C12]] step %[[C2]] {
// CHECK:               aie.use_lock(%[[OBJFIFO_LOCK_0]], Acquire, 0)
// CHECK:               func.call @some_work(%[[OBJFIFO_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[OBJFIFO_LOCK_0]], Release, 1)
// CHECK:               aie.use_lock(%[[OBJFIFO_LOCK_1]], Acquire, 0)
// CHECK:               func.call @some_work(%[[OBJFIFO_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[OBJFIFO_LOCK_1]], Release, 1)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_3_3:.*]] = aie.core(%[[TILE_3_3]]) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C12:.*]] = arith.constant 12 : index
// CHECK:             %[[C2:.*]] = arith.constant 2 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C12]] step %[[C2]] {
// CHECK:               aie.use_lock(%[[OBJFIFO_CONS_LOCK_0]], Acquire, 1)
// CHECK:               func.call @some_work(%[[OBJFIFO_CONS_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[OBJFIFO_CONS_LOCK_0]], Release, 0)
// CHECK:               aie.use_lock(%[[OBJFIFO_CONS_LOCK_1]], Acquire, 1)
// CHECK:               func.call @some_work(%[[OBJFIFO_CONS_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[OBJFIFO_CONS_LOCK_1]], Release, 0)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_1_2:.*]] = aie.mem(%[[TILE_1_2]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OBJFIFO_LOCK_0]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[OBJFIFO_BUFF_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OBJFIFO_LOCK_0]], Release, 0)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[OBJFIFO_LOCK_1]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[OBJFIFO_BUFF_1]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OBJFIFO_LOCK_1]], Release, 0)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_3_3:.*]] = aie.mem(%[[TILE_3_3]]) {
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OBJFIFO_CONS_LOCK_0]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[OBJFIFO_CONS_BUFF_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OBJFIFO_CONS_LOCK_0]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[OBJFIFO_CONS_LOCK_1]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[OBJFIFO_CONS_BUFF_1]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OBJFIFO_CONS_LOCK_1]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @non_adjacency {
    aie.device(xcvc1902) {
        %tile12 = aie.tile(1, 2)
        %tile33 = aie.tile(3, 3)
        aie.objectfifo @objfifo (%tile12, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
        func.func @some_work(%lineOut : memref<16xi32>) -> () {
            return
        }
        %core12 = aie.core(%tile12) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %height = arith.constant 12 : index
            scf.for %indexInHeight = %c0 to %height step %c1 {
                %subview = aie.objectfifo.acquire @objfifo (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @some_work(%elem0) : (memref<16xi32>) -> ()
                aie.objectfifo.release @objfifo (Produce, 1)
            }
            aie.end
        }
        %core33 = aie.core(%tile33) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %height = arith.constant 12 : index
            scf.for %indexInHeight = %c0 to %height step %c1 {
                %subview = aie.objectfifo.acquire @objfifo (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @some_work(%elem0) : (memref<16xi32>) -> ()
                aie.objectfifo.release @objfifo (Consume, 1)
            }
            aie.end
        }
    }
}
