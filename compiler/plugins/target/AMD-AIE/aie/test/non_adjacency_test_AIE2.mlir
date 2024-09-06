
// RUN: iree-opt --amdaie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           memref.global "public" @of_cons : memref<16xi32>
// CHECK:           memref.global "public" @of : memref<16xi32>
// CHECK:           %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[OF_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "of_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[OF_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "of_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[OF_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_3_3]]) {init = 2 : i8, sym_name = "of_cons_prod_lock"}
// CHECK:           %[[OF_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_3_3]]) {init = 0 : i8, sym_name = "of_cons_cons_lock"}
// CHECK:           %[[OF_BUFF_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of_buff_0"} : memref<16xi32>
// CHECK:           %[[OF_BUFF_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of_buff_1"} : memref<16xi32>
// CHECK:           %[[OF_PROD_LOCK:.*]] = aie.lock(%[[TILE_1_2]]) {init = 2 : i8, sym_name = "of_prod_lock"}
// CHECK:           %[[OF_CONS_LOCK:.*]] = aie.lock(%[[TILE_1_2]]) {init = 0 : i8, sym_name = "of_cons_lock"}
// CHECK:           aie.flow(%[[TILE_1_2]], DMA : 0, %[[TILE_3_3]], DMA : 0)
// CHECK:           func.func @some_work(%[[ARG0:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[CORE_1_2:.*]] = aie.core(%[[TILE_1_2]]) {
// CHECK-DAG:         %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:         %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:         %[[C12:.*]] = arith.constant 12 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C12]] step %[[C2]] {
// CHECK:               aie.use_lock(%[[OF_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               func.call @some_work(%[[OF_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[OF_CONS_LOCK]], Release, 1)
// CHECK:               aie.use_lock(%[[OF_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               func.call @some_work(%[[OF_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[OF_CONS_LOCK]], Release, 1)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_3_3:.*]] = aie.core(%[[TILE_3_3]]) {
// CHECK-DAG:         %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:         %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:         %[[C12:.*]] = arith.constant 12 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C12]] step %[[C2]] {
// CHECK:               aie.use_lock(%[[OF_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               func.call @some_work(%[[OF_CONS_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[OF_CONS_PROD_LOCK]], Release, 1)
// CHECK:               aie.use_lock(%[[OF_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               func.call @some_work(%[[OF_CONS_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[OF_CONS_PROD_LOCK]], Release, 1)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_1_2:.*]] = aie.mem(%[[TILE_1_2]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OF_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_BUFF_0]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[OF_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[OF_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_BUFF_1]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[OF_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_3_3:.*]] = aie.mem(%[[TILE_3_3]]) {
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OF_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_CONS_BUFF_0]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[OF_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[OF_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF_CONS_BUFF_1]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[OF_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @non_adjacency_AIE2 {
    aie.device(xcve2302) {
        %tile12 = aie.tile(1, 2)
        %tile33 = aie.tile(3, 3)
        aie.flow(%tile12, DMA : 0, %tile33, DMA : 0) {symbol = @of}
        aie.objectfifo @of (%tile12, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
        func.func @some_work(%lineOut : memref<16xi32>) -> () {
            return
        }
        %core12 = aie.core(%tile12) {
            %c0 = arith.constant 0 : index
            %c2 = arith.constant 2 : index
            %height = arith.constant 12 : index
            scf.for %indexInHeight = %c0 to %height step %c2 {
                %subview = aie.objectfifo.acquire @of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @some_work(%elem0) : (memref<16xi32>) -> ()
                aie.objectfifo.release @of (Produce, 1)
                %subview1 = aie.objectfifo.acquire @of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem1 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @some_work(%elem1) : (memref<16xi32>) -> ()
                aie.objectfifo.release @of (Produce, 1)
            }
            aie.end
        }
        %core33 = aie.core(%tile33) {
            %c0 = arith.constant 0 : index
            %c2 = arith.constant 2 : index
            %height = arith.constant 12 : index
            scf.for %indexInHeight = %c0 to %height step %c2 {
                %subview = aie.objectfifo.acquire @of (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @some_work(%elem0) : (memref<16xi32>) -> ()
                aie.objectfifo.release @of (Consume, 1)
                %subview1 = aie.objectfifo.acquire @of (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem1 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @some_work(%elem1) : (memref<16xi32>) -> ()
                aie.objectfifo.release @of (Consume, 1)
            }
            aie.end
        }
    }
}
