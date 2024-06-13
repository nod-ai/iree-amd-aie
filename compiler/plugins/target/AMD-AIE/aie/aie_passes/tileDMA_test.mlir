
// RUN: iree-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @objfifo_cons : memref<16xi32>
// CHECK:           memref.global "public" @objfifo : memref<16xi32>
// CHECK:           %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[OBJFIFO_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "objfifo_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[OBJFIFO_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "objfifo_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[OBJFIFO_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_3_3]], 0) {init = 2 : i32, sym_name = "objfifo_cons_prod_lock"}
// CHECK:           %[[OBJFIFO_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_3_3]], 1) {init = 0 : i32, sym_name = "objfifo_cons_cons_lock"}
// CHECK:           %[[OBJFIFO_BUFF_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "objfifo_buff_0"} : memref<16xi32>
// CHECK:           %[[OBJFIFO_BUFF_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "objfifo_buff_1"} : memref<16xi32>
// CHECK:           %[[OBJFIFO_PROD_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 3) {init = 2 : i32, sym_name = "objfifo_prod_lock"}
// CHECK:           %[[OBJFIFO_CONS_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 4) {init = 0 : i32, sym_name = "objfifo_cons_lock"}
// CHECK:           %[[BUFFER_1_2:.*]] = aie.buffer(%[[TILE_1_2]]) : memref<16xi32>
// CHECK:           %[[LOCK_1_2:.*]] = aie.lock(%[[TILE_1_2]], 0)
// CHECK:           %[[BUFFER_1_2_0:.*]] = aie.buffer(%[[TILE_1_2]]) : memref<16xi32>
// CHECK:           %[[LOCK_1_2_1:.*]] = aie.lock(%[[TILE_1_2]], 1)
// CHECK:           %[[BUFFER_1_2_2:.*]] = aie.buffer(%[[TILE_1_2]]) : memref<16xi32>
// CHECK:           %[[LOCK_1_2_3:.*]] = aie.lock(%[[TILE_1_2]], 2)
// CHECK:           aie.flow(%[[TILE_1_2]], DMA : 1, %[[TILE_3_3]], DMA : 0)
// CHECK:           func.func @some_work(%[[ARG0:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[CORE_1_2:.*]] = aie.core(%[[TILE_1_2]]) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C12:.*]] = arith.constant 12 : index
// CHECK:             %[[C2:.*]] = arith.constant 2 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C12]] step %[[C2]] {
// CHECK:               aie.use_lock(%[[OBJFIFO_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               func.call @some_work(%[[OBJFIFO_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[OBJFIFO_CONS_LOCK]], Release, 1)
// CHECK:               aie.use_lock(%[[OBJFIFO_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               func.call @some_work(%[[OBJFIFO_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[OBJFIFO_CONS_LOCK]], Release, 1)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_1_2:.*]] = aie.mem(%[[TILE_1_2]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_1_2]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[LOCK_1_2]], Release, 0)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_1_2_1]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[LOCK_1_2_1]], Release, 0)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LOCK_1_2_3]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_2]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[LOCK_1_2_3]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb5:
// CHECK:             %[[VAL_2:.*]] = aie.dma_start(MM2S, 1, ^bb6, ^bb8)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[OBJFIFO_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OBJFIFO_BUFF_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OBJFIFO_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[OBJFIFO_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OBJFIFO_BUFF_1]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OBJFIFO_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb8:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_3_3:.*]] = aie.mem(%[[TILE_3_3]]) {
// CHECK:             %[[VAL_3:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OBJFIFO_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OBJFIFO_CONS_BUFF_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OBJFIFO_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[OBJFIFO_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OBJFIFO_CONS_BUFF_1]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[OBJFIFO_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @tileDMA_channels {
    aie.device(npu1_4col) {
        %tile12 = aie.tile(1, 2)
        %tile33 = aie.tile(3, 3)
        %buff0 = aie.buffer(%tile12) : memref<16xi32>
        %lock0 = aie.lock(%tile12, 0)
        %buff1 = aie.buffer(%tile12) : memref<16xi32>
        %lock1 = aie.lock(%tile12, 1)
        %buff2 = aie.buffer(%tile12) : memref<16xi32>
        %lock2 = aie.lock(%tile12, 2)
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
        %mem12 = aie.mem(%tile12) {
            %dma1 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
        ^bb1:
            aie.use_lock(%lock0, Acquire, 1)
            aie.dma_bd(%buff0 : memref<16xi32>, 0, 16)
            aie.use_lock(%lock0, Release, 0)
            aie.next_bd ^bb2
        ^bb2:
            aie.use_lock(%lock1, Acquire, 1)
            aie.dma_bd(%buff1 : memref<16xi32>, 0, 16)
            aie.use_lock(%lock1, Release, 0)
            aie.next_bd ^bb1
        ^bb3:
            %dma2 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
        ^bb4:
            aie.use_lock(%lock2, Acquire, 0)
            aie.dma_bd(%buff2 : memref<16xi32>, 0, 16)
            aie.use_lock(%lock2, Release, 1)
            aie.next_bd ^bb4
        ^bb5:
            aie.end
        }
    }
}
