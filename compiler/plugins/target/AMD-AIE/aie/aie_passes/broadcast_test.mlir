
// RUN: iree-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           memref.global "public" @broadcast_of_0_cons : memref<16xi32>
// CHECK:           memref.global "public" @broadcast_of_1_cons : memref<16xi32>
// CHECK:           memref.global "public" @broadcast_of_2_cons : memref<16xi32>
// CHECK:           memref.global "public" @broadcast_of_3_cons : memref<16xi32>
// CHECK:           memref.global "public" @broadcast_of : memref<16xi32>
// CHECK:           %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:           %[[TILE_1_3:.*]] = aie.tile(1, 3)
// CHECK:           %[[TILE_1_4:.*]] = aie.tile(1, 4)
// CHECK:           %[[TILE_3_2:.*]] = aie.tile(3, 2)
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[BROADCAST_OF_0_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "broadcast_of_0_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[BROADCAST_OF_0_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "broadcast_of_0_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[BROADCAST_OF_0_CONS_LOCK_0:.*]] = aie.lock(%[[TILE_1_2]], 0) {init = 0 : i32, sym_name = "broadcast_of_0_cons_lock_0"}
// CHECK:           %[[BROADCAST_OF_0_CONS_LOCK_1:.*]] = aie.lock(%[[TILE_1_2]], 1) {init = 0 : i32, sym_name = "broadcast_of_0_cons_lock_1"}
// CHECK:           %[[BROADCAST_OF_1_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_1_4]]) {sym_name = "broadcast_of_1_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[BROADCAST_OF_1_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_1_4]]) {sym_name = "broadcast_of_1_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[BROADCAST_OF_1_CONS_BUFF_2:.*]] = aie.buffer(%[[TILE_1_4]]) {sym_name = "broadcast_of_1_cons_buff_2"} : memref<16xi32>
// CHECK:           %[[BROADCAST_OF_1_CONS_LOCK_0:.*]] = aie.lock(%[[TILE_1_4]], 0) {init = 0 : i32, sym_name = "broadcast_of_1_cons_lock_0"}
// CHECK:           %[[BROADCAST_OF_1_CONS_LOCK_1:.*]] = aie.lock(%[[TILE_1_4]], 1) {init = 0 : i32, sym_name = "broadcast_of_1_cons_lock_1"}
// CHECK:           %[[BROADCAST_OF_1_CONS_LOCK_2:.*]] = aie.lock(%[[TILE_1_4]], 2) {init = 0 : i32, sym_name = "broadcast_of_1_cons_lock_2"}
// CHECK:           %[[BROADCAST_OF_2_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_3_2]]) {sym_name = "broadcast_of_2_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[BROADCAST_OF_2_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_3_2]]) {sym_name = "broadcast_of_2_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[BROADCAST_OF_2_CONS_BUFF_2:.*]] = aie.buffer(%[[TILE_3_2]]) {sym_name = "broadcast_of_2_cons_buff_2"} : memref<16xi32>
// CHECK:           %[[BROADCAST_OF_2_CONS_BUFF_3:.*]] = aie.buffer(%[[TILE_3_2]]) {sym_name = "broadcast_of_2_cons_buff_3"} : memref<16xi32>
// CHECK:           %[[BROADCAST_OF_2_CONS_LOCK_0:.*]] = aie.lock(%[[TILE_3_2]], 0) {init = 0 : i32, sym_name = "broadcast_of_2_cons_lock_0"}
// CHECK:           %[[BROADCAST_OF_2_CONS_LOCK_1:.*]] = aie.lock(%[[TILE_3_2]], 1) {init = 0 : i32, sym_name = "broadcast_of_2_cons_lock_1"}
// CHECK:           %[[BROADCAST_OF_2_CONS_LOCK_2:.*]] = aie.lock(%[[TILE_3_2]], 2) {init = 0 : i32, sym_name = "broadcast_of_2_cons_lock_2"}
// CHECK:           %[[BROADCAST_OF_2_CONS_LOCK_3:.*]] = aie.lock(%[[TILE_3_2]], 3) {init = 0 : i32, sym_name = "broadcast_of_2_cons_lock_3"}
// CHECK:           %[[BROADCAST_OF_3_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "broadcast_of_3_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[BROADCAST_OF_3_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "broadcast_of_3_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[BROADCAST_OF_3_CONS_BUFF_2:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "broadcast_of_3_cons_buff_2"} : memref<16xi32>
// CHECK:           %[[BROADCAST_OF_3_CONS_LOCK_0:.*]] = aie.lock(%[[TILE_3_3]], 0) {init = 0 : i32, sym_name = "broadcast_of_3_cons_lock_0"}
// CHECK:           %[[BROADCAST_OF_3_CONS_LOCK_1:.*]] = aie.lock(%[[TILE_3_3]], 1) {init = 0 : i32, sym_name = "broadcast_of_3_cons_lock_1"}
// CHECK:           %[[BROADCAST_OF_3_CONS_LOCK_2:.*]] = aie.lock(%[[TILE_3_3]], 2) {init = 0 : i32, sym_name = "broadcast_of_3_cons_lock_2"}
// CHECK:           %[[BROADCAST_OF_BUFF_0:.*]] = aie.buffer(%[[TILE_1_3]]) {sym_name = "broadcast_of_buff_0"} : memref<16xi32>
// CHECK:           %[[BROADCAST_OF_BUFF_1:.*]] = aie.buffer(%[[TILE_1_3]]) {sym_name = "broadcast_of_buff_1"} : memref<16xi32>
// CHECK:           %[[BROADCAST_OF_LOCK_0:.*]] = aie.lock(%[[TILE_1_3]], 0) {init = 0 : i32, sym_name = "broadcast_of_lock_0"}
// CHECK:           %[[BROADCAST_OF_LOCK_1:.*]] = aie.lock(%[[TILE_1_3]], 1) {init = 0 : i32, sym_name = "broadcast_of_lock_1"}
// CHECK:           aie.flow(%[[TILE_1_3]], DMA : 0, %[[TILE_3_3]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_1_3]], DMA : 0, %[[TILE_3_2]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_1_3]], DMA : 0, %[[TILE_1_4]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_1_3]], DMA : 0, %[[TILE_1_2]], DMA : 0)
// CHECK:           func.func @some_work(%[[ARG0:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[CORE_1_3:.*]] = aie.core(%[[TILE_1_3]]) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C12:.*]] = arith.constant 12 : index
// CHECK:             %[[C2:.*]] = arith.constant 2 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C12]] step %[[C2]] {
// CHECK:               aie.use_lock(%[[BROADCAST_OF_LOCK_0]], Acquire, 0)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_LOCK_0]], Release, 1)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_LOCK_1]], Acquire, 0)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_LOCK_1]], Release, 1)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_1_2:.*]] = aie.core(%[[TILE_1_2]]) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C12:.*]] = arith.constant 12 : index
// CHECK:             %[[C2:.*]] = arith.constant 2 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C12]] step %[[C2]] {
// CHECK:               aie.use_lock(%[[BROADCAST_OF_0_CONS_LOCK_0]], Acquire, 1)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_0_CONS_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_0_CONS_LOCK_0]], Release, 0)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_0_CONS_LOCK_1]], Acquire, 1)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_0_CONS_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_0_CONS_LOCK_1]], Release, 0)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_1_4:.*]] = aie.core(%[[TILE_1_4]]) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C12:.*]] = arith.constant 12 : index
// CHECK:             %[[C3:.*]] = arith.constant 3 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C12]] step %[[C3]] {
// CHECK:               aie.use_lock(%[[BROADCAST_OF_1_CONS_LOCK_0]], Acquire, 1)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_1_CONS_LOCK_1]], Acquire, 1)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_1_CONS_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_1_CONS_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_1_CONS_LOCK_0]], Release, 0)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_1_CONS_LOCK_1]], Release, 0)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_1_CONS_LOCK_2]], Acquire, 1)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_1_CONS_LOCK_0]], Acquire, 1)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_1_CONS_BUFF_2]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_1_CONS_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_1_CONS_LOCK_2]], Release, 0)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_1_CONS_LOCK_0]], Release, 0)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_1_CONS_LOCK_1]], Acquire, 1)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_1_CONS_LOCK_2]], Acquire, 1)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_1_CONS_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_1_CONS_BUFF_2]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_1_CONS_LOCK_1]], Release, 0)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_1_CONS_LOCK_2]], Release, 0)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_3_2:.*]] = aie.core(%[[TILE_3_2]]) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C12:.*]] = arith.constant 12 : index
// CHECK:             %[[C4:.*]] = arith.constant 4 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C12]] step %[[C4]] {
// CHECK:               aie.use_lock(%[[BROADCAST_OF_2_CONS_LOCK_0]], Acquire, 1)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_2_CONS_LOCK_1]], Acquire, 1)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_2_CONS_LOCK_2]], Acquire, 1)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_2_CONS_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_2_CONS_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_2_CONS_BUFF_2]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_2_CONS_LOCK_0]], Release, 0)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_2_CONS_LOCK_3]], Acquire, 1)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_2_CONS_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_2_CONS_BUFF_2]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_2_CONS_BUFF_3]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_2_CONS_LOCK_1]], Release, 0)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_2_CONS_LOCK_0]], Acquire, 1)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_2_CONS_BUFF_2]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_2_CONS_BUFF_3]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_2_CONS_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_2_CONS_LOCK_2]], Release, 0)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_2_CONS_LOCK_1]], Acquire, 1)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_2_CONS_BUFF_3]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_2_CONS_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_2_CONS_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_2_CONS_LOCK_3]], Release, 0)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_3_3:.*]] = aie.core(%[[TILE_3_3]]) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C12:.*]] = arith.constant 12 : index
// CHECK:             %[[C3:.*]] = arith.constant 3 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C12]] step %[[C3]] {
// CHECK:               aie.use_lock(%[[BROADCAST_OF_3_CONS_LOCK_0]], Acquire, 1)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_3_CONS_LOCK_1]], Acquire, 1)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_3_CONS_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_3_CONS_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_3_CONS_LOCK_0]], Release, 0)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_3_CONS_LOCK_2]], Acquire, 1)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_3_CONS_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_3_CONS_BUFF_2]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_3_CONS_LOCK_1]], Release, 0)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_3_CONS_LOCK_0]], Acquire, 1)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_3_CONS_BUFF_2]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_3_CONS_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_3_CONS_LOCK_2]], Release, 0)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_1_3:.*]] = aie.mem(%[[TILE_1_3]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_LOCK_0]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_BUFF_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[BROADCAST_OF_LOCK_0]], Release, 0)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_LOCK_1]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_BUFF_1]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[BROADCAST_OF_LOCK_1]], Release, 0)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_1_2:.*]] = aie.mem(%[[TILE_1_2]]) {
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_0_CONS_LOCK_0]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_0_CONS_BUFF_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[BROADCAST_OF_0_CONS_LOCK_0]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_0_CONS_LOCK_1]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_0_CONS_BUFF_1]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[BROADCAST_OF_0_CONS_LOCK_1]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_1_4:.*]] = aie.mem(%[[TILE_1_4]]) {
// CHECK:             %[[VAL_2:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_1_CONS_LOCK_0]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_1_CONS_BUFF_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[BROADCAST_OF_1_CONS_LOCK_0]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_1_CONS_LOCK_1]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_1_CONS_BUFF_1]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[BROADCAST_OF_1_CONS_LOCK_1]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_1_CONS_LOCK_2]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_1_CONS_BUFF_2]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[BROADCAST_OF_1_CONS_LOCK_2]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_3_2:.*]] = aie.mem(%[[TILE_3_2]]) {
// CHECK:             %[[VAL_3:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_2_CONS_LOCK_0]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_2_CONS_BUFF_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[BROADCAST_OF_2_CONS_LOCK_0]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_2_CONS_LOCK_1]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_2_CONS_BUFF_1]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[BROADCAST_OF_2_CONS_LOCK_1]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_2_CONS_LOCK_2]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_2_CONS_BUFF_2]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[BROADCAST_OF_2_CONS_LOCK_2]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_2_CONS_LOCK_3]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_2_CONS_BUFF_3]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[BROADCAST_OF_2_CONS_LOCK_3]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_3_3:.*]] = aie.mem(%[[TILE_3_3]]) {
// CHECK:             %[[VAL_4:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_3_CONS_LOCK_0]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_3_CONS_BUFF_0]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[BROADCAST_OF_3_CONS_LOCK_0]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_3_CONS_LOCK_1]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_3_CONS_BUFF_1]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[BROADCAST_OF_3_CONS_LOCK_1]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_3_CONS_LOCK_2]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_3_CONS_BUFF_2]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[BROADCAST_OF_3_CONS_LOCK_2]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @broadcast {
 aie.device(xcvc1902) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile14 = aie.tile(1, 4)
    %tile32 = aie.tile(3, 2)
    %tile33 = aie.tile(3, 3)
    aie.objectfifo @broadcast_of (%tile13, {%tile12, %tile14, %tile32, %tile33}, [2, 2, 3, 4, 3]) : !aie.objectfifo<memref<16xi32>>
    func.func @some_work(%lineOut : memref<16xi32>) -> () {
        return
    }
    %core13 = aie.core(%tile13) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index
        scf.for %indexInHeight = %c0 to %height step %c1 {
            %subview = aie.objectfifo.acquire @broadcast_of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
            %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Produce, 1)
        }
        aie.end
    }
    %core12 = aie.core(%tile12) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index
        scf.for %indexInHeight = %c0 to %height step %c1 {
            %subview = aie.objectfifo.acquire @broadcast_of (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
            %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Consume, 1)
        }
        aie.end
    }
    %core14 = aie.core(%tile14) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index
        scf.for %indexInHeight = %c0 to %height step %c1 {
            %subview = aie.objectfifo.acquire @broadcast_of (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem1 = aie.objectfifo.subview.access %subview[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            func.call @some_work(%elem1) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Consume, 2)
        }
        aie.end
    }
    %core32 = aie.core(%tile32) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index
        scf.for %indexInHeight = %c0 to %height step %c1 {
            %subview = aie.objectfifo.acquire @broadcast_of (Consume, 3) : !aie.objectfifosubview<memref<16xi32>>
            %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem1 = aie.objectfifo.subview.access %subview[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem2 = aie.objectfifo.subview.access %subview[2] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            func.call @some_work(%elem1) : (memref<16xi32>) -> ()
            func.call @some_work(%elem2) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Consume, 1)
        }
        aie.end
    }
    %core33 = aie.core(%tile33) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index
        scf.for %indexInHeight = %c0 to %height step %c1 {
            %subview = aie.objectfifo.acquire @broadcast_of (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem1 = aie.objectfifo.subview.access %subview[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            func.call @some_work(%elem1) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Consume, 1)
        }
        aie.end
    }
 }
}
