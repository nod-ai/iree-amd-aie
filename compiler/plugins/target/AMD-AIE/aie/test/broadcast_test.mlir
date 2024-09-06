
// RUN: iree-opt --amdaie-objectFifo-stateful-transform %s | FileCheck %s

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// The script is designed to make adding checks to
// a test case fast, it is *not* designed to be authoritative
// about what constitutes a good test! The CHECK should be
// minimized and named to reflect the test intent.


// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @broadcast_of : memref<16xi32>
// CHECK-DAG:       %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK-DAG:       %[[TILE_1_3:.*]] = aie.tile(1, 3)
// CHECK-DAG:       %[[TILE_1_4:.*]] = aie.tile(1, 4)
// CHECK-DAG:       %[[TILE_3_2:.*]] = aie.tile(3, 2)
// CHECK-DAG:       %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK-DAG:       %[[BUFFER_1_3:.*]] = aie.buffer(%[[TILE_1_3]]) {sym_name = "broadcast_of_prod_buff_0_0"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_1_3_0:.*]] = aie.buffer(%[[TILE_1_3]]) {sym_name = "broadcast_of_prod_buff_0_1"} : memref<16xi32>
// CHECK-DAG:       %[[LOCK_1_3:.*]] = aie.lock(%[[TILE_1_3]]) {init = 2 : i8, sym_name = "broadcast_of_prod_prod_lock_0"}
// CHECK-DAG:       %[[LOCK_1_3_1:.*]] = aie.lock(%[[TILE_1_3]]) {init = 0 : i8, sym_name = "broadcast_of_prod_cons_lock_0"}
// CHECK-DAG:       %[[BUFFER_1_2:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "broadcast_of_cons_buff_0_0"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_1_2_2:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "broadcast_of_cons_buff_0_1"} : memref<16xi32>
// CHECK-DAG:       %[[LOCK_1_2:.*]] = aie.lock(%[[TILE_1_2]]) {init = 2 : i8, sym_name = "broadcast_of_cons_prod_lock_0"}
// CHECK-DAG:       %[[LOCK_1_2_3:.*]] = aie.lock(%[[TILE_1_2]]) {init = 0 : i8, sym_name = "broadcast_of_cons_cons_lock_0"}
// CHECK-DAG:       %[[BUFFER_1_4:.*]] = aie.buffer(%[[TILE_1_4]]) {sym_name = "broadcast_of_cons_buff_1_0"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_1_4_4:.*]] = aie.buffer(%[[TILE_1_4]]) {sym_name = "broadcast_of_cons_buff_1_1"} : memref<16xi32>
// CHECK-DAG:       %[[LOCK_1_4:.*]] = aie.lock(%[[TILE_1_4]]) {init = 2 : i8, sym_name = "broadcast_of_cons_prod_lock_1"}
// CHECK-DAG:       %[[LOCK_1_4_5:.*]] = aie.lock(%[[TILE_1_4]]) {init = 0 : i8, sym_name = "broadcast_of_cons_cons_lock_1"}
// CHECK-DAG:       %[[BUFFER_3_2:.*]] = aie.buffer(%[[TILE_3_2]]) {sym_name = "broadcast_of_cons_buff_2_0"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_3_2_6:.*]] = aie.buffer(%[[TILE_3_2]]) {sym_name = "broadcast_of_cons_buff_2_1"} : memref<16xi32>
// CHECK-DAG:       %[[LOCK_3_2:.*]] = aie.lock(%[[TILE_3_2]]) {init = 2 : i8, sym_name = "broadcast_of_cons_prod_lock_2"}
// CHECK-DAG:       %[[LOCK_3_2_7:.*]] = aie.lock(%[[TILE_3_2]]) {init = 0 : i8, sym_name = "broadcast_of_cons_cons_lock_2"}
// CHECK-DAG:       %[[BUFFER_3_3:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "broadcast_of_cons_buff_3_0"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_3_3_8:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "broadcast_of_cons_buff_3_1"} : memref<16xi32>
// CHECK-DAG:       %[[LOCK_3_3:.*]] = aie.lock(%[[TILE_3_3]]) {init = 2 : i8, sym_name = "broadcast_of_cons_prod_lock_3"}
// CHECK-DAG:       %[[LOCK_3_3_9:.*]] = aie.lock(%[[TILE_3_3]]) {init = 0 : i8, sym_name = "broadcast_of_cons_cons_lock_3"}
// CHECK-DAG:       aie.flow(%[[TILE_1_3]], DMA : 0, %[[TILE_3_3]], DMA : 0) {symbol = @broadcast_of}
// CHECK-DAG:       aie.flow(%[[TILE_1_3]], DMA : 0, %[[TILE_3_2]], DMA : 0) {symbol = @broadcast_of}
// CHECK-DAG:       aie.flow(%[[TILE_1_3]], DMA : 0, %[[TILE_1_4]], DMA : 0) {symbol = @broadcast_of}
// CHECK-DAG:       aie.flow(%[[TILE_1_3]], DMA : 0, %[[TILE_1_2]], DMA : 0) {symbol = @broadcast_of}
// CHECK:           func.func @some_work(%[[ARG0:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[CORE_1_3:.*]] = aie.core(%[[TILE_1_3]]) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C2:.*]] = arith.constant 2 : index
// CHECK:             %[[C12:.*]] = arith.constant 12 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C12]] step %[[C2]] {
// CHECK:               aie.use_lock(%[[LOCK_1_3]], AcquireGreaterEqual, 1)
// CHECK:               func.call @some_work(%[[BUFFER_1_3]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[LOCK_1_3_1]], Release, 1)
// CHECK:               aie.use_lock(%[[LOCK_1_3]], AcquireGreaterEqual, 1)
// CHECK:               func.call @some_work(%[[BUFFER_1_3_0]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[LOCK_1_3_1]], Release, 1)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_1_2:.*]] = aie.core(%[[TILE_1_2]]) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C2:.*]] = arith.constant 2 : index
// CHECK:             %[[C12:.*]] = arith.constant 12 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C12]] step %[[C2]] {
// CHECK:               aie.use_lock(%[[LOCK_1_2_3]], AcquireGreaterEqual, 1)
// CHECK:               func.call @some_work(%[[BUFFER_1_2]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[LOCK_1_2]], Release, 1)
// CHECK:               aie.use_lock(%[[LOCK_1_2_3]], AcquireGreaterEqual, 1)
// CHECK:               func.call @some_work(%[[BUFFER_1_2_2]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[LOCK_1_2]], Release, 1)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_1_4:.*]] = aie.core(%[[TILE_1_4]]) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C3:.*]] = arith.constant 3 : index
// CHECK:             %[[C12:.*]] = arith.constant 12 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C12]] step %[[C3]] {
// CHECK:               aie.use_lock(%[[LOCK_1_4_5]], AcquireGreaterEqual, 2)
// CHECK:               func.call @some_work(%[[BUFFER_1_4]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BUFFER_1_4_4]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[LOCK_1_4]], Release, 2)
// CHECK:               aie.use_lock(%[[LOCK_1_4_5]], AcquireGreaterEqual, 2)
// CHECK:               func.call @some_work(%[[BUFFER_1_4]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BUFFER_1_4_4]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[LOCK_1_4]], Release, 2)
// CHECK:               aie.use_lock(%[[LOCK_1_4_5]], AcquireGreaterEqual, 2)
// CHECK:               func.call @some_work(%[[BUFFER_1_4]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BUFFER_1_4_4]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[LOCK_1_4]], Release, 2)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_3_2:.*]] = aie.core(%[[TILE_3_2]]) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C4:.*]] = arith.constant 4 : index
// CHECK:             %[[C12:.*]] = arith.constant 12 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C12]] step %[[C4]] {
// CHECK:               aie.use_lock(%[[LOCK_3_2_7]], AcquireGreaterEqual, 3)
// CHECK:               func.call @some_work(%[[BUFFER_3_2]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BUFFER_3_2_6]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BUFFER_3_2]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[LOCK_3_2]], Release, 1)
// CHECK:               aie.use_lock(%[[LOCK_3_2_7]], AcquireGreaterEqual, 3)
// CHECK:               func.call @some_work(%[[BUFFER_3_2_6]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BUFFER_3_2]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BUFFER_3_2_6]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[LOCK_3_2]], Release, 1)
// CHECK:               aie.use_lock(%[[LOCK_3_2_7]], AcquireGreaterEqual, 3)
// CHECK:               func.call @some_work(%[[BUFFER_3_2]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BUFFER_3_2_6]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BUFFER_3_2]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[LOCK_3_2]], Release, 1)
// CHECK:               aie.use_lock(%[[LOCK_3_2_7]], AcquireGreaterEqual, 3)
// CHECK:               func.call @some_work(%[[BUFFER_3_2_6]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BUFFER_3_2]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BUFFER_3_2_6]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[LOCK_3_2]], Release, 1)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_3_3:.*]] = aie.core(%[[TILE_3_3]]) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C3:.*]] = arith.constant 3 : index
// CHECK:             %[[C12:.*]] = arith.constant 12 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C12]] step %[[C3]] {
// CHECK:               aie.use_lock(%[[LOCK_3_3_9]], AcquireGreaterEqual, 2)
// CHECK:               func.call @some_work(%[[BUFFER_3_3]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BUFFER_3_3_8]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[LOCK_3_3]], Release, 1)
// CHECK:               aie.use_lock(%[[LOCK_3_3_9]], AcquireGreaterEqual, 2)
// CHECK:               func.call @some_work(%[[BUFFER_3_3]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BUFFER_3_3_8]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[LOCK_3_3]], Release, 1)
// CHECK:               aie.use_lock(%[[LOCK_3_3_9]], AcquireGreaterEqual, 2)
// CHECK:               func.call @some_work(%[[BUFFER_3_3]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BUFFER_3_3_8]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[LOCK_3_3]], Release, 1)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_1_3:.*]] = aie.mem(%[[TILE_1_3]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_1_3_1]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_3]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_3]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_1_3_1]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_3_0]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_3]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_1_2:.*]] = aie.mem(%[[TILE_1_2]]) {
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_1_2]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2_3]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_1_2]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_2]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2_3]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_1_4:.*]] = aie.mem(%[[TILE_1_4]]) {
// CHECK:             %[[VAL_2:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_1_4]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_4]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_4_5]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_1_4]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_4_4]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_4_5]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_3_2:.*]] = aie.mem(%[[TILE_3_2]]) {
// CHECK:             %[[VAL_3:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_3_2]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_3_2]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_3_2_7]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_3_2]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_3_2_6]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_3_2_7]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_3_3:.*]] = aie.mem(%[[TILE_3_3]]) {
// CHECK:             %[[VAL_4:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_3_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_3_3]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_3_3_9]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_3_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_3_3_8]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_3_3_9]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }


module @broadcast {
 aie.device(npu1_4col) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile14 = aie.tile(1, 4)
    %tile32 = aie.tile(3, 2)
    %tile33 = aie.tile(3, 3)
    aie.flow(%tile13, DMA : 0, %tile33, DMA : 0) {symbol = @broadcast_of}
    aie.flow(%tile13, DMA : 0, %tile32, DMA : 0) {symbol = @broadcast_of}
    aie.flow(%tile13, DMA : 0, %tile14, DMA : 0) {symbol = @broadcast_of}
    aie.flow(%tile13, DMA : 0, %tile12, DMA : 0) {symbol = @broadcast_of}
    aie.objectfifo @broadcast_of (%tile13, {%tile12, %tile14, %tile32, %tile33}, [2]) : !aie.objectfifo<memref<16xi32>>
    func.func @some_work(%lineOut : memref<16xi32>) -> () {
        return
    }
    %core13 = aie.core(%tile13) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %height = arith.constant 12 : index
        scf.for %indexInHeight = %c0 to %height step %c2 {
            %subview = aie.objectfifo.acquire @broadcast_of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
            %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Produce, 1)
            %subview1 = aie.objectfifo.acquire @broadcast_of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
            %elem1 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem1) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Produce, 1)
        }
        aie.end
    }
    %core12 = aie.core(%tile12) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %height = arith.constant 12 : index
        scf.for %indexInHeight = %c0 to %height step %c2 {
            %subview = aie.objectfifo.acquire @broadcast_of (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
            %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Consume, 1)
            %subview1 = aie.objectfifo.acquire @broadcast_of (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
            %elem1 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem1) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Consume, 1)
        }
        aie.end
    }
    %core14 = aie.core(%tile14) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c3 = arith.constant 3 : index
        %height = arith.constant 12 : index
        scf.for %indexInHeight = %c0 to %height step %c3 {
            %subview = aie.objectfifo.acquire @broadcast_of (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem1 = aie.objectfifo.subview.access %subview[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            func.call @some_work(%elem1) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Consume, 2)
            %subview1 = aie.objectfifo.acquire @broadcast_of (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elem2 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem3 = aie.objectfifo.subview.access %subview1[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem2) : (memref<16xi32>) -> ()
            func.call @some_work(%elem3) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Consume, 2)
            %subview2 = aie.objectfifo.acquire @broadcast_of (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elem4 = aie.objectfifo.subview.access %subview2[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem5 = aie.objectfifo.subview.access %subview2[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem4) : (memref<16xi32>) -> ()
            func.call @some_work(%elem5) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Consume, 2)
        }
        aie.end
    }
    %core32 = aie.core(%tile32) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %height = arith.constant 12 : index
        scf.for %indexInHeight = %c0 to %height step %c4 {
            %subview = aie.objectfifo.acquire @broadcast_of (Consume, 3) : !aie.objectfifosubview<memref<16xi32>>
            %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem1 = aie.objectfifo.subview.access %subview[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem2 = aie.objectfifo.subview.access %subview[2] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            func.call @some_work(%elem1) : (memref<16xi32>) -> ()
            func.call @some_work(%elem2) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Consume, 1)
            %subview1 = aie.objectfifo.acquire @broadcast_of (Consume, 3) : !aie.objectfifosubview<memref<16xi32>>
            %elem3 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem4 = aie.objectfifo.subview.access %subview1[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem5 = aie.objectfifo.subview.access %subview1[2] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem3) : (memref<16xi32>) -> ()
            func.call @some_work(%elem4) : (memref<16xi32>) -> ()
            func.call @some_work(%elem5) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Consume, 1)

            %subview2 = aie.objectfifo.acquire @broadcast_of (Consume, 3) : !aie.objectfifosubview<memref<16xi32>>
            %elem6 = aie.objectfifo.subview.access %subview2[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem7 = aie.objectfifo.subview.access %subview2[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem8 = aie.objectfifo.subview.access %subview2[2] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem6) : (memref<16xi32>) -> ()
            func.call @some_work(%elem7) : (memref<16xi32>) -> ()
            func.call @some_work(%elem8) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Consume, 1)

            %subview3 = aie.objectfifo.acquire @broadcast_of (Consume, 3) : !aie.objectfifosubview<memref<16xi32>>
            %elem9 = aie.objectfifo.subview.access %subview3[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem10 = aie.objectfifo.subview.access %subview3[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem11 = aie.objectfifo.subview.access %subview3[2] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem9) : (memref<16xi32>) -> ()
            func.call @some_work(%elem10) : (memref<16xi32>) -> ()
            func.call @some_work(%elem11) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Consume, 1)
        }
        aie.end
    }
    %core33 = aie.core(%tile33) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c3 = arith.constant 3 : index
        %height = arith.constant 12 : index
        scf.for %indexInHeight = %c0 to %height step %c3 {
            %subview = aie.objectfifo.acquire @broadcast_of (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem1 = aie.objectfifo.subview.access %subview[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            func.call @some_work(%elem1) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Consume, 1)

            %subview1 = aie.objectfifo.acquire @broadcast_of (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elem2 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem3 = aie.objectfifo.subview.access %subview1[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem2) : (memref<16xi32>) -> ()
            func.call @some_work(%elem3) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Consume, 1)

            %subview2 = aie.objectfifo.acquire @broadcast_of (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elem4 = aie.objectfifo.subview.access %subview2[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem5 = aie.objectfifo.subview.access %subview2[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem4) : (memref<16xi32>) -> ()
            func.call @some_work(%elem5) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Consume, 1)
        }
        aie.end
    }
 }
}
