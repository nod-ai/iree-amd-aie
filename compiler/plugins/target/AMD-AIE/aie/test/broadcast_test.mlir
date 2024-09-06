
// RUN: iree-opt --amdaie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
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
// CHECK:           %[[BROADCAST_OF_0_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_1_2]]) {init = 2 : i8, sym_name = "broadcast_of_0_cons_prod_lock"}
// CHECK:           %[[BROADCAST_OF_0_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_1_2]]) {init = 0 : i8, sym_name = "broadcast_of_0_cons_cons_lock"}
// CHECK:           %[[BROADCAST_OF_1_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_1_4]]) {sym_name = "broadcast_of_1_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[BROADCAST_OF_1_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_1_4]]) {sym_name = "broadcast_of_1_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[BROADCAST_OF_1_CONS_BUFF_2:.*]] = aie.buffer(%[[TILE_1_4]]) {sym_name = "broadcast_of_1_cons_buff_2"} : memref<16xi32>
// CHECK:           %[[BROADCAST_OF_1_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_1_4]]) {init = 3 : i8, sym_name = "broadcast_of_1_cons_prod_lock"}
// CHECK:           %[[BROADCAST_OF_1_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_1_4]]) {init = 0 : i8, sym_name = "broadcast_of_1_cons_cons_lock"}
// CHECK:           %[[BROADCAST_OF_2_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_3_2]]) {sym_name = "broadcast_of_2_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[BROADCAST_OF_2_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_3_2]]) {sym_name = "broadcast_of_2_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[BROADCAST_OF_2_CONS_BUFF_2:.*]] = aie.buffer(%[[TILE_3_2]]) {sym_name = "broadcast_of_2_cons_buff_2"} : memref<16xi32>
// CHECK:           %[[BROADCAST_OF_2_CONS_BUFF_3:.*]] = aie.buffer(%[[TILE_3_2]]) {sym_name = "broadcast_of_2_cons_buff_3"} : memref<16xi32>
// CHECK:           %[[BROADCAST_OF_2_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_3_2]]) {init = 4 : i8, sym_name = "broadcast_of_2_cons_prod_lock"}
// CHECK:           %[[BROADCAST_OF_2_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_3_2]]) {init = 0 : i8, sym_name = "broadcast_of_2_cons_cons_lock"}
// CHECK:           %[[BROADCAST_OF_3_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "broadcast_of_3_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[BROADCAST_OF_3_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "broadcast_of_3_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[BROADCAST_OF_3_CONS_BUFF_2:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "broadcast_of_3_cons_buff_2"} : memref<16xi32>
// CHECK:           %[[BROADCAST_OF_3_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_3_3]]) {init = 3 : i8, sym_name = "broadcast_of_3_cons_prod_lock"}
// CHECK:           %[[BROADCAST_OF_3_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_3_3]]) {init = 0 : i8, sym_name = "broadcast_of_3_cons_cons_lock"}
// CHECK:           %[[BROADCAST_OF_BUFF_0:.*]] = aie.buffer(%[[TILE_1_3]]) {sym_name = "broadcast_of_buff_0"} : memref<16xi32>
// CHECK:           %[[BROADCAST_OF_BUFF_1:.*]] = aie.buffer(%[[TILE_1_3]]) {sym_name = "broadcast_of_buff_1"} : memref<16xi32>
// CHECK:           %[[BROADCAST_OF_PROD_LOCK:.*]] = aie.lock(%[[TILE_1_3]]) {init = 2 : i8, sym_name = "broadcast_of_prod_lock"}
// CHECK:           %[[BROADCAST_OF_CONS_LOCK:.*]] = aie.lock(%[[TILE_1_3]]) {init = 0 : i8, sym_name = "broadcast_of_cons_lock"}
// CHECK:           aie.flow(%[[TILE_1_3]], DMA : 0, %[[TILE_3_3]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_1_3]], DMA : 0, %[[TILE_3_2]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_1_3]], DMA : 0, %[[TILE_1_4]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_1_3]], DMA : 0, %[[TILE_1_2]], DMA : 0)
// CHECK:           func.func @some_work(%[[ARG0:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[CORE_1_3:.*]] = aie.core(%[[TILE_1_3]]) {
// CHECK-DAG:         %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:         %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:         %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:         %[[C12:.*]] = arith.constant 12 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C12]] step %[[C2]] {
// CHECK:               aie.use_lock(%[[BROADCAST_OF_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_CONS_LOCK]], Release, 1)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_CONS_LOCK]], Release, 1)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_1_2:.*]] = aie.core(%[[TILE_1_2]]) {
// CHECK-DAG:         %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:         %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:         %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:         %[[C12:.*]] = arith.constant 12 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C12]] step %[[C2]] {
// CHECK:               aie.use_lock(%[[BROADCAST_OF_0_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_0_CONS_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_0_CONS_PROD_LOCK]], Release, 1)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_0_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_0_CONS_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_0_CONS_PROD_LOCK]], Release, 1)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_1_4:.*]] = aie.core(%[[TILE_1_4]]) {
// CHECK-DAG:         %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:         %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:         %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:         %[[C12:.*]] = arith.constant 12 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C12]] step %[[C3]] {
// CHECK:               aie.use_lock(%[[BROADCAST_OF_1_CONS_CONS_LOCK]], AcquireGreaterEqual, 2)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_1_CONS_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_1_CONS_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_1_CONS_PROD_LOCK]], Release, 2)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_1_CONS_CONS_LOCK]], AcquireGreaterEqual, 2)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_1_CONS_BUFF_2]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_1_CONS_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_1_CONS_PROD_LOCK]], Release, 2)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_1_CONS_CONS_LOCK]], AcquireGreaterEqual, 2)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_1_CONS_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_1_CONS_BUFF_2]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_1_CONS_PROD_LOCK]], Release, 2)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_3_2:.*]] = aie.core(%[[TILE_3_2]]) {
// CHECK-DAG:         %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:         %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:         %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:         %[[C12:.*]] = arith.constant 12 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C12]] step %[[C4]] {
// CHECK:               aie.use_lock(%[[BROADCAST_OF_2_CONS_CONS_LOCK]], AcquireGreaterEqual, 3)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_2_CONS_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_2_CONS_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_2_CONS_BUFF_2]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_2_CONS_PROD_LOCK]], Release, 1)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_2_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_2_CONS_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_2_CONS_BUFF_2]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_2_CONS_BUFF_3]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_2_CONS_PROD_LOCK]], Release, 1)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_2_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_2_CONS_BUFF_2]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_2_CONS_BUFF_3]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_2_CONS_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_2_CONS_PROD_LOCK]], Release, 1)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_2_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_2_CONS_BUFF_3]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_2_CONS_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_2_CONS_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_2_CONS_PROD_LOCK]], Release, 1)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_3_3:.*]] = aie.core(%[[TILE_3_3]]) {
// CHECK-DAG:         %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:         %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:         %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:         %[[C12:.*]] = arith.constant 12 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C12]] step %[[C3]] {
// CHECK:               aie.use_lock(%[[BROADCAST_OF_3_CONS_CONS_LOCK]], AcquireGreaterEqual, 2)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_3_CONS_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_3_CONS_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_3_CONS_PROD_LOCK]], Release, 1)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_3_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_3_CONS_BUFF_1]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_3_CONS_BUFF_2]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_3_CONS_PROD_LOCK]], Release, 1)
// CHECK:               aie.use_lock(%[[BROADCAST_OF_3_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               func.call @some_work(%[[BROADCAST_OF_3_CONS_BUFF_2]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[BROADCAST_OF_3_CONS_BUFF_0]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[BROADCAST_OF_3_CONS_PROD_LOCK]], Release, 1)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_1_3:.*]] = aie.mem(%[[TILE_1_3]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_BUFF_0]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[BROADCAST_OF_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_BUFF_1]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[BROADCAST_OF_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_1_2:.*]] = aie.mem(%[[TILE_1_2]]) {
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_0_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_0_CONS_BUFF_0]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[BROADCAST_OF_0_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_0_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_0_CONS_BUFF_1]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[BROADCAST_OF_0_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_1_4:.*]] = aie.mem(%[[TILE_1_4]]) {
// CHECK:             %[[VAL_2:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_1_CONS_BUFF_0]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[BROADCAST_OF_1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_1_CONS_BUFF_1]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[BROADCAST_OF_1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_1_CONS_BUFF_2]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[BROADCAST_OF_1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_3_2:.*]] = aie.mem(%[[TILE_3_2]]) {
// CHECK:             %[[VAL_3:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_2_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_2_CONS_BUFF_0]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[BROADCAST_OF_2_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_2_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_2_CONS_BUFF_1]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[BROADCAST_OF_2_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_2_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_2_CONS_BUFF_2]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[BROADCAST_OF_2_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_2_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_2_CONS_BUFF_3]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[BROADCAST_OF_2_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_3_3:.*]] = aie.mem(%[[TILE_3_3]]) {
// CHECK:             %[[VAL_4:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_3_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_3_CONS_BUFF_0]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[BROADCAST_OF_3_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_3_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_3_CONS_BUFF_1]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[BROADCAST_OF_3_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[BROADCAST_OF_3_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BROADCAST_OF_3_CONS_BUFF_2]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[BROADCAST_OF_3_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
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
    aie.objectfifo @broadcast_of (%tile13, {%tile12, %tile14, %tile32, %tile33}, [2, 2, 3, 4, 3]) : !aie.objectfifo<memref<16xi32>>
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
