
// RUN: iree-opt --amdaie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           memref.global "public" @of1 : memref<16xi32>
// CHECK:           memref.global "public" @of0 : memref<16xi32>
// CHECK-DAG:       %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK-DAG:       %[[TILE_1_3:.*]] = aie.tile(1, 3)
// CHECK-DAG:       %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK-DAG:       %[[BUFFER_1_2:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of1_prod_buff_0_0"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_1_2_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of1_prod_buff_0_1"} : memref<16xi32>
// CHECK-DAG:       %[[LOCK_1_2:.*]] = aie.lock(%[[TILE_1_2]]) {init = 2 : i8, sym_name = "of1_prod_prod_lock_0"}
// CHECK-DAG:       %[[LOCK_1_2_1:.*]] = aie.lock(%[[TILE_1_2]]) {init = 0 : i8, sym_name = "of1_prod_cons_lock_0"}
// CHECK-DAG:       %[[BUFFER_3_3:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "of1_cons_buff_0_0"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_3_3_2:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "of1_cons_buff_0_1"} : memref<16xi32>
// CHECK-DAG:       %[[LOCK_3_3:.*]] = aie.lock(%[[TILE_3_3]]) {init = 2 : i8, sym_name = "of1_cons_prod_lock_0"}
// CHECK-DAG:       %[[LOCK_3_3_3:.*]] = aie.lock(%[[TILE_3_3]]) {init = 0 : i8, sym_name = "of1_cons_cons_lock_0"}
// CHECK-DAG:       %[[BUFFER_1_2_4:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of0_prod_buff_0_0"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_1_2_5:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of0_prod_buff_0_1"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_1_2_6:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of0_prod_buff_0_2"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_1_2_7:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of0_prod_buff_0_3"} : memref<16xi32>
// CHECK-DAG:       %[[LOCK_1_2_8:.*]] = aie.lock(%[[TILE_1_2]]) {init = 4 : i8, sym_name = "of0_prod_prod_lock_0"}
// CHECK-DAG:       %[[LOCK_1_2_9:.*]] = aie.lock(%[[TILE_1_2]]) {init = 0 : i8, sym_name = "of0_prod_cons_lock_0"}
// CHECK-DAG:       %[[BUFFER_1_3:.*]] = aie.buffer(%[[TILE_1_3]]) {sym_name = "of0_cons_buff_0_0"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_1_3_10:.*]] = aie.buffer(%[[TILE_1_3]]) {sym_name = "of0_cons_buff_0_1"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_1_3_11:.*]] = aie.buffer(%[[TILE_1_3]]) {sym_name = "of0_cons_buff_0_2"} : memref<16xi32>
// CHECK-DAG:       %[[BUFFER_1_3_12:.*]] = aie.buffer(%[[TILE_1_3]]) {sym_name = "of0_cons_buff_0_3"} : memref<16xi32>
// CHECK-DAG:       %[[LOCK_1_3:.*]] = aie.lock(%[[TILE_1_3]]) {init = 4 : i8, sym_name = "of0_cons_prod_lock_0"}
// CHECK-DAG:       %[[LOCK_1_3_13:.*]] = aie.lock(%[[TILE_1_3]]) {init = 0 : i8, sym_name = "of0_cons_cons_lock_0"}
// CHECK-DAG:       aie.flow(%[[TILE_1_2]], DMA : 0, %[[TILE_3_3]], DMA : 0) {symbol = @of1}
// CHECK-DAG:       aie.flow(%[[TILE_1_2]], DMA : 1, %[[TILE_1_3]], DMA : 0) {symbol = @of0}
// CHECK:           %[[MEM_1_2:.*]] = aie.mem(%[[TILE_1_2]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(MM2S, 1, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_1_2_9]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_4]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2_8]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_1_2_9]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_5]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2_8]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LOCK_1_2_9]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_6]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2_8]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LOCK_1_2_9]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_7]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2_8]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(MM2S, 0, ^bb6, ^bb8)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[LOCK_1_2_1]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2]], Release, 1)
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[LOCK_1_2_1]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_0]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2]], Release, 1)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb8:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_1_3:.*]] = aie.mem(%[[TILE_1_3]]) {
// CHECK:             %[[VAL_2:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_1_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_3]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_3_13]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_1_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_3_10]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_3_13]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LOCK_1_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_3_11]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_3_13]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LOCK_1_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_3_12]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_3_13]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_3_3:.*]] = aie.mem(%[[TILE_3_3]]) {
// CHECK:             %[[VAL_3:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
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
module @elementGenerationAIE2 {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile33 = aie.tile(3, 3)
    aie.flow(%tile12, DMA : 0, %tile33, DMA : 0) {symbol = @of1}
    aie.flow(%tile12, DMA : 1, %tile13, DMA : 0) {symbol = @of0}
    // In the shared memory case, the number of elements does not change.
    aie.objectfifo @of0 (%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    // In the non-adjacent memory case, the number of elements depends on the max amount acquired by
    // the processes running on each core (here nothing is specified so it cannot be derived).
    aie.objectfifo @of1 (%tile12, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}
