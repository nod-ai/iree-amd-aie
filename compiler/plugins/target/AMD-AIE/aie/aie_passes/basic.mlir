
// RUN: iree-opt --aie-assign-bd-ids --split-input-file %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           %[[TILE_0_0:.*]] = aie.tile(0, 0)
// CHECK:           %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[DOUBLE_BUFFER:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "double_buffer"} : memref<32xi32>
// CHECK:           %[[BUFFER_0_1:.*]] = aie.buffer(%[[TILE_0_1]]) : memref<32xi32>
// CHECK:           %[[LOCK_X:.*]] = aie.lock(%[[TILE_0_2]]) {init = 1 : i32, sym_name = "lock_X"}
// CHECK:           %[[LOCK_Y:.*]] = aie.lock(%[[TILE_0_2]]) {init = 0 : i32, sym_name = "lock_Y"}
// CHECK:           %[[MEM_0_2:.*]] = aie.mem(%[[TILE_0_2]]) {
// CHECK:             %[[PLAYER_A:.*]] = aie.dma(S2MM, 0) {sym_name = "player_a"} [{
// CHECK:               aie.use_lock(%[[LOCK_Y]], Acquire, 0)
// CHECK:               aie.dma_bd(%[[DOUBLE_BUFFER]] : memref<32xi32>, 0) {bd_id = 0 : i32, next_bd_id = 1 : i32}
// CHECK:               aie.use_lock(%[[LOCK_Y]], Release, 0)
// CHECK:             }, {
// CHECK:               aie.use_lock(%[[LOCK_X]], Acquire, 1)
// CHECK:               aie.dma_bd(%[[DOUBLE_BUFFER]] : memref<32xi32>) {bd_id = 1 : i32, next_bd_id = 2 : i32}
// CHECK:               aie.use_lock(%[[LOCK_X]], Release, -1)
// CHECK:             }, {
// CHECK:               aie.use_lock(%[[LOCK_Y]], Acquire) {acq_en = false}
// CHECK:               aie.dma_bd(%[[DOUBLE_BUFFER]] : memref<32xi32>) {bd_id = 2 : i32, next_bd_id = 0 : i32}
// CHECK:               aie.use_lock(%[[LOCK_Y]], Release, 1)
// CHECK:             }]
// CHECK:             %[[PLAYER_B:.*]] = aie.dma(S2MM, 1) {sym_name = "player_b"} [{
// CHECK:               aie.use_lock(%[[LOCK_Y]], Acquire, 1)
// CHECK:               aie.dma_bd(%[[DOUBLE_BUFFER]] : memref<32xi32>, 0) {bd_id = 3 : i32, next_bd_id = 4 : i32}
// CHECK:               aie.use_lock(%[[LOCK_Y]], Release, 0)
// CHECK:             }, {
// CHECK:               aie.use_lock(%[[LOCK_X]], Acquire, 1)
// CHECK:               aie.dma_bd(%[[DOUBLE_BUFFER]] : memref<32xi32>) {bd_id = 4 : i32, next_bd_id = 5 : i32}
// CHECK:               aie.use_lock(%[[LOCK_X]], Release, -1)
// CHECK:             }, {
// CHECK:               aie.use_lock(%[[LOCK_Y]], Acquire) {acq_en = false}
// CHECK:               aie.dma_bd(%[[DOUBLE_BUFFER]] : memref<32xi32>) {bd_id = 5 : i32, next_bd_id = 3 : i32}
// CHECK:               aie.use_lock(%[[LOCK_Y]], Release, -1)
// CHECK:             }]
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEMTILE_DMA_0_1:.*]] = aie.memtile_dma(%[[TILE_0_1]]) {
// CHECK:             %[[LOCK_0_1:.*]] = aie.lock(%[[TILE_0_1]]) {init = 1 : i32}
// CHECK:             %[[LOCK_0_1_0:.*]] = aie.lock(%[[TILE_0_1]]) {init = 0 : i32}
// CHECK:             %[[VAL_0:.*]] = aie.dma(S2MM, 0) {loop = false, repeat_count = 10 : i32} [{
// CHECK:               aie.use_lock(%[[LOCK_0_1]], AcquireGreaterEqual)
// CHECK:               aie.dma_bd(%[[BUFFER_0_1]] : memref<32xi32>) {bd_id = 0 : i32}
// CHECK:               aie.use_lock(%[[LOCK_0_1_0]], Release)
// CHECK:             }]
// CHECK:             %[[VAL_1:.*]] = aie.dma(MM2S, 0) {loop = false, repeat_count = 10 : i32} [{
// CHECK:               aie.use_lock(%[[LOCK_0_1_0]], AcquireGreaterEqual)
// CHECK:               aie.dma_bd(%[[BUFFER_0_1]] : memref<32xi32>) {bd_id = 1 : i32}
// CHECK:               aie.use_lock(%[[LOCK_0_1]], Release)
// CHECK:             }]
// CHECK:             %[[LOCK_0_1_1:.*]] = aie.lock(%[[TILE_0_1]]) {init = 1 : i32}
// CHECK:             %[[LOCK_0_1_2:.*]] = aie.lock(%[[TILE_0_1]]) {init = 0 : i32}
// CHECK:             %[[VAL_2:.*]] = aie.dma(S2MM, 1) {loop = false, repeat_count = 10 : i32} [{
// CHECK:               aie.use_lock(%[[LOCK_0_1_1]], AcquireGreaterEqual)
// CHECK:               aie.dma_bd(%[[BUFFER_0_1]] : memref<32xi32>) {bd_id = 24 : i32}
// CHECK:               aie.use_lock(%[[LOCK_0_1_2]], Release)
// CHECK:             }]
// CHECK:             %[[VAL_3:.*]] = aie.dma(MM2S, 1) {loop = false, repeat_count = 10 : i32} [{
// CHECK:               aie.use_lock(%[[LOCK_0_1_2]], AcquireGreaterEqual)
// CHECK:               aie.dma_bd(%[[BUFFER_0_1]] : memref<32xi32>) {bd_id = 25 : i32}
// CHECK:               aie.use_lock(%[[LOCK_0_1_1]], Release)
// CHECK:             }]
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %double_buffer = aie.buffer(%tile_0_2) {sym_name = "double_buffer"} : memref<32xi32>
    %buffer_0_1 = aie.buffer(%tile_0_1) : memref<32xi32>
    %lock_X = aie.lock(%tile_0_2) {init = 1 : i32, sym_name = "lock_X"}
    %lock_Y = aie.lock(%tile_0_2) {init = 0 : i32, sym_name = "lock_Y"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %player_a = aie.dma(S2MM, 0) {sym_name = "player_a"} [{
        aie.use_lock(%lock_Y, Acquire, 0)
        aie.dma_bd(%double_buffer : memref<32xi32>, 0)
        aie.use_lock(%lock_Y, Release, 0)
      }, {
        aie.use_lock(%lock_X, Acquire, 1)
        aie.dma_bd(%double_buffer : memref<32xi32>)
        aie.use_lock(%lock_X, Release, -1)
      }, {
        aie.use_lock(%lock_Y, Acquire) {acq_en = false}
        aie.dma_bd(%double_buffer : memref<32xi32>)
        aie.use_lock(%lock_Y, Release, 1)
      }]
      %player_b = aie.dma(S2MM, 1) {sym_name = "player_b"} [{
        aie.use_lock(%lock_Y, Acquire, 1)
        aie.dma_bd(%double_buffer : memref<32xi32>, 0)
        aie.use_lock(%lock_Y, Release, 0)
      }, {
        aie.use_lock(%lock_X, Acquire, 1)
        aie.dma_bd(%double_buffer : memref<32xi32>)
        aie.use_lock(%lock_X, Release, -1)
      }, {
        aie.use_lock(%lock_Y, Acquire) {acq_en = false}
        aie.dma_bd(%double_buffer : memref<32xi32>)
        aie.use_lock(%lock_Y, Release, -1)
      }]
      aie.end
    }
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %lock_0_1 = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock_0_1_0 = aie.lock(%tile_0_1) {init = 0 : i32}
      %0 = aie.dma(S2MM, 0) {loop = false, repeat_count = 10 : i32} [{
        aie.use_lock(%lock_0_1, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock_0_1_0, Release)
      }]
      %1 = aie.dma(MM2S, 0) {loop = false, repeat_count = 10 : i32} [{
        aie.use_lock(%lock_0_1_0, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock_0_1, Release)
      }]
      %lock_0_1_1 = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock_0_1_2 = aie.lock(%tile_0_1) {init = 0 : i32}
      %2 = aie.dma(S2MM, 1) {loop = false, repeat_count = 10 : i32} [{
        aie.use_lock(%lock_0_1_1, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock_0_1_2, Release)
      }]
      %3 = aie.dma(MM2S, 1) {loop = false, repeat_count = 10 : i32} [{
        aie.use_lock(%lock_0_1_2, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock_0_1_1, Release)
      }]
      aie.end
    }
  }
}

// -----

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[TILE_2_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[IN:.*]] = aie.buffer(%[[TILE_2_1]]) {address = 8192 : i32, sym_name = "in"} : memref<16xi32>
// CHECK:           %[[OUT:.*]] = aie.buffer(%[[TILE_2_1]]) {address = 1824 : i32, sym_name = "out"} : memref<16xi32>
// CHECK:           %[[LOCK_2_1:.*]] = aie.lock(%[[TILE_2_1]], 0) {init = 1 : i32}
// CHECK:           %[[LOCK_2_1_0:.*]] = aie.lock(%[[TILE_2_1]], 1)
// CHECK:           %[[LOCK_2_1_1:.*]] = aie.lock(%[[TILE_2_1]], 2) {init = 1 : i32}
// CHECK:           %[[LOCK_2_1_2:.*]] = aie.lock(%[[TILE_2_1]], 3)
// CHECK:           %[[MEMTILE_DMA_2_1:.*]] = aie.memtile_dma(%[[TILE_2_1]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(S2MM, 0, ^bb4, ^bb1)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(MM2S, 1, ^bb5, ^bb2)
// CHECK:           ^bb2:
// CHECK:             %[[VAL_2:.*]] = aie.dma_start(S2MM, 1, ^bb6, ^bb3)
// CHECK:           ^bb3:
// CHECK:             %[[VAL_3:.*]] = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LOCK_2_1]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[IN]] : memref<16xi32>, 0, 128, [<size = 2, stride = 1>, <size = 3, stride = 2>, <size = 2, stride = 4>, <size = 1, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 0 : i32}
// CHECK:             aie.use_lock(%[[LOCK_2_1_0]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[LOCK_2_1_0]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[IN]] : memref<16xi32>, 0, 16) {bd_id = 24 : i32, next_bd_id = 24 : i32}
// CHECK:             aie.use_lock(%[[LOCK_2_1]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[LOCK_2_1_1]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OUT]] : memref<16xi32>, 0, 16) {bd_id = 25 : i32, next_bd_id = 25 : i32}
// CHECK:             aie.use_lock(%[[LOCK_2_1_2]], Release, 1)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[LOCK_2_1_2]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OUT]] : memref<16xi32>, 0, 16) {bd_id = 1 : i32, next_bd_id = 1 : i32}
// CHECK:             aie.use_lock(%[[LOCK_2_1_1]], Release, 1)
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb8:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @aie_module  {
  aie.device(xcve2302) {
    %t01 = aie.tile(2, 1)
    %buf01_0 = aie.buffer(%t01) { address = 8192 : i32, sym_name = "in" } : memref<16xi32>
    %buf01_1 = aie.buffer(%t01) { address = 1824 : i32, sym_name = "out" } : memref<16xi32>
    %l01_0 = aie.lock(%t01, 0) { init = 1 : i32 }
    %l01_1 = aie.lock(%t01, 1)
    %l01_2 = aie.lock(%t01, 2) { init = 1 : i32 }
    %l01_3 = aie.lock(%t01, 3)
    %m01 = aie.memtile_dma(%t01) {
        %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^dma0)
      ^dma0:
        %memSrcDma = aie.dma_start(MM2S, 1, ^bd1, ^dma1)
      ^dma1:
        %memDstDma = aie.dma_start(S2MM, 1, ^bd2, ^dma2)
      ^dma2:
        %dstDma = aie.dma_start(MM2S, 0, ^bd3, ^end)
      ^bd0:
        aie.use_lock(%l01_0, "AcquireGreaterEqual", 1)
        aie.dma_bd(%buf01_0 : memref<16xi32>, 0, 128, [<size = 2, stride = 1>, <size = 3, stride = 2>, <size = 2, stride = 4>, <size = 1, stride = 1>])
        aie.use_lock(%l01_1, "Release", 1)
        aie.next_bd ^bd0
      ^bd1:
        aie.use_lock(%l01_1, "AcquireGreaterEqual", 1)
        aie.dma_bd(%buf01_0 : memref<16xi32>, 0, 16)
        aie.use_lock(%l01_0, "Release", 1)
        aie.next_bd ^bd1
      ^bd2:
        aie.use_lock(%l01_2, "AcquireGreaterEqual", 1)
        aie.dma_bd(%buf01_1 : memref<16xi32>, 0, 16)
        aie.use_lock(%l01_3, "Release", 1)
        aie.next_bd ^bd2
      ^bd3:
        aie.use_lock(%l01_3, "AcquireGreaterEqual", 1)
        aie.dma_bd(%buf01_1 : memref<16xi32>, 0, 16)
        aie.use_lock(%l01_2, "Release", 1)
        aie.next_bd ^bd3
      ^end:
        aie.end
    }
  }
}
