#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  aie.device(npu1_4col) {
    memref.global "public" @shim_4 : memref<32x64xi32>
    memref.global "public" @shim_3 : memref<32x64xi32>
    memref.global "public" @shim_2 : memref<1024x64xi32>
    memref.global "public" @shim_1 : memref<1024x64xi32>
    memref.global "public" @shim_0 : memref<32x1024xi32>
    %tile_0_1 = aie.tile(0, 1)
    %tile_1_1 = aie.tile(1, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_0_0 = aie.tile(0, 0)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %buffer_1_2 = aie.buffer(%tile_1_2) {sym_name = "buff_0"} : memref<1024xi32>
    %buffer_1_2_0 = aie.buffer(%tile_1_2) {sym_name = "buff_1"} : memref<1024xi32>
    %lock_1_2 = aie.lock(%tile_1_2, 4) {init = 2 : i8, sym_name = "lock_0"}
    %lock_1_2_1 = aie.lock(%tile_1_2, 5) {init = 0 : i8, sym_name = "lock_1"}
    %buffer_1_2_2 = aie.buffer(%tile_1_2) {sym_name = "buff_2"} : memref<2048xi32>
    %buffer_1_2_3 = aie.buffer(%tile_1_2) {sym_name = "buff_3"} : memref<2048xi32>
    %lock_1_2_4 = aie.lock(%tile_1_2, 2) {init = 2 : i8, sym_name = "lock_2"}
    %lock_1_2_5 = aie.lock(%tile_1_2, 3) {init = 0 : i8, sym_name = "lock_3"}
    %buffer_1_2_6 = aie.buffer(%tile_1_2) {sym_name = "buff_4"} : memref<2048xi32>
    %buffer_1_2_7 = aie.buffer(%tile_1_2) {sym_name = "buff_5"} : memref<2048xi32>
    %lock_1_2_8 = aie.lock(%tile_1_2, 0) {init = 2 : i8, sym_name = "lock_4"}
    %lock_1_2_9 = aie.lock(%tile_1_2, 1) {init = 0 : i8, sym_name = "lock_5"}
    %buffer_0_2 = aie.buffer(%tile_0_2) {sym_name = "buff_6"} : memref<1024xi32>
    %buffer_0_2_10 = aie.buffer(%tile_0_2) {sym_name = "buff_7"} : memref<1024xi32>
    %lock_0_2 = aie.lock(%tile_0_2, 4) {init = 2 : i8, sym_name = "lock_6"}
    %lock_0_2_11 = aie.lock(%tile_0_2, 5) {init = 0 : i8, sym_name = "lock_7"}
    %buffer_0_2_12 = aie.buffer(%tile_0_2) {sym_name = "buff_8"} : memref<2048xi32>
    %buffer_0_2_13 = aie.buffer(%tile_0_2) {sym_name = "buff_9"} : memref<2048xi32>
    %lock_0_2_14 = aie.lock(%tile_0_2, 2) {init = 2 : i8, sym_name = "lock_8"}
    %lock_0_2_15 = aie.lock(%tile_0_2, 3) {init = 0 : i8, sym_name = "lock_9"}
    %buffer_0_2_16 = aie.buffer(%tile_0_2) {sym_name = "buff_10"} : memref<2048xi32>
    %buffer_0_2_17 = aie.buffer(%tile_0_2) {sym_name = "buff_11"} : memref<2048xi32>
    %lock_0_2_18 = aie.lock(%tile_0_2, 0) {init = 2 : i8, sym_name = "lock_10"}
    %lock_0_2_19 = aie.lock(%tile_0_2, 1) {init = 0 : i8, sym_name = "lock_11"}
    %buffer_1_1 = aie.buffer(%tile_1_1) {sym_name = "buff_12"} : memref<1024xi32>
    %buffer_1_1_20 = aie.buffer(%tile_1_1) {sym_name = "buff_13"} : memref<1024xi32>
    %lock_1_1 = aie.lock(%tile_1_1, 2) {init = 2 : i8, sym_name = "lock_12"}
    %lock_1_1_21 = aie.lock(%tile_1_1, 3) {init = 0 : i8, sym_name = "lock_13"}
    %buffer_1_1_22 = aie.buffer(%tile_1_1) {sym_name = "buff_14"} : memref<2048xi32>
    %buffer_1_1_23 = aie.buffer(%tile_1_1) {sym_name = "buff_15"} : memref<2048xi32>
    %lock_1_1_24 = aie.lock(%tile_1_1, 0) {init = 2 : i8, sym_name = "lock_14"}
    %lock_1_1_25 = aie.lock(%tile_1_1, 1) {init = 0 : i8, sym_name = "lock_15"}
    %buffer_0_1 = aie.buffer(%tile_0_1) {sym_name = "buff_16"} : memref<1024xi32>
    %buffer_0_1_26 = aie.buffer(%tile_0_1) {sym_name = "buff_17"} : memref<1024xi32>
    %lock_0_1 = aie.lock(%tile_0_1, 4) {init = 2 : i8, sym_name = "lock_16"}
    %lock_0_1_27 = aie.lock(%tile_0_1, 5) {init = 0 : i8, sym_name = "lock_17"}
    %buffer_0_1_28 = aie.buffer(%tile_0_1) {sym_name = "buff_18"} : memref<2048xi32>
    %buffer_0_1_29 = aie.buffer(%tile_0_1) {sym_name = "buff_19"} : memref<2048xi32>
    %lock_0_1_30 = aie.lock(%tile_0_1, 2) {init = 2 : i8, sym_name = "lock_18"}
    %lock_0_1_31 = aie.lock(%tile_0_1, 3) {init = 0 : i8, sym_name = "lock_19"}
    %buffer_0_1_32 = aie.buffer(%tile_0_1) {sym_name = "buff_20"} : memref<2048xi32>
    %buffer_0_1_33 = aie.buffer(%tile_0_1) {sym_name = "buff_21"} : memref<2048xi32>
    %lock_0_1_34 = aie.lock(%tile_0_1, 0) {init = 2 : i8, sym_name = "lock_20"}
    %lock_0_1_35 = aie.lock(%tile_0_1, 1) {init = 0 : i8, sym_name = "lock_21"}
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.shim_dma_allocation @shim_0(MM2S, 0, 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_1_2, DMA : 0)
    aie.flow(%tile_0_0, DMA : 1, %tile_0_1, DMA : 1)
    aie.shim_dma_allocation @shim_1(MM2S, 1, 0)
    aie.flow(%tile_0_1, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_0, DMA : 2, %tile_1_1, DMA : 0)
    aie.shim_dma_allocation @shim_2(MM2S, 2, 0)
    aie.flow(%tile_1_1, DMA : 0, %tile_1_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 2)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_2_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_2_12 : memref<2048xi32>) {len = 2048 : i32}
      aie.use_lock(%lock_0_2_15, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_2_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_2_13 : memref<2048xi32>) {len = 2048 : i32}
      aie.use_lock(%lock_0_2_15, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_0_2_18, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_2_16 : memref<2048xi32>) {len = 2048 : i32}
      aie.use_lock(%lock_0_2_19, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_2_18, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_2_17 : memref<2048xi32>) {len = 2048 : i32}
      aie.use_lock(%lock_0_2_19, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%lock_0_2_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_2 : memref<1024xi32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 8>, <size = 4, stride = 256>, <size = 8, stride = 1>]>, len = 1024 : i32}
      aie.use_lock(%lock_0_2, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%lock_0_2_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_2_10 : memref<1024xi32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 8>, <size = 4, stride = 256>, <size = 8, stride = 1>]>, len = 1024 : i32}
      aie.use_lock(%lock_0_2, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.flow(%tile_0_1, DMA : 2, %tile_0_0, DMA : 0)
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_1_30, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_1_28 : memref<2048xi32>) {len = 2048 : i32}
      aie.use_lock(%lock_0_1_31, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_1_30, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_1_29 : memref<2048xi32>) {len = 2048 : i32}
      aie.use_lock(%lock_0_1_31, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_0_1_31, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_1_28 : memref<2048xi32>) {dimensions = #aie<bd_dim_layout_array[<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]>, len = 2048 : i32}
      aie.use_lock(%lock_0_1_30, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_1_31, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_1_29 : memref<2048xi32>) {dimensions = #aie<bd_dim_layout_array[<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]>, len = 2048 : i32}
      aie.use_lock(%lock_0_1_30, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%lock_0_1_34, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_1_32 : memref<2048xi32>) {len = 2048 : i32}
      aie.use_lock(%lock_0_1_35, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%lock_0_1_34, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_1_33 : memref<2048xi32>) {len = 2048 : i32}
      aie.use_lock(%lock_0_1_35, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%lock_0_1_35, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_1_32 : memref<2048xi32>) {dimensions = #aie<bd_dim_layout_array[<size = 4, stride = 8>, <size = 64, stride = 32>, <size = 8, stride = 1>]>, len = 2048 : i32}
      aie.use_lock(%lock_0_1_34, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%lock_0_1_35, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_1_33 : memref<2048xi32>) {dimensions = #aie<bd_dim_layout_array[<size = 4, stride = 8>, <size = 64, stride = 32>, <size = 8, stride = 1>]>, len = 2048 : i32}
      aie.use_lock(%lock_0_1_34, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_1 : memref<1024xi32>) {len = 1024 : i32}
      aie.use_lock(%lock_0_1_27, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_1_26 : memref<1024xi32>) {len = 1024 : i32}
      aie.use_lock(%lock_0_1_27, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(MM2S, 2, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%lock_0_1_27, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_1 : memref<1024xi32>) {len = 1024 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%lock_0_1_27, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_1_26 : memref<1024xi32>) {len = 1024 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      aie.end
    }
    aie.shim_dma_allocation @shim_3(S2MM, 0, 0)
    aie.flow(%tile_1_2, DMA : 0, %tile_1_1, DMA : 1)
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_1_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_1_2_2 : memref<2048xi32>) {len = 2048 : i32}
      aie.use_lock(%lock_1_2_5, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_1_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_1_2_3 : memref<2048xi32>) {len = 2048 : i32}
      aie.use_lock(%lock_1_2_5, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_1_2_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_1_2_6 : memref<2048xi32>) {len = 2048 : i32}
      aie.use_lock(%lock_1_2_9, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_1_2_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_1_2_7 : memref<2048xi32>) {len = 2048 : i32}
      aie.use_lock(%lock_1_2_9, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%lock_1_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_1_2 : memref<1024xi32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 8>, <size = 4, stride = 256>, <size = 8, stride = 1>]>, len = 1024 : i32}
      aie.use_lock(%lock_1_2, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%lock_1_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_1_2_0 : memref<1024xi32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 8>, <size = 4, stride = 256>, <size = 8, stride = 1>]>, len = 1024 : i32}
      aie.use_lock(%lock_1_2, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.flow(%tile_1_1, DMA : 1, %tile_0_0, DMA : 1)
    %memtile_dma_1_1 = aie.memtile_dma(%tile_1_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_1_1_24, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_1_1_22 : memref<2048xi32>) {len = 2048 : i32}
      aie.use_lock(%lock_1_1_25, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_1_1_24, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_1_1_23 : memref<2048xi32>) {len = 2048 : i32}
      aie.use_lock(%lock_1_1_25, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_1_1_25, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_1_1_22 : memref<2048xi32>) {dimensions = #aie<bd_dim_layout_array[<size = 4, stride = 8>, <size = 64, stride = 32>, <size = 8, stride = 1>]>, len = 2048 : i32}
      aie.use_lock(%lock_1_1_24, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_1_1_25, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_1_1_23 : memref<2048xi32>) {dimensions = #aie<bd_dim_layout_array[<size = 4, stride = 8>, <size = 64, stride = 32>, <size = 8, stride = 1>]>, len = 2048 : i32}
      aie.use_lock(%lock_1_1_24, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_1_1 : memref<1024xi32>) {len = 1024 : i32}
      aie.use_lock(%lock_1_1_21, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_1_1_20 : memref<1024xi32>) {len = 1024 : i32}
      aie.use_lock(%lock_1_1_21, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%lock_1_1_21, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_1_1 : memref<1024xi32>) {len = 1024 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%lock_1_1_21, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_1_1_20 : memref<1024xi32>) {len = 1024 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      aie.end
    }
    aie.shim_dma_allocation @shim_4(S2MM, 1, 0)
    %core_0_2 = aie.core(%tile_0_2) {
      %c14 = arith.constant 14 : index
      %c0_36 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %c2_37 = arith.constant 2 : index
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      %reinterpret_cast = memref.reinterpret_cast %buffer_0_2 to offset: [0], sizes: [4, 8, 4, 8], strides: [256, 32, 8, 1] : memref<1024xi32> to memref<4x8x4x8xi32>
      linalg.fill ins(%c0_i32 : i32) outs(%reinterpret_cast : memref<4x8x4x8xi32>)
      aie.use_lock(%lock_0_2_15, AcquireGreaterEqual, 1)
      %reinterpret_cast_38 = memref.reinterpret_cast %buffer_0_2_12 to offset: [0], sizes: [8, 8, 4, 8], strides: [256, 32, 8, 1] : memref<2048xi32> to memref<8x8x4x8xi32>
      aie.use_lock(%lock_0_2_19, AcquireGreaterEqual, 1)
      %reinterpret_cast_39 = memref.reinterpret_cast %buffer_0_2_16 to offset: [0], sizes: [4, 8, 8, 8], strides: [512, 64, 8, 1] : memref<2048xi32> to memref<4x8x8x8xi32>
      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%reinterpret_cast_38, %reinterpret_cast_39 : memref<8x8x4x8xi32>, memref<4x8x8x8xi32>) outs(%reinterpret_cast : memref<4x8x4x8xi32>) {
      ^bb0(%in: i32, %in_42: i32, %out: i32):
        %0 = arith.muli %in, %in_42 : i32
        %1 = arith.addi %out, %0 : i32
        linalg.yield %1 : i32
      }
      scf.for %arg0 = %c0_36 to %c14 step %c2_37 {
        aie.use_lock(%lock_0_2_14, Release, 1)
        aie.use_lock(%lock_0_2_15, AcquireGreaterEqual, 1)
        %reinterpret_cast_42 = memref.reinterpret_cast %buffer_0_2_13 to offset: [0], sizes: [8, 8, 4, 8], strides: [256, 32, 8, 1] : memref<2048xi32> to memref<8x8x4x8xi32>
        aie.use_lock(%lock_0_2_18, Release, 1)
        aie.use_lock(%lock_0_2_19, AcquireGreaterEqual, 1)
        %reinterpret_cast_43 = memref.reinterpret_cast %buffer_0_2_17 to offset: [0], sizes: [4, 8, 8, 8], strides: [512, 64, 8, 1] : memref<2048xi32> to memref<4x8x8x8xi32>
        linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%reinterpret_cast_42, %reinterpret_cast_43 : memref<8x8x4x8xi32>, memref<4x8x8x8xi32>) outs(%reinterpret_cast : memref<4x8x4x8xi32>) {
        ^bb0(%in: i32, %in_46: i32, %out: i32):
          %0 = arith.muli %in, %in_46 : i32
          %1 = arith.addi %out, %0 : i32
          linalg.yield %1 : i32
        }
        aie.use_lock(%lock_0_2_14, Release, 1)
        aie.use_lock(%lock_0_2_15, AcquireGreaterEqual, 1)
        %reinterpret_cast_44 = memref.reinterpret_cast %buffer_0_2_12 to offset: [0], sizes: [8, 8, 4, 8], strides: [256, 32, 8, 1] : memref<2048xi32> to memref<8x8x4x8xi32>
        aie.use_lock(%lock_0_2_18, Release, 1)
        aie.use_lock(%lock_0_2_19, AcquireGreaterEqual, 1)
        %reinterpret_cast_45 = memref.reinterpret_cast %buffer_0_2_16 to offset: [0], sizes: [4, 8, 8, 8], strides: [512, 64, 8, 1] : memref<2048xi32> to memref<4x8x8x8xi32>
        linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%reinterpret_cast_44, %reinterpret_cast_45 : memref<8x8x4x8xi32>, memref<4x8x8x8xi32>) outs(%reinterpret_cast : memref<4x8x4x8xi32>) {
        ^bb0(%in: i32, %in_46: i32, %out: i32):
          %0 = arith.muli %in, %in_46 : i32
          %1 = arith.addi %out, %0 : i32
          linalg.yield %1 : i32
        }
      }
      aie.use_lock(%lock_0_2_14, Release, 1)
      aie.use_lock(%lock_0_2_15, AcquireGreaterEqual, 1)
      %reinterpret_cast_40 = memref.reinterpret_cast %buffer_0_2_13 to offset: [0], sizes: [8, 8, 4, 8], strides: [256, 32, 8, 1] : memref<2048xi32> to memref<8x8x4x8xi32>
      aie.use_lock(%lock_0_2_18, Release, 1)
      aie.use_lock(%lock_0_2_19, AcquireGreaterEqual, 1)
      %reinterpret_cast_41 = memref.reinterpret_cast %buffer_0_2_17 to offset: [0], sizes: [4, 8, 8, 8], strides: [512, 64, 8, 1] : memref<2048xi32> to memref<4x8x8x8xi32>
      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%reinterpret_cast_40, %reinterpret_cast_41 : memref<8x8x4x8xi32>, memref<4x8x8x8xi32>) outs(%reinterpret_cast : memref<4x8x4x8xi32>) {
      ^bb0(%in: i32, %in_42: i32, %out: i32):
        %0 = arith.muli %in, %in_42 : i32
        %1 = arith.addi %out, %0 : i32
        linalg.yield %1 : i32
      }
      aie.use_lock(%lock_0_2_14, Release, 1)
      aie.use_lock(%lock_0_2_18, Release, 1)
      aie.use_lock(%lock_0_2_11, Release, 1)
      aie.end
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c14 = arith.constant 14 : index
      %c0_36 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %c2_37 = arith.constant 2 : index
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      %reinterpret_cast = memref.reinterpret_cast %buffer_1_2 to offset: [0], sizes: [4, 8, 4, 8], strides: [256, 32, 8, 1] : memref<1024xi32> to memref<4x8x4x8xi32>
      linalg.fill ins(%c0_i32 : i32) outs(%reinterpret_cast : memref<4x8x4x8xi32>)
      aie.use_lock(%lock_1_2_5, AcquireGreaterEqual, 1)
      %reinterpret_cast_38 = memref.reinterpret_cast %buffer_1_2_2 to offset: [0], sizes: [8, 8, 4, 8], strides: [256, 32, 8, 1] : memref<2048xi32> to memref<8x8x4x8xi32>
      aie.use_lock(%lock_1_2_9, AcquireGreaterEqual, 1)
      %reinterpret_cast_39 = memref.reinterpret_cast %buffer_1_2_6 to offset: [0], sizes: [4, 8, 8, 8], strides: [512, 64, 8, 1] : memref<2048xi32> to memref<4x8x8x8xi32>
      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%reinterpret_cast_38, %reinterpret_cast_39 : memref<8x8x4x8xi32>, memref<4x8x8x8xi32>) outs(%reinterpret_cast : memref<4x8x4x8xi32>) {
      ^bb0(%in: i32, %in_42: i32, %out: i32):
        %0 = arith.muli %in, %in_42 : i32
        %1 = arith.addi %out, %0 : i32
        linalg.yield %1 : i32
      }
      scf.for %arg0 = %c0_36 to %c14 step %c2_37 {
        aie.use_lock(%lock_1_2_4, Release, 1)
        aie.use_lock(%lock_1_2_5, AcquireGreaterEqual, 1)
        %reinterpret_cast_42 = memref.reinterpret_cast %buffer_1_2_3 to offset: [0], sizes: [8, 8, 4, 8], strides: [256, 32, 8, 1] : memref<2048xi32> to memref<8x8x4x8xi32>
        aie.use_lock(%lock_1_2_8, Release, 1)
        aie.use_lock(%lock_1_2_9, AcquireGreaterEqual, 1)
        %reinterpret_cast_43 = memref.reinterpret_cast %buffer_1_2_7 to offset: [0], sizes: [4, 8, 8, 8], strides: [512, 64, 8, 1] : memref<2048xi32> to memref<4x8x8x8xi32>
        linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%reinterpret_cast_42, %reinterpret_cast_43 : memref<8x8x4x8xi32>, memref<4x8x8x8xi32>) outs(%reinterpret_cast : memref<4x8x4x8xi32>) {
        ^bb0(%in: i32, %in_46: i32, %out: i32):
          %0 = arith.muli %in, %in_46 : i32
          %1 = arith.addi %out, %0 : i32
          linalg.yield %1 : i32
        }
        aie.use_lock(%lock_1_2_4, Release, 1)
        aie.use_lock(%lock_1_2_5, AcquireGreaterEqual, 1)
        %reinterpret_cast_44 = memref.reinterpret_cast %buffer_1_2_2 to offset: [0], sizes: [8, 8, 4, 8], strides: [256, 32, 8, 1] : memref<2048xi32> to memref<8x8x4x8xi32>
        aie.use_lock(%lock_1_2_8, Release, 1)
        aie.use_lock(%lock_1_2_9, AcquireGreaterEqual, 1)
        %reinterpret_cast_45 = memref.reinterpret_cast %buffer_1_2_6 to offset: [0], sizes: [4, 8, 8, 8], strides: [512, 64, 8, 1] : memref<2048xi32> to memref<4x8x8x8xi32>
        linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%reinterpret_cast_44, %reinterpret_cast_45 : memref<8x8x4x8xi32>, memref<4x8x8x8xi32>) outs(%reinterpret_cast : memref<4x8x4x8xi32>) {
        ^bb0(%in: i32, %in_46: i32, %out: i32):
          %0 = arith.muli %in, %in_46 : i32
          %1 = arith.addi %out, %0 : i32
          linalg.yield %1 : i32
        }
      }
      aie.use_lock(%lock_1_2_4, Release, 1)
      aie.use_lock(%lock_1_2_5, AcquireGreaterEqual, 1)
      %reinterpret_cast_40 = memref.reinterpret_cast %buffer_1_2_3 to offset: [0], sizes: [8, 8, 4, 8], strides: [256, 32, 8, 1] : memref<2048xi32> to memref<8x8x4x8xi32>
      aie.use_lock(%lock_1_2_8, Release, 1)
      aie.use_lock(%lock_1_2_9, AcquireGreaterEqual, 1)
      %reinterpret_cast_41 = memref.reinterpret_cast %buffer_1_2_7 to offset: [0], sizes: [4, 8, 8, 8], strides: [512, 64, 8, 1] : memref<2048xi32> to memref<4x8x8x8xi32>
      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%reinterpret_cast_40, %reinterpret_cast_41 : memref<8x8x4x8xi32>, memref<4x8x8x8xi32>) outs(%reinterpret_cast : memref<4x8x4x8xi32>) {
      ^bb0(%in: i32, %in_42: i32, %out: i32):
        %0 = arith.muli %in, %in_42 : i32
        %1 = arith.addi %out, %0 : i32
        linalg.yield %1 : i32
      }
      aie.use_lock(%lock_1_2_4, Release, 1)
      aie.use_lock(%lock_1_2_8, Release, 1)
      aie.use_lock(%lock_1_2_1, Release, 1)
      aie.end
    }
    aiex.runtime_sequence @matmul_i32(%arg0: memref<32x1024xi32>, %arg1: memref<1024x64xi32>, %arg2: memref<32x64xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 16, 32, 64][0, 64, 1024, 1]) {id = 0 : i64, issue_token = true, metadata = @shim_0} : memref<32x1024xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1024, 32][0, 0, 64, 1]) {id = 1 : i64, issue_token = true, metadata = @shim_1} : memref<1024x64xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 32][1, 1, 1024, 32][0, 0, 64, 1]) {id = 2 : i64, issue_token = true, metadata = @shim_2} : memref<1024x64xi32>
      aiex.npu.dma_wait {symbol = @shim_0}
      aiex.npu.dma_wait {symbol = @shim_1}
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 32, 32][0, 0, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @shim_3} : memref<32x64xi32>
      aiex.npu.dma_wait {symbol = @shim_3}
      aiex.npu.dma_wait {symbol = @shim_2}
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 32][1, 1, 32, 32][0, 0, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @shim_4} : memref<32x64xi32>
      aiex.npu.dma_wait {symbol = @shim_4}
    }
  }
}

