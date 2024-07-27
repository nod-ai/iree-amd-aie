// RUN: iree-opt --convert-vector-to-aievec %s

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
aie.device(npu1_4col) {
  %tile_0_1 = aie.tile(0, 1)
  %tile_1_1 = aie.tile(1, 1)
  %tile_2_1 = aie.tile(2, 1)
  %tile_3_1 = aie.tile(3, 1)
  %tile_0_2 = aie.tile(0, 2)
  %tile_0_3 = aie.tile(0, 3)
  %tile_0_4 = aie.tile(0, 4)
  %tile_0_5 = aie.tile(0, 5)
  %lock_0_4 = aie.lock(%tile_0_4, 5) {init = 2 : i32}
  %lock_0_4_0 = aie.lock(%tile_0_4, 4) {init = 0 : i32}
  %lock_0_4_1 = aie.lock(%tile_0_4, 3) {init = 2 : i32}
  %lock_0_4_2 = aie.lock(%tile_0_4, 2) {init = 0 : i32}
  %lock_0_4_3 = aie.lock(%tile_0_4, 1) {init = 1 : i32}
  %lock_0_4_4 = aie.lock(%tile_0_4, 0) {init = 0 : i32}
  %buf19 = aie.buffer(%tile_0_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "buf19"} : memref<1x1x4x8xi32>
  %buf18 = aie.buffer(%tile_0_5) {address = 1152 : i32, mem_bank = 0 : i32, sym_name = "buf18"} : memref<1x1x8x8xi8>
  %buf17 = aie.buffer(%tile_0_5) {address = 1216 : i32, mem_bank = 0 : i32, sym_name = "buf17"} : memref<1x1x7x8xi8>
  %buf16 = aie.buffer(%tile_0_5) {address = 1272 : i32, mem_bank = 0 : i32, sym_name = "buf16"} : memref<1x1x8x8xi8>
  %buf15 = aie.buffer(%tile_0_5) {address = 1336 : i32, mem_bank = 0 : i32, sym_name = "buf15"} : memref<1x1x7x8xi8>
  %buf14 = aie.buffer(%tile_0_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "buf14"} : memref<1x1x4x8xi32>
  %buf13 = aie.buffer(%tile_0_4) {address = 1152 : i32, mem_bank = 0 : i32, sym_name = "buf13"} : memref<1x1x8x8xi8>
  %buf12 = aie.buffer(%tile_0_4) {address = 1216 : i32, mem_bank = 0 : i32, sym_name = "buf12"} : memref<1x1x7x8xi8>
  %buf11 = aie.buffer(%tile_0_4) {address = 1272 : i32, mem_bank = 0 : i32, sym_name = "buf11"} : memref<1x1x8x8xi8>
  %buf10 = aie.buffer(%tile_0_4) {address = 1336 : i32, mem_bank = 0 : i32, sym_name = "buf10"} : memref<1x1x7x8xi8>
  %core_0_5 = aie.core(%tile_0_5) {
    %c48 = arith.constant 48 : index
    %c49 = arith.constant 49 : index
    %c50 = arith.constant 50 : index
    %c51 = arith.constant 51 : index
    %c52 = arith.constant 52 : index
    %c53 = arith.constant 53 : index
    %c0_i32 = arith.constant 0 : i32
    %c0_i8 = arith.constant 0 : i8
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb10
    aie.use_lock(%c49, AcquireGreaterEqual, 1)
    cf.br ^bb2(%c0 : index)
  ^bb2(%0: index):  // 2 preds: ^bb1, ^bb5
    %1 = arith.cmpi slt, %0, %c4 : index
    cf.cond_br %1, ^bb3(%c0 : index), ^bb6(%c0 : index)
  ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
    %3 = arith.cmpi slt, %2, %c8 : index
    cf.cond_br %3, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    memref.store %c0_i32, %buf19[%c0, %c0, %0, %2] : memref<1x1x4x8xi32>
    %4 = arith.addi %2, %c1 : index
    cf.br ^bb3(%4 : index)
  ^bb5:  // pred: ^bb3
    %5 = arith.addi %0, %c1 : index
    cf.br ^bb2(%5 : index)
  ^bb6(%6: index):  // 2 preds: ^bb2, ^bb9
    %7 = arith.cmpi slt, %6, %c3 : index
    cf.cond_br %7, ^bb7(%c0 : index), ^bb10
  ^bb7(%8: index):  // 2 preds: ^bb6, ^bb8
    %9 = arith.cmpi slt, %8, %c3 : index
    cf.cond_br %9, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    aie.use_lock(%c52, AcquireGreaterEqual, 1)
    aie.use_lock(%c50, AcquireGreaterEqual, 1)
    %10 = vector.transfer_read %buf15[%c0, %c0, %c0, %c0], %c0_i8 {in_bounds = [true, true, true]} : memref<1x1x7x8xi8>, vector<1x7x8xi8>
    %11 = vector.transfer_read %buf16[%c0, %c0, %c0, %c0], %c0_i8 {in_bounds = [true, true, true]} : memref<1x1x8x8xi8>, vector<1x8x8xi8>
    %12 = vector.transfer_read %buf19[%c0, %c0, %c0, %c0], %c0_i32 {in_bounds = [true, true, true]} : memref<1x1x4x8xi32>, vector<1x4x8xi32>
    %13 = vector.extract_strided_slice %10 {offsets = [0, 0, 0], sizes = [1, 1, 8], strides = [1, 1, 1]} : vector<1x7x8xi8> to vector<1x1x8xi8>
    %14 = vector.extract_strided_slice %10 {offsets = [0, 2, 0], sizes = [1, 1, 8], strides = [1, 1, 1]} : vector<1x7x8xi8> to vector<1x1x8xi8>
    %15 = vector.extract_strided_slice %10 {offsets = [0, 4, 0], sizes = [1, 1, 8], strides = [1, 1, 1]} : vector<1x7x8xi8> to vector<1x1x8xi8>
    %16 = vector.extract_strided_slice %10 {offsets = [0, 6, 0], sizes = [1, 1, 8], strides = [1, 1, 1]} : vector<1x7x8xi8> to vector<1x1x8xi8>
    %17 = vector.extract %11[0] : vector<8x8xi8> from vector<1x8x8xi8>
    %18 = vector.extract_strided_slice %12 {offsets = [0, 0, 0], sizes = [1, 1, 8], strides = [1, 1, 1]} : vector<1x4x8xi32> to vector<1x1x8xi32>
    %19 = vector.extract_strided_slice %12 {offsets = [0, 1, 0], sizes = [1, 1, 8], strides = [1, 1, 1]} : vector<1x4x8xi32> to vector<1x1x8xi32>
    %20 = vector.extract_strided_slice %12 {offsets = [0, 2, 0], sizes = [1, 1, 8], strides = [1, 1, 1]} : vector<1x4x8xi32> to vector<1x1x8xi32>
    %21 = vector.extract_strided_slice %12 {offsets = [0, 3, 0], sizes = [1, 1, 8], strides = [1, 1, 1]} : vector<1x4x8xi32> to vector<1x1x8xi32>
    %22 = arith.extsi %13 : vector<1x1x8xi8> to vector<1x1x8xi32>
    %23 = arith.extsi %17 : vector<8x8xi8> to vector<8x8xi32>
    %24 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %23, %18 : vector<1x1x8xi32>, vector<8x8xi32> into vector<1x1x8xi32>
    %25 = arith.extsi %14 : vector<1x1x8xi8> to vector<1x1x8xi32>
    %26 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %25, %23, %19 : vector<1x1x8xi32>, vector<8x8xi32> into vector<1x1x8xi32>
    %27 = arith.extsi %15 : vector<1x1x8xi8> to vector<1x1x8xi32>
    %28 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %27, %23, %20 : vector<1x1x8xi32>, vector<8x8xi32> into vector<1x1x8xi32>
    %29 = arith.extsi %16 : vector<1x1x8xi8> to vector<1x1x8xi32>
    %30 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %29, %23, %21 : vector<1x1x8xi32>, vector<8x8xi32> into vector<1x1x8xi32>
    %31 = vector.insert_strided_slice %24, %12 {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<1x1x8xi32> into vector<1x4x8xi32>
    %32 = vector.insert_strided_slice %26, %31 {offsets = [0, 1, 0], strides = [1, 1, 1]} : vector<1x1x8xi32> into vector<1x4x8xi32>
    %33 = vector.insert_strided_slice %28, %32 {offsets = [0, 2, 0], strides = [1, 1, 1]} : vector<1x1x8xi32> into vector<1x4x8xi32>
    %34 = vector.insert_strided_slice %30, %33 {offsets = [0, 3, 0], strides = [1, 1, 1]} : vector<1x1x8xi32> into vector<1x4x8xi32>
    vector.transfer_write %34, %buf19[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x4x8xi32>, memref<1x1x4x8xi32>
    aie.use_lock(%c53, Release, 1)
    aie.use_lock(%c51, Release, 1)
    aie.use_lock(%c52, AcquireGreaterEqual, 1)
    aie.use_lock(%c50, AcquireGreaterEqual, 1)
    %35 = vector.transfer_read %buf17[%c0, %c0, %c0, %c0], %c0_i8 {in_bounds = [true, true, true]} : memref<1x1x7x8xi8>, vector<1x7x8xi8>
    %36 = vector.transfer_read %buf18[%c0, %c0, %c0, %c0], %c0_i8 {in_bounds = [true, true, true]} : memref<1x1x8x8xi8>, vector<1x8x8xi8>
    %37 = vector.transfer_read %buf19[%c0, %c0, %c0, %c0], %c0_i32 {in_bounds = [true, true, true]} : memref<1x1x4x8xi32>, vector<1x4x8xi32>
    %38 = vector.extract_strided_slice %35 {offsets = [0, 0, 0], sizes = [1, 1, 8], strides = [1, 1, 1]} : vector<1x7x8xi8> to vector<1x1x8xi8>
    %39 = vector.extract_strided_slice %35 {offsets = [0, 2, 0], sizes = [1, 1, 8], strides = [1, 1, 1]} : vector<1x7x8xi8> to vector<1x1x8xi8>
    %40 = vector.extract_strided_slice %35 {offsets = [0, 4, 0], sizes = [1, 1, 8], strides = [1, 1, 1]} : vector<1x7x8xi8> to vector<1x1x8xi8>
    %41 = vector.extract_strided_slice %35 {offsets = [0, 6, 0], sizes = [1, 1, 8], strides = [1, 1, 1]} : vector<1x7x8xi8> to vector<1x1x8xi8>
    %42 = vector.extract %36[0] : vector<8x8xi8> from vector<1x8x8xi8>
    %43 = vector.extract_strided_slice %37 {offsets = [0, 0, 0], sizes = [1, 1, 8], strides = [1, 1, 1]} : vector<1x4x8xi32> to vector<1x1x8xi32>
    %44 = vector.extract_strided_slice %37 {offsets = [0, 1, 0], sizes = [1, 1, 8], strides = [1, 1, 1]} : vector<1x4x8xi32> to vector<1x1x8xi32>
    %45 = vector.extract_strided_slice %37 {offsets = [0, 2, 0], sizes = [1, 1, 8], strides = [1, 1, 1]} : vector<1x4x8xi32> to vector<1x1x8xi32>
    %46 = vector.extract_strided_slice %37 {offsets = [0, 3, 0], sizes = [1, 1, 8], strides = [1, 1, 1]} : vector<1x4x8xi32> to vector<1x1x8xi32>
    %47 = arith.extsi %38 : vector<1x1x8xi8> to vector<1x1x8xi32>
    %48 = arith.extsi %42 : vector<8x8xi8> to vector<8x8xi32>
    %49 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %47, %48, %43 : vector<1x1x8xi32>, vector<8x8xi32> into vector<1x1x8xi32>
    %50 = arith.extsi %39 : vector<1x1x8xi8> to vector<1x1x8xi32>
    %51 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %50, %48, %44 : vector<1x1x8xi32>, vector<8x8xi32> into vector<1x1x8xi32>
    %52 = arith.extsi %40 : vector<1x1x8xi8> to vector<1x1x8xi32>
    %53 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %52, %48, %45 : vector<1x1x8xi32>, vector<8x8xi32> into vector<1x1x8xi32>
    %54 = arith.extsi %41 : vector<1x1x8xi8> to vector<1x1x8xi32>
    %55 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %54, %48, %46 : vector<1x1x8xi32>, vector<8x8xi32> into vector<1x1x8xi32>
    %56 = vector.insert_strided_slice %49, %37 {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<1x1x8xi32> into vector<1x4x8xi32>
    %57 = vector.insert_strided_slice %51, %56 {offsets = [0, 1, 0], strides = [1, 1, 1]} : vector<1x1x8xi32> into vector<1x4x8xi32>
    %58 = vector.insert_strided_slice %53, %57 {offsets = [0, 2, 0], strides = [1, 1, 1]} : vector<1x1x8xi32> into vector<1x4x8xi32>
    %59 = vector.insert_strided_slice %55, %58 {offsets = [0, 3, 0], strides = [1, 1, 1]} : vector<1x1x8xi32> into vector<1x4x8xi32>
    vector.transfer_write %59, %buf19[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x4x8xi32>, memref<1x1x4x8xi32>
    aie.use_lock(%c53, Release, 1)
    aie.use_lock(%c51, Release, 1)
    %60 = arith.addi %8, %c1 : index
    cf.br ^bb7(%60 : index)
  ^bb9:  // pred: ^bb7
    %61 = arith.addi %6, %c1 : index
    cf.br ^bb6(%61 : index)
  ^bb10:  // pred: ^bb6
    aie.use_lock(%c48, Release, 1)
    cf.br ^bb1
  } {elf_file = "conv_2d_nhwc_hwcf_dispatch_0_conv_2d_nhwc_hwcf_1x64x64x32x3x3x16_i8xi8xi32_0_core_0_5.elf"}
  %mem_0_4 = aie.mem(%tile_0_4) {
    %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
  ^bb1:  // 2 preds: ^bb0, ^bb2
    aie.use_lock(%lock_0_4_1, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf11 : memref<1x1x8x8xi8>, 0, 64) {bd_id = 0 : i32, next_bd_id = 1 : i32}
    aie.use_lock(%lock_0_4_2, Release, 1)
    aie.next_bd ^bb2
  ^bb2:  // pred: ^bb1
    aie.use_lock(%lock_0_4_1, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf13 : memref<1x1x8x8xi8>, 0, 64) {bd_id = 1 : i32, next_bd_id = 0 : i32}
    aie.use_lock(%lock_0_4_2, Release, 1)
    aie.next_bd ^bb1
  ^bb3:  // pred: ^bb4
    aie.end
  ^bb4:  // pred: ^bb7
    %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb3, repeat_count = 1)
  ^bb5:  // 2 preds: ^bb4, ^bb6
    aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf10 : memref<1x1x7x8xi8>, 0, 56) {bd_id = 2 : i32, next_bd_id = 3 : i32}
    aie.use_lock(%lock_0_4_0, Release, 1)
    aie.next_bd ^bb6
  ^bb6:  // pred: ^bb5
    aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf12 : memref<1x1x7x8xi8>, 0, 56) {bd_id = 3 : i32, next_bd_id = 2 : i32}
    aie.use_lock(%lock_0_4_0, Release, 1)
    aie.next_bd ^bb5
  ^bb7:  // pred: ^bb0
    %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb4, repeat_count = 1)
  ^bb8:  // 2 preds: ^bb7, ^bb8
    aie.use_lock(%lock_0_4_4, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf14 : memref<1x1x4x8xi32>, 0, 32) {bd_id = 4 : i32, next_bd_id = 4 : i32}
    aie.use_lock(%lock_0_4_3, Release, 1)
    aie.next_bd ^bb8
  }
} {sym_name = "conv_2d_nhwc_hwcf_dispatch_0_conv_2d_nhwc_hwcf_1x64x64x32x3x3x16_i8xi8xi32_0"}