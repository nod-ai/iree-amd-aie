// RUN: iree-opt --convert-vector-to-aievec %s

// CHECK: aievec.matmul %[[VAL_21:.*]], %[[VAL_26:.*]], %[[VAL_24:.*]] : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
// CHECK: aievec.matmul %[[VAL_39:.*]], %[[VAL_44:.*]], %[[VAL_42:.*]] : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d4, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
aie.device(npu1_4col) {
  %tile_3_5 = aie.tile(3, 5)
  %buf79 = aie.buffer(%tile_3_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "buf79"} : memref<16x16x4x4xf32>
  %buf78 = aie.buffer(%tile_3_5) {address = 17408 : i32, mem_bank = 0 : i32, sym_name = "buf78"} : memref<8x16x4x8xbf16>
  %buf77 = aie.buffer(%tile_3_5) {address = 25600 : i32, mem_bank = 0 : i32, sym_name = "buf77"} : memref<8x16x4x8xbf16>
  %buf76 = aie.buffer(%tile_3_5) {address = 33792 : i32, mem_bank = 0 : i32, sym_name = "buf76"} : memref<8x16x4x8xbf16>
  %buf75 = aie.buffer(%tile_3_5) {address = 41984 : i32, mem_bank = 0 : i32, sym_name = "buf75"} : memref<8x16x4x8xbf16>
  %core_3_5 = aie.core(%tile_3_5) {
    %c48 = arith.constant 48 : index
    %c49 = arith.constant 49 : index
    %c50 = arith.constant 50 : index
    %c51 = arith.constant 51 : index
    %c52 = arith.constant 52 : index
    %c53 = arith.constant 53 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c4 = arith.constant 4 : index
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb26
    aie.use_lock(%c49, AcquireGreaterEqual, 1)
    cf.br ^bb2(%c0 : index)
  ^bb2(%0: index):  // 2 preds: ^bb1, ^bb9
    %1 = arith.cmpi slt, %0, %c16 : index
    cf.cond_br %1, ^bb3(%c0 : index), ^bb10(%c0 : index)
  ^bb3(%2: index):  // 2 preds: ^bb2, ^bb8
    %3 = arith.cmpi slt, %2, %c16 : index
    cf.cond_br %3, ^bb4(%c0 : index), ^bb9
  ^bb4(%4: index):  // 2 preds: ^bb3, ^bb7
    %5 = arith.cmpi slt, %4, %c4 : index
    cf.cond_br %5, ^bb5(%c0 : index), ^bb8
  ^bb5(%6: index):  // 2 preds: ^bb4, ^bb6
    %7 = arith.cmpi slt, %6, %c4 : index
    cf.cond_br %7, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    memref.store %cst_0, %buf79[%0, %2, %4, %6] : memref<16x16x4x4xf32>
    %8 = arith.addi %6, %c1 : index
    cf.br ^bb5(%8 : index)
  ^bb7:  // pred: ^bb5
    %9 = arith.addi %4, %c1 : index
    cf.br ^bb4(%9 : index)
  ^bb8:  // pred: ^bb4
    %10 = arith.addi %2, %c1 : index
    cf.br ^bb3(%10 : index)
  ^bb9:  // pred: ^bb3
    %11 = arith.addi %0, %c1 : index
    cf.br ^bb2(%11 : index)
  ^bb10(%12: index):  // 2 preds: ^bb2, ^bb25
    %13 = arith.cmpi slt, %12, %c32 : index
    cf.cond_br %13, ^bb11, ^bb26
  ^bb11:  // pred: ^bb10
    aie.use_lock(%c50, AcquireGreaterEqual, 1)
    aie.use_lock(%c52, AcquireGreaterEqual, 1)
    cf.br ^bb12(%c0 : index)
  ^bb12(%14: index):  // 2 preds: ^bb11, ^bb17
    %15 = arith.cmpi slt, %14, %c16 : index
    cf.cond_br %15, ^bb13(%c0 : index), ^bb18
  ^bb13(%16: index):  // 2 preds: ^bb12, ^bb16
    %17 = arith.cmpi slt, %16, %c16 : index
    cf.cond_br %17, ^bb14(%c0 : index), ^bb17
  ^bb14(%18: index):  // 2 preds: ^bb13, ^bb15
    %19 = arith.cmpi slt, %18, %c8 : index
    cf.cond_br %19, ^bb15, ^bb16
  ^bb15:  // pred: ^bb14
    %20 = vector.transfer_read %buf76[%18, %14, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<8x16x4x8xbf16>, vector<1x1x4x8xbf16>
    %21 = vector.transfer_read %buf75[%18, %16, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<8x16x4x8xbf16>, vector<1x1x4x8xbf16>
    %22 = vector.transfer_read %buf79[%16, %14, %c0, %c0], %cst_0 {in_bounds = [true, true, true, true]} : memref<16x16x4x4xf32>, vector<1x1x4x4xf32>
    %23 = arith.extf %20 : vector<1x1x4x8xbf16> to vector<1x1x4x8xf32>
    %24 = arith.extf %21 : vector<1x1x4x8xbf16> to vector<1x1x4x8xf32>
    %25 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %23, %24, %22 : vector<1x1x4x8xf32>, vector<1x1x4x8xf32> into vector<1x1x4x4xf32>
    vector.transfer_write %25, %buf79[%16, %14, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x4x4xf32>, memref<16x16x4x4xf32>
    %26 = arith.addi %18, %c1 : index
    cf.br ^bb14(%26 : index)
  ^bb16:  // pred: ^bb14
    %27 = arith.addi %16, %c1 : index
    cf.br ^bb13(%27 : index)
  ^bb17:  // pred: ^bb13
    %28 = arith.addi %14, %c1 : index
    cf.br ^bb12(%28 : index)
  ^bb18:  // pred: ^bb12
    aie.use_lock(%c51, Release, 1)
    aie.use_lock(%c53, Release, 1)
    aie.use_lock(%c50, AcquireGreaterEqual, 1)
    aie.use_lock(%c52, AcquireGreaterEqual, 1)
    cf.br ^bb19(%c0 : index)
  ^bb19(%29: index):  // 2 preds: ^bb18, ^bb24
    %30 = arith.cmpi slt, %29, %c16 : index
    cf.cond_br %30, ^bb20(%c0 : index), ^bb25
  ^bb20(%31: index):  // 2 preds: ^bb19, ^bb23
    %32 = arith.cmpi slt, %31, %c16 : index
    cf.cond_br %32, ^bb21(%c0 : index), ^bb24
  ^bb21(%33: index):  // 2 preds: ^bb20, ^bb22
    %34 = arith.cmpi slt, %33, %c8 : index
    cf.cond_br %34, ^bb22, ^bb23
  ^bb22:  // pred: ^bb21
    %35 = vector.transfer_read %buf77[%33, %29, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<8x16x4x8xbf16>, vector<1x1x4x8xbf16>
    %36 = vector.transfer_read %buf78[%33, %31, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<8x16x4x8xbf16>, vector<1x1x4x8xbf16>
    %37 = vector.transfer_read %buf79[%31, %29, %c0, %c0], %cst_0 {in_bounds = [true, true, true, true]} : memref<16x16x4x4xf32>, vector<1x1x4x4xf32>
    %38 = arith.extf %35 : vector<1x1x4x8xbf16> to vector<1x1x4x8xf32>
    %39 = arith.extf %36 : vector<1x1x4x8xbf16> to vector<1x1x4x8xf32>
    %40 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %38, %39, %37 : vector<1x1x4x8xf32>, vector<1x1x4x8xf32> into vector<1x1x4x4xf32>
    vector.transfer_write %40, %buf79[%31, %29, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x4x4xf32>, memref<16x16x4x4xf32>
    %41 = arith.addi %33, %c1 : index
    cf.br ^bb21(%41 : index)
  ^bb23:  // pred: ^bb21
    %42 = arith.addi %31, %c1 : index
    cf.br ^bb20(%42 : index)
  ^bb24:  // pred: ^bb20
    %43 = arith.addi %29, %c1 : index
    cf.br ^bb19(%43 : index)
  ^bb25:  // pred: ^bb19
    aie.use_lock(%c51, Release, 1)
    aie.use_lock(%c53, Release, 1)
    %44 = arith.addi %12, %c16 : index
    cf.br ^bb10(%44 : index)
  ^bb26:  // pred: ^bb10
    aie.use_lock(%c48, Release, 1)
    cf.br ^bb1
  } {elf_file = "matmul_256x256_256xbf16__dispatch_0_matmul_transpose_b_256x256x256_bf16xbf16xf32_0_core_3_5.elf"}
} {sym_name = "matmul_256x256_256xbf16__dispatch_0_matmul_transpose_b_256x256x256_bf16xbf16xf32_0"}
