// RUN: iree-opt --convert-vector-to-aievec %s

// CHECK: aievec.matmul %[[VAL_19:.*]], %[[VAL_21:.*]], %[[VAL_23:.*]] : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>

aie.device(npu1_4col) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_1 = aie.tile(0, 1)
  %tile_1_1 = aie.tile(1, 1)
  %tile_2_1 = aie.tile(2, 1)
  %tile_0_2 = aie.tile(0, 2)
  %lock_1_1 = aie.lock(%tile_1_1, 1) {init = 1 : i32}
  %lock_1_1_0 = aie.lock(%tile_1_1, 0) {init = 0 : i32}
  %lock_0_1 = aie.lock(%tile_0_1, 1) {init = 1 : i32}
  %lock_0_1_1 = aie.lock(%tile_0_1, 0) {init = 0 : i32}
  %lock_2_1 = aie.lock(%tile_2_1, 1) {init = 1 : i32}
  %lock_2_1_2 = aie.lock(%tile_2_1, 0) {init = 0 : i32}
  %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
  %lock_0_2_3 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
  %lock_0_2_4 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
  %lock_0_2_5 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
  %lock_0_2_6 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
  %lock_0_2_7 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
  %buf5 = aie.buffer(%tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "buf5"} : memref<64x64xbf16> 
  %buf4 = aie.buffer(%tile_1_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "buf4"} : memref<64x64xbf16> 
  %buf3 = aie.buffer(%tile_2_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "buf3"} : memref<64x64xf32> 
  %buf2 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "buf2"} : memref<8x16x4x8xbf16> 
  %buf1 = aie.buffer(%tile_0_2) {address = 9216 : i32, mem_bank = 0 : i32, sym_name = "buf1"} : memref<16x8x8x4xbf16> 
  %buf0 = aie.buffer(%tile_0_2) {address = 17408 : i32, mem_bank = 0 : i32, sym_name = "buf0"} : memref<16x16x4x4xf32> 
  %core_0_2 = aie.core(%tile_0_2) {
    %c48 = arith.constant 48 : index
    %c49 = arith.constant 49 : index
    %c50 = arith.constant 50 : index
    %c51 = arith.constant 51 : index
    %c52 = arith.constant 52 : index
    %c53 = arith.constant 53 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_8 = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb16
    aie.use_lock(%c49, AcquireGreaterEqual, 1)
    aie.use_lock(%c50, AcquireGreaterEqual, 1)
    aie.use_lock(%c52, AcquireGreaterEqual, 1)
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
    memref.store %cst_8, %buf0[%0, %2, %4, %6] : memref<16x16x4x4xf32>
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
  ^bb10(%12: index):  // 2 preds: ^bb2, ^bb15
    %13 = arith.cmpi slt, %12, %c16 : index
    cf.cond_br %13, ^bb11(%c0 : index), ^bb16
  ^bb11(%14: index):  // 2 preds: ^bb10, ^bb14
    %15 = arith.cmpi slt, %14, %c16 : index
    cf.cond_br %15, ^bb12(%c0 : index), ^bb15
  ^bb12(%16: index):  // 2 preds: ^bb11, ^bb13
    %17 = arith.cmpi slt, %16, %c8 : index
    cf.cond_br %17, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    %18 = vector.transfer_read %buf2[%16, %12, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<8x16x4x8xbf16>, vector<1x1x4x8xbf16>
    %19 = vector.transfer_read %buf1[%14, %16, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<16x8x8x4xbf16>, vector<1x1x8x4xbf16>
    %20 = vector.transfer_read %buf0[%14, %12, %c0, %c0], %cst_8 {in_bounds = [true, true, true, true]} : memref<16x16x4x4xf32>, vector<1x1x4x4xf32>
    %21 = arith.extf %18 : vector<1x1x4x8xbf16> to vector<1x1x4x8xf32>
    %22 = arith.extf %19 : vector<1x1x8x4xbf16> to vector<1x1x8x4xf32>
    %23 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %21, %22, %20 : vector<1x1x4x8xf32>, vector<1x1x8x4xf32> into vector<1x1x4x4xf32>
    vector.transfer_write %23, %buf0[%14, %12, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x4x4xf32>, memref<16x16x4x4xf32>
    %24 = arith.addi %16, %c1 : index
    cf.br ^bb12(%24 : index)
  ^bb14:  // pred: ^bb12
    %25 = arith.addi %14, %c1 : index
    cf.br ^bb11(%25 : index)
  ^bb15:  // pred: ^bb11
    %26 = arith.addi %12, %c1 : index
    cf.br ^bb10(%26 : index)
  ^bb16:  // pred: ^bb10
    aie.use_lock(%c48, Release, 1)
    aie.use_lock(%c51, Release, 1)
    aie.use_lock(%c53, Release, 1)
    cf.br ^bb1
  } {elf_file = "matmul_64x64_64xbf16__dispatch_0_matmul_64x64x64_bf16xbf16xf32_0_core_0_2.elf"}
} {sym_name = "matmul_64x64_64xbf16__dispatch_0_matmul_64x64x64_bf16xbf16xf32_0"}