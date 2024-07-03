// RUN: (aie_cdo_gen_test %s %S) 2>&1 | FileCheck %s

module {
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
  %mem_0_2 = aie.mem(%tile_0_2) {
    %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5, repeat_count = 1)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf2 : memref<8x16x4x8xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 0 : i32}
    aie.use_lock(%lock_0_2_5, Release, 1)
    aie.next_bd ^bb1
  ^bb2:  // pred: ^bb3
    aie.end
  ^bb3:  // pred: ^bb5
    %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2, repeat_count = 1)
  ^bb4:  // 2 preds: ^bb3, ^bb4
    aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf1 : memref<16x8x8x4xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 1 : i32}
    aie.use_lock(%lock_0_2_3, Release, 1)
    aie.next_bd ^bb4
  ^bb5:  // pred: ^bb0
    %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb3, repeat_count = 1)
  ^bb6:  // 2 preds: ^bb5, ^bb6
    aie.use_lock(%lock_0_2_7, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf0 : memref<16x16x4x4xf32>, 0, 4096, [<size = 64, stride = 4>, <size = 16, stride = 256>, <size = 4, stride = 1>]) {bd_id = 2 : i32, next_bd_id = 2 : i32}
    aie.use_lock(%lock_0_2_6, Release, 1)
    aie.next_bd ^bb6
  }
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
  %switchbox_0_0 = aie.switchbox(%tile_0_0) {
    aie.connect<South : 3, North : 0>
    aie.connect<South : 7, East : 0>
    aie.connect<East : 0, South : 2>
  }
  %shim_mux_0_0 = aie.shim_mux(%tile_0_0) {
    aie.connect<DMA : 0, North : 3>
    aie.connect<DMA : 1, North : 7>
    aie.connect<North : 2, DMA : 0>
  }
  %switchbox_0_1 = aie.switchbox(%tile_0_1) {
    aie.connect<South : 0, DMA : 0>
    aie.connect<DMA : 0, North : 0>
  }
  %tile_1_0 = aie.tile(1, 0)
  %switchbox_1_0 = aie.switchbox(%tile_1_0) {
    aie.connect<West : 0, North : 0>
    aie.connect<East : 0, West : 0>
  }
  %switchbox_1_1 = aie.switchbox(%tile_1_1) {
    aie.connect<South : 0, DMA : 0>
    aie.connect<DMA : 0, North : 0>
  }
  %tile_2_0 = aie.tile(2, 0)
  %switchbox_2_0 = aie.switchbox(%tile_2_0) {
    aie.connect<North : 0, West : 0>
  }
  %switchbox_2_1 = aie.switchbox(%tile_2_1) {
    aie.connect<DMA : 0, South : 0>
    aie.connect<North : 0, DMA : 0>
  }
  %switchbox_0_2 = aie.switchbox(%tile_0_2) {
    aie.connect<South : 0, DMA : 0>
    aie.connect<East : 0, DMA : 1>
    aie.connect<DMA : 0, East : 0>
  }
  %tile_1_2 = aie.tile(1, 2)
  %switchbox_1_2 = aie.switchbox(%tile_1_2) {
    aie.connect<South : 0, West : 0>
    aie.connect<West : 0, East : 0>
  }
  %tile_2_2 = aie.tile(2, 2)
  %switchbox_2_2 = aie.switchbox(%tile_2_2) {
    aie.connect<West : 0, South : 0>
  }
  %memtile_dma_2_1 = aie.memtile_dma(%tile_2_1) {
    %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf3 : memref<64x64xf32>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 0 : i32}
    aie.use_lock(%lock_2_1_2, Release, 1)
    aie.next_bd ^bb1
  ^bb2:  // pred: ^bb3
    aie.end
  ^bb3:  // pred: ^bb0
    %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb2, repeat_count = 1)
  ^bb4:  // 2 preds: ^bb3, ^bb4
    aie.use_lock(%lock_2_1_2, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf3 : memref<64x64xf32>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 1 : i32}
    aie.use_lock(%lock_2_1, Release, 1)
    aie.next_bd ^bb4
  }
  %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
    %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf5 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 0 : i32}
    aie.use_lock(%lock_0_1_1, Release, 1)
    aie.next_bd ^bb1
  ^bb2:  // pred: ^bb3
    aie.end
  ^bb3:  // pred: ^bb0
    %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb2, repeat_count = 1)
  ^bb4:  // 2 preds: ^bb3, ^bb4
    aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf5 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 1 : i32}
    aie.use_lock(%lock_0_1, Release, 1)
    aie.next_bd ^bb4
  }
  %memtile_dma_1_1 = aie.memtile_dma(%tile_1_1) {
    %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf4 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 0 : i32}
    aie.use_lock(%lock_1_1_0, Release, 1)
    aie.next_bd ^bb1
  ^bb2:  // pred: ^bb3
    aie.end
  ^bb3:  // pred: ^bb0
    %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb2, repeat_count = 1)
  ^bb4:  // 2 preds: ^bb3, ^bb4
    aie.use_lock(%lock_1_1_0, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf4 : memref<64x64xbf16>, 0, 4096, [<size = 16, stride = 4>, <size = 64, stride = 64>, <size = 4, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 1 : i32}
    aie.use_lock(%lock_1_1, Release, 1)
    aie.next_bd ^bb4
  }
  aie.shim_dma_allocation @airMemcpyId12(S2MM, 0, 0)
  memref.global "public" @airMemcpyId12 : memref<64x64xf32>
  aie.shim_dma_allocation @airMemcpyId4(MM2S, 0, 0)
  memref.global "public" @airMemcpyId4 : memref<64x64xbf16>
  aie.shim_dma_allocation @airMemcpyId5(MM2S, 1, 0)
  memref.global "public" @airMemcpyId5 : memref<64x64xbf16>
  func.func @matmul_64x64_64xbf16__dispatch_0_matmul_64x64x64_bf16xbf16xf32(%arg0: memref<2048xi32>, %arg1: memref<2048xi32>, %arg2: memref<64x64xf32>) {
    memref.assume_alignment %arg0, 64 : memref<2048xi32>
    memref.assume_alignment %arg1, 64 : memref<2048xi32>
    memref.assume_alignment %arg2, 64 : memref<64x64xf32>
    aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 64, 32][0, 0, 32]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<2048xi32>
    aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 64, 32][0, 0, 32]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<2048xi32>
    aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 64, 64][0, 0, 64]) {id = 2 : i64, metadata = @airMemcpyId12} : memref<64x64xf32>
    aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    return
  }
  aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
  aie.wire(%tile_0_0 : DMA, %shim_mux_0_0 : DMA)
  aie.wire(%tile_0_1 : Core, %switchbox_0_1 : Core)
  aie.wire(%tile_0_1 : DMA, %switchbox_0_1 : DMA)
  aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
  aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
  aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
  aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
  aie.wire(%switchbox_0_0 : East, %switchbox_1_0 : West)
  aie.wire(%switchbox_0_1 : East, %switchbox_1_1 : West)
  aie.wire(%tile_1_1 : Core, %switchbox_1_1 : Core)
  aie.wire(%tile_1_1 : DMA, %switchbox_1_1 : DMA)
  aie.wire(%switchbox_1_0 : North, %switchbox_1_1 : South)
  aie.wire(%switchbox_0_2 : East, %switchbox_1_2 : West)
  aie.wire(%tile_1_2 : Core, %switchbox_1_2 : Core)
  aie.wire(%tile_1_2 : DMA, %switchbox_1_2 : DMA)
  aie.wire(%switchbox_1_1 : North, %switchbox_1_2 : South)
  aie.wire(%switchbox_1_0 : East, %switchbox_2_0 : West)
  aie.wire(%switchbox_1_1 : East, %switchbox_2_1 : West)
  aie.wire(%tile_2_1 : Core, %switchbox_2_1 : Core)
  aie.wire(%tile_2_1 : DMA, %switchbox_2_1 : DMA)
  aie.wire(%switchbox_2_0 : North, %switchbox_2_1 : South)
  aie.wire(%switchbox_1_2 : East, %switchbox_2_2 : West)
  aie.wire(%tile_2_2 : Core, %switchbox_2_2 : Core)
  aie.wire(%tile_2_2 : DMA, %switchbox_2_2 : DMA)
  aie.wire(%switchbox_2_1 : North, %switchbox_2_2 : South)
} {sym_name = "matmul_64x64_64xbf16__dispatch_0_matmul_64x64x64_bf16xbf16xf32_0"}
}

// CHECK: trying XAIE API: XAie_SetupPartitionConfig with args: &devInst={{.+}}, 0x0=0, partitionStartCol=1, partitionNumCols=4
// CHECK: trying XAIE API: XAie_CfgInitialize with args: &devInst={{.+}}, &configPtr=
// CHECK: trying XAIE API: XAie_SetIOBackend with args: &devInst={{.+}}, XAIE_IO_BACKEND_CDO=2
// CHECK: trying XAIE API: XAie_UpdateNpiAddr with args: &devInst={{.+}}, 0x0=0
// CHECK: trying XAIE API: XAie_LoadElf with args: &devInst={{.+}}, XAie_TileLoc(col=XAie_LocType(col: 0, row: 2), row)={{.+}}/matmul_64x64_64xbf16__dispatch_0_matmul_64x64x64_bf16xbf16xf32_0_core_0_2.elf, elfPath.str().c_str(), aieSim=0
// CHECK: trying XAIE API: XAie_CoreReset with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2)
// CHECK: trying XAIE API: XAie_CoreUnreset with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), locInit=XAie_Lock(id: 0, val: 0)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), locInit=XAie_Lock(id: 1, val: 0)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), locInit=XAie_Lock(id: 2, val: 0)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), locInit=XAie_Lock(id: 3, val: 0)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), locInit=XAie_Lock(id: 4, val: 0)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), locInit=XAie_Lock(id: 5, val: 0)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), locInit=XAie_Lock(id: 6, val: 0)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), locInit=XAie_Lock(id: 7, val: 0)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), locInit=XAie_Lock(id: 8, val: 0)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), locInit=XAie_Lock(id: 9, val: 0)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), locInit=XAie_Lock(id: 10, val: 0)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), locInit=XAie_Lock(id: 11, val: 0)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), locInit=XAie_Lock(id: 12, val: 0)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), locInit=XAie_Lock(id: 13, val: 0)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), locInit=XAie_Lock(id: 14, val: 0)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), locInit=XAie_Lock(id: 15, val: 0)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 1, row: 1), locInit=XAie_Lock(id: 1, val: 1)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 1, row: 1), locInit=XAie_Lock(id: 0, val: 0)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 1), locInit=XAie_Lock(id: 1, val: 1)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 1), locInit=XAie_Lock(id: 0, val: 0)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 2, row: 1), locInit=XAie_Lock(id: 1, val: 1)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 2, row: 1), locInit=XAie_Lock(id: 0, val: 0)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), locInit=XAie_Lock(id: 5, val: 1)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), locInit=XAie_Lock(id: 4, val: 0)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), locInit=XAie_Lock(id: 3, val: 1)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), locInit=XAie_Lock(id: 2, val: 0)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), locInit=XAie_Lock(id: 1, val: 1)
// CHECK: trying XAIE API: XAie_LockSetValue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), locInit=XAie_Lock(id: 0, val: 0)
// CHECK: trying XAIE API: XAie_DmaDescInit with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2)

// CHECK: start configuring bds
// CHECK: trying XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd={{.+}}, acqLock=XAie_Lock(id: 3, val: -1), relLock=XAie_Lock(id: 2, val: 1), acqEn=1, relEn=0
// CHECK: trying XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd={{.+}}, basePlusOffsetInBytes=1024, lenInBytes=8192
// CHECK: trying XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd={{.+}}, nextBdId.value()=0, enableNextBd=1
// CHECK: trying XAIE API: XAie_DmaEnableBd with args: &dmaTileBd={{.+}}
// CHECK: trying XAIE API: XAie_DmaWriteBd with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), bdId=0

// CHECK: end configuring bds
// CHECK: trying XAIE API: XAie_DmaDescInit with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2)

// CHECK: start configuring bds
// CHECK: trying XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd={{.+}}, acqLock=XAie_Lock(id: 5, val: -1), relLock=XAie_Lock(id: 4, val: 1), acqEn=1, relEn=0
// CHECK: trying XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd={{.+}}, basePlusOffsetInBytes=9216, lenInBytes=8192
// CHECK: trying XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd={{.+}}, nextBdId.value()=1, enableNextBd=1
// CHECK: trying XAIE API: XAie_DmaEnableBd with args: &dmaTileBd={{.+}}
// CHECK: trying XAIE API: XAie_DmaWriteBd with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), bdId=1

// CHECK: end configuring bds
// CHECK: trying XAIE API: XAie_DmaDescInit with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2)

// CHECK: start configuring bds
// CHECK: trying XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd={{.+}}, acqLock=XAie_Lock(id: 0, val: -1), relLock=XAie_Lock(id: 1, val: 1), acqEn=1, relEn=0
// CHECK: trying XAIE API: XAie_DmaSetMultiDimAddr with args: &dmaTileBd={{.+}}, &dmaTileBdTensor={{.+}}, basePlusOffsetInBytes=17408, lenInBytes=16384
// CHECK: trying XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd={{.+}}, nextBdId.value()=2, enableNextBd=1
// CHECK: trying XAIE API: XAie_DmaEnableBd with args: &dmaTileBd={{.+}}
// CHECK: trying XAIE API: XAie_DmaWriteBd with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), bdId=2

// CHECK: end configuring bds
// CHECK: trying XAIE API: XAie_DmaChannelSetStartQueue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), chNum=0, direction=0, bdId=0, repeatCount=2, enTokenIssue=0
// CHECK: trying XAIE API: XAie_DmaChannelEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), chNum=0, direction=0
// CHECK: trying XAIE API: XAie_DmaChannelSetStartQueue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), chNum=1, direction=0, bdId=1, repeatCount=2, enTokenIssue=0
// CHECK: trying XAIE API: XAie_DmaChannelEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), chNum=1, direction=0
// CHECK: trying XAIE API: XAie_DmaChannelSetStartQueue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), chNum=0, direction=1, bdId=2, repeatCount=2, enTokenIssue=0
// CHECK: trying XAIE API: XAie_DmaChannelEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), chNum=0, direction=1
// CHECK: trying XAIE API: XAie_DmaDescInit with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 2, row: 1)

// CHECK: start configuring bds
// CHECK: trying XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd={{.+}}, acqLock=XAie_Lock(id: 65, val: -1), relLock=XAie_Lock(id: 64, val: 1), acqEn=1, relEn=0
// CHECK: trying XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd={{.+}}, basePlusOffsetInBytes=524288, lenInBytes=16384
// CHECK: trying XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd={{.+}}, nextBdId.value()=0, enableNextBd=1
// CHECK: trying XAIE API: XAie_DmaEnableBd with args: &dmaTileBd={{.+}}
// CHECK: trying XAIE API: XAie_DmaWriteBd with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 2, row: 1), bdId=0

// CHECK: end configuring bds
// CHECK: trying XAIE API: XAie_DmaDescInit with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 2, row: 1)

// CHECK: start configuring bds
// CHECK: trying XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd={{.+}}, acqLock=XAie_Lock(id: 64, val: -1), relLock=XAie_Lock(id: 65, val: 1), acqEn=1, relEn=0
// CHECK: trying XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd={{.+}}, basePlusOffsetInBytes=524288, lenInBytes=16384
// CHECK: trying XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd={{.+}}, nextBdId.value()=1, enableNextBd=1
// CHECK: trying XAIE API: XAie_DmaEnableBd with args: &dmaTileBd={{.+}}
// CHECK: trying XAIE API: XAie_DmaWriteBd with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 2, row: 1), bdId=1

// CHECK: end configuring bds
// CHECK: trying XAIE API: XAie_DmaChannelSetStartQueue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 2, row: 1), chNum=0, direction=0, bdId=0, repeatCount=2, enTokenIssue=0
// CHECK: trying XAIE API: XAie_DmaChannelEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 2, row: 1), chNum=0, direction=0
// CHECK: trying XAIE API: XAie_DmaChannelSetStartQueue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 2, row: 1), chNum=0, direction=1, bdId=1, repeatCount=2, enTokenIssue=0
// CHECK: trying XAIE API: XAie_DmaChannelEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 2, row: 1), chNum=0, direction=1
// CHECK: trying XAIE API: XAie_DmaDescInit with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 0, row: 1)

// CHECK: start configuring bds
// CHECK: trying XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd={{.+}}, acqLock=XAie_Lock(id: 65, val: -1), relLock=XAie_Lock(id: 64, val: 1), acqEn=1, relEn=0
// CHECK: trying XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd={{.+}}, basePlusOffsetInBytes=524288, lenInBytes=8192
// CHECK: trying XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd={{.+}}, nextBdId.value()=0, enableNextBd=1
// CHECK: trying XAIE API: XAie_DmaEnableBd with args: &dmaTileBd={{.+}}
// CHECK: trying XAIE API: XAie_DmaWriteBd with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 0, row: 1), bdId=0

// CHECK: end configuring bds
// CHECK: trying XAIE API: XAie_DmaDescInit with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 0, row: 1)

// CHECK: start configuring bds
// CHECK: trying XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd={{.+}}, acqLock=XAie_Lock(id: 64, val: -1), relLock=XAie_Lock(id: 65, val: 1), acqEn=1, relEn=0
// CHECK: trying XAIE API: XAie_DmaSetMultiDimAddr with args: &dmaTileBd={{.+}}, &dmaTileBdTensor={{.+}}, basePlusOffsetInBytes=524288, lenInBytes=8192
// CHECK: trying XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd={{.+}}, nextBdId.value()=1, enableNextBd=1
// CHECK: trying XAIE API: XAie_DmaEnableBd with args: &dmaTileBd={{.+}}
// CHECK: trying XAIE API: XAie_DmaWriteBd with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 0, row: 1), bdId=1

// CHECK: end configuring bds
// CHECK: trying XAIE API: XAie_DmaChannelSetStartQueue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 1), chNum=0, direction=0, bdId=0, repeatCount=2, enTokenIssue=0
// CHECK: trying XAIE API: XAie_DmaChannelEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 1), chNum=0, direction=0
// CHECK: trying XAIE API: XAie_DmaChannelSetStartQueue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 1), chNum=0, direction=1, bdId=1, repeatCount=2, enTokenIssue=0
// CHECK: trying XAIE API: XAie_DmaChannelEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 1), chNum=0, direction=1
// CHECK: trying XAIE API: XAie_DmaDescInit with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 1, row: 1)

// CHECK: start configuring bds
// CHECK: trying XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd={{.+}}, acqLock=XAie_Lock(id: 65, val: -1), relLock=XAie_Lock(id: 64, val: 1), acqEn=1, relEn=0
// CHECK: trying XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd={{.+}}, basePlusOffsetInBytes=524288, lenInBytes=8192
// CHECK: trying XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd={{.+}}, nextBdId.value()=0, enableNextBd=1
// CHECK: trying XAIE API: XAie_DmaEnableBd with args: &dmaTileBd={{.+}}
// CHECK: trying XAIE API: XAie_DmaWriteBd with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 1, row: 1), bdId=0

// CHECK: end configuring bds
// CHECK: trying XAIE API: XAie_DmaDescInit with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 1, row: 1)

// CHECK: start configuring bds
// CHECK: trying XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd={{.+}}, acqLock=XAie_Lock(id: 64, val: -1), relLock=XAie_Lock(id: 65, val: 1), acqEn=1, relEn=0
// CHECK: trying XAIE API: XAie_DmaSetMultiDimAddr with args: &dmaTileBd={{.+}}, &dmaTileBdTensor={{.+}}, basePlusOffsetInBytes=524288, lenInBytes=8192
// CHECK: trying XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd={{.+}}, nextBdId.value()=1, enableNextBd=1
// CHECK: trying XAIE API: XAie_DmaEnableBd with args: &dmaTileBd={{.+}}
// CHECK: trying XAIE API: XAie_DmaWriteBd with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 1, row: 1), bdId=1

// CHECK: end configuring bds
// CHECK: trying XAIE API: XAie_DmaChannelSetStartQueue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 1, row: 1), chNum=0, direction=0, bdId=0, repeatCount=2, enTokenIssue=0
// CHECK: trying XAIE API: XAie_DmaChannelEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 1, row: 1), chNum=0, direction=0
// CHECK: trying XAIE API: XAie_DmaChannelSetStartQueue with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 1, row: 1), chNum=0, direction=1, bdId=1, repeatCount=2, enTokenIssue=0
// CHECK: trying XAIE API: XAie_DmaChannelEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 1, row: 1), chNum=0, direction=1
// CHECK: trying XAIE API: XAie_StrmConnCctEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 0), CTRL=2, slvPortNum=0, SOUTH=4, mstrPortNum=0
// CHECK: trying XAIE API: XAie_StrmConnCctEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 0), WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle())=4, connectOp.sourceIndex()=3, WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle())=6, connectOp.destIndex()=0
// CHECK: trying XAIE API: XAie_StrmConnCctEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 0), WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle())=4, connectOp.sourceIndex()=7, WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle())=7, connectOp.destIndex()=0
// CHECK: trying XAIE API: XAie_StrmConnCctEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 0), WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle())=7, connectOp.sourceIndex()=0, WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle())=4, connectOp.destIndex()=2
// CHECK: trying XAIE API: XAie_StrmConnCctEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 1), WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle())=4, connectOp.sourceIndex()=0, WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle())=1, connectOp.destIndex()=0
// CHECK: trying XAIE API: XAie_StrmConnCctEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 1), WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle())=1, connectOp.sourceIndex()=0, WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle())=6, connectOp.destIndex()=0
// CHECK: trying XAIE API: XAie_StrmConnCctEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 1, row: 0), CTRL=2, slvPortNum=0, SOUTH=4, mstrPortNum=0
// CHECK: trying XAIE API: XAie_StrmConnCctEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 1, row: 0), WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle())=5, connectOp.sourceIndex()=0, WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle())=6, connectOp.destIndex()=0
// CHECK: trying XAIE API: XAie_StrmConnCctEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 1, row: 0), WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle())=7, connectOp.sourceIndex()=0, WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle())=5, connectOp.destIndex()=0
// CHECK: trying XAIE API: XAie_StrmConnCctEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 1, row: 1), WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle())=4, connectOp.sourceIndex()=0, WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle())=1, connectOp.destIndex()=0
// CHECK: trying XAIE API: XAie_StrmConnCctEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 1, row: 1), WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle())=1, connectOp.sourceIndex()=0, WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle())=6, connectOp.destIndex()=0
// CHECK: trying XAIE API: XAie_StrmConnCctEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 2, row: 0), CTRL=2, slvPortNum=0, SOUTH=4, mstrPortNum=0
// CHECK: trying XAIE API: XAie_StrmConnCctEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 2, row: 0), WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle())=6, connectOp.sourceIndex()=0, WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle())=5, connectOp.destIndex()=0
// CHECK: trying XAIE API: XAie_StrmConnCctEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 2, row: 1), WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle())=1, connectOp.sourceIndex()=0, WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle())=4, connectOp.destIndex()=0
// CHECK: trying XAIE API: XAie_StrmConnCctEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 2, row: 1), WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle())=6, connectOp.sourceIndex()=0, WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle())=1, connectOp.destIndex()=0
// CHECK: trying XAIE API: XAie_StrmConnCctEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle())=4, connectOp.sourceIndex()=0, WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle())=1, connectOp.destIndex()=0
// CHECK: trying XAIE API: XAie_StrmConnCctEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle())=7, connectOp.sourceIndex()=0, WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle())=1, connectOp.destIndex()=1
// CHECK: trying XAIE API: XAie_StrmConnCctEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle())=1, connectOp.sourceIndex()=0, WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle())=7, connectOp.destIndex()=0
// CHECK: trying XAIE API: XAie_StrmConnCctEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 1, row: 2), WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle())=4, connectOp.sourceIndex()=0, WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle())=5, connectOp.destIndex()=0
// CHECK: trying XAIE API: XAie_StrmConnCctEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 1, row: 2), WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle())=5, connectOp.sourceIndex()=0, WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle())=7, connectOp.destIndex()=0
// CHECK: trying XAIE API: XAie_StrmConnCctEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 2, row: 2), WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle())=5, connectOp.sourceIndex()=0, WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle())=4, connectOp.destIndex()=0
// CHECK: trying XAIE API: XAie_EnableShimDmaToAieStrmPort with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 0), connectOp.destIndex()=3
// CHECK: trying XAIE API: XAie_EnableShimDmaToAieStrmPort with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 0), connectOp.destIndex()=7
// CHECK: trying XAIE API: XAie_EnableAieToShimDmaStrmPort with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 0), connectOp.sourceIndex()=2
// CHECK: trying XAIE API: XAie_CoreEnable with args: &devInst={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2)
// CHECK: Generating: {{.+}}/aie_cdo_elfs.bin
// CHECK: Generating: {{.+}}/aie_cdo_init.bin
// CHECK: Generating: {{.+}}/aie_cdo_enable.bin
// CHECK: Generating: {{.+}}/aie_cdo_elfs.bin
// CHECK: (NOP Command): Payload Length: 0
// CHECK: (NOP Command): Payload Length: 2
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x0000000000220000  Size: 128
// CHECK:     Address: 0x0000000000220000  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220004  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220008  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022000C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220010  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220014  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220018  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022001C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220020  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220024  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220028  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022002C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220030  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220034  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220038  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022003C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220040  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220044  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220048  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022004C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220050  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220054  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220058  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022005C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220060  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220064  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220068  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022006C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220070  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220074  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220078  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022007C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220080  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220084  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220088  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022008C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220090  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220094  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220098  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022009C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002200A0  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002200A4  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002200A8  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002200AC  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002200B0  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002200B4  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002200B8  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002200BC  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002200C0  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002200C4  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002200C8  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002200CC  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002200D0  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002200D4  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002200D8  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002200DC  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002200E0  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002200E4  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002200E8  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002200EC  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002200F0  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002200F4  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002200F8  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002200FC  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220100  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220104  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220108  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022010C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220110  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220114  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220118  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022011C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220120  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220124  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220128  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022012C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220130  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220134  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220138  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022013C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220140  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220144  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220148  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022014C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220150  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220154  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220158  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022015C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220160  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220164  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220168  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022016C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220170  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220174  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220178  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022017C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220180  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220184  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220188  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022018C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220190  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220194  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220198  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022019C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002201A0  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002201A4  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002201A8  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002201AC  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002201B0  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002201B4  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002201B8  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002201BC  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002201C0  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002201C4  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002201C8  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002201CC  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002201D0  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002201D4  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002201D8  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002201DC  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002201E0  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002201E4  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002201E8  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002201EC  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002201F0  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002201F4  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002201F8  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002201FC  Data@ {{.+}} is: 0x00000000

// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x0000000000220200  Size: 436
// CHECK:     Address: 0x0000000000220200  Data@ {{.+}} is: 0x000003C0
// CHECK:     Address: 0x0000000000220204  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220208  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022020C  Data@ {{.+}} is: 0x00680000
// CHECK:     Address: 0x0000000000220210  Data@ {{.+}} is: 0x0000003D
// CHECK:     Address: 0x0000000000220214  Data@ {{.+}} is: 0x017BFF9C
// CHECK:     Address: 0x0000000000220218  Data@ {{.+}} is: 0x62000002
// CHECK:     Address: 0x000000000022021C  Data@ {{.+}} is: 0x01018FFC
// CHECK:     Address: 0x0000000000220220  Data@ {{.+}} is: 0x9A00113B
// CHECK:     Address: 0x0000000000220224  Data@ {{.+}} is: 0x800001D0
// CHECK:     Address: 0x0000000000220228  Data@ {{.+}} is: 0x113BFF7C
// CHECK:     Address: 0x000000000022022C  Data@ {{.+}} is: 0x01CA9A00
// CHECK:     Address: 0x0000000000220230  Data@ {{.+}} is: 0xFF6CC000
// CHECK:     Address: 0x0000000000220234  Data@ {{.+}} is: 0x1400113B
// CHECK:     Address: 0x0000000000220238  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022023C  Data@ {{.+}} is: 0x113BFF5D
// CHECK:     Address: 0x0000000000220240  Data@ {{.+}} is: 0x00003600
// CHECK:     Address: 0x0000000000220244  Data@ {{.+}} is: 0xFF4D4000
// CHECK:     Address: 0x0000000000220248  Data@ {{.+}} is: 0x5000113B
// CHECK:     Address: 0x000000000022024C  Data@ {{.+}} is: 0x80000004
// CHECK:     Address: 0x0000000000220250  Data@ {{.+}} is: 0x113BFF3D
// CHECK:     Address: 0x0000000000220254  Data@ {{.+}} is: 0x00047200
// CHECK:     Address: 0x0000000000220258  Data@ {{.+}} is: 0xFF2DC000
// CHECK:     Address: 0x000000000022025C  Data@ {{.+}} is: 0x9400113B
// CHECK:     Address: 0x0000000000220260  Data@ {{.+}} is: 0xD0000004
// CHECK:     Address: 0x0000000000220264  Data@ {{.+}} is: 0x113BFF1E
// CHECK:     Address: 0x0000000000220268  Data@ {{.+}} is: 0x0004B600
// CHECK:     Address: 0x000000000022026C  Data@ {{.+}} is: 0xFF0FD000
// CHECK:     Address: 0x0000000000220270  Data@ {{.+}} is: 0x000003C0
// CHECK:     Address: 0x0000000000220274  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220278  Data@ {{.+}} is: 0x08000000
// CHECK:     Address: 0x000000000022027C  Data@ {{.+}} is: 0x0000FFE0
// CHECK:     Address: 0x0000000000220280  Data@ {{.+}} is: 0x0622021D
// CHECK:     Address: 0x0000000000220284  Data@ {{.+}} is: 0x00010000
// CHECK:     Address: 0x0000000000220288  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x000000000022028C  Data@ {{.+}} is: 0x16420219
// CHECK:     Address: 0x0000000000220290  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220294  Data@ {{.+}} is: 0x02190001
// CHECK:     Address: 0x0000000000220298  Data@ {{.+}} is: 0x011D1682
// CHECK:     Address: 0x000000000022029C  Data@ {{.+}} is: 0x0006C806
// CHECK:     Address: 0x00000000002202A0  Data@ {{.+}} is: 0x000003C0
// CHECK:     Address: 0x00000000002202A4  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002202A8  Data@ {{.+}} is: 0x07FE7600
// CHECK:     Address: 0x00000000002202AC  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002202B0  Data@ {{.+}} is: 0xC0C06DBD
// CHECK:     Address: 0x00000000002202B4  Data@ {{.+}} is: 0x017BFFB8
// CHECK:     Address: 0x00000000002202B8  Data@ {{.+}} is: 0x4000002A
// CHECK:     Address: 0x00000000002202BC  Data@ {{.+}} is: 0x0006CFFD
// CHECK:     Address: 0x00000000002202C0  Data@ {{.+}} is: 0x000000B7
// CHECK:     Address: 0x00000000002202C4  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002202C8  Data@ {{.+}} is: 0xFFA8C800
// CHECK:     Address: 0x00000000002202CC  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002202D0  Data@ {{.+}} is: 0x311D0001
// CHECK:     Address: 0x00000000002202D4  Data@ {{.+}} is: 0x02078828
// CHECK:     Address: 0x00000000002202D8  Data@ {{.+}} is: 0x480E511D
// CHECK:     Address: 0x00000000002202DC  Data@ {{.+}} is: 0x5D990081
// CHECK:     Address: 0x00000000002202E0  Data@ {{.+}} is: 0x309D1550
// CHECK:     Address: 0x00000000002202E4  Data@ {{.+}} is: 0x00408A10
// CHECK:     Address: 0x00000000002202E8  Data@ {{.+}} is: 0x8A102D9D
// CHECK:     Address: 0x00000000002202EC  Data@ {{.+}} is: 0x28430301
// CHECK:     Address: 0x00000000002202F0  Data@ {{.+}} is: 0x148A120B
// CHECK:     Address: 0x00000000002202F4  Data@ {{.+}} is: 0x55FB0001
// CHECK:     Address: 0x00000000002202F8  Data@ {{.+}} is: 0x82000212
// CHECK:     Address: 0x00000000002202FC  Data@ {{.+}} is: 0x05048910
// CHECK:     Address: 0x0000000000220300  Data@ {{.+}} is: 0x124B2843
// CHECK:     Address: 0x0000000000220304  Data@ {{.+}} is: 0x0010BA2E
// CHECK:     Address: 0x0000000000220308  Data@ {{.+}} is: 0x32CB2843
// CHECK:     Address: 0x000000000022030C  Data@ {{.+}} is: 0x0010CF2E
// CHECK:     Address: 0x0000000000220310  Data@ {{.+}} is: 0x530B2843
// CHECK:     Address: 0x0000000000220314  Data@ {{.+}} is: 0x0010D3AE
// CHECK:     Address: 0x0000000000220318  Data@ {{.+}} is: 0x734B2843
// CHECK:     Address: 0x000000000022031C  Data@ {{.+}} is: 0x0010E32E
// CHECK:     Address: 0x0000000000220320  Data@ {{.+}} is: 0x938B2843
// CHECK:     Address: 0x0000000000220324  Data@ {{.+}} is: 0x001188AE
// CHECK:     Address: 0x0000000000220328  Data@ {{.+}} is: 0xB60B2843
// CHECK:     Address: 0x000000000022032C  Data@ {{.+}} is: 0x0011992E
// CHECK:     Address: 0x0000000000220330  Data@ {{.+}} is: 0xD64B2843
// CHECK:     Address: 0x0000000000220334  Data@ {{.+}} is: 0x0001310A
// CHECK:     Address: 0x0000000000220338  Data@ {{.+}} is: 0x421535BD
// CHECK:     Address: 0x000000000022033C  Data@ {{.+}} is: 0x28432210
// CHECK:     Address: 0x0000000000220340  Data@ {{.+}} is: 0x038A128B
// CHECK:     Address: 0x0000000000220344  Data@ {{.+}} is: 0x05BD0000
// CHECK:     Address: 0x0000000000220348  Data@ {{.+}} is: 0x2210421E
// CHECK:     Address: 0x000000000022034C  Data@ {{.+}} is: 0x13CB2843
// CHECK:     Address: 0x0000000000220350  Data@ {{.+}} is: 0x0000470A
// CHECK:     Address: 0x0000000000220354  Data@ {{.+}} is: 0x420845BD
// CHECK:     Address: 0x0000000000220358  Data@ {{.+}} is: 0x28432610
// CHECK:     Address: 0x000000000022035C  Data@ {{.+}} is: 0x268A310B
// CHECK:     Address: 0x0000000000220360  Data@ {{.+}} is: 0x25BD0000
// CHECK:     Address: 0x0000000000220364  Data@ {{.+}} is: 0x2A104204
// CHECK:     Address: 0x0000000000220368  Data@ {{.+}} is: 0x508B2843
// CHECK:     Address: 0x000000000022036C  Data@ {{.+}} is: 0x0000060A
// CHECK:     Address: 0x0000000000220370  Data@ {{.+}} is: 0x420005BD
// CHECK:     Address: 0x0000000000220374  Data@ {{.+}} is: 0x28432E10
// CHECK:     Address: 0x0000000000220378  Data@ {{.+}} is: 0xF58A700B
// CHECK:     Address: 0x000000000022037C  Data@ {{.+}} is: 0xF5BD0001
// CHECK:     Address: 0x0000000000220380  Data@ {{.+}} is: 0x32104235
// CHECK:     Address: 0x0000000000220384  Data@ {{.+}} is: 0x968B2843
// CHECK:     Address: 0x0000000000220388  Data@ {{.+}} is: 0x0001040A
// CHECK:     Address: 0x000000000022038C  Data@ {{.+}} is: 0x422D05BD
// CHECK:     Address: 0x0000000000220390  Data@ {{.+}} is: 0x28432210
// CHECK:     Address: 0x0000000000220394  Data@ {{.+}} is: 0xD78A158B
// CHECK:     Address: 0x0000000000220398  Data@ {{.+}} is: 0xD5BD0001
// CHECK:     Address: 0x000000000022039C  Data@ {{.+}} is: 0x2210423B
// CHECK:     Address: 0x00000000002203A0  Data@ {{.+}} is: 0x1C2E9659
// CHECK:     Address: 0x00000000002203A4  Data@ {{.+}} is: 0x09B08219
// CHECK:     Address: 0x00000000002203A8  Data@ {{.+}} is: 0x09D08219
// CHECK:     Address: 0x00000000002203AC  Data@ {{.+}} is: 0xC9908235
// CHECK:     Address: 0x00000000002203B0  Data@ {{.+}} is: 0x75BD0025
// CHECK:     Address: 0x00000000002203B4  Data@ {{.+}} is: 0x2E10456F
// CHECK:     Address: 0x00000000002203B8  Data@ {{.+}} is: 0x45D05DBD
// CHECK:     Address: 0x00000000002203BC  Data@ {{.+}} is: 0x30FB2A10
// CHECK:     Address: 0x00000000002203C0  Data@ {{.+}} is: 0x82000210
// CHECK:     Address: 0x00000000002203C4  Data@ {{.+}} is: 0x00470930
// CHECK:     Address: 0x00000000002203C8  Data@ {{.+}} is: 0x4211CDBD
// CHECK:     Address: 0x00000000002203CC  Data@ {{.+}} is: 0x16592210
// CHECK:     Address: 0x00000000002203D0  Data@ {{.+}} is: 0x55BD1C24
// CHECK:     Address: 0x00000000002203D4  Data@ {{.+}} is: 0x22104212
// CHECK:     Address: 0x00000000002203D8  Data@ {{.+}} is: 0x1C249659
// CHECK:     Address: 0x00000000002203DC  Data@ {{.+}} is: 0x421535BD
// CHECK:     Address: 0x00000000002203E0  Data@ {{.+}} is: 0x16592210
// CHECK:     Address: 0x00000000002203E4  Data@ {{.+}} is: 0x45BD1C25
// CHECK:     Address: 0x00000000002203E8  Data@ {{.+}} is: 0x22104217
// CHECK:     Address: 0x00000000002203EC  Data@ {{.+}} is: 0x1C259659
// CHECK:     Address: 0x00000000002203F0  Data@ {{.+}} is: 0x4219E5BD
// CHECK:     Address: 0x00000000002203F4  Data@ {{.+}} is: 0x16592210
// CHECK:     Address: 0x00000000002203F8  Data@ {{.+}} is: 0x75BD1C26
// CHECK:     Address: 0x00000000002203FC  Data@ {{.+}} is: 0x2210421A
// CHECK:     Address: 0x0000000000220400  Data@ {{.+}} is: 0x1C269659
// CHECK:     Address: 0x0000000000220404  Data@ {{.+}} is: 0x421C65BD
// CHECK:     Address: 0x0000000000220408  Data@ {{.+}} is: 0x28432210
// CHECK:     Address: 0x000000000022040C  Data@ {{.+}} is: 0x538A138B
// CHECK:     Address: 0x0000000000220410  Data@ {{.+}} is: 0x55BD0000
// CHECK:     Address: 0x0000000000220414  Data@ {{.+}} is: 0x22104200
// CHECK:     Address: 0x0000000000220418  Data@ {{.+}} is: 0x1C201659
// CHECK:     Address: 0x000000000022041C  Data@ {{.+}} is: 0x420505BD
// CHECK:     Address: 0x0000000000220420  Data@ {{.+}} is: 0x28432210
// CHECK:     Address: 0x0000000000220424  Data@ {{.+}} is: 0x505A108B
// CHECK:     Address: 0x0000000000220428  Data@ {{.+}} is: 0x15BD002B
// CHECK:     Address: 0x000000000022042C  Data@ {{.+}} is: 0x22104209
// CHECK:     Address: 0x0000000000220430  Data@ {{.+}} is: 0x110B2843
// CHECK:     Address: 0x0000000000220434  Data@ {{.+}} is: 0x00118FAE
// CHECK:     Address: 0x0000000000220438  Data@ {{.+}} is: 0x421F25BD
// CHECK:     Address: 0x000000000022043C  Data@ {{.+}} is: 0x28432210
// CHECK:     Address: 0x0000000000220440  Data@ {{.+}} is: 0x660A13CB
// CHECK:     Address: 0x0000000000220444  Data@ {{.+}} is: 0x65BD0000
// CHECK:     Address: 0x0000000000220448  Data@ {{.+}} is: 0x22104232
// CHECK:     Address: 0x000000000022044C  Data@ {{.+}} is: 0x160B2843
// CHECK:     Address: 0x0000000000220450  Data@ {{.+}} is: 0x0000668A
// CHECK:     Address: 0x0000000000220454  Data@ {{.+}} is: 0x423465BD
// CHECK:     Address: 0x0000000000220458  Data@ {{.+}} is: 0x28432210
// CHECK:     Address: 0x000000000022045C  Data@ {{.+}} is: 0x81EA164B
// CHECK:     Address: 0x0000000000220460  Data@ {{.+}} is: 0x8CBD0001
// CHECK:     Address: 0x0000000000220464  Data@ {{.+}} is: 0x221045C5
// CHECK:     Address: 0x0000000000220468  Data@ {{.+}} is: 0x168B2843
// CHECK:     Address: 0x000000000022046C  Data@ {{.+}} is: 0x000420B2
// CHECK:     Address: 0x0000000000220470  Data@ {{.+}} is: 0x100060BB
// CHECK:     Address: 0x0000000000220474  Data@ {{.+}} is: 0x88040058
// CHECK:     Address: 0x0000000000220478  Data@ {{.+}} is: 0x65BD0701
// CHECK:     Address: 0x000000000022047C  Data@ {{.+}} is: 0x2210423A
// CHECK:     Address: 0x0000000000220480  Data@ {{.+}} is: 0x174B2843
// CHECK:     Address: 0x0000000000220484  Data@ {{.+}} is: 0x0000678A
// CHECK:     Address: 0x0000000000220488  Data@ {{.+}} is: 0x022C65FB
// CHECK:     Address: 0x000000000022048C  Data@ {{.+}} is: 0xC9108200
// CHECK:     Address: 0x0000000000220490  Data@ {{.+}} is: 0x28430040
// CHECK:     Address: 0x0000000000220494  Data@ {{.+}} is: 0x01E6158B
// CHECK:     Address: 0x0000000000220498  Data@ {{.+}} is: 0x00BD002A
// CHECK:     Address: 0x000000000022049C  Data@ {{.+}} is: 0x221046F6
// CHECK:     Address: 0x00000000002204A0  Data@ {{.+}} is: 0x0000007F
// CHECK:     Address: 0x00000000002204A4  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002204A8  Data@ {{.+}} is: 0xC8000000
// CHECK:     Address: 0x00000000002204AC  Data@ {{.+}} is: 0x7659FFB8
// CHECK:     Address: 0x00000000002204B0  Data@ {{.+}} is: 0x000107FE
// CHECK:     Address: 0x00000000002204B4  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002204B8  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002204BC  Data@ {{.+}} is: 0x10C18C99
// CHECK:     Address: 0x00000000002204C0  Data@ {{.+}} is: 0x10001619
// CHECK:     Address: 0x00000000002204C4  Data@ {{.+}} is: 0x50400195
// CHECK:     Address: 0x00000000002204C8  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002204CC  Data@ {{.+}} is: 0x071D0001
// CHECK:     Address: 0x00000000002204D0  Data@ {{.+}} is: 0x002088C6
// CHECK:     Address: 0x00000000002204D4  Data@ {{.+}} is: 0x88C42C9D
// CHECK:     Address: 0x00000000002204D8  Data@ {{.+}} is: 0x209D0023
// CHECK:     Address: 0x00000000002204DC  Data@ {{.+}} is: 0x01018EF6
// CHECK:     Address: 0x00000000002204E0  Data@ {{.+}} is: 0x00100113
// CHECK:     Address: 0x00000000002204E4  Data@ {{.+}} is: 0x0006C800
// CHECK:     Address: 0x00000000002204E8  Data@ {{.+}} is: 0xC806091D
// CHECK:     Address: 0x00000000002204EC  Data@ {{.+}} is: 0x011D0123
// CHECK:     Address: 0x00000000002204F0  Data@ {{.+}} is: 0x10078852
// CHECK:     Address: 0x00000000002204F4  Data@ {{.+}} is: 0x08D4011D
// CHECK:     Address: 0x00000000002204F8  Data@ {{.+}} is: 0x011D2003
// CHECK:     Address: 0x00000000002204FC  Data@ {{.+}} is: 0x30034962
// CHECK:     Address: 0x0000000000220500  Data@ {{.+}} is: 0x00060059
// CHECK:     Address: 0x0000000000220504  Data@ {{.+}} is: 0x04680055
// CHECK:     Address: 0x0000000000220508  Data@ {{.+}} is: 0x28430007
// CHECK:     Address: 0x000000000022050C  Data@ {{.+}} is: 0x006C1A3B
// CHECK:     Address: 0x0000000000220510  Data@ {{.+}} is: 0x16590010
// CHECK:     Address: 0x0000000000220514  Data@ {{.+}} is: 0x081919A0
// CHECK:     Address: 0x0000000000220518  Data@ {{.+}} is: 0x83D938CB
// CHECK:     Address: 0x000000000022051C  Data@ {{.+}} is: 0x764D0006
// CHECK:     Address: 0x0000000000220520  Data@ {{.+}} is: 0x00503834
// CHECK:     Address: 0x0000000000220524  Data@ {{.+}} is: 0x38CB0819
// CHECK:     Address: 0x0000000000220528  Data@ {{.+}} is: 0x0FFD4699
// CHECK:     Address: 0x000000000022052C  Data@ {{.+}} is: 0x18347659
// CHECK:     Address: 0x0000000000220530  Data@ {{.+}} is: 0x38CB0819
// CHECK:     Address: 0x0000000000220534  Data@ {{.+}} is: 0x0FFDC699
// CHECK:     Address: 0x0000000000220538  Data@ {{.+}} is: 0x18347659
// CHECK:     Address: 0x000000000022053C  Data@ {{.+}} is: 0x38CB0819
// CHECK:     Address: 0x0000000000220540  Data@ {{.+}} is: 0x19007659
// CHECK:     Address: 0x0000000000220544  Data@ {{.+}} is: 0x18347659
// CHECK:     Address: 0x0000000000220548  Data@ {{.+}} is: 0x0FFE7605
// CHECK:     Address: 0x000000000022054C  Data@ {{.+}} is: 0x765908CB
// CHECK:     Address: 0x0000000000220550  Data@ {{.+}} is: 0x76591E80
// CHECK:     Address: 0x0000000000220554  Data@ {{.+}} is: 0xD0051834
// CHECK:     Address: 0x0000000000220558  Data@ {{.+}} is: 0x08CB0FFE
// CHECK:     Address: 0x000000000022055C  Data@ {{.+}} is: 0xA03B2843
// CHECK:     Address: 0x0000000000220560  Data@ {{.+}} is: 0x0001700B
// CHECK:     Address: 0x0000000000220564  Data@ {{.+}} is: 0x1A3B283B
// CHECK:     Address: 0x0000000000220568  Data@ {{.+}} is: 0x0801B008
// CHECK:     Address: 0x000000000022056C  Data@ {{.+}} is: 0x283B02CB
// CHECK:     Address: 0x0000000000220570  Data@ {{.+}} is: 0x008A423B
// CHECK:     Address: 0x0000000000220574  Data@ {{.+}} is: 0x00CB0800
// CHECK:     Address: 0x0000000000220578  Data@ {{.+}} is: 0xE03B2843
// CHECK:     Address: 0x000000000022057C  Data@ {{.+}} is: 0x0011606F
// CHECK:     Address: 0x0000000000220580  Data@ {{.+}} is: 0x193B28B7
// CHECK:     Address: 0x0000000000220584  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220588  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022058C  Data@ {{.+}} is: 0x413B28BB
// CHECK:     Address: 0x0000000000220590  Data@ {{.+}} is: 0xD82E036C
// CHECK:     Address: 0x0000000000220594  Data@ {{.+}} is: 0x2843FFAD
// CHECK:     Address: 0x0000000000220598  Data@ {{.+}} is: 0x0B049D3B
// CHECK:     Address: 0x000000000022059C  Data@ {{.+}} is: 0x28430000
// CHECK:     Address: 0x00000000002205A0  Data@ {{.+}} is: 0xB1EF990B
// CHECK:     Address: 0x00000000002205A4  Data@ {{.+}} is: 0x2EBB0000
// CHECK:     Address: 0x00000000002205A8  Data@ {{.+}} is: 0x0800D2CB
// CHECK:     Address: 0x00000000002205AC  Data@ {{.+}} is: 0xFFBEDF2B
// CHECK:     Address: 0x00000000002205B0  Data@ {{.+}} is: 0x853B28B7
// CHECK:     Address: 0x00000000002205B4  Data@ {{.+}} is: 0x082FC7EE
// CHECK:     Address: 0x00000000002205B8  Data@ {{.+}} is: 0xE0D378CB
// CHECK:     Address: 0x00000000002205BC  Data@ {{.+}} is: 0x71AE164D
// CHECK:     Address: 0x00000000002205C0  Data@ {{.+}} is: 0x959300D0
// CHECK:     Address: 0x00000000002205C4  Data@ {{.+}} is: 0x51CB0F2A
// CHECK:     Address: 0x00000000002205C8  Data@ {{.+}} is: 0x964D0050
// CHECK:     Address: 0x00000000002205CC  Data@ {{.+}} is: 0x20D279AA
// CHECK:     Address: 0x00000000002205D0  Data@ {{.+}} is: 0x393A164D
// CHECK:     Address: 0x00000000002205D4  Data@ {{.+}} is: 0x087D2052
// CHECK:     Address: 0x00000000002205D8  Data@ {{.+}} is: 0xA80671CB
// CHECK:     Address: 0x00000000002205DC  Data@ {{.+}} is: 0x0106ABD9
// CHECK:     Address: 0x00000000002205E0  Data@ {{.+}} is: 0x0102A9D9
// CHECK:     Address: 0x00000000002205E4  Data@ {{.+}} is: 0x7F11E59D
// CHECK:     Address: 0x00000000002205E8  Data@ {{.+}} is: 0x2EBBA0D4
// CHECK:     Address: 0x00000000002205EC  Data@ {{.+}} is: 0x0800D20B
// CHECK:     Address: 0x00000000002205F0  Data@ {{.+}} is: 0xA0543E0B
// CHECK:     Address: 0x00000000002205F4  Data@ {{.+}} is: 0x793A164D
// CHECK:     Address: 0x00000000002205F8  Data@ {{.+}} is: 0xA593C0D1
// CHECK:     Address: 0x00000000002205FC  Data@ {{.+}} is: 0x39CB0F12
// CHECK:     Address: 0x0000000000220600  Data@ {{.+}} is: 0x024BC051
// CHECK:     Address: 0x0000000000220604  Data@ {{.+}} is: 0x96422841
// CHECK:     Address: 0x0000000000220608  Data@ {{.+}} is: 0x20D279A4
// CHECK:     Address: 0x000000000022060C  Data@ {{.+}} is: 0x393A164D
// CHECK:     Address: 0x0000000000220610  Data@ {{.+}} is: 0x2EBB2052
// CHECK:     Address: 0x0000000000220614  Data@ {{.+}} is: 0x0800E63B
// CHECK:     Address: 0x0000000000220618  Data@ {{.+}} is: 0xE05339CB
// CHECK:     Address: 0x000000000022061C  Data@ {{.+}} is: 0x7E3D164D
// CHECK:     Address: 0x0000000000220620  Data@ {{.+}} is: 0x824F20D5
// CHECK:     Address: 0x0000000000220624  Data@ {{.+}} is: 0xA1652862
// CHECK:     Address: 0x0000000000220628  Data@ {{.+}} is: 0x0F14C5A3
// CHECK:     Address: 0x000000000022062C  Data@ {{.+}} is: 0x20553E4B
// CHECK:     Address: 0x0000000000220630  Data@ {{.+}} is: 0x79A5164D
// CHECK:     Address: 0x0000000000220634  Data@ {{.+}} is: 0x28B7C0D4
// CHECK:     Address: 0x0000000000220638  Data@ {{.+}} is: 0xB8AD9F4B
// CHECK:     Address: 0x000000000022063C  Data@ {{.+}} is: 0x3ACB0838
// CHECK:     Address: 0x0000000000220640  Data@ {{.+}} is: 0x964DC054
// CHECK:     Address: 0x0000000000220644  Data@ {{.+}} is: 0x40D179A5
// CHECK:     Address: 0x0000000000220648  Data@ {{.+}} is: 0x28490267
// CHECK:     Address: 0x000000000022064C  Data@ {{.+}} is: 0x0A3A1642
// CHECK:     Address: 0x0000000000220650  Data@ {{.+}} is: 0x40513B6B
// CHECK:     Address: 0x0000000000220654  Data@ {{.+}} is: 0x7ACB087D
// CHECK:     Address: 0x0000000000220658  Data@ {{.+}} is: 0x164D60D2
// CHECK:     Address: 0x000000000022065C  Data@ {{.+}} is: 0x40D57C3A
// CHECK:     Address: 0x0000000000220660  Data@ {{.+}} is: 0x1FCB28BB
// CHECK:     Address: 0x0000000000220664  Data@ {{.+}} is: 0x3838C6AD
// CHECK:     Address: 0x0000000000220668  Data@ {{.+}} is: 0x824F4055
// CHECK:     Address: 0x000000000022066C  Data@ {{.+}} is: 0x6165285A
// CHECK:     Address: 0x0000000000220670  Data@ {{.+}} is: 0x09C0011A
// CHECK:     Address: 0x0000000000220674  Data@ {{.+}} is: 0x60523A8B
// CHECK:     Address: 0x0000000000220678  Data@ {{.+}} is: 0x0F1A0593
// CHECK:     Address: 0x000000000022067C  Data@ {{.+}} is: 0x40D37CCB
// CHECK:     Address: 0x0000000000220680  Data@ {{.+}} is: 0x79A6964D
// CHECK:     Address: 0x0000000000220684  Data@ {{.+}} is: 0x164D80D4
// CHECK:     Address: 0x0000000000220688  Data@ {{.+}} is: 0x80543C3A
// CHECK:     Address: 0x000000000022068C  Data@ {{.+}} is: 0x2860824B
// CHECK:     Address: 0x0000000000220690  Data@ {{.+}} is: 0x3CCB081D
// CHECK:     Address: 0x0000000000220694  Data@ {{.+}} is: 0x8BD94053
// CHECK:     Address: 0x0000000000220698  Data@ {{.+}} is: 0x164D0406
// CHECK:     Address: 0x000000000022069C  Data@ {{.+}} is: 0x80513C39
// CHECK:     Address: 0x00000000002206A0  Data@ {{.+}} is: 0x3CAB0819
// CHECK:     Address: 0x00000000002206A4  Data@ {{.+}} is: 0x28528263
// CHECK:     Address: 0x00000000002206A8  Data@ {{.+}} is: 0x80D27804
// CHECK:     Address: 0x00000000002206AC  Data@ {{.+}} is: 0x040291D9
// CHECK:     Address: 0x00000000002206B0  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002206B4  Data@ {{.+}} is: 0x285A0249
// CHECK:     Address: 0x00000000002206B8  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002206BC  Data@ {{.+}} is: 0x15F8E599
// CHECK:     Address: 0x00000000002206C0  Data@ {{.+}} is: 0x2850825B
// CHECK:     Address: 0x00000000002206C4  Data@ {{.+}} is: 0x6DE40765
// CHECK:     Address: 0x00000000002206C8  Data@ {{.+}} is: 0x28430700
// CHECK:     Address: 0x00000000002206CC  Data@ {{.+}} is: 0x5B041D0B
// CHECK:     Address: 0x00000000002206D0  Data@ {{.+}} is: 0xFD9D0001
// CHECK:     Address: 0x00000000002206D4  Data@ {{.+}} is: 0x38000F12
// CHECK:     Address: 0x00000000002206D8  Data@ {{.+}} is: 0xD24B2843
// CHECK:     Address: 0x00000000002206DC  Data@ {{.+}} is: 0x0012F02C
// CHECK:     Address: 0x00000000002206E0  Data@ {{.+}} is: 0x9E4B28B7
// CHECK:     Address: 0x00000000002206E4  Data@ {{.+}} is: 0x082A01EC
// CHECK:     Address: 0x00000000002206E8  Data@ {{.+}} is: 0xA0D278CB
// CHECK:     Address: 0x00000000002206EC  Data@ {{.+}} is: 0x79A0164D
// CHECK:     Address: 0x00000000002206F0  Data@ {{.+}} is: 0x287700D1
// CHECK:     Address: 0x00000000002206F4  Data@ {{.+}} is: 0xE008188B
// CHECK:     Address: 0x00000000002206F8  Data@ {{.+}} is: 0x39068343
// CHECK:     Address: 0x00000000002206FC  Data@ {{.+}} is: 0x487F0051
// CHECK:     Address: 0x0000000000220700  Data@ {{.+}} is: 0xA1652050
// CHECK:     Address: 0x0000000000220704  Data@ {{.+}} is: 0x0A51E5D3
// CHECK:     Address: 0x0000000000220708  Data@ {{.+}} is: 0xA05238CB
// CHECK:     Address: 0x000000000022070C  Data@ {{.+}} is: 0xD20B28BB
// CHECK:     Address: 0x0000000000220710  Data@ {{.+}} is: 0x7005E008
// CHECK:     Address: 0x0000000000220714  Data@ {{.+}} is: 0xE59300D0
// CHECK:     Address: 0x0000000000220718  Data@ {{.+}} is: 0x55CB0A55
// CHECK:     Address: 0x000000000022071C  Data@ {{.+}} is: 0x164D0050
// CHECK:     Address: 0x0000000000220720  Data@ {{.+}} is: 0xA0D379A5
// CHECK:     Address: 0x0000000000220724  Data@ {{.+}} is: 0x3933964D
// CHECK:     Address: 0x0000000000220728  Data@ {{.+}} is: 0x164DA053
// CHECK:     Address: 0x000000000022072C  Data@ {{.+}} is: 0x20D47D3A
// CHECK:     Address: 0x0000000000220730  Data@ {{.+}} is: 0x08E00113
// CHECK:     Address: 0x0000000000220734  Data@ {{.+}} is: 0x20543DCB
// CHECK:     Address: 0x0000000000220738  Data@ {{.+}} is: 0x2840826B
// CHECK:     Address: 0x000000000022073C  Data@ {{.+}} is: 0x7812B82C
// CHECK:     Address: 0x0000000000220740  Data@ {{.+}} is: 0x964DA0D1
// CHECK:     Address: 0x0000000000220744  Data@ {{.+}} is: 0xA05139A5
// CHECK:     Address: 0x0000000000220748  Data@ {{.+}} is: 0x7D3A164D
// CHECK:     Address: 0x000000000022074C  Data@ {{.+}} is: 0x087DE0D5
// CHECK:     Address: 0x0000000000220750  Data@ {{.+}} is: 0xE0553DCB
// CHECK:     Address: 0x0000000000220754  Data@ {{.+}} is: 0x2851826B
// CHECK:     Address: 0x0000000000220758  Data@ {{.+}} is: 0x78091008
// CHECK:     Address: 0x000000000022075C  Data@ {{.+}} is: 0x28BBA0D2
// CHECK:     Address: 0x0000000000220760  Data@ {{.+}} is: 0xC8AC9D0B
// CHECK:     Address: 0x0000000000220764  Data@ {{.+}} is: 0xA0523812
// CHECK:     Address: 0x0000000000220768  Data@ {{.+}} is: 0x79A6164D
// CHECK:     Address: 0x000000000022076C  Data@ {{.+}} is: 0x0113C0D3
// CHECK:     Address: 0x0000000000220770  Data@ {{.+}} is: 0x39CB0962
// CHECK:     Address: 0x0000000000220774  Data@ {{.+}} is: 0x826BC053
// CHECK:     Address: 0x0000000000220778  Data@ {{.+}} is: 0xD8AC2860
// CHECK:     Address: 0x000000000022077C  Data@ {{.+}} is: 0x20D17812
// CHECK:     Address: 0x0000000000220780  Data@ {{.+}} is: 0x39A6964D
// CHECK:     Address: 0x0000000000220784  Data@ {{.+}} is: 0x164D2051
// CHECK:     Address: 0x0000000000220788  Data@ {{.+}} is: 0x60D4793A
// CHECK:     Address: 0x000000000022078C  Data@ {{.+}} is: 0x09A60113
// CHECK:     Address: 0x0000000000220790  Data@ {{.+}} is: 0x605439CB
// CHECK:     Address: 0x0000000000220794  Data@ {{.+}} is: 0x2869026B
// CHECK:     Address: 0x0000000000220798  Data@ {{.+}} is: 0x7812E9AC
// CHECK:     Address: 0x000000000022079C  Data@ {{.+}} is: 0x164D20D2
// CHECK:     Address: 0x00000000002207A0  Data@ {{.+}} is: 0x205239A7
// CHECK:     Address: 0x00000000002207A4  Data@ {{.+}} is: 0x793A164D
// CHECK:     Address: 0x00000000002207A8  Data@ {{.+}} is: 0x087D40D5
// CHECK:     Address: 0x00000000002207AC  Data@ {{.+}} is: 0x405539CB
// CHECK:     Address: 0x00000000002207B0  Data@ {{.+}} is: 0x28588263
// CHECK:     Address: 0x00000000002207B4  Data@ {{.+}} is: 0x20D17804
// CHECK:     Address: 0x00000000002207B8  Data@ {{.+}} is: 0x39A7964D
// CHECK:     Address: 0x00000000002207BC  Data@ {{.+}} is: 0x164D2051
// CHECK:     Address: 0x00000000002207C0  Data@ {{.+}} is: 0x80D3793A
// CHECK:     Address: 0x00000000002207C4  Data@ {{.+}} is: 0x39CB087D
// CHECK:     Address: 0x00000000002207C8  Data@ {{.+}} is: 0x02638053
// CHECK:     Address: 0x00000000002207CC  Data@ {{.+}} is: 0x78042861
// CHECK:     Address: 0x00000000002207D0  Data@ {{.+}} is: 0x91D920D2
// CHECK:     Address: 0x00000000002207D4  Data@ {{.+}} is: 0x00010102
// CHECK:     Address: 0x00000000002207D8  Data@ {{.+}} is: 0x82490001
// CHECK:     Address: 0x00000000002207DC  Data@ {{.+}} is: 0x00012868
// CHECK:     Address: 0x00000000002207E0  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002207E4  Data@ {{.+}} is: 0x2859025B
// CHECK:     Address: 0x00000000002207E8  Data@ {{.+}} is: 0x8C931165
// CHECK:     Address: 0x00000000002207EC  Data@ {{.+}} is: 0x16190705
// CHECK:     Address: 0x00000000002207F0  Data@ {{.+}} is: 0x01951084
// CHECK:     Address: 0x00000000002207F4  Data@ {{.+}} is: 0x1002C040
// CHECK:     Address: 0x00000000002207F8  Data@ {{.+}} is: 0x4DEE0B1D
// CHECK:     Address: 0x00000000002207FC  Data@ {{.+}} is: 0x3C9D0802
// CHECK:     Address: 0x0000000000220800  Data@ {{.+}} is: 0x18028DC0
// CHECK:     Address: 0x0000000000220804  Data@ {{.+}} is: 0x9D0B2843
// CHECK:     Address: 0x0000000000220808  Data@ {{.+}} is: 0x0008C00A
// CHECK:     Address: 0x000000000022080C  Data@ {{.+}} is: 0x019A017B
// CHECK:     Address: 0x0000000000220810  Data@ {{.+}} is: 0x88068340
// CHECK:     Address: 0x0000000000220814  Data@ {{.+}} is: 0x00FB0023
// CHECK:     Address: 0x0000000000220818  Data@ {{.+}} is: 0x824006F6
// CHECK:     Address: 0x000000000022081C  Data@ {{.+}} is: 0x0123C802
// CHECK:     Address: 0x0000000000220820  Data@ {{.+}} is: 0x07FED059
// CHECK:     Address: 0x0000000000220824  Data@ {{.+}} is: 0x07FE7659
// CHECK:     Address: 0x0000000000220828  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x000000000022082C  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220830  Data@ {{.+}} is: 0x8C990001
// CHECK:     Address: 0x0000000000220834  Data@ {{.+}} is: 0x16191201
// CHECK:     Address: 0x0000000000220838  Data@ {{.+}} is: 0x01951000
// CHECK:     Address: 0x000000000022083C  Data@ {{.+}} is: 0x00028040
// CHECK:     Address: 0x0000000000220840  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220844  Data@ {{.+}} is: 0x12100719
// CHECK:     Address: 0x0000000000220848  Data@ {{.+}} is: 0x1204EC99
// CHECK:     Address: 0x000000000022084C  Data@ {{.+}} is: 0x16F62099
// CHECK:     Address: 0x0000000000220850  Data@ {{.+}} is: 0x880003C0
// CHECK:     Address: 0x0000000000220854  Data@ {{.+}} is: 0x07100003
// CHECK:     Address: 0x0000000000220858  Data@ {{.+}} is: 0x00000030
// CHECK:     Address: 0x000000000022085C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220860  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220864  Data@ {{.+}} is: 0xE2190001
// CHECK:     Address: 0x0000000000220868  Data@ {{.+}} is: 0x00011660
// CHECK:     Address: 0x000000000022086C  Data@ {{.+}} is: 0x00950001
// CHECK:     Address: 0x0000000000220870  Data@ {{.+}} is: 0x00013800
// CHECK:     Address: 0x0000000000220874  Data@ {{.+}} is: 0x16A0E219
// CHECK:     Address: 0x0000000000220878  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x000000000022087C  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220880  Data@ {{.+}} is: 0x0000007F
// CHECK:     Address: 0x0000000000220884  Data@ {{.+}} is: 0x08000008
// CHECK:     Address: 0x0000000000220888  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022088C  Data@ {{.+}} is: 0x00010000
// CHECK:     Address: 0x0000000000220890  Data@ {{.+}} is: 0x38032019
// CHECK:     Address: 0x0000000000220894  Data@ {{.+}} is: 0x0FFC4299
// CHECK:     Address: 0x0000000000220898  Data@ {{.+}} is: 0x011D0001
// CHECK:     Address: 0x000000000022089C  Data@ {{.+}} is: 0x0000D802
// CHECK:     Address: 0x00000000002208A0  Data@ {{.+}} is: 0x0000081D
// CHECK:     Address: 0x00000000002208A4  Data@ {{.+}} is: 0x00010000
// CHECK:     Address: 0x00000000002208A8  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002208AC  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002208B0  Data@ {{.+}} is: 0x07FC42D9
// CHECK:     Address: 0x00000000002208B4  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002208B8  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002208BC  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002208C0  Data@ {{.+}} is: 0x10001819
// CHECK:     Address: 0x00000000002208C4  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002208C8  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002208CC  Data@ {{.+}} is: 0x3FFFE019

// CHECK: Generating: {{.+}}/aie_cdo_init.bin
// CHECK: (NOP Command): Payload Length: 0
// CHECK: (MaskWrite64): Address: 0x0000000000232000  Mask: 0x00000002  Data: 0x00000002
// CHECK: (MaskWrite64): Address: 0x0000000000232000  Mask: 0x00000002  Data: 0x00000000
// CHECK: (Write64): Address:  0x000000000021F000 Data:  0x00000000
// CHECK: (Write64): Address:  0x000000000021F010 Data:  0x00000000
// CHECK: (Write64): Address:  0x000000000021F020 Data:  0x00000000
// CHECK: (Write64): Address:  0x000000000021F030 Data:  0x00000000
// CHECK: (Write64): Address:  0x000000000021F040 Data:  0x00000000
// CHECK: (Write64): Address:  0x000000000021F050 Data:  0x00000000
// CHECK: (Write64): Address:  0x000000000021F060 Data:  0x00000000
// CHECK: (Write64): Address:  0x000000000021F070 Data:  0x00000000
// CHECK: (Write64): Address:  0x000000000021F080 Data:  0x00000000
// CHECK: (Write64): Address:  0x000000000021F090 Data:  0x00000000
// CHECK: (Write64): Address:  0x000000000021F0A0 Data:  0x00000000
// CHECK: (Write64): Address:  0x000000000021F0B0 Data:  0x00000000
// CHECK: (Write64): Address:  0x000000000021F0C0 Data:  0x00000000
// CHECK: (Write64): Address:  0x000000000021F0D0 Data:  0x00000000
// CHECK: (Write64): Address:  0x000000000021F0E0 Data:  0x00000000
// CHECK: (Write64): Address:  0x000000000021F0F0 Data:  0x00000000
// CHECK: (Write64): Address:  0x00000000021C0010 Data:  0x00000001
// CHECK: (Write64): Address:  0x00000000021C0000 Data:  0x00000000
// CHECK: (Write64): Address:  0x00000000001C0010 Data:  0x00000001
// CHECK: (Write64): Address:  0x00000000001C0000 Data:  0x00000000
// CHECK: (Write64): Address:  0x00000000041C0010 Data:  0x00000001
// CHECK: (Write64): Address:  0x00000000041C0000 Data:  0x00000000
// CHECK: (Write64): Address:  0x000000000021F050 Data:  0x00000001
// CHECK: (Write64): Address:  0x000000000021F040 Data:  0x00000000
// CHECK: (Write64): Address:  0x000000000021F030 Data:  0x00000001
// CHECK: (Write64): Address:  0x000000000021F020 Data:  0x00000000
// CHECK: (Write64): Address:  0x000000000021F010 Data:  0x00000001
// CHECK: (Write64): Address:  0x000000000021F000 Data:  0x00000000
// CHECK: (NOP Command): Payload Length: 0
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x000000000021D000  Size: 6
// CHECK:     Address: 0x000000000021D000  Data@ {{.+}} is: 0x00400800
// CHECK:     Address: 0x000000000021D004  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D008  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D00C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D010  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D014  Data@ {{.+}} is: 0x06045FE3

// CHECK: (NOP Command): Payload Length: 2
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x000000000021D020  Size: 6
// CHECK:     Address: 0x000000000021D020  Data@ {{.+}} is: 0x02400800
// CHECK:     Address: 0x000000000021D024  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D028  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D02C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D030  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D034  Data@ {{.+}} is: 0x0E049FE5

// CHECK: (NOP Command): Payload Length: 2
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x000000000021D040  Size: 6
// CHECK:     Address: 0x000000000021D040  Data@ {{.+}} is: 0x04401000
// CHECK:     Address: 0x000000000021D044  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D048  Data@ {{.+}} is: 0x001FE000
// CHECK:     Address: 0x000000000021D04C  Data@ {{.+}} is: 0x02008003
// CHECK:     Address: 0x000000000021D050  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D054  Data@ {{.+}} is: 0x16043FE0

// CHECK: (Write64): Address:  0x000000000021DE04 Data:  0x00010000
// CHECK: (MaskWrite64): Address: 0x000000000021DE00  Mask: 0x00000000  Data: 0x00000001
// CHECK: (Write64): Address:  0x000000000021DE0C Data:  0x00010001
// CHECK: (MaskWrite64): Address: 0x000000000021DE08  Mask: 0x00000000  Data: 0x00000001
// CHECK: (Write64): Address:  0x000000000021DE14 Data:  0x00010002
// CHECK: (MaskWrite64): Address: 0x000000000021DE10  Mask: 0x00000000  Data: 0x00000001
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x00000000041A0000  Size: 8
// CHECK:     Address: 0x00000000041A0000  Data@ {{.+}} is: 0x00001000
// CHECK:     Address: 0x00000000041A0004  Data@ {{.+}} is: 0x000A0000
// CHECK:     Address: 0x00000000041A0008  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000041A000C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000041A0010  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000041A0014  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000041A0018  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000041A001C  Data@ {{.+}} is: 0x8140FF41

// CHECK: (NOP Command): Payload Length: 0
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x00000000041A0020  Size: 8
// CHECK:     Address: 0x00000000041A0020  Data@ {{.+}} is: 0x00001000
// CHECK:     Address: 0x00000000041A0024  Data@ {{.+}} is: 0x001A0000
// CHECK:     Address: 0x00000000041A0028  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000041A002C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000041A0030  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000041A0034  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000041A0038  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000041A003C  Data@ {{.+}} is: 0x8141FF40

// CHECK: (Write64): Address:  0x00000000041A0604 Data:  0x00010000
// CHECK: (MaskWrite64): Address: 0x00000000041A0600  Mask: 0x00000000  Data: 0x00000001
// CHECK: (Write64): Address:  0x00000000041A0634 Data:  0x00010001
// CHECK: (MaskWrite64): Address: 0x00000000041A0630  Mask: 0x00000000  Data: 0x00000001
// CHECK: (NOP Command): Payload Length: 2
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x00000000001A0000  Size: 8
// CHECK:     Address: 0x00000000001A0000  Data@ {{.+}} is: 0x00000800
// CHECK:     Address: 0x00000000001A0004  Data@ {{.+}} is: 0x000A0000
// CHECK:     Address: 0x00000000001A0008  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000001A000C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000001A0010  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000001A0014  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000001A0018  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000001A001C  Data@ {{.+}} is: 0x8140FF41

// CHECK: (NOP Command): Payload Length: 0
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x00000000001A0020  Size: 8
// CHECK:     Address: 0x00000000001A0020  Data@ {{.+}} is: 0x00000800
// CHECK:     Address: 0x00000000001A0024  Data@ {{.+}} is: 0x001A0000
// CHECK:     Address: 0x00000000001A0028  Data@ {{.+}} is: 0x00080000
// CHECK:     Address: 0x00000000001A002C  Data@ {{.+}} is: 0x0080001F
// CHECK:     Address: 0x00000000001A0030  Data@ {{.+}} is: 0x00100003
// CHECK:     Address: 0x00000000001A0034  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000001A0038  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000001A003C  Data@ {{.+}} is: 0x8141FF40

// CHECK: (Write64): Address:  0x00000000001A0604 Data:  0x00010000
// CHECK: (MaskWrite64): Address: 0x00000000001A0600  Mask: 0x00000000  Data: 0x00000001
// CHECK: (Write64): Address:  0x00000000001A0634 Data:  0x00010001
// CHECK: (MaskWrite64): Address: 0x00000000001A0630  Mask: 0x00000000  Data: 0x00000001
// CHECK: (NOP Command): Payload Length: 2
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x00000000021A0000  Size: 8
// CHECK:     Address: 0x00000000021A0000  Data@ {{.+}} is: 0x00000800
// CHECK:     Address: 0x00000000021A0004  Data@ {{.+}} is: 0x000A0000
// CHECK:     Address: 0x00000000021A0008  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000021A000C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000021A0010  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000021A0014  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000021A0018  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000021A001C  Data@ {{.+}} is: 0x8140FF41

// CHECK: (NOP Command): Payload Length: 0
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x00000000021A0020  Size: 8
// CHECK:     Address: 0x00000000021A0020  Data@ {{.+}} is: 0x00000800
// CHECK:     Address: 0x00000000021A0024  Data@ {{.+}} is: 0x001A0000
// CHECK:     Address: 0x00000000021A0028  Data@ {{.+}} is: 0x00040000
// CHECK:     Address: 0x00000000021A002C  Data@ {{.+}} is: 0x0080001F
// CHECK:     Address: 0x00000000021A0030  Data@ {{.+}} is: 0x00200001
// CHECK:     Address: 0x00000000021A0034  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000021A0038  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000021A003C  Data@ {{.+}} is: 0x8141FF40

// CHECK: (Write64): Address:  0x00000000021A0604 Data:  0x00010000
// CHECK: (MaskWrite64): Address: 0x00000000021A0600  Mask: 0x00000000  Data: 0x00000001
// CHECK: (Write64): Address:  0x00000000021A0634 Data:  0x00010001
// CHECK: (MaskWrite64): Address: 0x00000000021A0630  Mask: 0x00000000  Data: 0x00000001
// CHECK: (Write64): Address:  0x000000000003F008 Data:  0x80000000
// CHECK: (Write64): Address:  0x000000000003F100 Data:  0x80000000
// CHECK: (Write64): Address:  0x000000000003F030 Data:  0x80000005
// CHECK: (Write64): Address:  0x000000000003F114 Data:  0x80000000
// CHECK: (Write64): Address:  0x000000000003F048 Data:  0x80000009
// CHECK: (Write64): Address:  0x000000000003F124 Data:  0x80000000
// CHECK: (Write64): Address:  0x000000000003F010 Data:  0x80000012
// CHECK: (Write64): Address:  0x000000000003F148 Data:  0x80000000
// CHECK: (Write64): Address:  0x00000000001B0000 Data:  0x80000007
// CHECK: (Write64): Address:  0x00000000001B011C Data:  0x80000000
// CHECK: (Write64): Address:  0x00000000001B002C Data:  0x80000000
// CHECK: (Write64): Address:  0x00000000001B0100 Data:  0x80000000
// CHECK: (Write64): Address:  0x000000000203F008 Data:  0x80000000
// CHECK: (Write64): Address:  0x000000000203F100 Data:  0x80000000
// CHECK: (Write64): Address:  0x000000000203F030 Data:  0x8000000A
// CHECK: (Write64): Address:  0x000000000203F128 Data:  0x80000000
// CHECK: (Write64): Address:  0x000000000203F020 Data:  0x80000012
// CHECK: (Write64): Address:  0x000000000203F148 Data:  0x80000000
// CHECK: (Write64): Address:  0x00000000021B0000 Data:  0x80000007
// CHECK: (Write64): Address:  0x00000000021B011C Data:  0x80000000
// CHECK: (Write64): Address:  0x00000000021B002C Data:  0x80000000
// CHECK: (Write64): Address:  0x00000000021B0100 Data:  0x80000000
// CHECK: (Write64): Address:  0x000000000403F008 Data:  0x80000000
// CHECK: (Write64): Address:  0x000000000403F100 Data:  0x80000000
// CHECK: (Write64): Address:  0x000000000403F020 Data:  0x8000000E
// CHECK: (Write64): Address:  0x000000000403F138 Data:  0x80000000
// CHECK: (Write64): Address:  0x00000000041B001C Data:  0x80000000
// CHECK: (Write64): Address:  0x00000000041B0100 Data:  0x80000000
// CHECK: (Write64): Address:  0x00000000041B0000 Data:  0x8000000D
// CHECK: (Write64): Address:  0x00000000041B0134 Data:  0x80000000
// CHECK: (Write64): Address:  0x000000000023F004 Data:  0x80000005
// CHECK: (Write64): Address:  0x000000000023F114 Data:  0x80000000
// CHECK: (Write64): Address:  0x000000000023F008 Data:  0x80000013
// CHECK: (Write64): Address:  0x000000000023F14C Data:  0x80000000
// CHECK: (Write64): Address:  0x000000000023F04C Data:  0x80000001
// CHECK: (Write64): Address:  0x000000000023F104 Data:  0x80000000
// CHECK: (Write64): Address:  0x000000000223F024 Data:  0x80000005
// CHECK: (Write64): Address:  0x000000000223F114 Data:  0x80000000
// CHECK: (Write64): Address:  0x000000000223F04C Data:  0x8000000B
// CHECK: (Write64): Address:  0x000000000223F12C Data:  0x80000000
// CHECK: (Write64): Address:  0x000000000423F014 Data:  0x8000000B
// CHECK: (Write64): Address:  0x000000000423F12C Data:  0x80000000
// CHECK: (MaskWrite64): Address: 0x000000000001F000  Mask: 0x00000C00  Data: 0x00000400
// CHECK: (MaskWrite64): Address: 0x000000000001F000  Mask: 0x0000C000  Data: 0x00004000
// CHECK: (MaskWrite64): Address: 0x000000000001F004  Mask: 0x00000030  Data: 0x00000010
// CHECK: Generating: {{.+}}/aie_cdo_enable.bin
// CHECK: (NOP Command): Payload Length: 0
// CHECK: (MaskWrite64): Address: 0x0000000000232000  Mask: 0x00000001  Data: 0x00000001

