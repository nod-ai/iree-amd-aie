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
  %buf5 = aie.buffer(%tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "buf5"} : memref<64x64xi8> 
  %buf4 = aie.buffer(%tile_1_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "buf4"} : memref<64x64xi8> 
  %buf3 = aie.buffer(%tile_2_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "buf3"} : memref<64x64xi32> 
  %buf2 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "buf2"} : memref<8x16x4x8xi8> 
  %buf1 = aie.buffer(%tile_0_2) {address = 5120 : i32, mem_bank = 0 : i32, sym_name = "buf1"} : memref<8x8x8x8xi8> 
  %buf0 = aie.buffer(%tile_0_2) {address = 9216 : i32, mem_bank = 0 : i32, sym_name = "buf0"} : memref<8x16x4x8xi32> 
  %mem_0_2 = aie.mem(%tile_0_2) {
    %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5, repeat_count = 1)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf2 : memref<8x16x4x8xi8>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 0 : i32}
    aie.use_lock(%lock_0_2_5, Release, 1)
    aie.next_bd ^bb1
  ^bb2:  // pred: ^bb3
    aie.end
  ^bb3:  // pred: ^bb5
    %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2, repeat_count = 1)
  ^bb4:  // 2 preds: ^bb3, ^bb4
    aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf1 : memref<8x8x8x8xi8>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 1 : i32}
    aie.use_lock(%lock_0_2_3, Release, 1)
    aie.next_bd ^bb4
  ^bb5:  // pred: ^bb0
    %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb3, repeat_count = 1)
  ^bb6:  // 2 preds: ^bb5, ^bb6
    aie.use_lock(%lock_0_2_7, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf0 : memref<8x16x4x8xi32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {bd_id = 2 : i32, next_bd_id = 2 : i32}
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
    %c0_i8 = arith.constant 0 : i8
    %c0_i32 = arith.constant 0 : i32
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
    %1 = arith.cmpi slt, %0, %c8 : index
    cf.cond_br %1, ^bb3(%c0 : index), ^bb10(%c0 : index)
  ^bb3(%2: index):  // 2 preds: ^bb2, ^bb8
    %3 = arith.cmpi slt, %2, %c16 : index
    cf.cond_br %3, ^bb4(%c0 : index), ^bb9
  ^bb4(%4: index):  // 2 preds: ^bb3, ^bb7
    %5 = arith.cmpi slt, %4, %c4 : index
    cf.cond_br %5, ^bb5(%c0 : index), ^bb8
  ^bb5(%6: index):  // 2 preds: ^bb4, ^bb6
    %7 = arith.cmpi slt, %6, %c8 : index
    cf.cond_br %7, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    memref.store %c0_i32, %buf0[%0, %2, %4, %6] : memref<8x16x4x8xi32>
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
    %15 = arith.cmpi slt, %14, %c8 : index
    cf.cond_br %15, ^bb12(%c0 : index), ^bb15
  ^bb12(%16: index):  // 2 preds: ^bb11, ^bb13
    %17 = arith.cmpi slt, %16, %c8 : index
    cf.cond_br %17, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    %18 = vector.transfer_read %buf2[%16, %12, %c0, %c0], %c0_i8 {in_bounds = [true, true, true, true]} : memref<8x16x4x8xi8>, vector<1x1x4x8xi8>
    %19 = vector.transfer_read %buf1[%14, %16, %c0, %c0], %c0_i8 {in_bounds = [true, true, true, true]} : memref<8x8x8x8xi8>, vector<1x1x8x8xi8>
    %20 = vector.transfer_read %buf0[%14, %12, %c0, %c0], %c0_i32 {in_bounds = [true, true, true, true]} : memref<8x16x4x8xi32>, vector<1x1x4x8xi32>
    %21 = arith.extsi %18 : vector<1x1x4x8xi8> to vector<1x1x4x8xi32>
    %22 = arith.extsi %19 : vector<1x1x8x8xi8> to vector<1x1x8x8xi32>
    %23 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %21, %22, %20 : vector<1x1x4x8xi32>, vector<1x1x8x8xi32> into vector<1x1x4x8xi32>
    vector.transfer_write %23, %buf0[%14, %12, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x4x8xi32>, memref<8x16x4x8xi32>
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
  } {elf_file = "matmul_64x64_64xi8__dispatch_0_matmul_64x64x64_i8xi8xi32_0_core_0_2.elf"}
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
    aie.dma_bd(%buf3 : memref<64x64xi32>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 0 : i32}
    aie.use_lock(%lock_2_1_2, Release, 1)
    aie.next_bd ^bb1
  ^bb2:  // pred: ^bb3
    aie.end
  ^bb3:  // pred: ^bb0
    %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb2, repeat_count = 1)
  ^bb4:  // 2 preds: ^bb3, ^bb4
    aie.use_lock(%lock_2_1_2, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf3 : memref<64x64xi32>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 1 : i32}
    aie.use_lock(%lock_2_1, Release, 1)
    aie.next_bd ^bb4
  }
  %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
    %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf5 : memref<64x64xi8>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 0 : i32}
    aie.use_lock(%lock_0_1_1, Release, 1)
    aie.next_bd ^bb1
  ^bb2:  // pred: ^bb3
    aie.end
  ^bb3:  // pred: ^bb0
    %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb2, repeat_count = 1)
  ^bb4:  // 2 preds: ^bb3, ^bb4
    aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf5 : memref<64x64xi8>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 1 : i32}
    aie.use_lock(%lock_0_1, Release, 1)
    aie.next_bd ^bb4
  }
  %memtile_dma_1_1 = aie.memtile_dma(%tile_1_1) {
    %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf4 : memref<64x64xi8>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 0 : i32}
    aie.use_lock(%lock_1_1_0, Release, 1)
    aie.next_bd ^bb1
  ^bb2:  // pred: ^bb3
    aie.end
  ^bb3:  // pred: ^bb0
    %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb2, repeat_count = 1)
  ^bb4:  // 2 preds: ^bb3, ^bb4
    aie.use_lock(%lock_1_1_0, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf4 : memref<64x64xi8>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 1 : i32}
    aie.use_lock(%lock_1_1, Release, 1)
    aie.next_bd ^bb4
  }
  aie.shim_dma_allocation @airMemcpyId12(S2MM, 0, 0)
  memref.global "public" @airMemcpyId12 : memref<64x64xi32>
  aie.shim_dma_allocation @airMemcpyId4(MM2S, 0, 0)
  memref.global "public" @airMemcpyId4 : memref<64x64xi8>
  aie.shim_dma_allocation @airMemcpyId5(MM2S, 1, 0)
  memref.global "public" @airMemcpyId5 : memref<64x64xi8>
  func.func @matmul_64x64_64xi8__dispatch_0_matmul_64x64x64_i8xi8xi32(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>, %arg2: memref<64x64xi32>) {
    memref.assume_alignment %arg0, 64 : memref<1024xi32>
    memref.assume_alignment %arg1, 64 : memref<1024xi32>
    memref.assume_alignment %arg2, 64 : memref<64x64xi32>
    aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 64, 16][0, 0, 16]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<1024xi32>
    aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 64, 16][0, 0, 16]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<1024xi32>
    aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 64, 64][0, 0, 64]) {id = 2 : i64, metadata = @airMemcpyId12} : memref<64x64xi32>
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
} {sym_name = "matmul_64x64_64xi8__dispatch_0_matmul_64x64x64_i8xi8xi32_0"}
}

// CHECK: trying XAIE API: XAie_SetupPartitionConfig with args: &devInst={{.+}}, 0x0=0, partitionStartCol=1, partitionNumCols=4
// CHECK: trying XAIE API: XAie_CfgInitialize with args: &devInst={{.+}}, &configPtr=
// CHECK: trying XAIE API: XAie_SetIOBackend with args: &devInst={{.+}}, XAIE_IO_BACKEND_CDO=2
// CHECK: trying XAIE API: XAie_UpdateNpiAddr with args: &devInst={{.+}}, 0x0=0
// CHECK: trying XAIE API: XAie_LoadElf with args: &devInst={{.+}}, XAie_TileLoc(col=XAie_LocType(col: 0, row: 2), row)={{.+}}/matmul_64x64_64xi8__dispatch_0_matmul_64x64x64_i8xi8xi32_0_core_0_2.elf, elfPath.str().c_str(), aieSim=0
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
// CHECK: trying XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd={{.+}}, basePlusOffsetInBytes=1024, lenInBytes=4096
// CHECK: trying XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd={{.+}}, nextBdId.value()=0, enableNextBd=1
// CHECK: trying XAIE API: XAie_DmaEnableBd with args: &dmaTileBd={{.+}}
// CHECK: trying XAIE API: XAie_DmaWriteBd with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), bdId=0

// CHECK: end configuring bds
// CHECK: trying XAIE API: XAie_DmaDescInit with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2)

// CHECK: start configuring bds
// CHECK: trying XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd={{.+}}, acqLock=XAie_Lock(id: 5, val: -1), relLock=XAie_Lock(id: 4, val: 1), acqEn=1, relEn=0
// CHECK: trying XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd={{.+}}, basePlusOffsetInBytes=5120, lenInBytes=4096
// CHECK: trying XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd={{.+}}, nextBdId.value()=1, enableNextBd=1
// CHECK: trying XAIE API: XAie_DmaEnableBd with args: &dmaTileBd={{.+}}
// CHECK: trying XAIE API: XAie_DmaWriteBd with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), bdId=1

// CHECK: end configuring bds
// CHECK: trying XAIE API: XAie_DmaDescInit with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2)

// CHECK: start configuring bds
// CHECK: trying XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd={{.+}}, acqLock=XAie_Lock(id: 0, val: -1), relLock=XAie_Lock(id: 1, val: 1), acqEn=1, relEn=0
// CHECK: trying XAIE API: XAie_DmaSetMultiDimAddr with args: &dmaTileBd={{.+}}, &dmaTileBdTensor={{.+}}, basePlusOffsetInBytes=9216, lenInBytes=16384
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
// CHECK: trying XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd={{.+}}, basePlusOffsetInBytes=524288, lenInBytes=4096
// CHECK: trying XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd={{.+}}, nextBdId.value()=0, enableNextBd=1
// CHECK: trying XAIE API: XAie_DmaEnableBd with args: &dmaTileBd={{.+}}
// CHECK: trying XAIE API: XAie_DmaWriteBd with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 0, row: 1), bdId=0

// CHECK: end configuring bds
// CHECK: trying XAIE API: XAie_DmaDescInit with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 0, row: 1)

// CHECK: start configuring bds
// CHECK: trying XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd={{.+}}, acqLock=XAie_Lock(id: 64, val: -1), relLock=XAie_Lock(id: 65, val: 1), acqEn=1, relEn=0
// CHECK: trying XAIE API: XAie_DmaSetMultiDimAddr with args: &dmaTileBd={{.+}}, &dmaTileBdTensor={{.+}}, basePlusOffsetInBytes=524288, lenInBytes=4096
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
// CHECK: trying XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd={{.+}}, basePlusOffsetInBytes=524288, lenInBytes=4096
// CHECK: trying XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd={{.+}}, nextBdId.value()=0, enableNextBd=1
// CHECK: trying XAIE API: XAie_DmaEnableBd with args: &dmaTileBd={{.+}}
// CHECK: trying XAIE API: XAie_DmaWriteBd with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 1, row: 1), bdId=0

// CHECK: end configuring bds
// CHECK: trying XAIE API: XAie_DmaDescInit with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 1, row: 1)

// CHECK: start configuring bds
// CHECK: trying XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd={{.+}}, acqLock=XAie_Lock(id: 64, val: -1), relLock=XAie_Lock(id: 65, val: 1), acqEn=1, relEn=0
// CHECK: trying XAIE API: XAie_DmaSetMultiDimAddr with args: &dmaTileBd={{.+}}, &dmaTileBdTensor={{.+}}, basePlusOffsetInBytes=524288, lenInBytes=4096
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

// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x0000000000220200  Size: 412
// CHECK:     Address: 0x0000000000220200  Data@ {{.+}} is: 0x4802011D
// CHECK:     Address: 0x0000000000220204  Data@ {{.+}} is: 0x251D0043
// CHECK:     Address: 0x0000000000220208  Data@ {{.+}} is: 0x00A3C81C
// CHECK:     Address: 0x000000000022020C  Data@ {{.+}} is: 0x4830111D
// CHECK:     Address: 0x0000000000220210  Data@ {{.+}} is: 0x311D0106
// CHECK:     Address: 0x0000000000220214  Data@ {{.+}} is: 0x02070834
// CHECK:     Address: 0x0000000000220218  Data@ {{.+}} is: 0xC83A511D
// CHECK:     Address: 0x000000000022021C  Data@ {{.+}} is: 0xA11D0480
// CHECK:     Address: 0x0000000000220220  Data@ {{.+}} is: 0x05814808
// CHECK:     Address: 0x0000000000220224  Data@ {{.+}} is: 0xC80CC11D
// CHECK:     Address: 0x0000000000220228  Data@ {{.+}} is: 0xE11D0681
// CHECK:     Address: 0x000000000022022C  Data@ {{.+}} is: 0x07878810
// CHECK:     Address: 0x0000000000220230  Data@ {{.+}} is: 0x20680055
// CHECK:     Address: 0x0000000000220234  Data@ {{.+}} is: 0x103B0007
// CHECK:     Address: 0x0000000000220238  Data@ {{.+}} is: 0x01C09A00
// CHECK:     Address: 0x000000000022023C  Data@ {{.+}} is: 0x00034000
// CHECK:     Address: 0x0000000000220240  Data@ {{.+}} is: 0x1A00113B
// CHECK:     Address: 0x0000000000220244  Data@ {{.+}} is: 0x000001C5
// CHECK:     Address: 0x0000000000220248  Data@ {{.+}} is: 0x113BFF7C
// CHECK:     Address: 0x000000000022024C  Data@ {{.+}} is: 0x00021200
// CHECK:     Address: 0x0000000000220250  Data@ {{.+}} is: 0xFF5C8000
// CHECK:     Address: 0x0000000000220254  Data@ {{.+}} is: 0x3300113B
// CHECK:     Address: 0x0000000000220258  Data@ {{.+}} is: 0x40000002
// CHECK:     Address: 0x000000000022025C  Data@ {{.+}} is: 0x113BFF6C
// CHECK:     Address: 0x0000000000220260  Data@ {{.+}} is: 0x00025400
// CHECK:     Address: 0x0000000000220264  Data@ {{.+}} is: 0xFF4CC000
// CHECK:     Address: 0x0000000000220268  Data@ {{.+}} is: 0x7500113B
// CHECK:     Address: 0x000000000022026C  Data@ {{.+}} is: 0x00000002
// CHECK:     Address: 0x0000000000220270  Data@ {{.+}} is: 0x113BFF3D
// CHECK:     Address: 0x0000000000220274  Data@ {{.+}} is: 0x00029600
// CHECK:     Address: 0x0000000000220278  Data@ {{.+}} is: 0xFF2D4000
// CHECK:     Address: 0x000000000022027C  Data@ {{.+}} is: 0xB700113B
// CHECK:     Address: 0x0000000000220280  Data@ {{.+}} is: 0x80000002
// CHECK:     Address: 0x0000000000220284  Data@ {{.+}} is: 0x617BFF1D
// CHECK:     Address: 0x0000000000220288  Data@ {{.+}} is: 0x6E000020
// CHECK:     Address: 0x000000000022028C  Data@ {{.+}} is: 0x03848FF8
// CHECK:     Address: 0x0000000000220290  Data@ {{.+}} is: 0x000003C0
// CHECK:     Address: 0x0000000000220294  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220298  Data@ {{.+}} is: 0x08000000
// CHECK:     Address: 0x000000000022029C  Data@ {{.+}} is: 0x0000FFE0
// CHECK:     Address: 0x00000000002202A0  Data@ {{.+}} is: 0x0622021D
// CHECK:     Address: 0x00000000002202A4  Data@ {{.+}} is: 0x00010000
// CHECK:     Address: 0x00000000002202A8  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002202AC  Data@ {{.+}} is: 0x16420219
// CHECK:     Address: 0x00000000002202B0  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002202B4  Data@ {{.+}} is: 0x02190001
// CHECK:     Address: 0x00000000002202B8  Data@ {{.+}} is: 0x011D1682
// CHECK:     Address: 0x00000000002202BC  Data@ {{.+}} is: 0x0006C816
// CHECK:     Address: 0x00000000002202C0  Data@ {{.+}} is: 0x000003C0
// CHECK:     Address: 0x00000000002202C4  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002202C8  Data@ {{.+}} is: 0x07FEF600
// CHECK:     Address: 0x00000000002202CC  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002202D0  Data@ {{.+}} is: 0xC2C0EDBD
// CHECK:     Address: 0x00000000002202D4  Data@ {{.+}} is: 0x017BFFCA
// CHECK:     Address: 0x00000000002202D8  Data@ {{.+}} is: 0xC000003E
// CHECK:     Address: 0x00000000002202DC  Data@ {{.+}} is: 0x0006CFFD
// CHECK:     Address: 0x00000000002202E0  Data@ {{.+}} is: 0x00000013
// CHECK:     Address: 0x00000000002202E4  Data@ {{.+}} is: 0xFFB88800
// CHECK:     Address: 0x00000000002202E8  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002202EC  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002202F0  Data@ {{.+}} is: 0xFD990001
// CHECK:     Address: 0x00000000002202F4  Data@ {{.+}} is: 0x209917C0
// CHECK:     Address: 0x00000000002202F8  Data@ {{.+}} is: 0xDD991000
// CHECK:     Address: 0x00000000002202FC  Data@ {{.+}} is: 0x85991022
// CHECK:     Address: 0x0000000000220300  Data@ {{.+}} is: 0x95BD1441
// CHECK:     Address: 0x0000000000220304  Data@ {{.+}} is: 0xFF880441
// CHECK:     Address: 0x0000000000220308  Data@ {{.+}} is: 0x0441A5FB
// CHECK:     Address: 0x000000000022030C  Data@ {{.+}} is: 0xAFFCC000
// CHECK:     Address: 0x0000000000220310  Data@ {{.+}} is: 0x4035FF8D
// CHECK:     Address: 0x0000000000220314  Data@ {{.+}} is: 0xFF9DAFFD
// CHECK:     Address: 0x0000000000220318  Data@ {{.+}} is: 0x07FD6D59
// CHECK:     Address: 0x000000000022031C  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220320  Data@ {{.+}} is: 0x1DA89659
// CHECK:     Address: 0x0000000000220324  Data@ {{.+}} is: 0x08D08219
// CHECK:     Address: 0x0000000000220328  Data@ {{.+}} is: 0x08D08219
// CHECK:     Address: 0x000000000022032C  Data@ {{.+}} is: 0x08D08219
// CHECK:     Address: 0x0000000000220330  Data@ {{.+}} is: 0x4469C5BD
// CHECK:     Address: 0x0000000000220334  Data@ {{.+}} is: 0x28431A10
// CHECK:     Address: 0x0000000000220338  Data@ {{.+}} is: 0xA22ED50B
// CHECK:     Address: 0x000000000022033C  Data@ {{.+}} is: 0xD5BD0022
// CHECK:     Address: 0x0000000000220340  Data@ {{.+}} is: 0x1A10446B
// CHECK:     Address: 0x0000000000220344  Data@ {{.+}} is: 0xD54B2843
// CHECK:     Address: 0x0000000000220348  Data@ {{.+}} is: 0x0022B2AE
// CHECK:     Address: 0x000000000022034C  Data@ {{.+}} is: 0x446D05BD
// CHECK:     Address: 0x0000000000220350  Data@ {{.+}} is: 0x28431A10
// CHECK:     Address: 0x0000000000220354  Data@ {{.+}} is: 0xC32ED58B
// CHECK:     Address: 0x0000000000220358  Data@ {{.+}} is: 0x25BD0022
// CHECK:     Address: 0x000000000022035C  Data@ {{.+}} is: 0x1A10446F
// CHECK:     Address: 0x0000000000220360  Data@ {{.+}} is: 0xD5CB2843
// CHECK:     Address: 0x0000000000220364  Data@ {{.+}} is: 0x0000040A
// CHECK:     Address: 0x0000000000220368  Data@ {{.+}} is: 0x446605BD
// CHECK:     Address: 0x000000000022036C  Data@ {{.+}} is: 0x28431A10
// CHECK:     Address: 0x0000000000220370  Data@ {{.+}} is: 0xD3AED4CB
// CHECK:     Address: 0x0000000000220374  Data@ {{.+}} is: 0x35BD0022
// CHECK:     Address: 0x0000000000220378  Data@ {{.+}} is: 0x1A104464
// CHECK:     Address: 0x000000000022037C  Data@ {{.+}} is: 0xD48B2843
// CHECK:     Address: 0x0000000000220380  Data@ {{.+}} is: 0x0022E42E
// CHECK:     Address: 0x0000000000220384  Data@ {{.+}} is: 0x445FE5BD
// CHECK:     Address: 0x0000000000220388  Data@ {{.+}} is: 0x28431A10
// CHECK:     Address: 0x000000000022038C  Data@ {{.+}} is: 0x460AD28B
// CHECK:     Address: 0x0000000000220390  Data@ {{.+}} is: 0x45BD0002
// CHECK:     Address: 0x0000000000220394  Data@ {{.+}} is: 0x1A104478
// CHECK:     Address: 0x0000000000220398  Data@ {{.+}} is: 0xD2CB2843
// CHECK:     Address: 0x000000000022039C  Data@ {{.+}} is: 0x0002468A
// CHECK:     Address: 0x00000000002203A0  Data@ {{.+}} is: 0x044845FB
// CHECK:     Address: 0x00000000002203A4  Data@ {{.+}} is: 0x88D08200
// CHECK:     Address: 0x00000000002203A8  Data@ {{.+}} is: 0x28430F06
// CHECK:     Address: 0x00000000002203AC  Data@ {{.+}} is: 0x0D2ED30B
// CHECK:     Address: 0x00000000002203B0  Data@ {{.+}} is: 0xF17B0023
// CHECK:     Address: 0x00000000002203B4  Data@ {{.+}} is: 0x82000074
// CHECK:     Address: 0x00000000002203B8  Data@ {{.+}} is: 0x080008D0
// CHECK:     Address: 0x00000000002203BC  Data@ {{.+}} is: 0xD34B2843
// CHECK:     Address: 0x00000000002203C0  Data@ {{.+}} is: 0x0023AD2E
// CHECK:     Address: 0x00000000002203C4  Data@ {{.+}} is: 0x445205BD
// CHECK:     Address: 0x00000000002203C8  Data@ {{.+}} is: 0x28431A10
// CHECK:     Address: 0x00000000002203CC  Data@ {{.+}} is: 0x008AD38B
// CHECK:     Address: 0x00000000002203D0  Data@ {{.+}} is: 0x05BD0002
// CHECK:     Address: 0x00000000002203D4  Data@ {{.+}} is: 0x1A104450
// CHECK:     Address: 0x00000000002203D8  Data@ {{.+}} is: 0xD3CB2843
// CHECK:     Address: 0x00000000002203DC  Data@ {{.+}} is: 0x0002010A
// CHECK:     Address: 0x00000000002203E0  Data@ {{.+}} is: 0x444E05BD
// CHECK:     Address: 0x00000000002203E4  Data@ {{.+}} is: 0x28431A10
// CHECK:     Address: 0x00000000002203E8  Data@ {{.+}} is: 0x018AD24B
// CHECK:     Address: 0x00000000002203EC  Data@ {{.+}} is: 0x05BD0002
// CHECK:     Address: 0x00000000002203F0  Data@ {{.+}} is: 0x1A104470
// CHECK:     Address: 0x00000000002203F4  Data@ {{.+}} is: 0xD20B2843
// CHECK:     Address: 0x00000000002203F8  Data@ {{.+}} is: 0x0002020A
// CHECK:     Address: 0x00000000002203FC  Data@ {{.+}} is: 0x447205BD
// CHECK:     Address: 0x0000000000220400  Data@ {{.+}} is: 0x28431A10
// CHECK:     Address: 0x0000000000220404  Data@ {{.+}} is: 0x028AD1CB
// CHECK:     Address: 0x0000000000220408  Data@ {{.+}} is: 0x05BD0002
// CHECK:     Address: 0x000000000022040C  Data@ {{.+}} is: 0x1A10447A
// CHECK:     Address: 0x0000000000220410  Data@ {{.+}} is: 0xD60B2843
// CHECK:     Address: 0x0000000000220414  Data@ {{.+}} is: 0x0002030A
// CHECK:     Address: 0x0000000000220418  Data@ {{.+}} is: 0x444C05BD
// CHECK:     Address: 0x000000000022041C  Data@ {{.+}} is: 0x28431A10
// CHECK:     Address: 0x0000000000220420  Data@ {{.+}} is: 0x038AD64B
// CHECK:     Address: 0x0000000000220424  Data@ {{.+}} is: 0x05BD0002
// CHECK:     Address: 0x0000000000220428  Data@ {{.+}} is: 0x1A10447C
// CHECK:     Address: 0x000000000022042C  Data@ {{.+}} is: 0xD74B2843
// CHECK:     Address: 0x0000000000220430  Data@ {{.+}} is: 0x0002040A
// CHECK:     Address: 0x0000000000220434  Data@ {{.+}} is: 0x444A05BD
// CHECK:     Address: 0x0000000000220438  Data@ {{.+}} is: 0x28431A10
// CHECK:     Address: 0x000000000022043C  Data@ {{.+}} is: 0x048AD18B
// CHECK:     Address: 0x0000000000220440  Data@ {{.+}} is: 0x05BD0002
// CHECK:     Address: 0x0000000000220444  Data@ {{.+}} is: 0x1A104446
// CHECK:     Address: 0x0000000000220448  Data@ {{.+}} is: 0xD78B2843
// CHECK:     Address: 0x000000000022044C  Data@ {{.+}} is: 0x0002050A
// CHECK:     Address: 0x0000000000220450  Data@ {{.+}} is: 0x444405BD
// CHECK:     Address: 0x0000000000220454  Data@ {{.+}} is: 0x28431A10
// CHECK:     Address: 0x0000000000220458  Data@ {{.+}} is: 0x058AD14B
// CHECK:     Address: 0x000000000022045C  Data@ {{.+}} is: 0x05FB0002
// CHECK:     Address: 0x0000000000220460  Data@ {{.+}} is: 0x82000440
// CHECK:     Address: 0x0000000000220464  Data@ {{.+}} is: 0x038488D0
// CHECK:     Address: 0x0000000000220468  Data@ {{.+}} is: 0xD0CB2843
// CHECK:     Address: 0x000000000022046C  Data@ {{.+}} is: 0x0000A1EA
// CHECK:     Address: 0x0000000000220470  Data@ {{.+}} is: 0x001A097B
// CHECK:     Address: 0x0000000000220474  Data@ {{.+}} is: 0xC8D08200
// CHECK:     Address: 0x0000000000220478  Data@ {{.+}} is: 0x284300A3
// CHECK:     Address: 0x000000000022047C  Data@ {{.+}} is: 0x902AD08B
// CHECK:     Address: 0x0000000000220480  Data@ {{.+}} is: 0xE17B0000
// CHECK:     Address: 0x0000000000220484  Data@ {{.+}} is: 0x82000010
// CHECK:     Address: 0x0000000000220488  Data@ {{.+}} is: 0x0681C8D0
// CHECK:     Address: 0x000000000022048C  Data@ {{.+}} is: 0xD00B2843
// CHECK:     Address: 0x0000000000220490  Data@ {{.+}} is: 0x0001808A
// CHECK:     Address: 0x0000000000220494  Data@ {{.+}} is: 0x0032217B
// CHECK:     Address: 0x0000000000220498  Data@ {{.+}} is: 0x48D08200
// CHECK:     Address: 0x000000000022049C  Data@ {{.+}} is: 0x28430287
// CHECK:     Address: 0x00000000002204A0  Data@ {{.+}} is: 0x660AD70B
// CHECK:     Address: 0x00000000002204A4  Data@ {{.+}} is: 0xACFB0000
// CHECK:     Address: 0x00000000002204A8  Data@ {{.+}} is: 0x820007C0
// CHECK:     Address: 0x00000000002204AC  Data@ {{.+}} is: 0x078788D0
// CHECK:     Address: 0x00000000002204B0  Data@ {{.+}} is: 0xD10B2843
// CHECK:     Address: 0x00000000002204B4  Data@ {{.+}} is: 0x000000B2
// CHECK:     Address: 0x00000000002204B8  Data@ {{.+}} is: 0x100060BB
// CHECK:     Address: 0x00000000002204BC  Data@ {{.+}} is: 0x4800005C
// CHECK:     Address: 0x00000000002204C0  Data@ {{.+}} is: 0x077B0581
// CHECK:     Address: 0x00000000002204C4  Data@ {{.+}} is: 0x820007FE
// CHECK:     Address: 0x00000000002204C8  Data@ {{.+}} is: 0x0480C8D0
// CHECK:     Address: 0x00000000002204CC  Data@ {{.+}} is: 0xD40B2843
// CHECK:     Address: 0x00000000002204D0  Data@ {{.+}} is: 0x0001C20A
// CHECK:     Address: 0x00000000002204D4  Data@ {{.+}} is: 0x07C49CFB
// CHECK:     Address: 0x00000000002204D8  Data@ {{.+}} is: 0x08D08200
// CHECK:     Address: 0x00000000002204DC  Data@ {{.+}} is: 0x28430501
// CHECK:     Address: 0x00000000002204E0  Data@ {{.+}} is: 0xB106D68B
// CHECK:     Address: 0x00000000002204E4  Data@ {{.+}} is: 0x617B0037
// CHECK:     Address: 0x00000000002204E8  Data@ {{.+}} is: 0x82000020
// CHECK:     Address: 0x00000000002204EC  Data@ {{.+}} is: 0x018688D0
// CHECK:     Address: 0x00000000002204F0  Data@ {{.+}} is: 0x000003C0
// CHECK:     Address: 0x00000000002204F4  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002204F8  Data@ {{.+}} is: 0xC8000000
// CHECK:     Address: 0x00000000002204FC  Data@ {{.+}} is: 0x0000FFCA
// CHECK:     Address: 0x0000000000220500  Data@ {{.+}} is: 0x07FEF659
// CHECK:     Address: 0x0000000000220504  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220508  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x000000000022050C  Data@ {{.+}} is: 0x10001D19
// CHECK:     Address: 0x0000000000220510  Data@ {{.+}} is: 0x12C00C99
// CHECK:     Address: 0x0000000000220514  Data@ {{.+}} is: 0x10001619
// CHECK:     Address: 0x0000000000220518  Data@ {{.+}} is: 0x60400195
// CHECK:     Address: 0x000000000022051C  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220520  Data@ {{.+}} is: 0x07190001
// CHECK:     Address: 0x0000000000220524  Data@ {{.+}} is: 0x9C9912D6
// CHECK:     Address: 0x0000000000220528  Data@ {{.+}} is: 0x209D12C4
// CHECK:     Address: 0x000000000022052C  Data@ {{.+}} is: 0x01238EF6
// CHECK:     Address: 0x0000000000220530  Data@ {{.+}} is: 0x00000013
// CHECK:     Address: 0x0000000000220534  Data@ {{.+}} is: 0x0000C800
// CHECK:     Address: 0x0000000000220538  Data@ {{.+}} is: 0x0836011D
// CHECK:     Address: 0x000000000022053C  Data@ {{.+}} is: 0x011D6103
// CHECK:     Address: 0x0000000000220540  Data@ {{.+}} is: 0x10050848
// CHECK:     Address: 0x0000000000220544  Data@ {{.+}} is: 0x88EA011D
// CHECK:     Address: 0x0000000000220548  Data@ {{.+}} is: 0x011D2005
// CHECK:     Address: 0x000000000022054C  Data@ {{.+}} is: 0x3004496E
// CHECK:     Address: 0x0000000000220550  Data@ {{.+}} is: 0xFF88C07F
// CHECK:     Address: 0x0000000000220554  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220558  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022055C  Data@ {{.+}} is: 0x28430000
// CHECK:     Address: 0x0000000000220560  Data@ {{.+}} is: 0x37ED993B
// CHECK:     Address: 0x0000000000220564  Data@ {{.+}} is: 0x96590006
// CHECK:     Address: 0x0000000000220568  Data@ {{.+}} is: 0x28431821
// CHECK:     Address: 0x000000000022056C  Data@ {{.+}} is: 0xE00AD0CB
// CHECK:     Address: 0x0000000000220570  Data@ {{.+}} is: 0xC07F0001
// CHECK:     Address: 0x0000000000220574  Data@ {{.+}} is: 0x0069FF9E
// CHECK:     Address: 0x0000000000220578  Data@ {{.+}} is: 0x09D6016C
// CHECK:     Address: 0x000000000022057C  Data@ {{.+}} is: 0x3A103B0B
// CHECK:     Address: 0x0000000000220580  Data@ {{.+}} is: 0x07BEED9D
// CHECK:     Address: 0x0000000000220584  Data@ {{.+}} is: 0x28430000
// CHECK:     Address: 0x0000000000220588  Data@ {{.+}} is: 0x0186183B
// CHECK:     Address: 0x000000000022058C  Data@ {{.+}} is: 0x2843003E
// CHECK:     Address: 0x0000000000220590  Data@ {{.+}} is: 0x26EE9A3B
// CHECK:     Address: 0x0000000000220594  Data@ {{.+}} is: 0x16590000
// CHECK:     Address: 0x0000000000220598  Data@ {{.+}} is: 0x96451821
// CHECK:     Address: 0x000000000022059C  Data@ {{.+}} is: 0x040B082F
// CHECK:     Address: 0x00000000002205A0  Data@ {{.+}} is: 0x0FC04585
// CHECK:     Address: 0x00000000002205A4  Data@ {{.+}} is: 0x164D150B
// CHECK:     Address: 0x00000000002205A8  Data@ {{.+}} is: 0xA0D47820
// CHECK:     Address: 0x00000000002205AC  Data@ {{.+}} is: 0x3D34764D
// CHECK:     Address: 0x00000000002205B0  Data@ {{.+}} is: 0x4593A054
// CHECK:     Address: 0x00000000002205B4  Data@ {{.+}} is: 0x3D0B0FCF
// CHECK:     Address: 0x00000000002205B8  Data@ {{.+}} is: 0x964D6852
// CHECK:     Address: 0x00000000002205BC  Data@ {{.+}} is: 0xA0D1F823
// CHECK:     Address: 0x00000000002205C0  Data@ {{.+}} is: 0xBD34764D
// CHECK:     Address: 0x00000000002205C4  Data@ {{.+}} is: 0x5593A051
// CHECK:     Address: 0x00000000002205C8  Data@ {{.+}} is: 0x750B0FC5
// CHECK:     Address: 0x00000000002205CC  Data@ {{.+}} is: 0x164D81D0
// CHECK:     Address: 0x00000000002205D0  Data@ {{.+}} is: 0xA0D2F821
// CHECK:     Address: 0x00000000002205D4  Data@ {{.+}} is: 0xBD34764D
// CHECK:     Address: 0x00000000002205D8  Data@ {{.+}} is: 0x6593A052
// CHECK:     Address: 0x00000000002205DC  Data@ {{.+}} is: 0x550B0FC9
// CHECK:     Address: 0x00000000002205E0  Data@ {{.+}} is: 0x164D8150
// CHECK:     Address: 0x00000000002205E4  Data@ {{.+}} is: 0xA0D0F822
// CHECK:     Address: 0x00000000002205E8  Data@ {{.+}} is: 0xBD34764D
// CHECK:     Address: 0x00000000002205EC  Data@ {{.+}} is: 0x7593A050
// CHECK:     Address: 0x00000000002205F0  Data@ {{.+}} is: 0x350B0FCB
// CHECK:     Address: 0x00000000002205F4  Data@ {{.+}} is: 0x964D80D0
// CHECK:     Address: 0x00000000002205F8  Data@ {{.+}} is: 0xA0D3F822
// CHECK:     Address: 0x00000000002205FC  Data@ {{.+}} is: 0xBD34764D
// CHECK:     Address: 0x0000000000220600  Data@ {{.+}} is: 0x1593A053
// CHECK:     Address: 0x0000000000220604  Data@ {{.+}} is: 0x150B0FCD
// CHECK:     Address: 0x0000000000220608  Data@ {{.+}} is: 0x164D8050
// CHECK:     Address: 0x000000000022060C  Data@ {{.+}} is: 0xA0D47823
// CHECK:     Address: 0x0000000000220610  Data@ {{.+}} is: 0x3D34764D
// CHECK:     Address: 0x0000000000220614  Data@ {{.+}} is: 0x087DA054
// CHECK:     Address: 0x0000000000220618  Data@ {{.+}} is: 0x62133D0B
// CHECK:     Address: 0x000000000022061C  Data@ {{.+}} is: 0x05069BD9
// CHECK:     Address: 0x0000000000220620  Data@ {{.+}} is: 0x89D90001
// CHECK:     Address: 0x0000000000220624  Data@ {{.+}} is: 0x00090330
// CHECK:     Address: 0x0000000000220628  Data@ {{.+}} is: 0xA9D96002
// CHECK:     Address: 0x000000000022062C  Data@ {{.+}} is: 0xC02B0350
// CHECK:     Address: 0x0000000000220630  Data@ {{.+}} is: 0x85AC6010
// CHECK:     Address: 0x0000000000220634  Data@ {{.+}} is: 0x6E14B83E
// CHECK:     Address: 0x0000000000220638  Data@ {{.+}} is: 0x3824164D
// CHECK:     Address: 0x000000000022063C  Data@ {{.+}} is: 0x400B7212
// CHECK:     Address: 0x0000000000220640  Data@ {{.+}} is: 0x76426019
// CHECK:     Address: 0x0000000000220644  Data@ {{.+}} is: 0xA0533D34
// CHECK:     Address: 0x0000000000220648  Data@ {{.+}} is: 0xBD0B087D
// CHECK:     Address: 0x000000000022064C  Data@ {{.+}} is: 0x40237611
// CHECK:     Address: 0x0000000000220650  Data@ {{.+}} is: 0x78046008
// CHECK:     Address: 0x0000000000220654  Data@ {{.+}} is: 0x89D9A0D1
// CHECK:     Address: 0x0000000000220658  Data@ {{.+}} is: 0xC0090502
// CHECK:     Address: 0x000000000022065C  Data@ {{.+}} is: 0x00016029
// CHECK:     Address: 0x0000000000220660  Data@ {{.+}} is: 0x60260009
// CHECK:     Address: 0x0000000000220664  Data@ {{.+}} is: 0x80090001
// CHECK:     Address: 0x0000000000220668  Data@ {{.+}} is: 0x00016011
// CHECK:     Address: 0x000000000022066C  Data@ {{.+}} is: 0x600C8009
// CHECK:     Address: 0x0000000000220670  Data@ {{.+}} is: 0x95990001
// CHECK:     Address: 0x0000000000220674  Data@ {{.+}} is: 0xED9917BE
// CHECK:     Address: 0x0000000000220678  Data@ {{.+}} is: 0x309917E2
// CHECK:     Address: 0x000000000022067C  Data@ {{.+}} is: 0xDDBD1440
// CHECK:     Address: 0x0000000000220680  Data@ {{.+}} is: 0x81D06808
// CHECK:     Address: 0x0000000000220684  Data@ {{.+}} is: 0x110B2003
// CHECK:     Address: 0x0000000000220688  Data@ {{.+}} is: 0x81504800
// CHECK:     Address: 0x000000000022068C  Data@ {{.+}} is: 0x9A3B2003
// CHECK:     Address: 0x0000000000220690  Data@ {{.+}} is: 0x80D02802
// CHECK:     Address: 0x0000000000220694  Data@ {{.+}} is: 0x183B2003
// CHECK:     Address: 0x0000000000220698  Data@ {{.+}} is: 0x80500802
// CHECK:     Address: 0x000000000022069C  Data@ {{.+}} is: 0x144B283B
// CHECK:     Address: 0x00000000002206A0  Data@ {{.+}} is: 0x08034008
// CHECK:     Address: 0x00000000002206A4  Data@ {{.+}} is: 0x4585040B
// CHECK:     Address: 0x00000000002206A8  Data@ {{.+}} is: 0x150B0C41
// CHECK:     Address: 0x00000000002206AC  Data@ {{.+}} is: 0x7820164D
// CHECK:     Address: 0x00000000002206B0  Data@ {{.+}} is: 0x28BBA0D2
// CHECK:     Address: 0x00000000002206B4  Data@ {{.+}} is: 0x400A9A3B
// CHECK:     Address: 0x00000000002206B8  Data@ {{.+}} is: 0xA0523805
// CHECK:     Address: 0x00000000002206BC  Data@ {{.+}} is: 0x0C454593
// CHECK:     Address: 0x00000000002206C0  Data@ {{.+}} is: 0x68533D0B
// CHECK:     Address: 0x00000000002206C4  Data@ {{.+}} is: 0xF821164D
// CHECK:     Address: 0x00000000002206C8  Data@ {{.+}} is: 0x764DA0D1
// CHECK:     Address: 0x00000000002206CC  Data@ {{.+}} is: 0xA051BD34
// CHECK:     Address: 0x00000000002206D0  Data@ {{.+}} is: 0x0C495593
// CHECK:     Address: 0x00000000002206D4  Data@ {{.+}} is: 0x81D0750B
// CHECK:     Address: 0x00000000002206D8  Data@ {{.+}} is: 0xF822164D
// CHECK:     Address: 0x00000000002206DC  Data@ {{.+}} is: 0x764DA0D2
// CHECK:     Address: 0x00000000002206E0  Data@ {{.+}} is: 0xA052BD34
// CHECK:     Address: 0x00000000002206E4  Data@ {{.+}} is: 0x0C4B6593
// CHECK:     Address: 0x00000000002206E8  Data@ {{.+}} is: 0x8150550B
// CHECK:     Address: 0x00000000002206EC  Data@ {{.+}} is: 0x7822964D
// CHECK:     Address: 0x00000000002206F0  Data@ {{.+}} is: 0x764DA0D5
// CHECK:     Address: 0x00000000002206F4  Data@ {{.+}} is: 0xA0553D34
// CHECK:     Address: 0x00000000002206F8  Data@ {{.+}} is: 0x0C4D7593
// CHECK:     Address: 0x00000000002206FC  Data@ {{.+}} is: 0x80D0350B
// CHECK:     Address: 0x0000000000220700  Data@ {{.+}} is: 0xF823164D
// CHECK:     Address: 0x0000000000220704  Data@ {{.+}} is: 0x28BBA0D3
// CHECK:     Address: 0x0000000000220708  Data@ {{.+}} is: 0x700A9A3B
// CHECK:     Address: 0x000000000022070C  Data@ {{.+}} is: 0xA053B80C
// CHECK:     Address: 0x0000000000220710  Data@ {{.+}} is: 0x0C4E7593
// CHECK:     Address: 0x0000000000220714  Data@ {{.+}} is: 0x8050150B
// CHECK:     Address: 0x0000000000220718  Data@ {{.+}} is: 0x7823964D
// CHECK:     Address: 0x000000000022071C  Data@ {{.+}} is: 0x764DA0D2
// CHECK:     Address: 0x0000000000220720  Data@ {{.+}} is: 0xA0523D34
// CHECK:     Address: 0x0000000000220724  Data@ {{.+}} is: 0x3D0B087D
// CHECK:     Address: 0x0000000000220728  Data@ {{.+}} is: 0xA3D96214
// CHECK:     Address: 0x000000000022072C  Data@ {{.+}} is: 0x00010506
// CHECK:     Address: 0x0000000000220730  Data@ {{.+}} is: 0x033089D9
// CHECK:     Address: 0x0000000000220734  Data@ {{.+}} is: 0x60010009
// CHECK:     Address: 0x0000000000220738  Data@ {{.+}} is: 0x035085D9
// CHECK:     Address: 0x000000000022073C  Data@ {{.+}} is: 0x6018C02B
// CHECK:     Address: 0x0000000000220740  Data@ {{.+}} is: 0xB82285AC
// CHECK:     Address: 0x0000000000220744  Data@ {{.+}} is: 0x164D6E14
// CHECK:     Address: 0x0000000000220748  Data@ {{.+}} is: 0x72133824
// CHECK:     Address: 0x000000000022074C  Data@ {{.+}} is: 0x6021400B
// CHECK:     Address: 0x0000000000220750  Data@ {{.+}} is: 0x3D347642
// CHECK:     Address: 0x0000000000220754  Data@ {{.+}} is: 0x087DA054
// CHECK:     Address: 0x0000000000220758  Data@ {{.+}} is: 0x7611BD0B
// CHECK:     Address: 0x000000000022075C  Data@ {{.+}} is: 0x600A8023
// CHECK:     Address: 0x0000000000220760  Data@ {{.+}} is: 0xA0D17804
// CHECK:     Address: 0x0000000000220764  Data@ {{.+}} is: 0x050289D9
// CHECK:     Address: 0x0000000000220768  Data@ {{.+}} is: 0x6005C009
// CHECK:     Address: 0x000000000022076C  Data@ {{.+}} is: 0x00090001
// CHECK:     Address: 0x0000000000220770  Data@ {{.+}} is: 0x00016025
// CHECK:     Address: 0x0000000000220774  Data@ {{.+}} is: 0x601A0009
// CHECK:     Address: 0x0000000000220778  Data@ {{.+}} is: 0x80230001
// CHECK:     Address: 0x000000000022077C  Data@ {{.+}} is: 0xC804600C
// CHECK:     Address: 0x0000000000220780  Data@ {{.+}} is: 0x3C9900E4
// CHECK:     Address: 0x0000000000220784  Data@ {{.+}} is: 0x161917C5
// CHECK:     Address: 0x0000000000220788  Data@ {{.+}} is: 0x01951084
// CHECK:     Address: 0x000000000022078C  Data@ {{.+}} is: 0x1002C040
// CHECK:     Address: 0x0000000000220790  Data@ {{.+}} is: 0x83590001
// CHECK:     Address: 0x0000000000220794  Data@ {{.+}} is: 0x0B3D0C0E
// CHECK:     Address: 0x0000000000220798  Data@ {{.+}} is: 0x81504FBC
// CHECK:     Address: 0x000000000022079C  Data@ {{.+}} is: 0x0780DCFB
// CHECK:     Address: 0x00000000002207A0  Data@ {{.+}} is: 0x4C068140
// CHECK:     Address: 0x00000000002207A4  Data@ {{.+}} is: 0x00FB3004
// CHECK:     Address: 0x00000000002207A8  Data@ {{.+}} is: 0x804006F6
// CHECK:     Address: 0x00000000002207AC  Data@ {{.+}} is: 0x08010C02
// CHECK:     Address: 0x00000000002207B0  Data@ {{.+}} is: 0x07FC4659
// CHECK:     Address: 0x00000000002207B4  Data@ {{.+}} is: 0x07FCF659
// CHECK:     Address: 0x00000000002207B8  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002207BC  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002207C0  Data@ {{.+}} is: 0xAC990001
// CHECK:     Address: 0x00000000002207C4  Data@ {{.+}} is: 0x161910C0
// CHECK:     Address: 0x00000000002207C8  Data@ {{.+}} is: 0x01951000
// CHECK:     Address: 0x00000000002207CC  Data@ {{.+}} is: 0x0002A840
// CHECK:     Address: 0x00000000002207D0  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002207D4  Data@ {{.+}} is: 0x10C60719
// CHECK:     Address: 0x00000000002207D8  Data@ {{.+}} is: 0x10C49C99
// CHECK:     Address: 0x00000000002207DC  Data@ {{.+}} is: 0x16F62099
// CHECK:     Address: 0x00000000002207E0  Data@ {{.+}} is: 0x880003C0
// CHECK:     Address: 0x00000000002207E4  Data@ {{.+}} is: 0x04900003
// CHECK:     Address: 0x00000000002207E8  Data@ {{.+}} is: 0x00000030
// CHECK:     Address: 0x00000000002207EC  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002207F0  Data@ {{.+}} is: 0x10000019
// CHECK:     Address: 0x00000000002207F4  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002207F8  Data@ {{.+}} is: 0x16609219
// CHECK:     Address: 0x00000000002207FC  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220800  Data@ {{.+}} is: 0x48000095
// CHECK:     Address: 0x0000000000220804  Data@ {{.+}} is: 0x92190001
// CHECK:     Address: 0x0000000000220808  Data@ {{.+}} is: 0x065916A0
// CHECK:     Address: 0x000000000022080C  Data@ {{.+}} is: 0xA11D0024
// CHECK:     Address: 0x0000000000220810  Data@ {{.+}} is: 0x05814808
// CHECK:     Address: 0x0000000000220814  Data@ {{.+}} is: 0xC80CC11D
// CHECK:     Address: 0x0000000000220818  Data@ {{.+}} is: 0xE11D0681
// CHECK:     Address: 0x000000000022081C  Data@ {{.+}} is: 0x07878810
// CHECK:     Address: 0x0000000000220820  Data@ {{.+}} is: 0x0000007F
// CHECK:     Address: 0x0000000000220824  Data@ {{.+}} is: 0x08000008
// CHECK:     Address: 0x0000000000220828  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022082C  Data@ {{.+}} is: 0x00010000
// CHECK:     Address: 0x0000000000220830  Data@ {{.+}} is: 0x38032019
// CHECK:     Address: 0x0000000000220834  Data@ {{.+}} is: 0x0FFC4299
// CHECK:     Address: 0x0000000000220838  Data@ {{.+}} is: 0x011D0001
// CHECK:     Address: 0x000000000022083C  Data@ {{.+}} is: 0x0000D802
// CHECK:     Address: 0x0000000000220840  Data@ {{.+}} is: 0x0000081D
// CHECK:     Address: 0x0000000000220844  Data@ {{.+}} is: 0x00010000
// CHECK:     Address: 0x0000000000220848  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x000000000022084C  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220850  Data@ {{.+}} is: 0x07FC42D9
// CHECK:     Address: 0x0000000000220854  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220858  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x000000000022085C  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220860  Data@ {{.+}} is: 0x10001819
// CHECK:     Address: 0x0000000000220864  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220868  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x000000000022086C  Data@ {{.+}} is: 0x3FFFE019

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
// CHECK:     Address: 0x000000000021D000  Data@ {{.+}} is: 0x00400400
// CHECK:     Address: 0x000000000021D004  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D008  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D00C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D010  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D014  Data@ {{.+}} is: 0x06045FE3

// CHECK: (NOP Command): Payload Length: 2
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x000000000021D020  Size: 6
// CHECK:     Address: 0x000000000021D020  Data@ {{.+}} is: 0x01400400
// CHECK:     Address: 0x000000000021D024  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D028  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D02C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D030  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D034  Data@ {{.+}} is: 0x0E049FE5

// CHECK: (NOP Command): Payload Length: 2
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x000000000021D040  Size: 6
// CHECK:     Address: 0x000000000021D040  Data@ {{.+}} is: 0x02401000
// CHECK:     Address: 0x000000000021D044  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D048  Data@ {{.+}} is: 0x003FE000
// CHECK:     Address: 0x000000000021D04C  Data@ {{.+}} is: 0x01010007
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
// CHECK:     Address: 0x00000000001A0000  Data@ {{.+}} is: 0x00000400
// CHECK:     Address: 0x00000000001A0004  Data@ {{.+}} is: 0x000A0000
// CHECK:     Address: 0x00000000001A0008  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000001A000C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000001A0010  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000001A0014  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000001A0018  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000001A001C  Data@ {{.+}} is: 0x8140FF41

// CHECK: (NOP Command): Payload Length: 0
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x00000000001A0020  Size: 8
// CHECK:     Address: 0x00000000001A0020  Data@ {{.+}} is: 0x00000400
// CHECK:     Address: 0x00000000001A0024  Data@ {{.+}} is: 0x001A0000
// CHECK:     Address: 0x00000000001A0028  Data@ {{.+}} is: 0x00040000
// CHECK:     Address: 0x00000000001A002C  Data@ {{.+}} is: 0x0080000F
// CHECK:     Address: 0x00000000001A0030  Data@ {{.+}} is: 0x00100001
// CHECK:     Address: 0x00000000001A0034  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000001A0038  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000001A003C  Data@ {{.+}} is: 0x8141FF40

// CHECK: (Write64): Address:  0x00000000001A0604 Data:  0x00010000
// CHECK: (MaskWrite64): Address: 0x00000000001A0600  Mask: 0x00000000  Data: 0x00000001
// CHECK: (Write64): Address:  0x00000000001A0634 Data:  0x00010001
// CHECK: (MaskWrite64): Address: 0x00000000001A0630  Mask: 0x00000000  Data: 0x00000001
// CHECK: (NOP Command): Payload Length: 2
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x00000000021A0000  Size: 8
// CHECK:     Address: 0x00000000021A0000  Data@ {{.+}} is: 0x00000400
// CHECK:     Address: 0x00000000021A0004  Data@ {{.+}} is: 0x000A0000
// CHECK:     Address: 0x00000000021A0008  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000021A000C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000021A0010  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000021A0014  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000021A0018  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000021A001C  Data@ {{.+}} is: 0x8140FF41

// CHECK: (NOP Command): Payload Length: 0
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x00000000021A0020  Size: 8
// CHECK:     Address: 0x00000000021A0020  Data@ {{.+}} is: 0x00000400
// CHECK:     Address: 0x00000000021A0024  Data@ {{.+}} is: 0x001A0000
// CHECK:     Address: 0x00000000021A0028  Data@ {{.+}} is: 0x00040000
// CHECK:     Address: 0x00000000021A002C  Data@ {{.+}} is: 0x0080000F
// CHECK:     Address: 0x00000000021A0030  Data@ {{.+}} is: 0x00100001
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