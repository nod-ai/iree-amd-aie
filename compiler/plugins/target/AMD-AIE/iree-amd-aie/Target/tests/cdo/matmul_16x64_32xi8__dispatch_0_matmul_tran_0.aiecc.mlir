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
  %buf5 = aie.buffer(%tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "buf5"} : memref<16x64xi8> 
  %buf4 = aie.buffer(%tile_1_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "buf4"} : memref<32x64xi8> 
  %buf3 = aie.buffer(%tile_2_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "buf3"} : memref<16x32xi32> 
  %buf2 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "buf2"} : memref<8x4x4x8xi8> 
  %buf1 = aie.buffer(%tile_0_2) {address = 2048 : i32, mem_bank = 0 : i32, sym_name = "buf1"} : memref<8x4x8x8xi8> 
  %buf0 = aie.buffer(%tile_0_2) {address = 4096 : i32, mem_bank = 0 : i32, sym_name = "buf0"} : memref<4x4x4x8xi32> 
  %mem_0_2 = aie.mem(%tile_0_2) {
    %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5, repeat_count = 1)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf2 : memref<8x4x4x8xi8>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 0 : i32}
    aie.use_lock(%lock_0_2_5, Release, 1)
    aie.next_bd ^bb1
  ^bb2:  // pred: ^bb3
    aie.end
  ^bb3:  // pred: ^bb5
    %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2, repeat_count = 1)
  ^bb4:  // 2 preds: ^bb3, ^bb4
    aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf1 : memref<8x4x8x8xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 1 : i32}
    aie.use_lock(%lock_0_2_3, Release, 1)
    aie.next_bd ^bb4
  ^bb5:  // pred: ^bb0
    %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb3, repeat_count = 1)
  ^bb6:  // 2 preds: ^bb5, ^bb6
    aie.use_lock(%lock_0_2_7, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf0 : memref<4x4x4x8xi32>, 0, 512, [<size = 16, stride = 8>, <size = 4, stride = 128>, <size = 8, stride = 1>]) {bd_id = 2 : i32, next_bd_id = 2 : i32}
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
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb16
    aie.use_lock(%c49, AcquireGreaterEqual, 1)
    aie.use_lock(%c50, AcquireGreaterEqual, 1)
    aie.use_lock(%c52, AcquireGreaterEqual, 1)
    cf.br ^bb2(%c0 : index)
  ^bb2(%0: index):  // 2 preds: ^bb1, ^bb9
    %1 = arith.cmpi slt, %0, %c4 : index
    cf.cond_br %1, ^bb3(%c0 : index), ^bb10(%c0 : index)
  ^bb3(%2: index):  // 2 preds: ^bb2, ^bb8
    %3 = arith.cmpi slt, %2, %c4 : index
    cf.cond_br %3, ^bb4(%c0 : index), ^bb9
  ^bb4(%4: index):  // 2 preds: ^bb3, ^bb7
    %5 = arith.cmpi slt, %4, %c4 : index
    cf.cond_br %5, ^bb5(%c0 : index), ^bb8
  ^bb5(%6: index):  // 2 preds: ^bb4, ^bb6
    %7 = arith.cmpi slt, %6, %c8 : index
    cf.cond_br %7, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    memref.store %c0_i32, %buf0[%0, %2, %4, %6] : memref<4x4x4x8xi32>
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
    %13 = arith.cmpi slt, %12, %c4 : index
    cf.cond_br %13, ^bb11(%c0 : index), ^bb16
  ^bb11(%14: index):  // 2 preds: ^bb10, ^bb14
    %15 = arith.cmpi slt, %14, %c4 : index
    cf.cond_br %15, ^bb12(%c0 : index), ^bb15
  ^bb12(%16: index):  // 2 preds: ^bb11, ^bb13
    %17 = arith.cmpi slt, %16, %c8 : index
    cf.cond_br %17, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    %18 = vector.transfer_read %buf2[%16, %12, %c0, %c0], %c0_i8 {in_bounds = [true, true, true, true]} : memref<8x4x4x8xi8>, vector<1x1x4x8xi8>
    %19 = vector.transfer_read %buf1[%16, %14, %c0, %c0], %c0_i8 {in_bounds = [true, true, true, true]} : memref<8x4x8x8xi8>, vector<1x1x8x8xi8>
    %20 = vector.transfer_read %buf0[%14, %12, %c0, %c0], %c0_i32 {in_bounds = [true, true, true, true]} : memref<4x4x4x8xi32>, vector<1x1x4x8xi32>
    %21 = arith.extsi %18 : vector<1x1x4x8xi8> to vector<1x1x4x8xi32>
    %22 = arith.extsi %19 : vector<1x1x8x8xi8> to vector<1x1x8x8xi32>
    %23 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %21, %22, %20 : vector<1x1x4x8xi32>, vector<1x1x8x8xi32> into vector<1x1x4x8xi32>
    vector.transfer_write %23, %buf0[%14, %12, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x4x8xi32>, memref<4x4x4x8xi32>
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
  } {elf_file = "matmul_16x64_32xi8__dispatch_0_matmul_transpose_b_16x32x64_i8xi8xi32_0_core_0_2.elf"}
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
    aie.dma_bd(%buf3 : memref<16x32xi32>, 0, 512) {bd_id = 0 : i32, next_bd_id = 0 : i32}
    aie.use_lock(%lock_2_1_2, Release, 1)
    aie.next_bd ^bb1
  ^bb2:  // pred: ^bb3
    aie.end
  ^bb3:  // pred: ^bb0
    %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb2, repeat_count = 1)
  ^bb4:  // 2 preds: ^bb3, ^bb4
    aie.use_lock(%lock_2_1_2, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf3 : memref<16x32xi32>, 0, 512) {bd_id = 1 : i32, next_bd_id = 1 : i32}
    aie.use_lock(%lock_2_1, Release, 1)
    aie.next_bd ^bb4
  }
  %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
    %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf5 : memref<16x64xi8>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 0 : i32}
    aie.use_lock(%lock_0_1_1, Release, 1)
    aie.next_bd ^bb1
  ^bb2:  // pred: ^bb3
    aie.end
  ^bb3:  // pred: ^bb0
    %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb2, repeat_count = 1)
  ^bb4:  // 2 preds: ^bb3, ^bb4
    aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf5 : memref<16x64xi8>, 0, 1024, [<size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 1 : i32}
    aie.use_lock(%lock_0_1, Release, 1)
    aie.next_bd ^bb4
  }
  %memtile_dma_1_1 = aie.memtile_dma(%tile_1_1) {
    %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf4 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 0 : i32}
    aie.use_lock(%lock_1_1_0, Release, 1)
    aie.next_bd ^bb1
  ^bb2:  // pred: ^bb3
    aie.end
  ^bb3:  // pred: ^bb0
    %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb2, repeat_count = 1)
  ^bb4:  // 2 preds: ^bb3, ^bb4
    aie.use_lock(%lock_1_1_0, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf4 : memref<32x64xi8>, 0, 2048, [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 1 : i32}
    aie.use_lock(%lock_1_1, Release, 1)
    aie.next_bd ^bb4
  }
  aie.shim_dma_allocation @airMemcpyId12(S2MM, 0, 0)
  memref.global "public" @airMemcpyId12 : memref<16x32xi32>
  aie.shim_dma_allocation @airMemcpyId4(MM2S, 0, 0)
  memref.global "public" @airMemcpyId4 : memref<16x64xi8>
  aie.shim_dma_allocation @airMemcpyId5(MM2S, 1, 0)
  memref.global "public" @airMemcpyId5 : memref<32x64xi8>
  func.func @matmul_16x64_32xi8__dispatch_0_matmul_transpose_b_16x32x64_i8xi8xi32(%arg0: memref<256xi32>, %arg1: memref<512xi32>, %arg2: memref<16x32xi32>) {
    memref.assume_alignment %arg0, 64 : memref<256xi32>
    memref.assume_alignment %arg1, 64 : memref<512xi32>
    memref.assume_alignment %arg2, 64 : memref<16x32xi32>
    aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 16, 16][0, 0, 16]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<256xi32>
    aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 32, 16][0, 0, 16]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<512xi32>
    aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 16, 32][0, 0, 32]) {id = 2 : i64, metadata = @airMemcpyId12} : memref<16x32xi32>
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
} {sym_name = "matmul_16x64_32xi8__dispatch_0_matmul_transpose_b_16x32x64_i8xi8xi32_0"}
}

// CHECK: XAIE API: XAie_SetupPartitionConfig with args: &devInst=ptr, 0x0=0, partitionStartCol=1, partitionNumCols=4
// CHECK: XAIE API: XAie_CfgInitialize with args: &devInst=ptr, &configPtr=ptr
// CHECK: XAIE API: XAie_SetIOBackend with args: &devInst=ptr, XAIE_IO_BACKEND_CDO=1
// CHECK: XAIE API: XAie_UpdateNpiAddr with args: &devInst=ptr, 0x0=0
// CHECK: XAIE API: XAie_LoadElf with args: &devInst=ptr, XAie_TileLoc(col=XAie_LocType(Col: 0, Row: 2), row)=ptr, elfPath.str().c_str(), aieSim=0
// CHECK: XAIE API: XAie_CoreReset with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2)
// CHECK: XAIE API: XAie_CoreUnreset with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 0, LockVal: 0)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 1, LockVal: 0)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 2, LockVal: 0)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 3, LockVal: 0)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 4, LockVal: 0)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 5, LockVal: 0)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 6, LockVal: 0)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 7, LockVal: 0)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 8, LockVal: 0)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 9, LockVal: 0)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 10, LockVal: 0)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 11, LockVal: 0)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 12, LockVal: 0)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 13, LockVal: 0)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 14, LockVal: 0)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 15, LockVal: 0)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 1, Row: 1), locInit=XAie_Lock(LockId: 1, LockVal: 1)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 1, Row: 1), locInit=XAie_Lock(LockId: 0, LockVal: 0)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), locInit=XAie_Lock(LockId: 1, LockVal: 1)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), locInit=XAie_Lock(LockId: 0, LockVal: 0)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 2, Row: 1), locInit=XAie_Lock(LockId: 1, LockVal: 1)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 2, Row: 1), locInit=XAie_Lock(LockId: 0, LockVal: 0)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 5, LockVal: 1)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 4, LockVal: 0)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 3, LockVal: 1)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 2, LockVal: 0)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 1, LockVal: 1)
// CHECK: XAIE API: XAie_LockSetValue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 0, LockVal: 0)
// CHECK: XAIE API: XAie_DmaDescInit with args: &devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2)

// CHECK: start configuring bds
// CHECK: XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd=ptr, acqLock=XAie_Lock(LockId: 3, LockVal: -1), relLock=XAie_Lock(LockId: 2, LockVal: 1), acqEn=1, relEn=0
// CHECK: XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd=ptr, basePlusOffsetInBytes=1024, lenInBytes=1024
// CHECK: XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd=ptr, nextBdId.value()=0, enableNextBd=1
// CHECK: XAIE API: XAie_DmaEnableBd with args: &dmaTileBd=ptr
// CHECK: XAIE API: XAie_DmaWriteBd with args: &devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), bdId=0

// CHECK: end configuring bds
// CHECK: XAIE API: XAie_DmaDescInit with args: &devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2)

// CHECK: start configuring bds
// CHECK: XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd=ptr, acqLock=XAie_Lock(LockId: 5, LockVal: -1), relLock=XAie_Lock(LockId: 4, LockVal: 1), acqEn=1, relEn=0
// CHECK: XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd=ptr, basePlusOffsetInBytes=2048, lenInBytes=2048
// CHECK: XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd=ptr, nextBdId.value()=1, enableNextBd=1
// CHECK: XAIE API: XAie_DmaEnableBd with args: &dmaTileBd=ptr
// CHECK: XAIE API: XAie_DmaWriteBd with args: &devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), bdId=1

// CHECK: end configuring bds
// CHECK: XAIE API: XAie_DmaDescInit with args: &devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2)

// CHECK: start configuring bds
// CHECK: XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd=ptr, acqLock=XAie_Lock(LockId: 0, LockVal: -1), relLock=XAie_Lock(LockId: 1, LockVal: 1), acqEn=1, relEn=0
// CHECK: XAIE API: XAie_DmaSetMultiDimAddr with args: &dmaTileBd=ptr, &dmaTileBdTensor=ptr, basePlusOffsetInBytes=4096, lenInBytes=2048
// CHECK: XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd=ptr, nextBdId.value()=2, enableNextBd=1
// CHECK: XAIE API: XAie_DmaEnableBd with args: &dmaTileBd=ptr
// CHECK: XAIE API: XAie_DmaWriteBd with args: &devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), bdId=2

// CHECK: end configuring bds
// CHECK: XAIE API: XAie_DmaChannelSetStartQueue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), chNum=0, direction=0, bdId=0, repeatCount=2, enTokenIssue=0
// CHECK: XAIE API: XAie_DmaChannelEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), chNum=0, direction=0
// CHECK: XAIE API: XAie_DmaChannelSetStartQueue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), chNum=1, direction=0, bdId=1, repeatCount=2, enTokenIssue=0
// CHECK: XAIE API: XAie_DmaChannelEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), chNum=1, direction=0
// CHECK: XAIE API: XAie_DmaChannelSetStartQueue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), chNum=0, direction=1, bdId=2, repeatCount=2, enTokenIssue=0
// CHECK: XAIE API: XAie_DmaChannelEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), chNum=0, direction=1
// CHECK: XAIE API: XAie_DmaDescInit with args: &devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 2, Row: 1)

// CHECK: start configuring bds
// CHECK: XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd=ptr, acqLock=XAie_Lock(LockId: 65, LockVal: -1), relLock=XAie_Lock(LockId: 64, LockVal: 1), acqEn=1, relEn=0
// CHECK: XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd=ptr, basePlusOffsetInBytes=524288, lenInBytes=2048
// CHECK: XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd=ptr, nextBdId.value()=0, enableNextBd=1
// CHECK: XAIE API: XAie_DmaEnableBd with args: &dmaTileBd=ptr
// CHECK: XAIE API: XAie_DmaWriteBd with args: &devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 2, Row: 1), bdId=0

// CHECK: end configuring bds
// CHECK: XAIE API: XAie_DmaDescInit with args: &devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 2, Row: 1)

// CHECK: start configuring bds
// CHECK: XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd=ptr, acqLock=XAie_Lock(LockId: 64, LockVal: -1), relLock=XAie_Lock(LockId: 65, LockVal: 1), acqEn=1, relEn=0
// CHECK: XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd=ptr, basePlusOffsetInBytes=524288, lenInBytes=2048
// CHECK: XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd=ptr, nextBdId.value()=1, enableNextBd=1
// CHECK: XAIE API: XAie_DmaEnableBd with args: &dmaTileBd=ptr
// CHECK: XAIE API: XAie_DmaWriteBd with args: &devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 2, Row: 1), bdId=1

// CHECK: end configuring bds
// CHECK: XAIE API: XAie_DmaChannelSetStartQueue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 2, Row: 1), chNum=0, direction=0, bdId=0, repeatCount=2, enTokenIssue=0
// CHECK: XAIE API: XAie_DmaChannelEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 2, Row: 1), chNum=0, direction=0
// CHECK: XAIE API: XAie_DmaChannelSetStartQueue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 2, Row: 1), chNum=0, direction=1, bdId=1, repeatCount=2, enTokenIssue=0
// CHECK: XAIE API: XAie_DmaChannelEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 2, Row: 1), chNum=0, direction=1
// CHECK: XAIE API: XAie_DmaDescInit with args: &devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1)

// CHECK: start configuring bds
// CHECK: XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd=ptr, acqLock=XAie_Lock(LockId: 65, LockVal: -1), relLock=XAie_Lock(LockId: 64, LockVal: 1), acqEn=1, relEn=0
// CHECK: XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd=ptr, basePlusOffsetInBytes=524288, lenInBytes=1024
// CHECK: XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd=ptr, nextBdId.value()=0, enableNextBd=1
// CHECK: XAIE API: XAie_DmaEnableBd with args: &dmaTileBd=ptr
// CHECK: XAIE API: XAie_DmaWriteBd with args: &devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), bdId=0

// CHECK: end configuring bds
// CHECK: XAIE API: XAie_DmaDescInit with args: &devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1)

// CHECK: start configuring bds
// CHECK: XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd=ptr, acqLock=XAie_Lock(LockId: 64, LockVal: -1), relLock=XAie_Lock(LockId: 65, LockVal: 1), acqEn=1, relEn=0
// CHECK: XAIE API: XAie_DmaSetMultiDimAddr with args: &dmaTileBd=ptr, &dmaTileBdTensor=ptr, basePlusOffsetInBytes=524288, lenInBytes=1024
// CHECK: XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd=ptr, nextBdId.value()=1, enableNextBd=1
// CHECK: XAIE API: XAie_DmaEnableBd with args: &dmaTileBd=ptr
// CHECK: XAIE API: XAie_DmaWriteBd with args: &devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), bdId=1

// CHECK: end configuring bds
// CHECK: XAIE API: XAie_DmaChannelSetStartQueue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), chNum=0, direction=0, bdId=0, repeatCount=2, enTokenIssue=0
// CHECK: XAIE API: XAie_DmaChannelEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), chNum=0, direction=0
// CHECK: XAIE API: XAie_DmaChannelSetStartQueue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), chNum=0, direction=1, bdId=1, repeatCount=2, enTokenIssue=0
// CHECK: XAIE API: XAie_DmaChannelEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), chNum=0, direction=1
// CHECK: XAIE API: XAie_DmaDescInit with args: &devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 1, Row: 1)

// CHECK: start configuring bds
// CHECK: XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd=ptr, acqLock=XAie_Lock(LockId: 65, LockVal: -1), relLock=XAie_Lock(LockId: 64, LockVal: 1), acqEn=1, relEn=0
// CHECK: XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd=ptr, basePlusOffsetInBytes=524288, lenInBytes=2048
// CHECK: XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd=ptr, nextBdId.value()=0, enableNextBd=1
// CHECK: XAIE API: XAie_DmaEnableBd with args: &dmaTileBd=ptr
// CHECK: XAIE API: XAie_DmaWriteBd with args: &devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 1, Row: 1), bdId=0

// CHECK: end configuring bds
// CHECK: XAIE API: XAie_DmaDescInit with args: &devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 1, Row: 1)

// CHECK: start configuring bds
// CHECK: XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd=ptr, acqLock=XAie_Lock(LockId: 64, LockVal: -1), relLock=XAie_Lock(LockId: 65, LockVal: 1), acqEn=1, relEn=0
// CHECK: XAIE API: XAie_DmaSetMultiDimAddr with args: &dmaTileBd=ptr, &dmaTileBdTensor=ptr, basePlusOffsetInBytes=524288, lenInBytes=2048
// CHECK: XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd=ptr, nextBdId.value()=1, enableNextBd=1
// CHECK: XAIE API: XAie_DmaEnableBd with args: &dmaTileBd=ptr
// CHECK: XAIE API: XAie_DmaWriteBd with args: &devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 1, Row: 1), bdId=1

// CHECK: end configuring bds
// CHECK: XAIE API: XAie_DmaChannelSetStartQueue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 1, Row: 1), chNum=0, direction=0, bdId=0, repeatCount=2, enTokenIssue=0
// CHECK: XAIE API: XAie_DmaChannelEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 1, Row: 1), chNum=0, direction=0
// CHECK: XAIE API: XAie_DmaChannelSetStartQueue with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 1, Row: 1), chNum=0, direction=1, bdId=1, repeatCount=2, enTokenIssue=0
// CHECK: XAIE API: XAie_DmaChannelEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 1, Row: 1), chNum=0, direction=1
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 0), CTRL=StrmSwPortType::CTRL, slvPortNum=0, SOUTH=StrmSwPortType::SOUTH, mstrPortNum=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 0), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::SOUTH, connectOp.sourceIndex()=3, toStrmT(connectOp.getDestBundle())=StrmSwPortType::NORTH, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 0), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::SOUTH, connectOp.sourceIndex()=7, toStrmT(connectOp.getDestBundle())=StrmSwPortType::EAST, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 0), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::EAST, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::SOUTH, connectOp.destIndex()=2
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::SOUTH, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::DMA, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::DMA, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::NORTH, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 1, Row: 0), CTRL=StrmSwPortType::CTRL, slvPortNum=0, SOUTH=StrmSwPortType::SOUTH, mstrPortNum=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 1, Row: 0), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::WEST, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::NORTH, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 1, Row: 0), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::EAST, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::WEST, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 1, Row: 1), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::SOUTH, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::DMA, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 1, Row: 1), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::DMA, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::NORTH, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 2, Row: 0), CTRL=StrmSwPortType::CTRL, slvPortNum=0, SOUTH=StrmSwPortType::SOUTH, mstrPortNum=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 2, Row: 0), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::NORTH, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::WEST, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 2, Row: 1), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::DMA, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::SOUTH, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 2, Row: 1), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::NORTH, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::DMA, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::SOUTH, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::DMA, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::EAST, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::DMA, connectOp.destIndex()=1
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::DMA, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::EAST, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 1, Row: 2), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::SOUTH, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::WEST, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 1, Row: 2), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::WEST, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::EAST, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 2, Row: 2), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::WEST, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::SOUTH, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_EnableShimDmaToAieStrmPort with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 0), connectOp.destIndex()=3
// CHECK: XAIE API: XAie_EnableShimDmaToAieStrmPort with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 0), connectOp.destIndex()=7
// CHECK: XAIE API: XAie_EnableAieToShimDmaStrmPort with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 0), connectOp.sourceIndex()=2
// CHECK: XAIE API: XAie_CoreEnable with args: &devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2)
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

// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x0000000000220200  Size: 464
// CHECK:     Address: 0x0000000000220200  Data@ {{.+}} is: 0x000003C0
// CHECK:     Address: 0x0000000000220204  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220208  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022020C  Data@ {{.+}} is: 0x00680000
// CHECK:     Address: 0x0000000000220210  Data@ {{.+}} is: 0x4FFCE035
// CHECK:     Address: 0x0000000000220214  Data@ {{.+}} is: 0x1D7B0003
// CHECK:     Address: 0x0000000000220218  Data@ {{.+}} is: 0x6200000C
// CHECK:     Address: 0x000000000022021C  Data@ {{.+}} is: 0x00A1CFFC
// CHECK:     Address: 0x0000000000220220  Data@ {{.+}} is: 0x0010097B
// CHECK:     Address: 0x0000000000220224  Data@ {{.+}} is: 0x4FFBE400
// CHECK:     Address: 0x0000000000220228  Data@ {{.+}} is: 0x217B0082
// CHECK:     Address: 0x000000000022022C  Data@ {{.+}} is: 0x66000014
// CHECK:     Address: 0x0000000000220230  Data@ {{.+}} is: 0x0182CFFB
// CHECK:     Address: 0x0000000000220234  Data@ {{.+}} is: 0x0018417B
// CHECK:     Address: 0x0000000000220238  Data@ {{.+}} is: 0x8FFAE800
// CHECK:     Address: 0x000000000022023C  Data@ {{.+}} is: 0x617B0283
// CHECK:     Address: 0x0000000000220240  Data@ {{.+}} is: 0x6A000032
// CHECK:     Address: 0x0000000000220244  Data@ {{.+}} is: 0x03868FFA
// CHECK:     Address: 0x0000000000220248  Data@ {{.+}} is: 0x0046017B
// CHECK:     Address: 0x000000000022024C  Data@ {{.+}} is: 0x4FF9EC00
// CHECK:     Address: 0x0000000000220250  Data@ {{.+}} is: 0x113B0881
// CHECK:     Address: 0x0000000000220254  Data@ {{.+}} is: 0x01C41800
// CHECK:     Address: 0x0000000000220258  Data@ {{.+}} is: 0xFF2DC000
// CHECK:     Address: 0x000000000022025C  Data@ {{.+}} is: 0x1C00113B
// CHECK:     Address: 0x0000000000220260  Data@ {{.+}} is: 0xD00001C2
// CHECK:     Address: 0x0000000000220264  Data@ {{.+}} is: 0x113BFF1E
// CHECK:     Address: 0x0000000000220268  Data@ {{.+}} is: 0x01C09A00
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
// CHECK:     Address: 0x000000000022029C  Data@ {{.+}} is: 0x0006C81E
// CHECK:     Address: 0x00000000002202A0  Data@ {{.+}} is: 0x000003C0
// CHECK:     Address: 0x00000000002202A4  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002202A8  Data@ {{.+}} is: 0x07FF5E00
// CHECK:     Address: 0x00000000002202AC  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002202B0  Data@ {{.+}} is: 0xC3C06DBD
// CHECK:     Address: 0x00000000002202B4  Data@ {{.+}} is: 0x017BFFFE
// CHECK:     Address: 0x00000000002202B8  Data@ {{.+}} is: 0xC000001E
// CHECK:     Address: 0x00000000002202BC  Data@ {{.+}} is: 0x0006CFFE
// CHECK:     Address: 0x00000000002202C0  Data@ {{.+}} is: 0x00000013
// CHECK:     Address: 0x00000000002202C4  Data@ {{.+}} is: 0xFFD84800
// CHECK:     Address: 0x00000000002202C8  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002202CC  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002202D0  Data@ {{.+}} is: 0x7D990001
// CHECK:     Address: 0x00000000002202D4  Data@ {{.+}} is: 0x109913C0
// CHECK:     Address: 0x00000000002202D8  Data@ {{.+}} is: 0x8D991000
// CHECK:     Address: 0x00000000002202DC  Data@ {{.+}} is: 0x95991030
// CHECK:     Address: 0x00000000002202E0  Data@ {{.+}} is: 0xA5BD1600
// CHECK:     Address: 0x00000000002202E4  Data@ {{.+}} is: 0xFFA80600
// CHECK:     Address: 0x00000000002202E8  Data@ {{.+}} is: 0x0600B5FB
// CHECK:     Address: 0x00000000002202EC  Data@ {{.+}} is: 0x2FFDC000
// CHECK:     Address: 0x00000000002202F0  Data@ {{.+}} is: 0x4035FFAC
// CHECK:     Address: 0x00000000002202F4  Data@ {{.+}} is: 0xFFBC2FFE
// CHECK:     Address: 0x00000000002202F8  Data@ {{.+}} is: 0x07FE6159
// CHECK:     Address: 0x00000000002202FC  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220300  Data@ {{.+}} is: 0x1C2C1659
// CHECK:     Address: 0x0000000000220304  Data@ {{.+}} is: 0x08109A19
// CHECK:     Address: 0x0000000000220308  Data@ {{.+}} is: 0x08109A19
// CHECK:     Address: 0x000000000022030C  Data@ {{.+}} is: 0x08109A19
// CHECK:     Address: 0x0000000000220310  Data@ {{.+}} is: 0x4638C5BD
// CHECK:     Address: 0x0000000000220314  Data@ {{.+}} is: 0x16590213
// CHECK:     Address: 0x0000000000220318  Data@ {{.+}} is: 0xE5BD1C2E
// CHECK:     Address: 0x000000000022031C  Data@ {{.+}} is: 0x0213463A
// CHECK:     Address: 0x0000000000220320  Data@ {{.+}} is: 0x1C2E9659
// CHECK:     Address: 0x0000000000220324  Data@ {{.+}} is: 0x463D95BD
// CHECK:     Address: 0x0000000000220328  Data@ {{.+}} is: 0x16590213
// CHECK:     Address: 0x000000000022032C  Data@ {{.+}} is: 0xA5BD1C2F
// CHECK:     Address: 0x0000000000220330  Data@ {{.+}} is: 0x0213463F
// CHECK:     Address: 0x0000000000220334  Data@ {{.+}} is: 0x17CB2843
// CHECK:     Address: 0x0000000000220338  Data@ {{.+}} is: 0x0000040A
// CHECK:     Address: 0x000000000022033C  Data@ {{.+}} is: 0x462005BD
// CHECK:     Address: 0x0000000000220340  Data@ {{.+}} is: 0x28430213
// CHECK:     Address: 0x0000000000220344  Data@ {{.+}} is: 0x048A140B
// CHECK:     Address: 0x0000000000220348  Data@ {{.+}} is: 0x05BD0000
// CHECK:     Address: 0x000000000022034C  Data@ {{.+}} is: 0x02134622
// CHECK:     Address: 0x0000000000220350  Data@ {{.+}} is: 0x144B2843
// CHECK:     Address: 0x0000000000220354  Data@ {{.+}} is: 0x0000050A
// CHECK:     Address: 0x0000000000220358  Data@ {{.+}} is: 0x462405BD
// CHECK:     Address: 0x000000000022035C  Data@ {{.+}} is: 0x28430213
// CHECK:     Address: 0x0000000000220360  Data@ {{.+}} is: 0x058A148B
// CHECK:     Address: 0x0000000000220364  Data@ {{.+}} is: 0x05BD0000
// CHECK:     Address: 0x0000000000220368  Data@ {{.+}} is: 0x02134626
// CHECK:     Address: 0x000000000022036C  Data@ {{.+}} is: 0x14CB2843
// CHECK:     Address: 0x0000000000220370  Data@ {{.+}} is: 0x0000060A
// CHECK:     Address: 0x0000000000220374  Data@ {{.+}} is: 0x462805BD
// CHECK:     Address: 0x0000000000220378  Data@ {{.+}} is: 0x28430213
// CHECK:     Address: 0x000000000022037C  Data@ {{.+}} is: 0x068A150B
// CHECK:     Address: 0x0000000000220380  Data@ {{.+}} is: 0x05BD0000
// CHECK:     Address: 0x0000000000220384  Data@ {{.+}} is: 0x0213462A
// CHECK:     Address: 0x0000000000220388  Data@ {{.+}} is: 0x154B2843
// CHECK:     Address: 0x000000000022038C  Data@ {{.+}} is: 0x0000070A
// CHECK:     Address: 0x0000000000220390  Data@ {{.+}} is: 0x062C05FB
// CHECK:     Address: 0x0000000000220394  Data@ {{.+}} is: 0x48109A00
// CHECK:     Address: 0x0000000000220398  Data@ {{.+}} is: 0x28430900
// CHECK:     Address: 0x000000000022039C  Data@ {{.+}} is: 0x078A158B
// CHECK:     Address: 0x00000000002203A0  Data@ {{.+}} is: 0x05FB0000
// CHECK:     Address: 0x00000000002203A4  Data@ {{.+}} is: 0x9A00062E
// CHECK:     Address: 0x00000000002203A8  Data@ {{.+}} is: 0x0F820810
// CHECK:     Address: 0x00000000002203AC  Data@ {{.+}} is: 0x15CB2843
// CHECK:     Address: 0x00000000002203B0  Data@ {{.+}} is: 0x003071AE
// CHECK:     Address: 0x00000000002203B4  Data@ {{.+}} is: 0x060C15FB
// CHECK:     Address: 0x00000000002203B8  Data@ {{.+}} is: 0x48109A00
// CHECK:     Address: 0x00000000002203BC  Data@ {{.+}} is: 0x28430980
// CHECK:     Address: 0x00000000002203C0  Data@ {{.+}} is: 0x842E11CB
// CHECK:     Address: 0x00000000002203C4  Data@ {{.+}} is: 0x55FB0030
// CHECK:     Address: 0x00000000002203C8  Data@ {{.+}} is: 0x9A000600
// CHECK:     Address: 0x00000000002203CC  Data@ {{.+}} is: 0x0D00C810
// CHECK:     Address: 0x00000000002203D0  Data@ {{.+}} is: 0x100B2843
// CHECK:     Address: 0x00000000002203D4  Data@ {{.+}} is: 0x003040AE
// CHECK:     Address: 0x00000000002203D8  Data@ {{.+}} is: 0x063235FB
// CHECK:     Address: 0x00000000002203DC  Data@ {{.+}} is: 0x48109A00
// CHECK:     Address: 0x00000000002203E0  Data@ {{.+}} is: 0x28430A00
// CHECK:     Address: 0x00000000002203E4  Data@ {{.+}} is: 0x358A118B
// CHECK:     Address: 0x00000000002203E8  Data@ {{.+}} is: 0x35FB0002
// CHECK:     Address: 0x00000000002203EC  Data@ {{.+}} is: 0x9A000634
// CHECK:     Address: 0x00000000002203F0  Data@ {{.+}} is: 0x0E00C810
// CHECK:     Address: 0x00000000002203F4  Data@ {{.+}} is: 0x110B2843
// CHECK:     Address: 0x00000000002203F8  Data@ {{.+}} is: 0x0030A1AE
// CHECK:     Address: 0x00000000002203FC  Data@ {{.+}} is: 0x460A15BD
// CHECK:     Address: 0x0000000000220400  Data@ {{.+}} is: 0x28430213
// CHECK:     Address: 0x0000000000220404  Data@ {{.+}} is: 0x128A114B
// CHECK:     Address: 0x0000000000220408  Data@ {{.+}} is: 0x15BD0002
// CHECK:     Address: 0x000000000022040C  Data@ {{.+}} is: 0x02134616
// CHECK:     Address: 0x0000000000220410  Data@ {{.+}} is: 0x12CB2843
// CHECK:     Address: 0x0000000000220414  Data@ {{.+}} is: 0x0002130A
// CHECK:     Address: 0x0000000000220418  Data@ {{.+}} is: 0x460415BD
// CHECK:     Address: 0x000000000022041C  Data@ {{.+}} is: 0x28430213
// CHECK:     Address: 0x0000000000220420  Data@ {{.+}} is: 0x138A108B
// CHECK:     Address: 0x0000000000220424  Data@ {{.+}} is: 0x15BD0002
// CHECK:     Address: 0x0000000000220428  Data@ {{.+}} is: 0x02134618
// CHECK:     Address: 0x000000000022042C  Data@ {{.+}} is: 0x130B2843
// CHECK:     Address: 0x0000000000220430  Data@ {{.+}} is: 0x0002140A
// CHECK:     Address: 0x0000000000220434  Data@ {{.+}} is: 0x461C15BD
// CHECK:     Address: 0x0000000000220438  Data@ {{.+}} is: 0x28430213
// CHECK:     Address: 0x000000000022043C  Data@ {{.+}} is: 0x148A138B
// CHECK:     Address: 0x0000000000220440  Data@ {{.+}} is: 0x15BD0002
// CHECK:     Address: 0x0000000000220444  Data@ {{.+}} is: 0x02134602
// CHECK:     Address: 0x0000000000220448  Data@ {{.+}} is: 0x104B2843
// CHECK:     Address: 0x000000000022044C  Data@ {{.+}} is: 0x0002368A
// CHECK:     Address: 0x0000000000220450  Data@ {{.+}} is: 0x061235FB
// CHECK:     Address: 0x0000000000220454  Data@ {{.+}} is: 0xC8109A00
// CHECK:     Address: 0x0000000000220458  Data@ {{.+}} is: 0x28430F00
// CHECK:     Address: 0x000000000022045C  Data@ {{.+}} is: 0x31AE164B
// CHECK:     Address: 0x0000000000220460  Data@ {{.+}} is: 0x157B0030
// CHECK:     Address: 0x0000000000220464  Data@ {{.+}} is: 0x9A00000E
// CHECK:     Address: 0x0000000000220468  Data@ {{.+}} is: 0x00610810
// CHECK:     Address: 0x000000000022046C  Data@ {{.+}} is: 0x168B2843
// CHECK:     Address: 0x0000000000220470  Data@ {{.+}} is: 0x001E0266
// CHECK:     Address: 0x0000000000220474  Data@ {{.+}} is: 0x03DE077B
// CHECK:     Address: 0x0000000000220478  Data@ {{.+}} is: 0x48109A00
// CHECK:     Address: 0x000000000022047C  Data@ {{.+}} is: 0x28430881
// CHECK:     Address: 0x0000000000220480  Data@ {{.+}} is: 0xB18A128B
// CHECK:     Address: 0x0000000000220484  Data@ {{.+}} is: 0xD67B0000
// CHECK:     Address: 0x0000000000220488  Data@ {{.+}} is: 0x9A000000
// CHECK:     Address: 0x000000000022048C  Data@ {{.+}} is: 0x00208810
// CHECK:     Address: 0x0000000000220490  Data@ {{.+}} is: 0x124B2843
// CHECK:     Address: 0x0000000000220494  Data@ {{.+}} is: 0x0000C20A
// CHECK:     Address: 0x0000000000220498  Data@ {{.+}} is: 0x100060BB
// CHECK:     Address: 0x000000000022049C  Data@ {{.+}} is: 0x88000058
// CHECK:     Address: 0x00000000002204A0  Data@ {{.+}} is: 0x2CFB0283
// CHECK:     Address: 0x00000000002204A4  Data@ {{.+}} is: 0x9A0003C2
// CHECK:     Address: 0x00000000002204A8  Data@ {{.+}} is: 0x03064810
// CHECK:     Address: 0x00000000002204AC  Data@ {{.+}} is: 0x10CB2843
// CHECK:     Address: 0x00000000002204B0  Data@ {{.+}} is: 0x0037B086
// CHECK:     Address: 0x00000000002204B4  Data@ {{.+}} is: 0x0034717B
// CHECK:     Address: 0x00000000002204B8  Data@ {{.+}} is: 0x88109A00
// CHECK:     Address: 0x00000000002204BC  Data@ {{.+}} is: 0x28430102
// CHECK:     Address: 0x00000000002204C0  Data@ {{.+}} is: 0x908A120B
// CHECK:     Address: 0x00000000002204C4  Data@ {{.+}} is: 0x017B0000
// CHECK:     Address: 0x00000000002204C8  Data@ {{.+}} is: 0x9A000046
// CHECK:     Address: 0x00000000002204CC  Data@ {{.+}} is: 0x00420810
// CHECK:     Address: 0x00000000002204D0  Data@ {{.+}} is: 0x000003C0
// CHECK:     Address: 0x00000000002204D4  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002204D8  Data@ {{.+}} is: 0xC8000000
// CHECK:     Address: 0x00000000002204DC  Data@ {{.+}} is: 0x0000FFEB
// CHECK:     Address: 0x00000000002204E0  Data@ {{.+}} is: 0xC800001D
// CHECK:     Address: 0x00000000002204E4  Data@ {{.+}} is: 0x0001FFFE
// CHECK:     Address: 0x00000000002204E8  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002204EC  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002204F0  Data@ {{.+}} is: 0x13C04C99
// CHECK:     Address: 0x00000000002204F4  Data@ {{.+}} is: 0x1000D619
// CHECK:     Address: 0x00000000002204F8  Data@ {{.+}} is: 0x50400195
// CHECK:     Address: 0x00000000002204FC  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220500  Data@ {{.+}} is: 0x07190001
// CHECK:     Address: 0x0000000000220504  Data@ {{.+}} is: 0x2C9913DE
// CHECK:     Address: 0x0000000000220508  Data@ {{.+}} is: 0x109D13C2
// CHECK:     Address: 0x000000000022050C  Data@ {{.+}} is: 0x00E18EF6
// CHECK:     Address: 0x0000000000220510  Data@ {{.+}} is: 0x0000007F
// CHECK:     Address: 0x0000000000220514  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220518  Data@ {{.+}} is: 0x78000000
// CHECK:     Address: 0x000000000022051C  Data@ {{.+}} is: 0x81D980D0
// CHECK:     Address: 0x0000000000220520  Data@ {{.+}} is: 0x00010402
// CHECK:     Address: 0x0000000000220524  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220528  Data@ {{.+}} is: 0x30590001
// CHECK:     Address: 0x000000000022052C  Data@ {{.+}} is: 0x28430000
// CHECK:     Address: 0x0000000000220530  Data@ {{.+}} is: 0x3469E13B
// CHECK:     Address: 0x0000000000220534  Data@ {{.+}} is: 0x38BB0000
// CHECK:     Address: 0x0000000000220538  Data@ {{.+}} is: 0xB00800C0
// CHECK:     Address: 0x000000000022053C  Data@ {{.+}} is: 0x61014801
// CHECK:     Address: 0x0000000000220540  Data@ {{.+}} is: 0x000003C0
// CHECK:     Address: 0x0000000000220544  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220548  Data@ {{.+}} is: 0x78000000
// CHECK:     Address: 0x000000000022054C  Data@ {{.+}} is: 0x000084D5
// CHECK:     Address: 0x0000000000220550  Data@ {{.+}} is: 0x0422A9D9
// CHECK:     Address: 0x0000000000220554  Data@ {{.+}} is: 0xFB30764D
// CHECK:     Address: 0x0000000000220558  Data@ {{.+}} is: 0x964D88D0
// CHECK:     Address: 0x000000000022055C  Data@ {{.+}} is: 0x8850BE37
// CHECK:     Address: 0x0000000000220560  Data@ {{.+}} is: 0xF938764D
// CHECK:     Address: 0x0000000000220564  Data@ {{.+}} is: 0x28BB8CD1
// CHECK:     Address: 0x0000000000220568  Data@ {{.+}} is: 0x036E9C3B
// CHECK:     Address: 0x000000000022056C  Data@ {{.+}} is: 0x8C51B830
// CHECK:     Address: 0x0000000000220570  Data@ {{.+}} is: 0x100B28BB
// CHECK:     Address: 0x0000000000220574  Data@ {{.+}} is: 0x703003EC
// CHECK:     Address: 0x0000000000220578  Data@ {{.+}} is: 0x164D2806
// CHECK:     Address: 0x000000000022057C  Data@ {{.+}} is: 0xAA067420
// CHECK:     Address: 0x0000000000220580  Data@ {{.+}} is: 0x3B0B087D
// CHECK:     Address: 0x0000000000220584  Data@ {{.+}} is: 0x8399C212
// CHECK:     Address: 0x0000000000220588  Data@ {{.+}} is: 0x964D030E
// CHECK:     Address: 0x000000000022058C  Data@ {{.+}} is: 0x61505737
// CHECK:     Address: 0x0000000000220590  Data@ {{.+}} is: 0x3060164D
// CHECK:     Address: 0x0000000000220594  Data@ {{.+}} is: 0x087D60D0
// CHECK:     Address: 0x0000000000220598  Data@ {{.+}} is: 0x6050172B
// CHECK:     Address: 0x000000000022059C  Data@ {{.+}} is: 0x3A37964D
// CHECK:     Address: 0x00000000002205A0  Data@ {{.+}} is: 0x087DE254
// CHECK:     Address: 0x00000000002205A4  Data@ {{.+}} is: 0xE2533A2B
// CHECK:     Address: 0x00000000002205A8  Data@ {{.+}} is: 0xBF13107D
// CHECK:     Address: 0x00000000002205AC  Data@ {{.+}} is: 0x107D4852
// CHECK:     Address: 0x00000000002205B0  Data@ {{.+}} is: 0xE253BF13
// CHECK:     Address: 0x00000000002205B4  Data@ {{.+}} is: 0x0712A5D9
// CHECK:     Address: 0x00000000002205B8  Data@ {{.+}} is: 0x010687D9
// CHECK:     Address: 0x00000000002205BC  Data@ {{.+}} is: 0xB938764D
// CHECK:     Address: 0x00000000002205C0  Data@ {{.+}} is: 0x33992050
// CHECK:     Address: 0x00000000002205C4  Data@ {{.+}} is: 0x000B0160
// CHECK:     Address: 0x00000000002205C8  Data@ {{.+}} is: 0xD0622810
// CHECK:     Address: 0x00000000002205CC  Data@ {{.+}} is: 0xA0D1FD01
// CHECK:     Address: 0x00000000002205D0  Data@ {{.+}} is: 0xBD01886D
// CHECK:     Address: 0x00000000002205D4  Data@ {{.+}} is: 0x800BA051
// CHECK:     Address: 0x00000000002205D8  Data@ {{.+}} is: 0x96422822
// CHECK:     Address: 0x00000000002205DC  Data@ {{.+}} is: 0x20D1FD37
// CHECK:     Address: 0x00000000002205E0  Data@ {{.+}} is: 0xB937964D
// CHECK:     Address: 0x00000000002205E4  Data@ {{.+}} is: 0x3EBB2051
// CHECK:     Address: 0x00000000002205E8  Data@ {{.+}} is: 0x080280CC
// CHECK:     Address: 0x00000000002205EC  Data@ {{.+}} is: 0xA506312B
// CHECK:     Address: 0x00000000002205F0  Data@ {{.+}} is: 0x281E800B
// CHECK:     Address: 0x00000000002205F4  Data@ {{.+}} is: 0x38027642
// CHECK:     Address: 0x00000000002205F8  Data@ {{.+}} is: 0x764DAA55
// CHECK:     Address: 0x00000000002205FC  Data@ {{.+}} is: 0x2C50B938
// CHECK:     Address: 0x0000000000220600  Data@ {{.+}} is: 0x2826800B
// CHECK:     Address: 0x0000000000220604  Data@ {{.+}} is: 0x70818862
// CHECK:     Address: 0x0000000000220608  Data@ {{.+}} is: 0x986D2E06
// CHECK:     Address: 0x000000000022060C  Data@ {{.+}} is: 0x20D1F881
// CHECK:     Address: 0x0000000000220610  Data@ {{.+}} is: 0x28144023
// CHECK:     Address: 0x0000000000220614  Data@ {{.+}} is: 0x2051B804
// CHECK:     Address: 0x0000000000220618  Data@ {{.+}} is: 0x3E2B087D
// CHECK:     Address: 0x000000000022061C  Data@ {{.+}} is: 0x8DD9C211
// CHECK:     Address: 0x0000000000220620  Data@ {{.+}} is: 0x40230672
// CHECK:     Address: 0x0000000000220624  Data@ {{.+}} is: 0x98662828
// CHECK:     Address: 0x0000000000220628  Data@ {{.+}} is: 0x00010181
// CHECK:     Address: 0x000000000022062C  Data@ {{.+}} is: 0x2804C009
// CHECK:     Address: 0x0000000000220630  Data@ {{.+}} is: 0x98790001
// CHECK:     Address: 0x0000000000220634  Data@ {{.+}} is: 0x00011D81
// CHECK:     Address: 0x0000000000220638  Data@ {{.+}} is: 0x280EC009
// CHECK:     Address: 0x000000000022063C  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220640  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220644  Data@ {{.+}} is: 0x0B0E8359
// CHECK:     Address: 0x0000000000220648  Data@ {{.+}} is: 0x983B2003
// CHECK:     Address: 0x000000000022064C  Data@ {{.+}} is: 0x61504800
// CHECK:     Address: 0x0000000000220650  Data@ {{.+}} is: 0x0B068173
// CHECK:     Address: 0x0000000000220654  Data@ {{.+}} is: 0x81D5F90B
// CHECK:     Address: 0x0000000000220658  Data@ {{.+}} is: 0x13028073
// CHECK:     Address: 0x000000000022065C  Data@ {{.+}} is: 0x8155B943
// CHECK:     Address: 0x0000000000220660  Data@ {{.+}} is: 0x010E8399
// CHECK:     Address: 0x0000000000220664  Data@ {{.+}} is: 0x010A8299
// CHECK:     Address: 0x0000000000220668  Data@ {{.+}} is: 0x01068199
// CHECK:     Address: 0x000000000022066C  Data@ {{.+}} is: 0x01028099
// CHECK:     Address: 0x0000000000220670  Data@ {{.+}} is: 0x93D90001
// CHECK:     Address: 0x0000000000220674  Data@ {{.+}} is: 0x91D9042E
// CHECK:     Address: 0x0000000000220678  Data@ {{.+}} is: 0x0001042A
// CHECK:     Address: 0x000000000022067C  Data@ {{.+}} is: 0xD86D0001
// CHECK:     Address: 0x0000000000220680  Data@ {{.+}} is: 0x89D27D81
// CHECK:     Address: 0x0000000000220684  Data@ {{.+}} is: 0x3B38764D
// CHECK:     Address: 0x0000000000220688  Data@ {{.+}} is: 0xC0238952
// CHECK:     Address: 0x000000000022068C  Data@ {{.+}} is: 0x70042812
// CHECK:     Address: 0x0000000000220690  Data@ {{.+}} is: 0xA3D96886
// CHECK:     Address: 0x0000000000220694  Data@ {{.+}} is: 0xA06D046E
// CHECK:     Address: 0x0000000000220698  Data@ {{.+}} is: 0x8D543A01
// CHECK:     Address: 0x000000000022069C  Data@ {{.+}} is: 0x030693D9
// CHECK:     Address: 0x00000000002206A0  Data@ {{.+}} is: 0x2821000B
// CHECK:     Address: 0x00000000002206A4  Data@ {{.+}} is: 0x3B387642
// CHECK:     Address: 0x00000000002206A8  Data@ {{.+}} is: 0x33996052
// CHECK:     Address: 0x00000000002206AC  Data@ {{.+}} is: 0xA06D0354
// CHECK:     Address: 0x00000000002206B0  Data@ {{.+}} is: 0x60D3FA01
// CHECK:     Address: 0x00000000002206B4  Data@ {{.+}} is: 0xBB38764D
// CHECK:     Address: 0x00000000002206B8  Data@ {{.+}} is: 0x33996053
// CHECK:     Address: 0x00000000002206BC  Data@ {{.+}} is: 0x00230364
// CHECK:     Address: 0x00000000002206C0  Data@ {{.+}} is: 0x7804281D
// CHECK:     Address: 0x00000000002206C4  Data@ {{.+}} is: 0x764D60D4
// CHECK:     Address: 0x00000000002206C8  Data@ {{.+}} is: 0x60543B38
// CHECK:     Address: 0x00000000002206CC  Data@ {{.+}} is: 0x03743399
// CHECK:     Address: 0x00000000002206D0  Data@ {{.+}} is: 0xFC01C06D
// CHECK:     Address: 0x00000000002206D4  Data@ {{.+}} is: 0xA5D960D4
// CHECK:     Address: 0x00000000002206D8  Data@ {{.+}} is: 0x00230302
// CHECK:     Address: 0x00000000002206DC  Data@ {{.+}} is: 0xA0662826
// CHECK:     Address: 0x00000000002206E0  Data@ {{.+}} is: 0x00010201
// CHECK:     Address: 0x00000000002206E4  Data@ {{.+}} is: 0x28150023
// CHECK:     Address: 0x00000000002206E8  Data@ {{.+}} is: 0x0201B866
// CHECK:     Address: 0x00000000002206EC  Data@ {{.+}} is: 0x00230001
// CHECK:     Address: 0x00000000002206F0  Data@ {{.+}} is: 0xC0662829
// CHECK:     Address: 0x00000000002206F4  Data@ {{.+}} is: 0x76590201
// CHECK:     Address: 0x00000000002206F8  Data@ {{.+}} is: 0x00231B30
// CHECK:     Address: 0x00000000002206FC  Data@ {{.+}} is: 0xC8662805
// CHECK:     Address: 0x0000000000220700  Data@ {{.+}} is: 0x08190201
// CHECK:     Address: 0x0000000000220704  Data@ {{.+}} is: 0x00233B0B
// CHECK:     Address: 0x0000000000220708  Data@ {{.+}} is: 0x7004280D
// CHECK:     Address: 0x000000000022070C  Data@ {{.+}} is: 0x97D96806
// CHECK:     Address: 0x0000000000220710  Data@ {{.+}} is: 0x95D90436
// CHECK:     Address: 0x0000000000220714  Data@ {{.+}} is: 0x9FD90432
// CHECK:     Address: 0x0000000000220718  Data@ {{.+}} is: 0x9DD90456
// CHECK:     Address: 0x000000000022071C  Data@ {{.+}} is: 0x83750452
// CHECK:     Address: 0x0000000000220720  Data@ {{.+}} is: 0x82D4790E
// CHECK:     Address: 0x0000000000220724  Data@ {{.+}} is: 0x390A8275
// CHECK:     Address: 0x0000000000220728  Data@ {{.+}} is: 0x81758254
// CHECK:     Address: 0x000000000022072C  Data@ {{.+}} is: 0xE1323906
// CHECK:     Address: 0x0000000000220730  Data@ {{.+}} is: 0x9C3B22BB
// CHECK:     Address: 0x0000000000220734  Data@ {{.+}} is: 0x39028040
// CHECK:     Address: 0x0000000000220738  Data@ {{.+}} is: 0x3399E054
// CHECK:     Address: 0x000000000022073C  Data@ {{.+}} is: 0x83990148
// CHECK:     Address: 0x0000000000220740  Data@ {{.+}} is: 0x8299030E
// CHECK:     Address: 0x0000000000220744  Data@ {{.+}} is: 0x8199030A
// CHECK:     Address: 0x0000000000220748  Data@ {{.+}} is: 0x80990306
// CHECK:     Address: 0x000000000022074C  Data@ {{.+}} is: 0xC06D0302
// CHECK:     Address: 0x0000000000220750  Data@ {{.+}} is: 0x8ED4FC81
// CHECK:     Address: 0x0000000000220754  Data@ {{.+}} is: 0x0472A5D9
// CHECK:     Address: 0x0000000000220758  Data@ {{.+}} is: 0x01069FD9
// CHECK:     Address: 0x000000000022075C  Data@ {{.+}} is: 0xB938764D
// CHECK:     Address: 0x0000000000220760  Data@ {{.+}} is: 0x33992053
// CHECK:     Address: 0x0000000000220764  Data@ {{.+}} is: 0x95D90158
// CHECK:     Address: 0x0000000000220768  Data@ {{.+}} is: 0x400B0242
// CHECK:     Address: 0x000000000022076C  Data@ {{.+}} is: 0xA862280A
// CHECK:     Address: 0x0000000000220770  Data@ {{.+}} is: 0x20D4FA81
// CHECK:     Address: 0x0000000000220774  Data@ {{.+}} is: 0xB938764D
// CHECK:     Address: 0x0000000000220778  Data@ {{.+}} is: 0x400B2054
// CHECK:     Address: 0x000000000022077C  Data@ {{.+}} is: 0xB8622819
// CHECK:     Address: 0x0000000000220780  Data@ {{.+}} is: 0x2D067281
// CHECK:     Address: 0x0000000000220784  Data@ {{.+}} is: 0xFC81C86D
// CHECK:     Address: 0x0000000000220788  Data@ {{.+}} is: 0x400B20D4
// CHECK:     Address: 0x000000000022078C  Data@ {{.+}} is: 0x76422811
// CHECK:     Address: 0x0000000000220790  Data@ {{.+}} is: 0x2054B938
// CHECK:     Address: 0x0000000000220794  Data@ {{.+}} is: 0x01783399
// CHECK:     Address: 0x0000000000220798  Data@ {{.+}} is: 0x28224023
// CHECK:     Address: 0x000000000022079C  Data@ {{.+}} is: 0x20D57804
// CHECK:     Address: 0x00000000002207A0  Data@ {{.+}} is: 0x0102A9D9
// CHECK:     Address: 0x00000000002207A4  Data@ {{.+}} is: 0x1B81B879
// CHECK:     Address: 0x00000000002207A8  Data@ {{.+}} is: 0x1B81C879
// CHECK:     Address: 0x00000000002207AC  Data@ {{.+}} is: 0x2815C009
// CHECK:     Address: 0x00000000002207B0  Data@ {{.+}} is: 0x1A307659
// CHECK:     Address: 0x00000000002207B4  Data@ {{.+}} is: 0x2829C023
// CHECK:     Address: 0x00000000002207B8  Data@ {{.+}} is: 0x0381C866
// CHECK:     Address: 0x00000000002207BC  Data@ {{.+}} is: 0x09387645
// CHECK:     Address: 0x00000000002207C0  Data@ {{.+}} is: 0xC00B020B
// CHECK:     Address: 0x00000000002207C4  Data@ {{.+}} is: 0xD0622805
// CHECK:     Address: 0x00000000002207C8  Data@ {{.+}} is: 0x4C067501
// CHECK:     Address: 0x00000000002207CC  Data@ {{.+}} is: 0x014C3399
// CHECK:     Address: 0x00000000002207D0  Data@ {{.+}} is: 0x280E8023
// CHECK:     Address: 0x00000000002207D4  Data@ {{.+}} is: 0x8BD4F804
// CHECK:     Address: 0x00000000002207D8  Data@ {{.+}} is: 0x045AA5D9
// CHECK:     Address: 0x00000000002207DC  Data@ {{.+}} is: 0x043E9FD9
// CHECK:     Address: 0x00000000002207E0  Data@ {{.+}} is: 0x043A9DD9
// CHECK:     Address: 0x00000000002207E4  Data@ {{.+}} is: 0x055285D9
// CHECK:     Address: 0x00000000002207E8  Data@ {{.+}} is: 0x7B0E8375
// CHECK:     Address: 0x00000000002207EC  Data@ {{.+}} is: 0x827520D5
// CHECK:     Address: 0x00000000002207F0  Data@ {{.+}} is: 0x20553B0A
// CHECK:     Address: 0x00000000002207F4  Data@ {{.+}} is: 0xFB068175
// CHECK:     Address: 0x00000000002207F8  Data@ {{.+}} is: 0x807583D1
// CHECK:     Address: 0x00000000002207FC  Data@ {{.+}} is: 0x8351BB02
// CHECK:     Address: 0x0000000000220800  Data@ {{.+}} is: 0x020E8399
// CHECK:     Address: 0x0000000000220804  Data@ {{.+}} is: 0x5138764D
// CHECK:     Address: 0x0000000000220808  Data@ {{.+}} is: 0x33994150
// CHECK:     Address: 0x000000000022080C  Data@ {{.+}} is: 0x8199015C
// CHECK:     Address: 0x0000000000220810  Data@ {{.+}} is: 0x80990206
// CHECK:     Address: 0x0000000000220814  Data@ {{.+}} is: 0x8BD90202
// CHECK:     Address: 0x0000000000220818  Data@ {{.+}} is: 0x8FD90106
// CHECK:     Address: 0x000000000022081C  Data@ {{.+}} is: 0x986D047E
// CHECK:     Address: 0x0000000000220820  Data@ {{.+}} is: 0x8F51B981
// CHECK:     Address: 0x0000000000220824  Data@ {{.+}} is: 0x3930164D
// CHECK:     Address: 0x0000000000220828  Data@ {{.+}} is: 0x764D2051
// CHECK:     Address: 0x000000000022082C  Data@ {{.+}} is: 0x2C533938
// CHECK:     Address: 0x0000000000220830  Data@ {{.+}} is: 0x2808C00B
// CHECK:     Address: 0x0000000000220834  Data@ {{.+}} is: 0x7381B862
// CHECK:     Address: 0x0000000000220838  Data@ {{.+}} is: 0xC86D2D86
// CHECK:     Address: 0x000000000022083C  Data@ {{.+}} is: 0x20D4FB81
// CHECK:     Address: 0x0000000000220840  Data@ {{.+}} is: 0x2819C00B
// CHECK:     Address: 0x0000000000220844  Data@ {{.+}} is: 0xB9387642
// CHECK:     Address: 0x0000000000220848  Data@ {{.+}} is: 0x33992054
// CHECK:     Address: 0x000000000022084C  Data@ {{.+}} is: 0xC00B017C
// CHECK:     Address: 0x0000000000220850  Data@ {{.+}} is: 0x98622811
// CHECK:     Address: 0x0000000000220854  Data@ {{.+}} is: 0x20D1FA01
// CHECK:     Address: 0x0000000000220858  Data@ {{.+}} is: 0x01028DD9
// CHECK:     Address: 0x000000000022085C  Data@ {{.+}} is: 0x28210023
// CHECK:     Address: 0x0000000000220860  Data@ {{.+}} is: 0x0201D066
// CHECK:     Address: 0x0000000000220864  Data@ {{.+}} is: 0x0672A1D9
// CHECK:     Address: 0x0000000000220868  Data@ {{.+}} is: 0x28150023
// CHECK:     Address: 0x000000000022086C  Data@ {{.+}} is: 0x01019066
// CHECK:     Address: 0x0000000000220870  Data@ {{.+}} is: 0x80230001
// CHECK:     Address: 0x0000000000220874  Data@ {{.+}} is: 0xC8662804
// CHECK:     Address: 0x0000000000220878  Data@ {{.+}} is: 0x00010101
// CHECK:     Address: 0x000000000022087C  Data@ {{.+}} is: 0x28188023
// CHECK:     Address: 0x0000000000220880  Data@ {{.+}} is: 0x01019866
// CHECK:     Address: 0x0000000000220884  Data@ {{.+}} is: 0x80090001
// CHECK:     Address: 0x0000000000220888  Data@ {{.+}} is: 0x4C992820
// CHECK:     Address: 0x000000000022088C  Data@ {{.+}} is: 0xD6191600
// CHECK:     Address: 0x0000000000220890  Data@ {{.+}} is: 0x01951000
// CHECK:     Address: 0x0000000000220894  Data@ {{.+}} is: 0x0002A040
// CHECK:     Address: 0x0000000000220898  Data@ {{.+}} is: 0x83590001
// CHECK:     Address: 0x000000000022089C  Data@ {{.+}} is: 0x073D0A0E
// CHECK:     Address: 0x00000000002208A0  Data@ {{.+}} is: 0x41504E30
// CHECK:     Address: 0x00000000002208A4  Data@ {{.+}} is: 0x2E022CBD
// CHECK:     Address: 0x00000000002208A8  Data@ {{.+}} is: 0x10BD40D0
// CHECK:     Address: 0x00000000002208AC  Data@ {{.+}} is: 0x40500EF6
// CHECK:     Address: 0x00000000002208B0  Data@ {{.+}} is: 0x000003C0
// CHECK:     Address: 0x00000000002208B4  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002208B8  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002208BC  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002208C0  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002208C4  Data@ {{.+}} is: 0x16002219
// CHECK:     Address: 0x00000000002208C8  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002208CC  Data@ {{.+}} is: 0x22190001
// CHECK:     Address: 0x00000000002208D0  Data@ {{.+}} is: 0x00011660
// CHECK:     Address: 0x00000000002208D4  Data@ {{.+}} is: 0x00950001
// CHECK:     Address: 0x00000000002208D8  Data@ {{.+}} is: 0x00013800
// CHECK:     Address: 0x00000000002208DC  Data@ {{.+}} is: 0x16A02219
// CHECK:     Address: 0x00000000002208E0  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002208E4  Data@ {{.+}} is: 0x00400659
// CHECK:     Address: 0x00000000002208E8  Data@ {{.+}} is: 0x9BCB2843
// CHECK:     Address: 0x00000000002208EC  Data@ {{.+}} is: 0x00025088
// CHECK:     Address: 0x00000000002208F0  Data@ {{.+}} is: 0x0000007F
// CHECK:     Address: 0x00000000002208F4  Data@ {{.+}} is: 0x08000008
// CHECK:     Address: 0x00000000002208F8  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002208FC  Data@ {{.+}} is: 0x00010000
// CHECK:     Address: 0x0000000000220900  Data@ {{.+}} is: 0x38032019
// CHECK:     Address: 0x0000000000220904  Data@ {{.+}} is: 0x0FFC4299
// CHECK:     Address: 0x0000000000220908  Data@ {{.+}} is: 0x011D0001
// CHECK:     Address: 0x000000000022090C  Data@ {{.+}} is: 0x0000D802
// CHECK:     Address: 0x0000000000220910  Data@ {{.+}} is: 0x0000081D
// CHECK:     Address: 0x0000000000220914  Data@ {{.+}} is: 0x00010000
// CHECK:     Address: 0x0000000000220918  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x000000000022091C  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220920  Data@ {{.+}} is: 0x07FC42D9
// CHECK:     Address: 0x0000000000220924  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220928  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x000000000022092C  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220930  Data@ {{.+}} is: 0x10001819
// CHECK:     Address: 0x0000000000220934  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220938  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x000000000022093C  Data@ {{.+}} is: 0x3FFFE019

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
// CHECK:     Address: 0x000000000021D000  Data@ {{.+}} is: 0x00400100
// CHECK:     Address: 0x000000000021D004  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D008  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D00C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D010  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D014  Data@ {{.+}} is: 0x06045FE3

// CHECK: (NOP Command): Payload Length: 2
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x000000000021D020  Size: 6
// CHECK:     Address: 0x000000000021D020  Data@ {{.+}} is: 0x00800200
// CHECK:     Address: 0x000000000021D024  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D028  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D02C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D030  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D034  Data@ {{.+}} is: 0x0E049FE5

// CHECK: (NOP Command): Payload Length: 2
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x000000000021D040  Size: 6
// CHECK:     Address: 0x000000000021D040  Data@ {{.+}} is: 0x01000200
// CHECK:     Address: 0x000000000021D044  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D048  Data@ {{.+}} is: 0x000FE000
// CHECK:     Address: 0x000000000021D04C  Data@ {{.+}} is: 0x00810007
// CHECK:     Address: 0x000000000021D050  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D054  Data@ {{.+}} is: 0x16043FE0

// CHECK: (Write64): Address:  0x000000000021DE04 Data:  0x00010000
// CHECK: (MaskWrite64): Address: 0x000000000021DE00  Mask: 0x00000000  Data: 0x00000001
// CHECK: (Write64): Address:  0x000000000021DE0C Data:  0x00010001
// CHECK: (MaskWrite64): Address: 0x000000000021DE08  Mask: 0x00000000  Data: 0x00000001
// CHECK: (Write64): Address:  0x000000000021DE14 Data:  0x00010002
// CHECK: (MaskWrite64): Address: 0x000000000021DE10  Mask: 0x00000000  Data: 0x00000001
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x00000000041A0000  Size: 8
// CHECK:     Address: 0x00000000041A0000  Data@ {{.+}} is: 0x00000200
// CHECK:     Address: 0x00000000041A0004  Data@ {{.+}} is: 0x000A0000
// CHECK:     Address: 0x00000000041A0008  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000041A000C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000041A0010  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000041A0014  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000041A0018  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000041A001C  Data@ {{.+}} is: 0x8140FF41

// CHECK: (NOP Command): Payload Length: 0
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x00000000041A0020  Size: 8
// CHECK:     Address: 0x00000000041A0020  Data@ {{.+}} is: 0x00000200
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
// CHECK:     Address: 0x00000000001A0000  Data@ {{.+}} is: 0x00000100
// CHECK:     Address: 0x00000000001A0004  Data@ {{.+}} is: 0x000A0000
// CHECK:     Address: 0x00000000001A0008  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000001A000C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000001A0010  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000001A0014  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000001A0018  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000001A001C  Data@ {{.+}} is: 0x8140FF41

// CHECK: (NOP Command): Payload Length: 0
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x00000000001A0020  Size: 8
// CHECK:     Address: 0x00000000001A0020  Data@ {{.+}} is: 0x00000100
// CHECK:     Address: 0x00000000001A0024  Data@ {{.+}} is: 0x001A0000
// CHECK:     Address: 0x00000000001A0028  Data@ {{.+}} is: 0x00040000
// CHECK:     Address: 0x00000000001A002C  Data@ {{.+}} is: 0x0020000F
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
// CHECK:     Address: 0x00000000021A0000  Data@ {{.+}} is: 0x00000200
// CHECK:     Address: 0x00000000021A0004  Data@ {{.+}} is: 0x000A0000
// CHECK:     Address: 0x00000000021A0008  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000021A000C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000021A0010  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000021A0014  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000021A0018  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000021A001C  Data@ {{.+}} is: 0x8140FF41

// CHECK: (NOP Command): Payload Length: 0
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x00000000021A0020  Size: 8
// CHECK:     Address: 0x00000000021A0020  Data@ {{.+}} is: 0x00000200
// CHECK:     Address: 0x00000000021A0024  Data@ {{.+}} is: 0x001A0000
// CHECK:     Address: 0x00000000021A0028  Data@ {{.+}} is: 0x00040000
// CHECK:     Address: 0x00000000021A002C  Data@ {{.+}} is: 0x0040000F
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