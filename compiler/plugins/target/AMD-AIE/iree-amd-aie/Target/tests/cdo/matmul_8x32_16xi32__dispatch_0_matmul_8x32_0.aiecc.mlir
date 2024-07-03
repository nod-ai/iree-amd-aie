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
  %buf5 = aie.buffer(%tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "buf5"} : memref<8x16xi32> 
  %buf4 = aie.buffer(%tile_1_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "buf4"} : memref<16x32xi32> 
  %buf3 = aie.buffer(%tile_2_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "buf3"} : memref<8x32xi32> 
  %buf2 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "buf2"} : memref<2x2x4x8xi32> 
  %buf1 = aie.buffer(%tile_0_2) {address = 1536 : i32, mem_bank = 0 : i32, sym_name = "buf1"} : memref<8x2x8x4xi32> 
  %buf0 = aie.buffer(%tile_0_2) {address = 3584 : i32, mem_bank = 0 : i32, sym_name = "buf0"} : memref<8x2x4x4xi32> 
  %mem_0_2 = aie.mem(%tile_0_2) {
    %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5, repeat_count = 1)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf2 : memref<2x2x4x8xi32>, 0, 128) {bd_id = 0 : i32, next_bd_id = 0 : i32}
    aie.use_lock(%lock_0_2_5, Release, 1)
    aie.next_bd ^bb1
  ^bb2:  // pred: ^bb3
    aie.end
  ^bb3:  // pred: ^bb5
    %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2, repeat_count = 1)
  ^bb4:  // 2 preds: ^bb3, ^bb4
    aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf1 : memref<8x2x8x4xi32>, 0, 512) {bd_id = 1 : i32, next_bd_id = 1 : i32}
    aie.use_lock(%lock_0_2_3, Release, 1)
    aie.next_bd ^bb4
  ^bb5:  // pred: ^bb0
    %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb3, repeat_count = 1)
  ^bb6:  // 2 preds: ^bb5, ^bb6
    aie.use_lock(%lock_0_2_7, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf0 : memref<8x2x4x4xi32>, 0, 256, [<size = 8, stride = 4>, <size = 8, stride = 32>, <size = 4, stride = 1>]) {bd_id = 2 : i32, next_bd_id = 2 : i32}
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
    %c0_i32 = arith.constant 0 : i32
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb22
    aie.use_lock(%c49, AcquireGreaterEqual, 1)
    aie.use_lock(%c50, AcquireGreaterEqual, 1)
    aie.use_lock(%c52, AcquireGreaterEqual, 1)
    cf.br ^bb2(%c0 : index)
  ^bb2(%0: index):  // 2 preds: ^bb1, ^bb9
    %1 = arith.cmpi slt, %0, %c8 : index
    cf.cond_br %1, ^bb3(%c0 : index), ^bb10(%c0 : index)
  ^bb3(%2: index):  // 2 preds: ^bb2, ^bb8
    %3 = arith.cmpi slt, %2, %c2 : index
    cf.cond_br %3, ^bb4(%c0 : index), ^bb9
  ^bb4(%4: index):  // 2 preds: ^bb3, ^bb7
    %5 = arith.cmpi slt, %4, %c4 : index
    cf.cond_br %5, ^bb5(%c0 : index), ^bb8
  ^bb5(%6: index):  // 2 preds: ^bb4, ^bb6
    %7 = arith.cmpi slt, %6, %c4 : index
    cf.cond_br %7, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    memref.store %c0_i32, %buf0[%0, %2, %4, %6] : memref<8x2x4x4xi32>
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
  ^bb10(%12: index):  // 2 preds: ^bb2, ^bb21
    %13 = arith.cmpi slt, %12, %c2 : index
    cf.cond_br %13, ^bb11(%c0 : index), ^bb22
  ^bb11(%14: index):  // 2 preds: ^bb10, ^bb20
    %15 = arith.cmpi slt, %14, %c8 : index
    cf.cond_br %15, ^bb12(%c0 : index), ^bb21
  ^bb12(%16: index):  // 2 preds: ^bb11, ^bb19
    %17 = arith.cmpi slt, %16, %c2 : index
    cf.cond_br %17, ^bb13(%c0 : index), ^bb20
  ^bb13(%18: index):  // 2 preds: ^bb12, ^bb18
    %19 = arith.cmpi slt, %18, %c4 : index
    cf.cond_br %19, ^bb14(%c0 : index), ^bb19
  ^bb14(%20: index):  // 2 preds: ^bb13, ^bb17
    %21 = arith.cmpi slt, %20, %c4 : index
    cf.cond_br %21, ^bb15(%c0 : index), ^bb18
  ^bb15(%22: index):  // 2 preds: ^bb14, ^bb16
    %23 = arith.cmpi slt, %22, %c8 : index
    cf.cond_br %23, ^bb16, ^bb17
  ^bb16:  // pred: ^bb15
    %24 = memref.load %buf2[%16, %12, %18, %22] : memref<2x2x4x8xi32>
    %25 = memref.load %buf1[%14, %16, %22, %20] : memref<8x2x8x4xi32>
    %26 = memref.load %buf0[%14, %12, %18, %20] : memref<8x2x4x4xi32>
    %27 = arith.muli %24, %25 : i32
    %28 = arith.addi %26, %27 : i32
    memref.store %28, %buf0[%14, %12, %18, %20] : memref<8x2x4x4xi32>
    %29 = arith.addi %22, %c1 : index
    cf.br ^bb15(%29 : index)
  ^bb17:  // pred: ^bb15
    %30 = arith.addi %20, %c1 : index
    cf.br ^bb14(%30 : index)
  ^bb18:  // pred: ^bb14
    %31 = arith.addi %18, %c1 : index
    cf.br ^bb13(%31 : index)
  ^bb19:  // pred: ^bb13
    %32 = arith.addi %16, %c1 : index
    cf.br ^bb12(%32 : index)
  ^bb20:  // pred: ^bb12
    %33 = arith.addi %14, %c1 : index
    cf.br ^bb11(%33 : index)
  ^bb21:  // pred: ^bb11
    %34 = arith.addi %12, %c1 : index
    cf.br ^bb10(%34 : index)
  ^bb22:  // pred: ^bb10
    aie.use_lock(%c48, Release, 1)
    aie.use_lock(%c51, Release, 1)
    aie.use_lock(%c53, Release, 1)
    cf.br ^bb1
  } {elf_file = "matmul_8x32_16xi32__dispatch_0_matmul_8x32x16_i32_0_core_0_2.elf"}
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
    aie.dma_bd(%buf3 : memref<8x32xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32}
    aie.use_lock(%lock_2_1_2, Release, 1)
    aie.next_bd ^bb1
  ^bb2:  // pred: ^bb3
    aie.end
  ^bb3:  // pred: ^bb0
    %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb2, repeat_count = 1)
  ^bb4:  // 2 preds: ^bb3, ^bb4
    aie.use_lock(%lock_2_1_2, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf3 : memref<8x32xi32>, 0, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32}
    aie.use_lock(%lock_2_1, Release, 1)
    aie.next_bd ^bb4
  }
  %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
    %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf5 : memref<8x16xi32>, 0, 128) {bd_id = 0 : i32, next_bd_id = 0 : i32}
    aie.use_lock(%lock_0_1_1, Release, 1)
    aie.next_bd ^bb1
  ^bb2:  // pred: ^bb3
    aie.end
  ^bb3:  // pred: ^bb0
    %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb2, repeat_count = 1)
  ^bb4:  // 2 preds: ^bb3, ^bb4
    aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf5 : memref<8x16xi32>, 0, 128, [<size = 2, stride = 8>, <size = 8, stride = 16>, <size = 8, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 1 : i32}
    aie.use_lock(%lock_0_1, Release, 1)
    aie.next_bd ^bb4
  }
  %memtile_dma_1_1 = aie.memtile_dma(%tile_1_1) {
    %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf4 : memref<16x32xi32>, 0, 512) {bd_id = 0 : i32, next_bd_id = 0 : i32}
    aie.use_lock(%lock_1_1_0, Release, 1)
    aie.next_bd ^bb1
  ^bb2:  // pred: ^bb3
    aie.end
  ^bb3:  // pred: ^bb0
    %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb2, repeat_count = 1)
  ^bb4:  // 2 preds: ^bb3, ^bb4
    aie.use_lock(%lock_1_1_0, AcquireGreaterEqual, 1)
    aie.dma_bd(%buf4 : memref<16x32xi32>, 0, 512, [<size = 8, stride = 4>, <size = 16, stride = 32>, <size = 4, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 1 : i32}
    aie.use_lock(%lock_1_1, Release, 1)
    aie.next_bd ^bb4
  }
  aie.shim_dma_allocation @airMemcpyId12(S2MM, 0, 0)
  memref.global "public" @airMemcpyId12 : memref<8x32xi32>
  aie.shim_dma_allocation @airMemcpyId4(MM2S, 0, 0)
  memref.global "public" @airMemcpyId4 : memref<8x16xi32>
  aie.shim_dma_allocation @airMemcpyId5(MM2S, 1, 0)
  memref.global "public" @airMemcpyId5 : memref<16x32xi32>
  func.func @matmul_8x32_16xi32__dispatch_0_matmul_8x32x16_i32(%arg0: memref<8x16xi32>, %arg1: memref<16x32xi32>, %arg2: memref<8x32xi32>) {
    memref.assume_alignment %arg0, 64 : memref<8x16xi32>
    memref.assume_alignment %arg1, 64 : memref<16x32xi32>
    memref.assume_alignment %arg2, 64 : memref<8x32xi32>
    aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 8, 16][0, 0, 16]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<8x16xi32>
    aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 16, 32][0, 0, 32]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<16x32xi32>
    aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 8, 32][0, 0, 32]) {id = 2 : i64, metadata = @airMemcpyId12} : memref<8x32xi32>
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
} {sym_name = "matmul_8x32_16xi32__dispatch_0_matmul_8x32x16_i32_0"}
}


// CHECK: trying XAIE API: XAie_SetupPartitionConfig with args: &devInst={{.+}}, 0x0=0, partitionStartCol=1, partitionNumCols=4
// CHECK: trying XAIE API: XAie_CfgInitialize with args: &devInst={{.+}}, &configPtr
// CHECK: trying XAIE API: XAie_SetIOBackend with args: &devInst={{.+}}, XAIE_IO_BACKEND_CDO=2
// CHECK: trying XAIE API: XAie_UpdateNpiAddr with args: &devInst={{.+}}, 0x0=0
// CHECK: trying XAIE API: XAie_LoadElf with args: &devInst={{.+}}, XAie_TileLoc(col=XAie_LocType(col: 0, row: 2), row)={{.+}}matmul_8x32_16xi32__dispatch_0_matmul_8x32x16_i32_0_core_0_2.elf, elfPath.str().c_str(), aieSim=0
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
// CHECK: trying XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd={{.+}}, basePlusOffsetInBytes=1024, lenInBytes=512
// CHECK: trying XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd={{.+}}, nextBdId.value()=0, enableNextBd=1
// CHECK: trying XAIE API: XAie_DmaEnableBd with args: &dmaTileBd={{.+}}
// CHECK: trying XAIE API: XAie_DmaWriteBd with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), bdId=0

// CHECK: end configuring bds
// CHECK: trying XAIE API: XAie_DmaDescInit with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2)

// CHECK: start configuring bds
// CHECK: trying XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd={{.+}}, acqLock=XAie_Lock(id: 5, val: -1), relLock=XAie_Lock(id: 4, val: 1), acqEn=1, relEn=0
// CHECK: trying XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd={{.+}}, basePlusOffsetInBytes=1536, lenInBytes=2048
// CHECK: trying XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd={{.+}}, nextBdId.value()=1, enableNextBd=1
// CHECK: trying XAIE API: XAie_DmaEnableBd with args: &dmaTileBd={{.+}}
// CHECK: trying XAIE API: XAie_DmaWriteBd with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2), bdId=1

// CHECK: end configuring bds
// CHECK: trying XAIE API: XAie_DmaDescInit with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 0, row: 2)

// CHECK: start configuring bds
// CHECK: trying XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd={{.+}}, acqLock=XAie_Lock(id: 0, val: -1), relLock=XAie_Lock(id: 1, val: 1), acqEn=1, relEn=0
// CHECK: trying XAIE API: XAie_DmaSetMultiDimAddr with args: &dmaTileBd={{.+}}, &dmaTileBdTensor={{.+}}, basePlusOffsetInBytes=3584, lenInBytes=1024
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
// CHECK: trying XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd={{.+}}, basePlusOffsetInBytes=524288, lenInBytes=1024
// CHECK: trying XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd={{.+}}, nextBdId.value()=0, enableNextBd=1
// CHECK: trying XAIE API: XAie_DmaEnableBd with args: &dmaTileBd={{.+}}
// CHECK: trying XAIE API: XAie_DmaWriteBd with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 2, row: 1), bdId=0

// CHECK: end configuring bds
// CHECK: trying XAIE API: XAie_DmaDescInit with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 2, row: 1)

// CHECK: start configuring bds
// CHECK: trying XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd={{.+}}, acqLock=XAie_Lock(id: 64, val: -1), relLock=XAie_Lock(id: 65, val: 1), acqEn=1, relEn=0
// CHECK: trying XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd={{.+}}, basePlusOffsetInBytes=524288, lenInBytes=1024
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
// CHECK: trying XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd={{.+}}, basePlusOffsetInBytes=524288, lenInBytes=512
// CHECK: trying XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd={{.+}}, nextBdId.value()=0, enableNextBd=1
// CHECK: trying XAIE API: XAie_DmaEnableBd with args: &dmaTileBd={{.+}}
// CHECK: trying XAIE API: XAie_DmaWriteBd with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 0, row: 1), bdId=0

// CHECK: end configuring bds
// CHECK: trying XAIE API: XAie_DmaDescInit with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 0, row: 1)

// CHECK: start configuring bds
// CHECK: trying XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd={{.+}}, acqLock=XAie_Lock(id: 64, val: -1), relLock=XAie_Lock(id: 65, val: 1), acqEn=1, relEn=0
// CHECK: trying XAIE API: XAie_DmaSetMultiDimAddr with args: &dmaTileBd={{.+}}, &dmaTileBdTensor={{.+}}, basePlusOffsetInBytes=524288, lenInBytes=512
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
// CHECK: trying XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd={{.+}}, basePlusOffsetInBytes=524288, lenInBytes=2048
// CHECK: trying XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd={{.+}}, nextBdId.value()=0, enableNextBd=1
// CHECK: trying XAIE API: XAie_DmaEnableBd with args: &dmaTileBd={{.+}}
// CHECK: trying XAIE API: XAie_DmaWriteBd with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 1, row: 1), bdId=0

// CHECK: end configuring bds
// CHECK: trying XAIE API: XAie_DmaDescInit with args: &devInst={{.+}}, &dmaTileBd={{.+}}, tileLoc=XAie_LocType(col: 1, row: 1)

// CHECK: start configuring bds
// CHECK: trying XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd={{.+}}, acqLock=XAie_Lock(id: 64, val: -1), relLock=XAie_Lock(id: 65, val: 1), acqEn=1, relEn=0
// CHECK: trying XAIE API: XAie_DmaSetMultiDimAddr with args: &dmaTileBd={{.+}}, &dmaTileBdTensor={{.+}}, basePlusOffsetInBytes=524288, lenInBytes=2048
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
// CHECK: Generating: {{.+}}aie_cdo_elfs.bin
// CHECK: Generating: {{.+}}aie_cdo_init.bin
// CHECK: Generating: {{.+}}aie_cdo_enable.bin
// CHECK: Generating: {{.+}}aie_cdo_elfs.bin
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

// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x0000000000220200  Size: 492
// CHECK:     Address: 0x0000000000220200  Data@ {{.+}} is: 0x3803A019
// CHECK:     Address: 0x0000000000220204  Data@ {{.+}} is: 0x0FEEE819
// CHECK:     Address: 0x0000000000220208  Data@ {{.+}} is: 0x0FEDEC19
// CHECK:     Address: 0x000000000022020C  Data@ {{.+}} is: 0x0FED6E19
// CHECK:     Address: 0x0000000000220210  Data@ {{.+}} is: 0x0FEE6A19
// CHECK:     Address: 0x0000000000220214  Data@ {{.+}} is: 0x00061D7B
// CHECK:     Address: 0x0000000000220218  Data@ {{.+}} is: 0x8FF0E000
// CHECK:     Address: 0x000000000022021C  Data@ {{.+}} is: 0x217B0087
// CHECK:     Address: 0x0000000000220220  Data@ {{.+}} is: 0x6200003E
// CHECK:     Address: 0x0000000000220224  Data@ {{.+}} is: 0x02064FF0
// CHECK:     Address: 0x0000000000220228  Data@ {{.+}} is: 0x1F00113B
// CHECK:     Address: 0x000000000022022C  Data@ {{.+}} is: 0x800001C0
// CHECK:     Address: 0x0000000000220230  Data@ {{.+}} is: 0x113BFDFC
// CHECK:     Address: 0x0000000000220234  Data@ {{.+}} is: 0x01C11A00
// CHECK:     Address: 0x0000000000220238  Data@ {{.+}} is: 0xFDECC000
// CHECK:     Address: 0x000000000022023C  Data@ {{.+}} is: 0x0028317B
// CHECK:     Address: 0x0000000000220240  Data@ {{.+}} is: 0x8FECF680
// CHECK:     Address: 0x0000000000220244  Data@ {{.+}} is: 0x617B0285
// CHECK:     Address: 0x0000000000220248  Data@ {{.+}} is: 0x7E80002E
// CHECK:     Address: 0x000000000022024C  Data@ {{.+}} is: 0x03854FEC
// CHECK:     Address: 0x0000000000220250  Data@ {{.+}} is: 0x07FF0059
// CHECK:     Address: 0x0000000000220254  Data@ {{.+}} is: 0x16220219
// CHECK:     Address: 0x0000000000220258  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x000000000022025C  Data@ {{.+}} is: 0x02190001
// CHECK:     Address: 0x0000000000220260  Data@ {{.+}} is: 0x00011642
// CHECK:     Address: 0x0000000000220264  Data@ {{.+}} is: 0x36590001
// CHECK:     Address: 0x0000000000220268  Data@ {{.+}} is: 0x021D0000
// CHECK:     Address: 0x000000000022026C  Data@ {{.+}} is: 0x00000E82
// CHECK:     Address: 0x0000000000220270  Data@ {{.+}} is: 0x000388BB
// CHECK:     Address: 0x0000000000220274  Data@ {{.+}} is: 0x000011EC
// CHECK:     Address: 0x0000000000220278  Data@ {{.+}} is: 0xE5990000
// CHECK:     Address: 0x000000000022027C  Data@ {{.+}} is: 0xF5BD1045
// CHECK:     Address: 0x0000000000220280  Data@ {{.+}} is: 0xFE288045
// CHECK:     Address: 0x0000000000220284  Data@ {{.+}} is: 0x004545FB
// CHECK:     Address: 0x0000000000220288  Data@ {{.+}} is: 0x2FF1C400
// CHECK:     Address: 0x000000000022028C  Data@ {{.+}} is: 0x4435FE2C
// CHECK:     Address: 0x0000000000220290  Data@ {{.+}} is: 0xFE3C2FF2
// CHECK:     Address: 0x0000000000220294  Data@ {{.+}} is: 0x07F26159
// CHECK:     Address: 0x0000000000220298  Data@ {{.+}} is: 0x96590001
// CHECK:     Address: 0x000000000022029C  Data@ {{.+}} is: 0x2843190D
// CHECK:     Address: 0x00000000002202A0  Data@ {{.+}} is: 0xB00A104B
// CHECK:     Address: 0x00000000002202A4  Data@ {{.+}} is: 0xB6190001
// CHECK:     Address: 0x00000000002202A8  Data@ {{.+}} is: 0xB6190810
// CHECK:     Address: 0x00000000002202AC  Data@ {{.+}} is: 0xB6190810
// CHECK:     Address: 0x00000000002202B0  Data@ {{.+}} is: 0x95BD0810
// CHECK:     Address: 0x00000000002202B4  Data@ {{.+}} is: 0x0216C04B
// CHECK:     Address: 0x00000000002202B8  Data@ {{.+}} is: 0x1C229659
// CHECK:     Address: 0x00000000002202BC  Data@ {{.+}} is: 0xC04D65BD
// CHECK:     Address: 0x00000000002202C0  Data@ {{.+}} is: 0x28430216
// CHECK:     Address: 0x00000000002202C4  Data@ {{.+}} is: 0x7BAE118B
// CHECK:     Address: 0x00000000002202C8  Data@ {{.+}} is: 0x55BD0002
// CHECK:     Address: 0x00000000002202CC  Data@ {{.+}} is: 0x0216C051
// CHECK:     Address: 0x00000000002202D0  Data@ {{.+}} is: 0x11CB2843
// CHECK:     Address: 0x00000000002202D4  Data@ {{.+}} is: 0x0003F60A
// CHECK:     Address: 0x00000000002202D8  Data@ {{.+}} is: 0xC063F5BD
// CHECK:     Address: 0x00000000002202DC  Data@ {{.+}} is: 0x28430216
// CHECK:     Address: 0x00000000002202E0  Data@ {{.+}} is: 0x240A120B
// CHECK:     Address: 0x00000000002202E4  Data@ {{.+}} is: 0x25BD0000
// CHECK:     Address: 0x00000000002202E8  Data@ {{.+}} is: 0x0216C052
// CHECK:     Address: 0x00000000002202EC  Data@ {{.+}} is: 0x124B2843
// CHECK:     Address: 0x00000000002202F0  Data@ {{.+}} is: 0x0000248A
// CHECK:     Address: 0x00000000002202F4  Data@ {{.+}} is: 0xC05425BD
// CHECK:     Address: 0x00000000002202F8  Data@ {{.+}} is: 0x28430216
// CHECK:     Address: 0x00000000002202FC  Data@ {{.+}} is: 0x250A128B
// CHECK:     Address: 0x0000000000220300  Data@ {{.+}} is: 0x25BD0000
// CHECK:     Address: 0x0000000000220304  Data@ {{.+}} is: 0x0216C056
// CHECK:     Address: 0x0000000000220308  Data@ {{.+}} is: 0x12CB2843
// CHECK:     Address: 0x000000000022030C  Data@ {{.+}} is: 0x0000258A
// CHECK:     Address: 0x0000000000220310  Data@ {{.+}} is: 0xC05825BD
// CHECK:     Address: 0x0000000000220314  Data@ {{.+}} is: 0x28430216
// CHECK:     Address: 0x0000000000220318  Data@ {{.+}} is: 0x260A130B
// CHECK:     Address: 0x000000000022031C  Data@ {{.+}} is: 0x25FB0000
// CHECK:     Address: 0x0000000000220320  Data@ {{.+}} is: 0xB600005A
// CHECK:     Address: 0x0000000000220324  Data@ {{.+}} is: 0x0E87C810
// CHECK:     Address: 0x0000000000220328  Data@ {{.+}} is: 0x134B2843
// CHECK:     Address: 0x000000000022032C  Data@ {{.+}} is: 0x00030FAE
// CHECK:     Address: 0x0000000000220330  Data@ {{.+}} is: 0x007EE17B
// CHECK:     Address: 0x0000000000220334  Data@ {{.+}} is: 0x8810B600
// CHECK:     Address: 0x0000000000220338  Data@ {{.+}} is: 0x28430680
// CHECK:     Address: 0x000000000022033C  Data@ {{.+}} is: 0xE12F610B
// CHECK:     Address: 0x0000000000220340  Data@ {{.+}} is: 0x28430002
// CHECK:     Address: 0x0000000000220344  Data@ {{.+}} is: 0x3FAE138B
// CHECK:     Address: 0x0000000000220348  Data@ {{.+}} is: 0xE11D0003
// CHECK:     Address: 0x000000000022034C  Data@ {{.+}} is: 0x00010804
// CHECK:     Address: 0x0000000000220350  Data@ {{.+}} is: 0x005E25BD
// CHECK:     Address: 0x0000000000220354  Data@ {{.+}} is: 0x28430211
// CHECK:     Address: 0x0000000000220358  Data@ {{.+}} is: 0x278A13CB
// CHECK:     Address: 0x000000000022035C  Data@ {{.+}} is: 0x25BD0000
// CHECK:     Address: 0x0000000000220360  Data@ {{.+}} is: 0x02110070
// CHECK:     Address: 0x0000000000220364  Data@ {{.+}} is: 0x160B2843
// CHECK:     Address: 0x0000000000220368  Data@ {{.+}} is: 0x0002200A
// CHECK:     Address: 0x000000000022036C  Data@ {{.+}} is: 0x007225BD
// CHECK:     Address: 0x0000000000220370  Data@ {{.+}} is: 0x28430211
// CHECK:     Address: 0x0000000000220374  Data@ {{.+}} is: 0x208A164B
// CHECK:     Address: 0x0000000000220378  Data@ {{.+}} is: 0x25BD0002
// CHECK:     Address: 0x000000000022037C  Data@ {{.+}} is: 0x02110074
// CHECK:     Address: 0x0000000000220380  Data@ {{.+}} is: 0x168B2843
// CHECK:     Address: 0x0000000000220384  Data@ {{.+}} is: 0x0002210A
// CHECK:     Address: 0x0000000000220388  Data@ {{.+}} is: 0x007825BD
// CHECK:     Address: 0x000000000022038C  Data@ {{.+}} is: 0x28430211
// CHECK:     Address: 0x0000000000220390  Data@ {{.+}} is: 0x218A170B
// CHECK:     Address: 0x0000000000220394  Data@ {{.+}} is: 0x25BD0002
// CHECK:     Address: 0x0000000000220398  Data@ {{.+}} is: 0x0211007A
// CHECK:     Address: 0x000000000022039C  Data@ {{.+}} is: 0x174B2843
// CHECK:     Address: 0x00000000002203A0  Data@ {{.+}} is: 0x0002220A
// CHECK:     Address: 0x00000000002203A4  Data@ {{.+}} is: 0x007C25BD
// CHECK:     Address: 0x00000000002203A8  Data@ {{.+}} is: 0x28430211
// CHECK:     Address: 0x00000000002203AC  Data@ {{.+}} is: 0x228A178B
// CHECK:     Address: 0x00000000002203B0  Data@ {{.+}} is: 0x25BD0002
// CHECK:     Address: 0x00000000002203B4  Data@ {{.+}} is: 0x02110064
// CHECK:     Address: 0x00000000002203B8  Data@ {{.+}} is: 0x148B2843
// CHECK:     Address: 0x00000000002203BC  Data@ {{.+}} is: 0x0002230A
// CHECK:     Address: 0x00000000002203C0  Data@ {{.+}} is: 0x006825BD
// CHECK:     Address: 0x00000000002203C4  Data@ {{.+}} is: 0x28430211
// CHECK:     Address: 0x00000000002203C8  Data@ {{.+}} is: 0x238A150B
// CHECK:     Address: 0x00000000002203CC  Data@ {{.+}} is: 0x25BD0002
// CHECK:     Address: 0x00000000002203D0  Data@ {{.+}} is: 0x0211006A
// CHECK:     Address: 0x00000000002203D4  Data@ {{.+}} is: 0x154B2843
// CHECK:     Address: 0x00000000002203D8  Data@ {{.+}} is: 0x0002240A
// CHECK:     Address: 0x00000000002203DC  Data@ {{.+}} is: 0x006C25BD
// CHECK:     Address: 0x00000000002203E0  Data@ {{.+}} is: 0x28430211
// CHECK:     Address: 0x00000000002203E4  Data@ {{.+}} is: 0x248A158B
// CHECK:     Address: 0x00000000002203E8  Data@ {{.+}} is: 0x25BD0002
// CHECK:     Address: 0x00000000002203EC  Data@ {{.+}} is: 0x0211006E
// CHECK:     Address: 0x00000000002203F0  Data@ {{.+}} is: 0x15CB2843
// CHECK:     Address: 0x00000000002203F4  Data@ {{.+}} is: 0x0002250A
// CHECK:     Address: 0x00000000002203F8  Data@ {{.+}} is: 0x004625BD
// CHECK:     Address: 0x00000000002203FC  Data@ {{.+}} is: 0x28430211
// CHECK:     Address: 0x0000000000220400  Data@ {{.+}} is: 0x258A10CB
// CHECK:     Address: 0x0000000000220404  Data@ {{.+}} is: 0x25BD0002
// CHECK:     Address: 0x0000000000220408  Data@ {{.+}} is: 0x02110044
// CHECK:     Address: 0x000000000022040C  Data@ {{.+}} is: 0x108B2843
// CHECK:     Address: 0x0000000000220410  Data@ {{.+}} is: 0x0003F78A
// CHECK:     Address: 0x0000000000220414  Data@ {{.+}} is: 0x007FF5FB
// CHECK:     Address: 0x0000000000220418  Data@ {{.+}} is: 0xC8108800
// CHECK:     Address: 0x000000000022041C  Data@ {{.+}} is: 0x28430023
// CHECK:     Address: 0x0000000000220420  Data@ {{.+}} is: 0x30EA144B
// CHECK:     Address: 0x0000000000220424  Data@ {{.+}} is: 0x3CFB0000
// CHECK:     Address: 0x0000000000220428  Data@ {{.+}} is: 0x88000002
// CHECK:     Address: 0x000000000022042C  Data@ {{.+}} is: 0x02064810
// CHECK:     Address: 0x0000000000220430  Data@ {{.+}} is: 0x140B2843
// CHECK:     Address: 0x0000000000220434  Data@ {{.+}} is: 0x00021232
// CHECK:     Address: 0x0000000000220438  Data@ {{.+}} is: 0x100060BB
// CHECK:     Address: 0x000000000022043C  Data@ {{.+}} is: 0x8802004E
// CHECK:     Address: 0x0000000000220440  Data@ {{.+}} is: 0x317B0087
// CHECK:     Address: 0x0000000000220444  Data@ {{.+}} is: 0x88000028
// CHECK:     Address: 0x0000000000220448  Data@ {{.+}} is: 0x03854810
// CHECK:     Address: 0x000000000022044C  Data@ {{.+}} is: 0x14CB2843
// CHECK:     Address: 0x0000000000220450  Data@ {{.+}} is: 0x0001628A
// CHECK:     Address: 0x0000000000220454  Data@ {{.+}} is: 0x0000077B
// CHECK:     Address: 0x0000000000220458  Data@ {{.+}} is: 0xC8108800
// CHECK:     Address: 0x000000000022045C  Data@ {{.+}} is: 0x28430305
// CHECK:     Address: 0x0000000000220460  Data@ {{.+}} is: 0x27E617CB
// CHECK:     Address: 0x0000000000220464  Data@ {{.+}} is: 0x20FB0000
// CHECK:     Address: 0x0000000000220468  Data@ {{.+}} is: 0x880006F6
// CHECK:     Address: 0x000000000022046C  Data@ {{.+}} is: 0x0107C810
// CHECK:     Address: 0x0000000000220470  Data@ {{.+}} is: 0x000388BB
// CHECK:     Address: 0x0000000000220474  Data@ {{.+}} is: 0x48000028
// CHECK:     Address: 0x0000000000220478  Data@ {{.+}} is: 0x091D0000
// CHECK:     Address: 0x000000000022047C  Data@ {{.+}} is: 0x00674830
// CHECK:     Address: 0x0000000000220480  Data@ {{.+}} is: 0x000000BB
// CHECK:     Address: 0x0000000000220484  Data@ {{.+}} is: 0x88000000
// CHECK:     Address: 0x0000000000220488  Data@ {{.+}} is: 0x2DBD00A0
// CHECK:     Address: 0x000000000022048C  Data@ {{.+}} is: 0xFFB80044
// CHECK:     Address: 0x0000000000220490  Data@ {{.+}} is: 0x8043EDBD
// CHECK:     Address: 0x0000000000220494  Data@ {{.+}} is: 0x017BFF78
// CHECK:     Address: 0x0000000000220498  Data@ {{.+}} is: 0x42000036
// CHECK:     Address: 0x000000000022049C  Data@ {{.+}} is: 0x00008FFD
// CHECK:     Address: 0x00000000002204A0  Data@ {{.+}} is: 0x00000013
// CHECK:     Address: 0x00000000002204A4  Data@ {{.+}} is: 0xFFA84800
// CHECK:     Address: 0x00000000002204A8  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002204AC  Data@ {{.+}} is: 0x00060059
// CHECK:     Address: 0x00000000002204B0  Data@ {{.+}} is: 0x10800D99
// CHECK:     Address: 0x00000000002204B4  Data@ {{.+}} is: 0x0FFAC035
// CHECK:     Address: 0x00000000002204B8  Data@ {{.+}} is: 0x0DBD00A0
// CHECK:     Address: 0x00000000002204BC  Data@ {{.+}} is: 0xFF9EC080
// CHECK:     Address: 0x00000000002204C0  Data@ {{.+}} is: 0x800015BD
// CHECK:     Address: 0x00000000002204C4  Data@ {{.+}} is: 0x017BFF88
// CHECK:     Address: 0x00000000002204C8  Data@ {{.+}} is: 0x4000001A
// CHECK:     Address: 0x00000000002204CC  Data@ {{.+}} is: 0x00200FFB
// CHECK:     Address: 0x00000000002204D0  Data@ {{.+}} is: 0x000003C0
// CHECK:     Address: 0x00000000002204D4  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002204D8  Data@ {{.+}} is: 0x88000000
// CHECK:     Address: 0x00000000002204DC  Data@ {{.+}} is: 0x0000FF58
// CHECK:     Address: 0x00000000002204E0  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002204E4  Data@ {{.+}} is: 0x02590001
// CHECK:     Address: 0x00000000002204E8  Data@ {{.+}} is: 0x00550005
// CHECK:     Address: 0x00000000002204EC  Data@ {{.+}} is: 0x0007026C
// CHECK:     Address: 0x00000000002204F0  Data@ {{.+}} is: 0x13421D99
// CHECK:     Address: 0x00000000002204F4  Data@ {{.+}} is: 0x10442599
// CHECK:     Address: 0x00000000002204F8  Data@ {{.+}} is: 0x10838D99
// CHECK:     Address: 0x00000000002204FC  Data@ {{.+}} is: 0x1C209659
// CHECK:     Address: 0x0000000000220500  Data@ {{.+}} is: 0x01108259
// CHECK:     Address: 0x0000000000220504  Data@ {{.+}} is: 0x1B327659
// CHECK:     Address: 0x0000000000220508  Data@ {{.+}} is: 0x3B039019
// CHECK:     Address: 0x000000000022050C  Data@ {{.+}} is: 0x03108259
// CHECK:     Address: 0x0000000000220510  Data@ {{.+}} is: 0x19067659
// CHECK:     Address: 0x0000000000220514  Data@ {{.+}} is: 0x1B327659
// CHECK:     Address: 0x0000000000220518  Data@ {{.+}} is: 0x3B071019
// CHECK:     Address: 0x000000000022051C  Data@ {{.+}} is: 0x19467659
// CHECK:     Address: 0x0000000000220520  Data@ {{.+}} is: 0x4B32764D
// CHECK:     Address: 0x0000000000220524  Data@ {{.+}} is: 0x42056210
// CHECK:     Address: 0x0000000000220528  Data@ {{.+}} is: 0x0B0797F1
// CHECK:     Address: 0x000000000022052C  Data@ {{.+}} is: 0x19867659
// CHECK:     Address: 0x0000000000220530  Data@ {{.+}} is: 0x4B32764D
// CHECK:     Address: 0x0000000000220534  Data@ {{.+}} is: 0x76456210
// CHECK:     Address: 0x0000000000220538  Data@ {{.+}} is: 0x030B1732
// CHECK:     Address: 0x000000000022053C  Data@ {{.+}} is: 0x4C32764D
// CHECK:     Address: 0x0000000000220540  Data@ {{.+}} is: 0xC2056210
// CHECK:     Address: 0x0000000000220544  Data@ {{.+}} is: 0x0F0B97F1
// CHECK:     Address: 0x0000000000220548  Data@ {{.+}} is: 0x4C0F107D
// CHECK:     Address: 0x000000000022054C  Data@ {{.+}} is: 0x8259E210
// CHECK:     Address: 0x0000000000220550  Data@ {{.+}} is: 0x42190410
// CHECK:     Address: 0x0000000000220554  Data@ {{.+}} is: 0x76590FF2
// CHECK:     Address: 0x0000000000220558  Data@ {{.+}} is: 0xC2191E36
// CHECK:     Address: 0x000000000022055C  Data@ {{.+}} is: 0x76590FF2
// CHECK:     Address: 0x0000000000220560  Data@ {{.+}} is: 0x42191B32
// CHECK:     Address: 0x0000000000220564  Data@ {{.+}} is: 0xC2190FF3
// CHECK:     Address: 0x0000000000220568  Data@ {{.+}} is: 0xF5BD0FF3
// CHECK:     Address: 0x000000000022056C  Data@ {{.+}} is: 0xFE884082
// CHECK:     Address: 0x0000000000220570  Data@ {{.+}} is: 0xE33B283B
// CHECK:     Address: 0x0000000000220574  Data@ {{.+}} is: 0x90021C6C
// CHECK:     Address: 0x0000000000220578  Data@ {{.+}} is: 0x964D010F
// CHECK:     Address: 0x000000000022057C  Data@ {{.+}} is: 0x2210CC20
// CHECK:     Address: 0x0000000000220580  Data@ {{.+}} is: 0x4B32164D
// CHECK:     Address: 0x0000000000220584  Data@ {{.+}} is: 0x964D6210
// CHECK:     Address: 0x0000000000220588  Data@ {{.+}} is: 0x62104D32
// CHECK:     Address: 0x000000000022058C  Data@ {{.+}} is: 0x4D33164D
// CHECK:     Address: 0x0000000000220590  Data@ {{.+}} is: 0x8259A210
// CHECK:     Address: 0x0000000000220594  Data@ {{.+}} is: 0x82590510
// CHECK:     Address: 0x0000000000220598  Data@ {{.+}} is: 0x82590610
// CHECK:     Address: 0x000000000022059C  Data@ {{.+}} is: 0x82590710
// CHECK:     Address: 0x00000000002205A0  Data@ {{.+}} is: 0x42190410
// CHECK:     Address: 0x00000000002205A4  Data@ {{.+}} is: 0xC2190FF5
// CHECK:     Address: 0x00000000002205A8  Data@ {{.+}} is: 0x42190FF5
// CHECK:     Address: 0x00000000002205AC  Data@ {{.+}} is: 0xC2190FF6
// CHECK:     Address: 0x00000000002205B0  Data@ {{.+}} is: 0x42190FF6
// CHECK:     Address: 0x00000000002205B4  Data@ {{.+}} is: 0xC2190FF7
// CHECK:     Address: 0x00000000002205B8  Data@ {{.+}} is: 0x85FB0FF7
// CHECK:     Address: 0x00000000002205BC  Data@ {{.+}} is: 0x42000083
// CHECK:     Address: 0x00000000002205C0  Data@ {{.+}} is: 0x2210CFF8
// CHECK:     Address: 0x00000000002205C4  Data@ {{.+}} is: 0x88438D9D
// CHECK:     Address: 0x00000000002205C8  Data@ {{.+}} is: 0x9659FF6E
// CHECK:     Address: 0x00000000002205CC  Data@ {{.+}} is: 0x764D1C20
// CHECK:     Address: 0x00000000002205D0  Data@ {{.+}} is: 0xA214090C
// CHECK:     Address: 0x00000000002205D4  Data@ {{.+}} is: 0x1E339659
// CHECK:     Address: 0x00000000002205D8  Data@ {{.+}} is: 0x06108259
// CHECK:     Address: 0x00000000002205DC  Data@ {{.+}} is: 0x0FF4C619
// CHECK:     Address: 0x00000000002205E0  Data@ {{.+}} is: 0x4FF8C635
// CHECK:     Address: 0x00000000002205E4  Data@ {{.+}} is: 0x7659E212
// CHECK:     Address: 0x00000000002205E8  Data@ {{.+}} is: 0x945918C6
// CHECK:     Address: 0x00000000002205EC  Data@ {{.+}} is: 0x164D0410
// CHECK:     Address: 0x00000000002205F0  Data@ {{.+}} is: 0x2212CD32
// CHECK:     Address: 0x00000000002205F4  Data@ {{.+}} is: 0x03108259
// CHECK:     Address: 0x00000000002205F8  Data@ {{.+}} is: 0x994B2843
// CHECK:     Address: 0x00000000002205FC  Data@ {{.+}} is: 0x00042EAD
// CHECK:     Address: 0x0000000000220600  Data@ {{.+}} is: 0x08858D9D
// CHECK:     Address: 0x0000000000220604  Data@ {{.+}} is: 0x164D6217
// CHECK:     Address: 0x0000000000220608  Data@ {{.+}} is: 0xA2104C21
// CHECK:     Address: 0x000000000022060C  Data@ {{.+}} is: 0x06109859
// CHECK:     Address: 0x0000000000220610  Data@ {{.+}} is: 0x42190001
// CHECK:     Address: 0x0000000000220614  Data@ {{.+}} is: 0x164D0FF9
// CHECK:     Address: 0x0000000000220618  Data@ {{.+}} is: 0xFF7B4943
// CHECK:     Address: 0x000000000022061C  Data@ {{.+}} is: 0x03108C59
// CHECK:     Address: 0x0000000000220620  Data@ {{.+}} is: 0x0FF9C235
// CHECK:     Address: 0x0000000000220624  Data@ {{.+}} is: 0x011DA212
// CHECK:     Address: 0x0000000000220628  Data@ {{.+}} is: 0xE2114826
// CHECK:     Address: 0x000000000022062C  Data@ {{.+}} is: 0x18CB28BB
// CHECK:     Address: 0x0000000000220630  Data@ {{.+}} is: 0x0801B00B
// CHECK:     Address: 0x0000000000220634  Data@ {{.+}} is: 0x28BB8211
// CHECK:     Address: 0x0000000000220638  Data@ {{.+}} is: 0x30C9994B
// CHECK:     Address: 0x000000000022063C  Data@ {{.+}} is: 0x22144800
// CHECK:     Address: 0x0000000000220640  Data@ {{.+}} is: 0x8B463D9D
// CHECK:     Address: 0x0000000000220644  Data@ {{.+}} is: 0xD5FBC210
// CHECK:     Address: 0x0000000000220648  Data@ {{.+}} is: 0x400000E4
// CHECK:     Address: 0x000000000022064C  Data@ {{.+}} is: 0x6211CFFA
// CHECK:     Address: 0x0000000000220650  Data@ {{.+}} is: 0x14C1DD99
// CHECK:     Address: 0x0000000000220654  Data@ {{.+}} is: 0x14800099
// CHECK:     Address: 0x0000000000220658  Data@ {{.+}} is: 0x101D8D99
// CHECK:     Address: 0x000000000022065C  Data@ {{.+}} is: 0x14C18D99
// CHECK:     Address: 0x0000000000220660  Data@ {{.+}} is: 0x139B9599
// CHECK:     Address: 0x0000000000220664  Data@ {{.+}} is: 0x16800099
// CHECK:     Address: 0x0000000000220668  Data@ {{.+}} is: 0x13BBF599
// CHECK:     Address: 0x000000000022066C  Data@ {{.+}} is: 0x201B48BB
// CHECK:     Address: 0x0000000000220670  Data@ {{.+}} is: 0x88003C6F
// CHECK:     Address: 0x0000000000220674  Data@ {{.+}} is: 0x2843FE2F
// CHECK:     Address: 0x0000000000220678  Data@ {{.+}} is: 0x8A2E138B
// CHECK:     Address: 0x000000000022067C  Data@ {{.+}} is: 0x48BB001D
// CHECK:     Address: 0x0000000000220680  Data@ {{.+}} is: 0x4F2C003B
// CHECK:     Address: 0x0000000000220684  Data@ {{.+}} is: 0x4215881D
// CHECK:     Address: 0x0000000000220688  Data@ {{.+}} is: 0x10CB2843
// CHECK:     Address: 0x000000000022068C  Data@ {{.+}} is: 0x001D502E
// CHECK:     Address: 0x0000000000220690  Data@ {{.+}} is: 0x003348BB
// CHECK:     Address: 0x0000000000220694  Data@ {{.+}} is: 0xC806FF2C
// CHECK:     Address: 0x0000000000220698  Data@ {{.+}} is: 0x28BB0210
// CHECK:     Address: 0x000000000022069C  Data@ {{.+}} is: 0x702E350B
// CHECK:     Address: 0x00000000002206A0  Data@ {{.+}} is: 0xFE3F881D
// CHECK:     Address: 0x00000000002206A4  Data@ {{.+}} is: 0xCB81659D
// CHECK:     Address: 0x00000000002206A8  Data@ {{.+}} is: 0x28434610
// CHECK:     Address: 0x00000000002206AC  Data@ {{.+}} is: 0xE10A374B
// CHECK:     Address: 0x00000000002206B0  Data@ {{.+}} is: 0xE59D0000
// CHECK:     Address: 0x00000000002206B4  Data@ {{.+}} is: 0x461748FE
// CHECK:     Address: 0x00000000002206B8  Data@ {{.+}} is: 0x360B28BB
// CHECK:     Address: 0x00000000002206BC  Data@ {{.+}} is: 0x0806ECAE
// CHECK:     Address: 0x00000000002206C0  Data@ {{.+}} is: 0x6F9DFE4D
// CHECK:     Address: 0x00000000002206C4  Data@ {{.+}} is: 0x46160FA9
// CHECK:     Address: 0x00000000002206C8  Data@ {{.+}} is: 0x409D0001
// CHECK:     Address: 0x00000000002206CC  Data@ {{.+}} is: 0xFE5D08E9
// CHECK:     Address: 0x00000000002206D0  Data@ {{.+}} is: 0x334B2843
// CHECK:     Address: 0x00000000002206D4  Data@ {{.+}} is: 0x003DE1FE
// CHECK:     Address: 0x00000000002206D8  Data@ {{.+}} is: 0x02309A59
// CHECK:     Address: 0x00000000002206DC  Data@ {{.+}} is: 0x0D3DE09D
// CHECK:     Address: 0x00000000002206E0  Data@ {{.+}} is: 0xDF99FE6D
// CHECK:     Address: 0x00000000002206E4  Data@ {{.+}} is: 0x16591529
// CHECK:     Address: 0x00000000002206E8  Data@ {{.+}} is: 0x40991C60
// CHECK:     Address: 0x00000000002206EC  Data@ {{.+}} is: 0x8F9D17BD
// CHECK:     Address: 0x00000000002206F0  Data@ {{.+}} is: 0x46178D29
// CHECK:     Address: 0x00000000002206F4  Data@ {{.+}} is: 0x07F3E859
// CHECK:     Address: 0x00000000002206F8  Data@ {{.+}} is: 0x17BD4099
// CHECK:     Address: 0x00000000002206FC  Data@ {{.+}} is: 0xCC6B964D
// CHECK:     Address: 0x0000000000220700  Data@ {{.+}} is: 0xDF9DFE8D
// CHECK:     Address: 0x0000000000220704  Data@ {{.+}} is: 0x46150D28
// CHECK:     Address: 0x0000000000220708  Data@ {{.+}} is: 0x1C6A9659
// CHECK:     Address: 0x000000000022070C  Data@ {{.+}} is: 0x4F81409D
// CHECK:     Address: 0x0000000000220710  Data@ {{.+}} is: 0xEE594615
// CHECK:     Address: 0x0000000000220714  Data@ {{.+}} is: 0xEF9907F4
// CHECK:     Address: 0x0000000000220718  Data@ {{.+}} is: 0x00011529
// CHECK:     Address: 0x000000000022071C  Data@ {{.+}} is: 0x10014099
// CHECK:     Address: 0x0000000000220720  Data@ {{.+}} is: 0x15EF4F99
// CHECK:     Address: 0x0000000000220724  Data@ {{.+}} is: 0x70990001
// CHECK:     Address: 0x0000000000220728  Data@ {{.+}} is: 0x5F991001
// CHECK:     Address: 0x000000000022072C  Data@ {{.+}} is: 0x000115EF
// CHECK:     Address: 0x0000000000220730  Data@ {{.+}} is: 0x10017099
// CHECK:     Address: 0x0000000000220734  Data@ {{.+}} is: 0xC8108035
// CHECK:     Address: 0x0000000000220738  Data@ {{.+}} is: 0x9659FEAB
// CHECK:     Address: 0x000000000022073C  Data@ {{.+}} is: 0x80591C27
// CHECK:     Address: 0x0000000000220740  Data@ {{.+}} is: 0xDE590010
// CHECK:     Address: 0x0000000000220744  Data@ {{.+}} is: 0x000107F5
// CHECK:     Address: 0x0000000000220748  Data@ {{.+}} is: 0x5E590001
// CHECK:     Address: 0x000000000022074C  Data@ {{.+}} is: 0x6F9907F6
// CHECK:     Address: 0x0000000000220750  Data@ {{.+}} is: 0x000113DF
// CHECK:     Address: 0x0000000000220754  Data@ {{.+}} is: 0xC800F09D
// CHECK:     Address: 0x0000000000220758  Data@ {{.+}} is: 0x3F99FEDB
// CHECK:     Address: 0x000000000022075C  Data@ {{.+}} is: 0x000113DE
// CHECK:     Address: 0x0000000000220760  Data@ {{.+}} is: 0xC800F09D
// CHECK:     Address: 0x0000000000220764  Data@ {{.+}} is: 0xDF99FEEB
// CHECK:     Address: 0x0000000000220768  Data@ {{.+}} is: 0x000113DF
// CHECK:     Address: 0x000000000022076C  Data@ {{.+}} is: 0xC800F09D
// CHECK:     Address: 0x0000000000220770  Data@ {{.+}} is: 0x8F99FEFB
// CHECK:     Address: 0x0000000000220774  Data@ {{.+}} is: 0x000113DF
// CHECK:     Address: 0x0000000000220778  Data@ {{.+}} is: 0xC800F09D
// CHECK:     Address: 0x000000000022077C  Data@ {{.+}} is: 0xDF99FF0B
// CHECK:     Address: 0x0000000000220780  Data@ {{.+}} is: 0x000113DE
// CHECK:     Address: 0x0000000000220784  Data@ {{.+}} is: 0xC800F09D
// CHECK:     Address: 0x0000000000220788  Data@ {{.+}} is: 0xEF99FF1B
// CHECK:     Address: 0x000000000022078C  Data@ {{.+}} is: 0x000113DF
// CHECK:     Address: 0x0000000000220790  Data@ {{.+}} is: 0x1000F099
// CHECK:     Address: 0x0000000000220794  Data@ {{.+}} is: 0x13DF4F99
// CHECK:     Address: 0x0000000000220798  Data@ {{.+}} is: 0xF0990001
// CHECK:     Address: 0x000000000022079C  Data@ {{.+}} is: 0x5F991000
// CHECK:     Address: 0x00000000002207A0  Data@ {{.+}} is: 0x000113DF
// CHECK:     Address: 0x00000000002207A4  Data@ {{.+}} is: 0x1000F099
// CHECK:     Address: 0x00000000002207A8  Data@ {{.+}} is: 0xC8108035
// CHECK:     Address: 0x00000000002207AC  Data@ {{.+}} is: 0x9659FF2B
// CHECK:     Address: 0x00000000002207B0  Data@ {{.+}} is: 0x80591C2F
// CHECK:     Address: 0x00000000002207B4  Data@ {{.+}} is: 0xDE590010
// CHECK:     Address: 0x00000000002207B8  Data@ {{.+}} is: 0x000107F9
// CHECK:     Address: 0x00000000002207BC  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002207C0  Data@ {{.+}} is: 0x13DF6F99
// CHECK:     Address: 0x00000000002207C4  Data@ {{.+}} is: 0xF0990001
// CHECK:     Address: 0x00000000002207C8  Data@ {{.+}} is: 0x3F991000
// CHECK:     Address: 0x00000000002207CC  Data@ {{.+}} is: 0x000113DE
// CHECK:     Address: 0x00000000002207D0  Data@ {{.+}} is: 0x1000F099
// CHECK:     Address: 0x00000000002207D4  Data@ {{.+}} is: 0x171FDF99
// CHECK:     Address: 0x00000000002207D8  Data@ {{.+}} is: 0xF0990001
// CHECK:     Address: 0x00000000002207DC  Data@ {{.+}} is: 0x8F991000
// CHECK:     Address: 0x00000000002207E0  Data@ {{.+}} is: 0x0001141F
// CHECK:     Address: 0x00000000002207E4  Data@ {{.+}} is: 0x1000F099
// CHECK:     Address: 0x00000000002207E8  Data@ {{.+}} is: 0x105EDF99
// CHECK:     Address: 0x00000000002207EC  Data@ {{.+}} is: 0xF0990001
// CHECK:     Address: 0x00000000002207F0  Data@ {{.+}} is: 0xEF991000
// CHECK:     Address: 0x00000000002207F4  Data@ {{.+}} is: 0x0001125F
// CHECK:     Address: 0x00000000002207F8  Data@ {{.+}} is: 0x1000F099
// CHECK:     Address: 0x00000000002207FC  Data@ {{.+}} is: 0x129F4F99
// CHECK:     Address: 0x0000000000220800  Data@ {{.+}} is: 0xF0990001
// CHECK:     Address: 0x0000000000220804  Data@ {{.+}} is: 0x5F991000
// CHECK:     Address: 0x0000000000220808  Data@ {{.+}} is: 0x000112DF
// CHECK:     Address: 0x000000000022080C  Data@ {{.+}} is: 0x1000F099
// CHECK:     Address: 0x0000000000220810  Data@ {{.+}} is: 0x047F5FBD
// CHECK:     Address: 0x0000000000220814  Data@ {{.+}} is: 0x28430210
// CHECK:     Address: 0x0000000000220818  Data@ {{.+}} is: 0x8C7E138B
// CHECK:     Address: 0x000000000022081C  Data@ {{.+}} is: 0xDF9D000F
// CHECK:     Address: 0x0000000000220820  Data@ {{.+}} is: 0x02138A1A
// CHECK:     Address: 0x0000000000220824  Data@ {{.+}} is: 0x131F6F99
// CHECK:     Address: 0x0000000000220828  Data@ {{.+}} is: 0x10803F99
// CHECK:     Address: 0x000000000022082C  Data@ {{.+}} is: 0x1187DF99
// CHECK:     Address: 0x0000000000220830  Data@ {{.+}} is: 0x117BEF99
// CHECK:     Address: 0x0000000000220834  Data@ {{.+}} is: 0x113D4F99
// CHECK:     Address: 0x0000000000220838  Data@ {{.+}} is: 0xF0990001
// CHECK:     Address: 0x000000000022083C  Data@ {{.+}} is: 0x0099139C
// CHECK:     Address: 0x0000000000220840  Data@ {{.+}} is: 0x30991380
// CHECK:     Address: 0x0000000000220844  Data@ {{.+}} is: 0x809D1000
// CHECK:     Address: 0x0000000000220848  Data@ {{.+}} is: 0x02064801
// CHECK:     Address: 0x000000000022084C  Data@ {{.+}} is: 0xC800D09D
// CHECK:     Address: 0x0000000000220850  Data@ {{.+}} is: 0xD09D0305
// CHECK:     Address: 0x0000000000220854  Data@ {{.+}} is: 0x00674801
// CHECK:     Address: 0x0000000000220858  Data@ {{.+}} is: 0x8CC7DC9D
// CHECK:     Address: 0x000000000022085C  Data@ {{.+}} is: 0xE61D0003
// CHECK:     Address: 0x0000000000220860  Data@ {{.+}} is: 0x038548C6
// CHECK:     Address: 0x0000000000220864  Data@ {{.+}} is: 0x100060BB
// CHECK:     Address: 0x0000000000220868  Data@ {{.+}} is: 0x880600CA
// CHECK:     Address: 0x000000000022086C  Data@ {{.+}} is: 0xE09D0285
// CHECK:     Address: 0x0000000000220870  Data@ {{.+}} is: 0x01850801
// CHECK:     Address: 0x0000000000220874  Data@ {{.+}} is: 0xCCE6071D
// CHECK:     Address: 0x0000000000220878  Data@ {{.+}} is: 0xF09D0023
// CHECK:     Address: 0x000000000022087C  Data@ {{.+}} is: 0x00460801
// CHECK:     Address: 0x0000000000220880  Data@ {{.+}} is: 0x8CDAFC9D
// CHECK:     Address: 0x0000000000220884  Data@ {{.+}} is: 0xD0FB0087
// CHECK:     Address: 0x0000000000220888  Data@ {{.+}} is: 0x800006F6
// CHECK:     Address: 0x000000000022088C  Data@ {{.+}} is: 0x0107C810
// CHECK:     Address: 0x0000000000220890  Data@ {{.+}} is: 0x000003C0
// CHECK:     Address: 0x0000000000220894  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220898  Data@ {{.+}} is: 0x48000000
// CHECK:     Address: 0x000000000022089C  Data@ {{.+}} is: 0x0000FF48
// CHECK:     Address: 0x00000000002208A0  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002208A4  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002208A8  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002208AC  Data@ {{.+}} is: 0x68400195
// CHECK:     Address: 0x00000000002208B0  Data@ {{.+}} is: 0x00010802
// CHECK:     Address: 0x00000000002208B4  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002208B8  Data@ {{.+}} is: 0x011D0001
// CHECK:     Address: 0x00000000002208BC  Data@ {{.+}} is: 0x00234800
// CHECK:     Address: 0x00000000002208C0  Data@ {{.+}} is: 0x000003C0
// CHECK:     Address: 0x00000000002208C4  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000002208C8  Data@ {{.+}} is: 0x88000000
// CHECK:     Address: 0x00000000002208CC  Data@ {{.+}} is: 0x0000FF88
// CHECK:     Address: 0x00000000002208D0  Data@ {{.+}} is: 0xC800001D
// CHECK:     Address: 0x00000000002208D4  Data@ {{.+}} is: 0x0001FF9E
// CHECK:     Address: 0x00000000002208D8  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002208DC  Data@ {{.+}} is: 0x1D190001
// CHECK:     Address: 0x00000000002208E0  Data@ {{.+}} is: 0x3C991006
// CHECK:     Address: 0x00000000002208E4  Data@ {{.+}} is: 0xE6191080
// CHECK:     Address: 0x00000000002208E8  Data@ {{.+}} is: 0x01951000
// CHECK:     Address: 0x00000000002208EC  Data@ {{.+}} is: 0x00025040
// CHECK:     Address: 0x00000000002208F0  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002208F4  Data@ {{.+}} is: 0x10840719
// CHECK:     Address: 0x00000000002208F8  Data@ {{.+}} is: 0x1082FC99
// CHECK:     Address: 0x00000000002208FC  Data@ {{.+}} is: 0x16F61099
// CHECK:     Address: 0x0000000000220900  Data@ {{.+}} is: 0x000003C0
// CHECK:     Address: 0x0000000000220904  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220908  Data@ {{.+}} is: 0x88000000
// CHECK:     Address: 0x000000000022090C  Data@ {{.+}} is: 0x0000FFB8
// CHECK:     Address: 0x0000000000220910  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220914  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220918  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x000000000022091C  Data@ {{.+}} is: 0x40400195
// CHECK:     Address: 0x0000000000220920  Data@ {{.+}} is: 0x00011002
// CHECK:     Address: 0x0000000000220924  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220928  Data@ {{.+}} is: 0x011D0001
// CHECK:     Address: 0x000000000022092C  Data@ {{.+}} is: 0x00204800
// CHECK:     Address: 0x0000000000220930  Data@ {{.+}} is: 0x880003C0
// CHECK:     Address: 0x0000000000220934  Data@ {{.+}} is: 0x07900003
// CHECK:     Address: 0x0000000000220938  Data@ {{.+}} is: 0x00000030
// CHECK:     Address: 0x000000000022093C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x0000000000220940  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220944  Data@ {{.+}} is: 0xF2190001
// CHECK:     Address: 0x0000000000220948  Data@ {{.+}} is: 0x00011660
// CHECK:     Address: 0x000000000022094C  Data@ {{.+}} is: 0x00950001
// CHECK:     Address: 0x0000000000220950  Data@ {{.+}} is: 0x00012800
// CHECK:     Address: 0x0000000000220954  Data@ {{.+}} is: 0x16A0F219
// CHECK:     Address: 0x0000000000220958  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x000000000022095C  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220960  Data@ {{.+}} is: 0x0000007F
// CHECK:     Address: 0x0000000000220964  Data@ {{.+}} is: 0x08000008
// CHECK:     Address: 0x0000000000220968  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000022096C  Data@ {{.+}} is: 0x00010000
// CHECK:     Address: 0x0000000000220970  Data@ {{.+}} is: 0x38032019
// CHECK:     Address: 0x0000000000220974  Data@ {{.+}} is: 0x0FFC4299
// CHECK:     Address: 0x0000000000220978  Data@ {{.+}} is: 0x011D0001
// CHECK:     Address: 0x000000000022097C  Data@ {{.+}} is: 0x0000D802
// CHECK:     Address: 0x0000000000220980  Data@ {{.+}} is: 0x0000081D
// CHECK:     Address: 0x0000000000220984  Data@ {{.+}} is: 0x00010000
// CHECK:     Address: 0x0000000000220988  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x000000000022098C  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220990  Data@ {{.+}} is: 0x07FC42D9
// CHECK:     Address: 0x0000000000220994  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x0000000000220998  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x000000000022099C  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002209A0  Data@ {{.+}} is: 0x10001819
// CHECK:     Address: 0x00000000002209A4  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002209A8  Data@ {{.+}} is: 0x00010001
// CHECK:     Address: 0x00000000002209AC  Data@ {{.+}} is: 0x3FFFE019

// CHECK: Generating: {{.+}}aie_cdo_init.bin
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
// CHECK:     Address: 0x000000000021D000  Data@ {{.+}} is: 0x00400080
// CHECK:     Address: 0x000000000021D004  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D008  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D00C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D010  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D014  Data@ {{.+}} is: 0x06045FE3

// CHECK: (NOP Command): Payload Length: 2
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x000000000021D020  Size: 6
// CHECK:     Address: 0x000000000021D020  Data@ {{.+}} is: 0x00600200
// CHECK:     Address: 0x000000000021D024  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D028  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D02C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D030  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D034  Data@ {{.+}} is: 0x0E049FE5

// CHECK: (NOP Command): Payload Length: 2
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x000000000021D040  Size: 6
// CHECK:     Address: 0x000000000021D040  Data@ {{.+}} is: 0x00E00100
// CHECK:     Address: 0x000000000021D044  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D048  Data@ {{.+}} is: 0x0003E000
// CHECK:     Address: 0x000000000021D04C  Data@ {{.+}} is: 0x01008003
// CHECK:     Address: 0x000000000021D050  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x000000000021D054  Data@ {{.+}} is: 0x16043FE0

// CHECK: (Write64): Address:  0x000000000021DE04 Data:  0x00010000
// CHECK: (MaskWrite64): Address: 0x000000000021DE00  Mask: 0x00000000  Data: 0x00000001
// CHECK: (Write64): Address:  0x000000000021DE0C Data:  0x00010001
// CHECK: (MaskWrite64): Address: 0x000000000021DE08  Mask: 0x00000000  Data: 0x00000001
// CHECK: (Write64): Address:  0x000000000021DE14 Data:  0x00010002
// CHECK: (MaskWrite64): Address: 0x000000000021DE10  Mask: 0x00000000  Data: 0x00000001
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x00000000041A0000  Size: 8
// CHECK:     Address: 0x00000000041A0000  Data@ {{.+}} is: 0x00000100
// CHECK:     Address: 0x00000000041A0004  Data@ {{.+}} is: 0x000A0000
// CHECK:     Address: 0x00000000041A0008  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000041A000C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000041A0010  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000041A0014  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000041A0018  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000041A001C  Data@ {{.+}} is: 0x8140FF41

// CHECK: (NOP Command): Payload Length: 0
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x00000000041A0020  Size: 8
// CHECK:     Address: 0x00000000041A0020  Data@ {{.+}} is: 0x00000100
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
// CHECK:     Address: 0x00000000001A0000  Data@ {{.+}} is: 0x00000080
// CHECK:     Address: 0x00000000001A0004  Data@ {{.+}} is: 0x000A0000
// CHECK:     Address: 0x00000000001A0008  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000001A000C  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000001A0010  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000001A0014  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000001A0018  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000001A001C  Data@ {{.+}} is: 0x8140FF41

// CHECK: (NOP Command): Payload Length: 0
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x00000000001A0020  Size: 8
// CHECK:     Address: 0x00000000001A0020  Data@ {{.+}} is: 0x00000080
// CHECK:     Address: 0x00000000001A0024  Data@ {{.+}} is: 0x001A0000
// CHECK:     Address: 0x00000000001A0028  Data@ {{.+}} is: 0x00100000
// CHECK:     Address: 0x00000000001A002C  Data@ {{.+}} is: 0x0010000F
// CHECK:     Address: 0x00000000001A0030  Data@ {{.+}} is: 0x00040007
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
// CHECK:     Address: 0x00000000021A0028  Data@ {{.+}} is: 0x00080000
// CHECK:     Address: 0x00000000021A002C  Data@ {{.+}} is: 0x0020001F
// CHECK:     Address: 0x00000000021A0030  Data@ {{.+}} is: 0x00100003
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
// CHECK: Generating: {{.+}}aie_cdo_enable.bin
// CHECK: (NOP Command): Payload Length: 0
// CHECK: (MaskWrite64): Address: 0x0000000000232000  Mask: 0x00000001  Data: 0x00000001
