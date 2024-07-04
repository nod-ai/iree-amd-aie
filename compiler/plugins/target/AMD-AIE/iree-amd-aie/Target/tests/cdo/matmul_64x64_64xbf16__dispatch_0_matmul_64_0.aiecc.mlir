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

// CHECK: XAIE API: XAie_SetupPartitionConfig with args: &devInst=ptr, 0x0=0, partitionStartCol_=0, partitionNumCols_=4
// CHECK: XAIE API: XAie_CfgInitialize with args: &devInst=ptr, &configPtr=ptr
// CHECK: XAIE API: XAie_SetIOBackend with args: &devInst=ptr, XAIE_IO_BACKEND_CDO=1
// CHECK: XAIE API: XAie_UpdateNpiAddr with args: &devInst=ptr, 0x0=0
// CHECK: XAIE API: XAie_TurnEccOff with args: &devInst=ptr

// CHECK: XAIE API: XAie_LockSetValue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 1, Row: 1), locInit=XAie_Lock(LockId: 1, LockVal: 1)
// CHECK: XAIE API: XAie_LockSetValue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 1, Row: 1), locInit=XAie_Lock(LockId: 0, LockVal: 0)
// CHECK: XAIE API: XAie_LockSetValue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), locInit=XAie_Lock(LockId: 1, LockVal: 1)
// CHECK: XAIE API: XAie_LockSetValue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), locInit=XAie_Lock(LockId: 0, LockVal: 0)
// CHECK: XAIE API: XAie_LockSetValue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 2, Row: 1), locInit=XAie_Lock(LockId: 1, LockVal: 1)
// CHECK: XAIE API: XAie_LockSetValue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 2, Row: 1), locInit=XAie_Lock(LockId: 0, LockVal: 0)
// CHECK: XAIE API: XAie_LockSetValue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 5, LockVal: 1)
// CHECK: XAIE API: XAie_LockSetValue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 4, LockVal: 0)
// CHECK: XAIE API: XAie_LockSetValue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 3, LockVal: 1)
// CHECK: XAIE API: XAie_LockSetValue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 2, LockVal: 0)
// CHECK: XAIE API: XAie_LockSetValue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 1, LockVal: 1)
// CHECK: XAIE API: XAie_LockSetValue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), locInit=XAie_Lock(LockId: 0, LockVal: 0)
// CHECK: XAIE API: XAie_DmaDescInit with args: &deviceModel.devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2)

// CHECK: start configuring bds
// CHECK: XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd=ptr, acqLock=XAie_Lock(LockId: 3, LockVal: -1), relLock=XAie_Lock(LockId: 2, LockVal: 1), acqEn=1, relEn=0
// CHECK: XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd=ptr, basePlusOffsetInBytes=1024, lenInBytes=8192
// CHECK: XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd=ptr, nextBdId.value()=0, enableNextBd=1
// CHECK: XAIE API: XAie_DmaEnableBd with args: &dmaTileBd=ptr
// CHECK: XAIE API: XAie_DmaWriteBd with args: &deviceModel.devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), bdId=0

// CHECK: end configuring bds
// CHECK: XAIE API: XAie_DmaDescInit with args: &deviceModel.devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2)

// CHECK: start configuring bds
// CHECK: XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd=ptr, acqLock=XAie_Lock(LockId: 5, LockVal: -1), relLock=XAie_Lock(LockId: 4, LockVal: 1), acqEn=1, relEn=0
// CHECK: XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd=ptr, basePlusOffsetInBytes=9216, lenInBytes=8192
// CHECK: XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd=ptr, nextBdId.value()=1, enableNextBd=1
// CHECK: XAIE API: XAie_DmaEnableBd with args: &dmaTileBd=ptr
// CHECK: XAIE API: XAie_DmaWriteBd with args: &deviceModel.devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), bdId=1

// CHECK: end configuring bds
// CHECK: XAIE API: XAie_DmaDescInit with args: &deviceModel.devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2)

// CHECK: start configuring bds
// CHECK: XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd=ptr, acqLock=XAie_Lock(LockId: 0, LockVal: -1), relLock=XAie_Lock(LockId: 1, LockVal: 1), acqEn=1, relEn=0
// CHECK: XAIE API: XAie_DmaSetMultiDimAddr with args: &dmaTileBd=ptr, &dmaTileBdTensor=ptr, basePlusOffsetInBytes=17408, lenInBytes=16384
// CHECK: XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd=ptr, nextBdId.value()=2, enableNextBd=1
// CHECK: XAIE API: XAie_DmaEnableBd with args: &dmaTileBd=ptr
// CHECK: XAIE API: XAie_DmaWriteBd with args: &deviceModel.devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), bdId=2

// CHECK: end configuring bds
// CHECK: XAIE API: XAie_DmaChannelSetStartQueue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), chNum=0, direction=0, bdId=0, repeatCount=2, enTokenIssue=0
// CHECK: XAIE API: XAie_DmaChannelEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), chNum=0, direction=0
// CHECK: XAIE API: XAie_DmaChannelSetStartQueue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), chNum=1, direction=0, bdId=1, repeatCount=2, enTokenIssue=0
// CHECK: XAIE API: XAie_DmaChannelEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), chNum=1, direction=0
// CHECK: XAIE API: XAie_DmaChannelSetStartQueue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), chNum=0, direction=1, bdId=2, repeatCount=2, enTokenIssue=0
// CHECK: XAIE API: XAie_DmaChannelEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), chNum=0, direction=1
// CHECK: XAIE API: XAie_DmaDescInit with args: &deviceModel.devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 2, Row: 1)

// CHECK: start configuring bds
// CHECK: XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd=ptr, acqLock=XAie_Lock(LockId: 65, LockVal: -1), relLock=XAie_Lock(LockId: 64, LockVal: 1), acqEn=1, relEn=0
// CHECK: XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd=ptr, basePlusOffsetInBytes=524288, lenInBytes=16384
// CHECK: XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd=ptr, nextBdId.value()=0, enableNextBd=1
// CHECK: XAIE API: XAie_DmaEnableBd with args: &dmaTileBd=ptr
// CHECK: XAIE API: XAie_DmaWriteBd with args: &deviceModel.devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 2, Row: 1), bdId=0

// CHECK: end configuring bds
// CHECK: XAIE API: XAie_DmaDescInit with args: &deviceModel.devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 2, Row: 1)

// CHECK: start configuring bds
// CHECK: XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd=ptr, acqLock=XAie_Lock(LockId: 64, LockVal: -1), relLock=XAie_Lock(LockId: 65, LockVal: 1), acqEn=1, relEn=0
// CHECK: XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd=ptr, basePlusOffsetInBytes=524288, lenInBytes=16384
// CHECK: XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd=ptr, nextBdId.value()=1, enableNextBd=1
// CHECK: XAIE API: XAie_DmaEnableBd with args: &dmaTileBd=ptr
// CHECK: XAIE API: XAie_DmaWriteBd with args: &deviceModel.devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 2, Row: 1), bdId=1

// CHECK: end configuring bds
// CHECK: XAIE API: XAie_DmaChannelSetStartQueue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 2, Row: 1), chNum=0, direction=0, bdId=0, repeatCount=2, enTokenIssue=0
// CHECK: XAIE API: XAie_DmaChannelEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 2, Row: 1), chNum=0, direction=0
// CHECK: XAIE API: XAie_DmaChannelSetStartQueue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 2, Row: 1), chNum=0, direction=1, bdId=1, repeatCount=2, enTokenIssue=0
// CHECK: XAIE API: XAie_DmaChannelEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 2, Row: 1), chNum=0, direction=1
// CHECK: XAIE API: XAie_DmaDescInit with args: &deviceModel.devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1)

// CHECK: start configuring bds
// CHECK: XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd=ptr, acqLock=XAie_Lock(LockId: 65, LockVal: -1), relLock=XAie_Lock(LockId: 64, LockVal: 1), acqEn=1, relEn=0
// CHECK: XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd=ptr, basePlusOffsetInBytes=524288, lenInBytes=8192
// CHECK: XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd=ptr, nextBdId.value()=0, enableNextBd=1
// CHECK: XAIE API: XAie_DmaEnableBd with args: &dmaTileBd=ptr
// CHECK: XAIE API: XAie_DmaWriteBd with args: &deviceModel.devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), bdId=0

// CHECK: end configuring bds
// CHECK: XAIE API: XAie_DmaDescInit with args: &deviceModel.devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1)

// CHECK: start configuring bds
// CHECK: XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd=ptr, acqLock=XAie_Lock(LockId: 64, LockVal: -1), relLock=XAie_Lock(LockId: 65, LockVal: 1), acqEn=1, relEn=0
// CHECK: XAIE API: XAie_DmaSetMultiDimAddr with args: &dmaTileBd=ptr, &dmaTileBdTensor=ptr, basePlusOffsetInBytes=524288, lenInBytes=8192
// CHECK: XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd=ptr, nextBdId.value()=1, enableNextBd=1
// CHECK: XAIE API: XAie_DmaEnableBd with args: &dmaTileBd=ptr
// CHECK: XAIE API: XAie_DmaWriteBd with args: &deviceModel.devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), bdId=1

// CHECK: end configuring bds
// CHECK: XAIE API: XAie_DmaChannelSetStartQueue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), chNum=0, direction=0, bdId=0, repeatCount=2, enTokenIssue=0
// CHECK: XAIE API: XAie_DmaChannelEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), chNum=0, direction=0
// CHECK: XAIE API: XAie_DmaChannelSetStartQueue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), chNum=0, direction=1, bdId=1, repeatCount=2, enTokenIssue=0
// CHECK: XAIE API: XAie_DmaChannelEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), chNum=0, direction=1
// CHECK: XAIE API: XAie_DmaDescInit with args: &deviceModel.devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 1, Row: 1)

// CHECK: start configuring bds
// CHECK: XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd=ptr, acqLock=XAie_Lock(LockId: 65, LockVal: -1), relLock=XAie_Lock(LockId: 64, LockVal: 1), acqEn=1, relEn=0
// CHECK: XAIE API: XAie_DmaSetAddrLen with args: &dmaTileBd=ptr, basePlusOffsetInBytes=524288, lenInBytes=8192
// CHECK: XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd=ptr, nextBdId.value()=0, enableNextBd=1
// CHECK: XAIE API: XAie_DmaEnableBd with args: &dmaTileBd=ptr
// CHECK: XAIE API: XAie_DmaWriteBd with args: &deviceModel.devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 1, Row: 1), bdId=0

// CHECK: end configuring bds
// CHECK: XAIE API: XAie_DmaDescInit with args: &deviceModel.devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 1, Row: 1)

// CHECK: start configuring bds
// CHECK: XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd=ptr, acqLock=XAie_Lock(LockId: 64, LockVal: -1), relLock=XAie_Lock(LockId: 65, LockVal: 1), acqEn=1, relEn=0
// CHECK: XAIE API: XAie_DmaSetMultiDimAddr with args: &dmaTileBd=ptr, &dmaTileBdTensor=ptr, basePlusOffsetInBytes=524288, lenInBytes=8192
// CHECK: XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd=ptr, nextBdId.value()=1, enableNextBd=1
// CHECK: XAIE API: XAie_DmaEnableBd with args: &dmaTileBd=ptr
// CHECK: XAIE API: XAie_DmaWriteBd with args: &deviceModel.devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 1, Row: 1), bdId=1

// CHECK: end configuring bds
// CHECK: XAIE API: XAie_DmaChannelSetStartQueue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 1, Row: 1), chNum=0, direction=0, bdId=0, repeatCount=2, enTokenIssue=0
// CHECK: XAIE API: XAie_DmaChannelEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 1, Row: 1), chNum=0, direction=0
// CHECK: XAIE API: XAie_DmaChannelSetStartQueue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 1, Row: 1), chNum=0, direction=1, bdId=1, repeatCount=2, enTokenIssue=0
// CHECK: XAIE API: XAie_DmaChannelEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 1, Row: 1), chNum=0, direction=1
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 0), CTRL=StrmSwPortType::CTRL, slvPortNum=0, SOUTH=StrmSwPortType::SOUTH, mstrPortNum=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 0), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::SOUTH, connectOp.sourceIndex()=3, toStrmT(connectOp.getDestBundle())=StrmSwPortType::NORTH, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 0), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::SOUTH, connectOp.sourceIndex()=7, toStrmT(connectOp.getDestBundle())=StrmSwPortType::EAST, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 0), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::EAST, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::SOUTH, connectOp.destIndex()=2
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::SOUTH, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::DMA, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::DMA, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::NORTH, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 1, Row: 0), CTRL=StrmSwPortType::CTRL, slvPortNum=0, SOUTH=StrmSwPortType::SOUTH, mstrPortNum=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 1, Row: 0), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::WEST, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::NORTH, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 1, Row: 0), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::EAST, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::WEST, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 1, Row: 1), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::SOUTH, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::DMA, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 1, Row: 1), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::DMA, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::NORTH, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 2, Row: 0), CTRL=StrmSwPortType::CTRL, slvPortNum=0, SOUTH=StrmSwPortType::SOUTH, mstrPortNum=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 2, Row: 0), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::NORTH, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::WEST, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 2, Row: 1), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::DMA, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::SOUTH, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 2, Row: 1), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::NORTH, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::DMA, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::SOUTH, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::DMA, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::EAST, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::DMA, connectOp.destIndex()=1
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 2), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::DMA, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::EAST, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 1, Row: 2), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::SOUTH, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::WEST, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 1, Row: 2), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::WEST, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::EAST, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_StrmConnCctEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 2, Row: 2), toStrmT(connectOp.getSourceBundle())=StrmSwPortType::WEST, connectOp.sourceIndex()=0, toStrmT(connectOp.getDestBundle())=StrmSwPortType::SOUTH, connectOp.destIndex()=0
// CHECK: XAIE API: XAie_EnableShimDmaToAieStrmPort with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 0), connectOp.destIndex()=3
// CHECK: XAIE API: XAie_EnableShimDmaToAieStrmPort with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 0), connectOp.destIndex()=7
// CHECK: XAIE API: XAie_EnableAieToShimDmaStrmPort with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 0), connectOp.sourceIndex()=2
// CHECK: Generating: {{.+}}/aie_cdo_elfs.bin
// CHECK: Generating: {{.+}}/aie_cdo_init.bin
// CHECK: Generating: {{.+}}/aie_cdo_elfs.bin
// CHECK: (NOP Command): Payload Length: 0
// CHECK: Generating: {{.+}}/aie_cdo_init.bin
// CHECK: (NOP Command): Payload Length: 0
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
// CHECK: (NOP Command): Payload Length: 2
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

