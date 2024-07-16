// RUN: (aie_cdo_gen_test %s %S) 2>&1 | FileCheck %s

module {
  aie.device(npu1_1col) {
    memref.global "public" @objFifo_in0 : memref<56x56xi8>
    memref.global "public" @objFifo_out0 : memref<64x64xi8>
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %objFifo_in1_cons_buff_0 = aie.buffer(%tile_0_2) {address = 0 : i32, sym_name = "objFifo_in1_cons_buff_0"} : memref<64x64xi8>
    %objFifo_in1_cons_buff_1 = aie.buffer(%tile_0_2) {address = 4096 : i32, sym_name = "objFifo_in1_cons_buff_1"} : memref<64x64xi8>
    %objFifo_out1_buff_0 = aie.buffer(%tile_0_2) {address = 8192 : i32, sym_name = "objFifo_out1_buff_0"} : memref<64x64xi8>
    %objFifo_out1_buff_1 = aie.buffer(%tile_0_2) {address = 12288 : i32, sym_name = "objFifo_out1_buff_1"} : memref<64x64xi8>
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_1, DMA : 1, %tile_0_0, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 1)
    aie.shim_dma_allocation @objFifo_in0(MM2S, 0, 0)
    func.func @bobsyouruncle(%arg0: memref<61x56xi8>, %arg1: memref<32xi8>, %arg2: memref<64x64xi8>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c56_i64 = arith.constant 56 : i64
      %c61_i64 = arith.constant 61 : i64
      %c64_i64 = arith.constant 64 : i64
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c61_i64, %c56_i64][%c0_i64, %c0_i64, %c56_i64, %c1_i64]) {id = 0 : i64, metadata = @objFifo_in0} : memref<61x56xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c64_i64, %c64_i64][%c0_i64, %c0_i64, %c64_i64, %c1_i64]) {id = 1 : i64, metadata = @objFifo_out0} : memref<64x64xi8>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
    %objFifo_in0_cons_buff_0 = aie.buffer(%tile_0_1) {address = 0 : i32, sym_name = "objFifo_in0_cons_buff_0"} : memref<64x64xi8>
    %objFifo_in0_cons_buff_1 = aie.buffer(%tile_0_1) {address = 4096 : i32, sym_name = "objFifo_in0_cons_buff_1"} : memref<64x64xi8>
    %objFifo_out0_buff_0 = aie.buffer(%tile_0_1) {address = 8192 : i32, sym_name = "objFifo_out0_buff_0"} : memref<64x64xi8>
    %objFifo_out0_buff_1 = aie.buffer(%tile_0_1) {address = 12288 : i32, sym_name = "objFifo_out0_buff_1"} : memref<64x64xi8>
    %objFifo_in0_cons_prod_lock = aie.lock(%tile_0_1, 0) {init = 1 : i32, sym_name = "objFifo_in0_cons_prod_lock"}
    %objFifo_in0_cons_cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "objFifo_in0_cons_cons_lock"}
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%objFifo_in0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<64x64xi8>, 0, 4096, [<size = 61, stride = 56>, <size = 56, stride = 1>], [<const_pad_before = 2, const_pad_after = 1>, <const_pad_before = 4, const_pad_after = 4>]) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%objFifo_in0_cons_prod_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
  }
}


// CHECK: XAIE API: XAie_SetupPartitionConfig with args: &devInst=ptr, partBaseAddr=0, partitionStartCol=1, partitionNumCols=1
// CHECK: XAIE API: XAie_CfgInitialize with args: &devInst=ptr, &configPtr=ptr
// CHECK: XAIE API: XAie_SetIOBackend with args: &devInst=ptr, XAIE_IO_BACKEND_CDO=1
// CHECK: XAIE API: XAie_UpdateNpiAddr with args: &devInst=ptr, npiAddr=0
// CHECK: XAIE API: XAie_TurnEccOff with args: &devInst=ptr

// CHECK: XAIE API: XAie_LockSetValue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), locInit=XAie_Lock(LockId: 0, LockVal: 1)
// CHECK: XAIE API: XAie_LockSetValue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), locInit=XAie_Lock(LockId: 1, LockVal: 0)
// CHECK: XAIE API: XAie_DmaDescInit with args: &deviceModel.devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1)

// CHECK: start configuring bds
// CHECK: XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd=ptr, acqLock=XAie_Lock(LockId: 65, LockVal: -1), relLock=XAie_Lock(LockId: 64, LockVal: 1), acqEn=1, relEn=0
// CHECK: XAIE API: XAie_DmaSetMultiDimAddr with args: &dmaTileBd=ptr, &dmaTileBdTensor=ptr, basePlusOffsetInBytes=524288, lenInBytes=4096
// CHECK: XAIE API: XAie_DmaSetPadding with args: &dmaTileBd=ptr, &dmaPadTensor=ptr
// CHECK: XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd=ptr, nextBdId.value()=0, enableNextBd=1
// CHECK: XAIE API: XAie_DmaEnableBd with args: &dmaTileBd=ptr
// CHECK: XAIE API: XAie_DmaWriteBd with args: &deviceModel.devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), bdId=0

// CHECK: end configuring bds
// CHECK: XAIE API: XAie_DmaChannelSetStartQueue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), chNum=0, direction=0, bdId=0, repeatCount=2, enTokenIssue=0
// CHECK: XAIE API: XAie_DmaChannelEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), chNum=0, direction=0

// CHECK: (NOP Command): Payload Length: 0

// CHECK: (NOP Command): Payload Length: 0
// CHECK: (Write64): Address:  0x00000000001C0000 Data:  0x00000001
// CHECK: (Write64): Address:  0x00000000001C0010 Data:  0x00000000
// CHECK: (NOP Command): Payload Length: 2
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x00000000001A0000  Size: 8
// CHECK:     Address: 0x00000000001A0000  Data@ {{.+}} is: 0x00000400
// CHECK:     Address: 0x00000000001A0004  Data@ {{.+}} is: 0x040A0000
// CHECK:     Address: 0x00000000001A0008  Data@ {{.+}} is: 0x001C0000
// CHECK:     Address: 0x00000000001A000C  Data@ {{.+}} is: 0x107A000D
// CHECK:     Address: 0x00000000001A0010  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000001A0014  Data@ {{.+}} is: 0x00820000
// CHECK:     Address: 0x00000000001A0018  Data@ {{.+}} is: 0x00000000
// CHECK:     Address: 0x00000000001A001C  Data@ {{.+}} is: 0x8140FF41

// CHECK: (Write64): Address:  0x00000000001A0604 Data:  0x00010000
// CHECK: (MaskWrite64): Address: 0x00000000001A0600  Mask: 0x00000000  Data: 0x00000001
