// RUN: (aie_cdo_gen_test %s %T) 2>&1 | FileCheck %s

module {
  aie.device(npu1_1col) {
    %tile_0_1 = aie.tile(0, 1)
    %objFifo_in0_cons_buff_0 = aie.buffer(%tile_0_1) {address = 0 : i32, sym_name = "objFifo_in0_cons_buff_0"} : memref<16xi32>
    %objFifo_in0_cons_buff_1 = aie.buffer(%tile_0_1) {address = 64 : i32, sym_name = "objFifo_in0_cons_buff_1"} : memref<16xi32>
    %objFifo_in0_cons_prod_lock = aie.lock(%tile_0_1, 0) {init = 2 : i8, sym_name = "objFifo_in0_cons_prod_lock"}
    %objFifo_in0_cons_cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i8, sym_name = "objFifo_in0_cons_cons_lock"}
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
    ^bb1:  // pred: ^bb0
      aie.use_lock(%objFifo_in0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<16xi32>) {bd_id = 0 : i32, dimensions = #aie<bd_dim_layout_array[<size = 13, stride = 1>]>, len = 16 : i32, next_bd_id = 1 : i32, pad_dimensions = #aie<bd_pad_layout_array[<const_pad_before = 2, const_pad_after = 1>]>}
      aie.use_lock(%objFifo_in0_cons_prod_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%objFifo_in0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_in0_cons_buff_1 : memref<16xi32>) {bd_id = 1 : i32, dimensions = #aie<bd_dim_layout_array[<size = 13, stride = 1>]>, len = 16 : i32, pad_dimensions = #aie<bd_pad_layout_array[<const_pad_before = 2, const_pad_after = 1>]>}
      aie.use_lock(%objFifo_in0_cons_prod_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // 2 preds: ^bb0, ^bb2
      aie.end
    }
  }
}

// CHECK: XAIE API: XAie_SetupPartitionConfig with args: &devInst=ptr, partBaseAddr=0, partitionStartCol=1, partitionNumCols=1
// CHECK: XAIE API: XAie_CfgInitialize with args: &devInst=ptr, &configPtr=ptr
// CHECK: XAIE API: XAie_SetIOBackend with args: &devInst=ptr, XAIE_IO_BACKEND_CDO=1
// CHECK: XAIE API: XAie_UpdateNpiAddr with args: &devInst=ptr, npiAddr=0
// CHECK: XAIE API: XAie_TurnEccOff with args: &devInst=ptr
// CHECK: XAIE API: XAie_LockSetValue with args: devInst=ptr, lock.tileLoc=TileLoc(col: 0, row: 1), locInit=XAie_Lock(LockId: 0, LockVal: 2)
// CHECK: XAIE API: XAie_LockSetValue with args: devInst=ptr, lock.tileLoc=TileLoc(col: 0, row: 1), locInit=XAie_Lock(LockId: 1, LockVal: 0)
// CHECK: XAIE API: XAie_DmaDescInit with args: devInst=ptr, &dmaDesc=ptr, tileLoc=TileLoc(col: 0, row: 1)
// CHECK: XAIE API: dmaDesc.DmaMod->SetLock with args: &dmaDesc=ptr, acqLock=XAie_Lock(LockId: 65, LockVal: -1), relLock=XAie_Lock(LockId: 64, LockVal: 1), acqEn=1, relEn=0
// CHECK: XAie_DmaTensor(XAie_AieMlDmaDimDesc(StepSize: 1, Wrap: 13))
// CHECK: XAIE API: XAie_DmaSetMultiDimAddr with args: &dmaDesc=ptr, &dmaTileBdTensor=ptr, basePlusOffsetInBytes=524288, lenInBytes=64
// CHECK: XAie_DmaPadTensor(XAie_PadDesc(Before: 2, After: 1))
// CHECK: XAIE API: XAie_DmaSetPadding with args: &dmaDesc=ptr, &dmaPadTensor=ptr
// CHECK: XAIE API: XAie_DmaSetNextBd with args: &dmaDesc=ptr, nextBdId.value()=1, enableNextBd=1
// CHECK: XAIE API: XAie_DmaEnableBd with args: &dmaDesc=ptr
// CHECK: XAIE API: XAie_DmaWriteBd with args: devInst=ptr, &dmaDesc=ptr, tileLoc=TileLoc(col: 0, row: 1), bdId=0
// CHECK: XAIE API: XAie_DmaDescInit with args: devInst=ptr, &dmaDesc=ptr, tileLoc=TileLoc(col: 0, row: 1)
// CHECK: XAIE API: dmaDesc.DmaMod->SetLock with args: &dmaDesc=ptr, acqLock=XAie_Lock(LockId: 65, LockVal: -1), relLock=XAie_Lock(LockId: 64, LockVal: 1), acqEn=1, relEn=0
// CHECK: XAie_DmaTensor(XAie_AieMlDmaDimDesc(StepSize: 1, Wrap: 13))
// CHECK: XAIE API: XAie_DmaSetMultiDimAddr with args: &dmaDesc=ptr, &dmaTileBdTensor=ptr, basePlusOffsetInBytes=524352, lenInBytes=64
// CHECK: XAie_DmaPadTensor(XAie_PadDesc(Before: 2, After: 1))
// CHECK: XAIE API: XAie_DmaSetPadding with args: &dmaDesc=ptr, &dmaPadTensor=ptr
// CHECK: XAIE API: XAie_DmaEnableBd with args: &dmaDesc=ptr
// CHECK: XAIE API: XAie_DmaWriteBd with args: devInst=ptr, &dmaDesc=ptr, tileLoc=TileLoc(col: 0, row: 1), bdId=1
// CHECK: XAIE API: XAie_DmaChannelSetStartQueue with args: devInst=ptr, tileLoc=TileLoc(col: 0, row: 1), chNum=0, direction=0, bdId=0, repeatCount=2, enTokenIssue=0
// CHECK: XAIE API: XAie_DmaChannelEnable with args: devInst=ptr, tileLoc=TileLoc(col: 0, row: 1), chNum=0, direction=0

// CHECK: cdo-driver: (NOP Command): Payload Length: 0

// CHECK: cdo-driver: (NOP Command): Payload Length: 0
// CHECK: cdo-driver: (Write64): Address:  0x00000000001C0000 Data:  0x00000002
// CHECK: cdo-driver: (Write64): Address:  0x00000000001C0010 Data:  0x00000000
// CHECK: cdo-driver: (NOP Command): Payload Length: 2
// CHECK: cdo-driver: (BlockWrite-DMAWriteCmd): Start Address: 0x00000000001A0000  Size: 8
// CHECK: cdo-driver:     Address: 0x00000000001A0000  Data is: 0x00000010
// CHECK: cdo-driver:     Address: 0x00000000001A0004  Data is: 0x081A0000
// CHECK: cdo-driver:     Address: 0x00000000001A0008  Data is: 0x001A0000
// CHECK: cdo-driver:     Address: 0x00000000001A000C  Data is: 0x00000000
// CHECK: cdo-driver:     Address: 0x00000000001A0010  Data is: 0x00000000
// CHECK: cdo-driver:     Address: 0x00000000001A0014  Data is: 0x00020000
// CHECK: cdo-driver:     Address: 0x00000000001A0018  Data is: 0x00000000
// CHECK: cdo-driver:     Address: 0x00000000001A001C  Data is: 0x8140FF41

// CHECK: cdo-driver: (NOP Command): Payload Length: 0
// CHECK: cdo-driver: (BlockWrite-DMAWriteCmd): Start Address: 0x00000000001A0020  Size: 8
// CHECK: cdo-driver:     Address: 0x00000000001A0020  Data is: 0x00000010
// CHECK: cdo-driver:     Address: 0x00000000001A0024  Data is: 0x08020010
// CHECK: cdo-driver:     Address: 0x00000000001A0028  Data is: 0x001A0000
// CHECK: cdo-driver:     Address: 0x00000000001A002C  Data is: 0x00000000
// CHECK: cdo-driver:     Address: 0x00000000001A0030  Data is: 0x00000000
// CHECK: cdo-driver:     Address: 0x00000000001A0034  Data is: 0x00020000
// CHECK: cdo-driver:     Address: 0x00000000001A0038  Data is: 0x00000000
// CHECK: cdo-driver:     Address: 0x00000000001A003C  Data is: 0x8140FF41

// CHECK: cdo-driver: (Write64): Address:  0x00000000001A0604 Data:  0x00010000
// CHECK: cdo-driver: (MaskWrite64): Address: 0x00000000001A0600  Mask: 0x00000000  Data: 0x00000001
