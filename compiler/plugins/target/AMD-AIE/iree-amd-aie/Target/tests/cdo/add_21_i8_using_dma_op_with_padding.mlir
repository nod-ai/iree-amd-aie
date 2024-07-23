// RUN: (aie_cdo_gen_test %s %S) 2>&1 | FileCheck %s

module {
  aie.device(npu1_1col) {
    %tile_0_1 = aie.tile(0, 1)
    %objFifo_in0_cons_buff_0 = aie.buffer(%tile_0_1) {address = 0 : i32, sym_name = "objFifo_in0_cons_buff_0"} : memref<16xi8>
    %objFifo_in0_cons_buff_1 = aie.buffer(%tile_0_1) {address = 16 : i32, sym_name = "objFifo_in0_cons_buff_1"} : memref<16xi8>
    %objFifo_in0_cons_prod_lock = aie.lock(%tile_0_1, 0) {init = 2 : i32, sym_name = "objFifo_in0_cons_prod_lock"}
    %objFifo_in0_cons_cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "objFifo_in0_cons_cons_lock"}
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
    ^bb1:  // pred: ^bb0
      aie.use_lock(%objFifo_in0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<16xi8>, 0, 16, [<size = 8, stride = 1>], [<const_pad_before = 4, const_pad_after = 4>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%objFifo_in0_cons_prod_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%objFifo_in0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_in0_cons_buff_1 : memref<16xi8>, 0, 16, [<size = 8, stride = 1>], [<const_pad_before = 4, const_pad_after = 4>]) {bd_id = 1 : i32}
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

// CHECK: XAIE API: XAie_LockSetValue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), locInit=XAie_Lock(LockId: 0, LockVal: 2)
// CHECK: XAIE API: XAie_LockSetValue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), locInit=XAie_Lock(LockId: 1, LockVal: 0)
// CHECK: XAIE API: XAie_DmaDescInit with args: &deviceModel.devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1)

// CHECK: start configuring bds
// CHECK: XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd=ptr, acqLock=XAie_Lock(LockId: 65, LockVal: -1), relLock=XAie_Lock(LockId: 64, LockVal: 1), acqEn=1, relEn=0
// CHECK: XAIE API: XAie_DmaSetMultiDimAddr with args: &dmaTileBd=ptr, &dmaTileBdTensor=ptr, basePlusOffsetInBytes=524288, lenInBytes=16
// CHECK: XAIE API: XAie_DmaSetPadding with args: &dmaTileBd=ptr, &dmaPadTensor=ptr
// CHECK: XAIE API: XAie_DmaSetNextBd with args: &dmaTileBd=ptr, nextBdId.value()=1, enableNextBd=1
// CHECK: XAIE API: XAie_DmaEnableBd with args: &dmaTileBd=ptr
// CHECK: XAIE API: XAie_DmaWriteBd with args: &deviceModel.devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), bdId=0

// CHECK: end configuring bds
// CHECK: XAIE API: XAie_DmaDescInit with args: &deviceModel.devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1)

// CHECK: start configuring bds
// CHECK: XAIE API: dmaTileBd.DmaMod->SetLock with args: &dmaTileBd=ptr, acqLock=XAie_Lock(LockId: 65, LockVal: -1), relLock=XAie_Lock(LockId: 64, LockVal: 1), acqEn=1, relEn=0
// CHECK: XAIE API: XAie_DmaSetMultiDimAddr with args: &dmaTileBd=ptr, &dmaTileBdTensor=ptr, basePlusOffsetInBytes=524304, lenInBytes=16
// CHECK: XAIE API: XAie_DmaSetPadding with args: &dmaTileBd=ptr, &dmaPadTensor=ptr
// CHECK: XAIE API: XAie_DmaEnableBd with args: &dmaTileBd=ptr
// CHECK: XAIE API: XAie_DmaWriteBd with args: &deviceModel.devInst=ptr, &dmaTileBd=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), bdId=1

// CHECK: end configuring bds
// CHECK: XAIE API: XAie_DmaChannelSetStartQueue with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), chNum=0, direction=0, bdId=0, repeatCount=2, enTokenIssue=0
// CHECK: XAIE API: XAie_DmaChannelEnable with args: &deviceModel.devInst=ptr, tileLoc=XAie_LocType(Col: 0, Row: 1), chNum=0, direction=0

// CHECK: (NOP Command): Payload Length: 0

// CHECK: (NOP Command): Payload Length: 0
// CHECK: (Write64): Address:  0x00000000001C0000 Data:  0x00000002
// CHECK: (Write64): Address:  0x00000000001C0010 Data:  0x00000000
// CHECK: (NOP Command): Payload Length: 2
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x00000000001A0000  Size: 8
// CHECK:     Address: 0x00000000001A0000  Data is: 0x00000004
// CHECK:     Address: 0x00000000001A0004  Data is: 0x041A0000
// CHECK:     Address: 0x00000000001A0008  Data is: 0x00040000
// CHECK:     Address: 0x00000000001A000C  Data is: 0x00000000
// CHECK:     Address: 0x00000000001A0010  Data is: 0x00000000
// CHECK:     Address: 0x00000000001A0014  Data is: 0x00020000
// CHECK:     Address: 0x00000000001A0018  Data is: 0x00000000
// CHECK:     Address: 0x00000000001A001C  Data is: 0x8140FF41

// CHECK: (NOP Command): Payload Length: 0
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x00000000001A0020  Size: 8
// CHECK:     Address: 0x00000000001A0020  Data is: 0x00000004
// CHECK:     Address: 0x00000000001A0024  Data is: 0x04020004
// CHECK:     Address: 0x00000000001A0028  Data is: 0x00040000
// CHECK:     Address: 0x00000000001A002C  Data is: 0x00000000
// CHECK:     Address: 0x00000000001A0030  Data is: 0x00000000
// CHECK:     Address: 0x00000000001A0034  Data is: 0x00020000
// CHECK:     Address: 0x00000000001A0038  Data is: 0x00000000
// CHECK:     Address: 0x00000000001A003C  Data is: 0x8140FF41

// CHECK: (Write64): Address:  0x00000000001A0604 Data:  0x00010000
// CHECK: (MaskWrite64): Address: 0x00000000001A0600  Mask: 0x00000000  Data: 0x00000001
