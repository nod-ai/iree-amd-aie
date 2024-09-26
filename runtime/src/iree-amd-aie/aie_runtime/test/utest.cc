// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*
clang-format off

RUN: (aie_runtime_utest %S/pi.elf) | FileCheck %s

CHECK: Generating: pi.cdo
CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x0000000000200404  Size: 9
CHECK:     Address: 0x0000000000200404  Data is: 0x00000170
CHECK:     Address: 0x0000000000200408  Data is: 0x00000000
CHECK:     Address: 0x000000000020040C  Data is: 0x00000000
CHECK:     Address: 0x0000000000200410  Data is: 0x00000000
CHECK:     Address: 0x0000000000200414  Data is: 0x00000000
CHECK:     Address: 0x0000000000200418  Data is: 0x00000000
CHECK:     Address: 0x000000000020041C  Data is: 0x00000000
CHECK:     Address: 0x0000000000200420  Data is: 0x00000000
CHECK:     Address: 0x0000000000200424  Data is: 0x00000008

CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x0000000000220000  Size: 176
CHECK:     Address: 0x0000000000220000  Data is: 0x38001043
CHECK:     Address: 0x0000000000220004  Data is: 0x000001C3
CHECK:     Address: 0x0000000000220008  Data is: 0x08000055
CHECK:     Address: 0x000000000022000C  Data is: 0x00550000
CHECK:     Address: 0x0000000000220010  Data is: 0x00000C00
CHECK:     Address: 0x0000000000220014  Data is: 0x16310799
CHECK:     Address: 0x0000000000220018  Data is: 0x40400195
CHECK:     Address: 0x000000000022001C  Data is: 0x0001C000
CHECK:     Address: 0x0000000000220020  Data is: 0x00010001
CHECK:     Address: 0x0000000000220024  Data is: 0x00010001
CHECK:     Address: 0x0000000000220028  Data is: 0xFC7FF855
CHECK:     Address: 0x000000000022002C  Data is: 0x7659FFFF
CHECK:     Address: 0x0000000000220030  Data is: 0x782F1F30
CHECK:     Address: 0x0000000000220034  Data is: 0x04B20000
CHECK:     Address: 0x0000000000220038  Data is: 0x00000062
CHECK:     Address: 0x000000000022003C  Data is: 0x00000000
CHECK:     Address: 0x0000000000220040  Data is: 0xDC8C764D
CHECK:     Address: 0x0000000000220044  Data is: 0x0001DFF0
CHECK:     Address: 0x0000000000220048  Data is: 0x00010001
CHECK:     Address: 0x000000000022004C  Data is: 0x00010001
CHECK:     Address: 0x0000000000220050  Data is: 0x14190001
CHECK:     Address: 0x0000000000220054  Data is: 0x00011000
CHECK:     Address: 0x0000000000220058  Data is: 0x00010001
CHECK:     Address: 0x000000000022005C  Data is: 0x00010001
CHECK:     Address: 0x0000000000220060  Data is: 0x14B10899
CHECK:     Address: 0x0000000000220064  Data is: 0x20400195
CHECK:     Address: 0x0000000000220068  Data is: 0x0001C000
CHECK:     Address: 0x000000000022006C  Data is: 0x00010001
CHECK:     Address: 0x0000000000220070  Data is: 0x00010001
CHECK:     Address: 0x0000000000220074  Data is: 0x183E7659
CHECK:     Address: 0x0000000000220078  Data is: 0x244B2003
CHECK:     Address: 0x000000000022007C  Data is: 0x00000000
CHECK:     Address: 0x0000000000220080  Data is: 0x70000115
CHECK:     Address: 0x0000000000220084  Data is: 0x00010000
CHECK:     Address: 0x0000000000220088  Data is: 0x00010001
CHECK:     Address: 0x000000000022008C  Data is: 0x00010001
CHECK:     Address: 0x0000000000220090  Data is: 0x10000115
CHECK:     Address: 0x0000000000220094  Data is: 0x06990001
CHECK:     Address: 0x0000000000220098  Data is: 0x00011830
CHECK:     Address: 0x000000000022009C  Data is: 0x00010001
CHECK:     Address: 0x00000000002200A0  Data is: 0x880003C0
CHECK:     Address: 0x00000000002200A4  Data is: 0x00000003
CHECK:     Address: 0x00000000002200A8  Data is: 0x00000000
CHECK:     Address: 0x00000000002200AC  Data is: 0x00000000
CHECK:     Address: 0x00000000002200B0  Data is: 0x00010001
CHECK:     Address: 0x00000000002200B4  Data is: 0x00010001
CHECK:     Address: 0x00000000002200B8  Data is: 0x00010001
CHECK:     Address: 0x00000000002200BC  Data is: 0x10000819
CHECK:     Address: 0x00000000002200C0  Data is: 0x00010001
CHECK:     Address: 0x00000000002200C4  Data is: 0x00010001
CHECK:     Address: 0x00000000002200C8  Data is: 0x00010001
CHECK:     Address: 0x00000000002200CC  Data is: 0x00000019
CHECK:     Address: 0x00000000002200D0  Data is: 0x68000095
CHECK:     Address: 0x00000000002200D4  Data is: 0x00010000
CHECK:     Address: 0x00000000002200D8  Data is: 0x00010001
CHECK:     Address: 0x00000000002200DC  Data is: 0x00010001
CHECK:     Address: 0x00000000002200E0  Data is: 0x38032019
CHECK:     Address: 0x00000000002200E4  Data is: 0x0FFFC299
CHECK:     Address: 0x00000000002200E8  Data is: 0x98000115
CHECK:     Address: 0x00000000002200EC  Data is: 0xC8430000
CHECK:     Address: 0x00000000002200F0  Data is: 0x060827FF
CHECK:     Address: 0x00000000002200F4  Data is: 0x00010000
CHECK:     Address: 0x00000000002200F8  Data is: 0x00010001
CHECK:     Address: 0x00000000002200FC  Data is: 0x00000019
CHECK:     Address: 0x0000000000220100  Data is: 0x07FFC2D9
CHECK:     Address: 0x0000000000220104  Data is: 0xA8000095
CHECK:     Address: 0x0000000000220108  Data is: 0x00010000
CHECK:     Address: 0x000000000022010C  Data is: 0x86550001
CHECK:     Address: 0x0000000000220110  Data is: 0x4048FC0B
CHECK:     Address: 0x0000000000220114  Data is: 0x1A0010B7
CHECK:     Address: 0x0000000000220118  Data is: 0x000001C0
CHECK:     Address: 0x000000000022011C  Data is: 0x00000000
CHECK:     Address: 0x0000000000220120  Data is: 0xC80003C0
CHECK:     Address: 0x0000000000220124  Data is: 0x06282003
CHECK:     Address: 0x0000000000220128  Data is: 0x0002B000
CHECK:     Address: 0x000000000022012C  Data is: 0xFFFC0000
CHECK:     Address: 0x0000000000220130  Data is: 0x10121219
CHECK:     Address: 0x0000000000220134  Data is: 0x10001819
CHECK:     Address: 0x0000000000220138  Data is: 0x00010001
CHECK:     Address: 0x000000000022013C  Data is: 0x00010001
CHECK:     Address: 0x0000000000220140  Data is: 0x880003C0
CHECK:     Address: 0x0000000000220144  Data is: 0x00000003
CHECK:     Address: 0x0000000000220148  Data is: 0x00000000
CHECK:     Address: 0x000000000022014C  Data is: 0x00000000
CHECK:     Address: 0x0000000000220150  Data is: 0x00010001
CHECK:     Address: 0x0000000000220154  Data is: 0x00010001
CHECK:     Address: 0x0000000000220158  Data is: 0x00010001
CHECK:     Address: 0x000000000022015C  Data is: 0x10101219
CHECK:     Address: 0x0000000000220160  Data is: 0x10001819
CHECK:     Address: 0x0000000000220164  Data is: 0x00010001
CHECK:     Address: 0x0000000000220168  Data is: 0x00010001
CHECK:     Address: 0x000000000022016C  Data is: 0x00000019
CHECK:     Address: 0x0000000000220170  Data is: 0x180B9659
CHECK:     Address: 0x0000000000220174  Data is: 0x0C000055
CHECK:     Address: 0x0000000000220178  Data is: 0x76590000
CHECK:     Address: 0x000000000022017C  Data is: 0x0055183E
CHECK:     Address: 0x0000000000220180  Data is: 0x00000B80
CHECK:     Address: 0x0000000000220184  Data is: 0x9E0B2843
CHECK:     Address: 0x0000000000220188  Data is: 0x002F8C3F
CHECK:     Address: 0x000000000022018C  Data is: 0xF8400195
CHECK:     Address: 0x0000000000220190  Data is: 0x2019C000
CHECK:     Address: 0x0000000000220194  Data is: 0xEC193803
CHECK:     Address: 0x0000000000220198  Data is: 0x42990FFF
CHECK:     Address: 0x000000000022019C  Data is: 0xC0190FFE
CHECK:     Address: 0x00000000002201A0  Data is: 0x782F0FFE
CHECK:     Address: 0x00000000002201A4  Data is: 0x00380000
CHECK:     Address: 0x00000000002201A8  Data is: 0x46800040
CHECK:     Address: 0x00000000002201AC  Data is: 0x000007FF
CHECK:     Address: 0x00000000002201B0  Data is: 0x070386D9
CHECK:     Address: 0x00000000002201B4  Data is: 0x00010001
CHECK:     Address: 0x00000000002201B8  Data is: 0x00010001
CHECK:     Address: 0x00000000002201BC  Data is: 0x00010001
CHECK:     Address: 0x00000000002201C0  Data is: 0x10001419
CHECK:     Address: 0x00000000002201C4  Data is: 0x1D8E7659
CHECK:     Address: 0x00000000002201C8  Data is: 0x00010001
CHECK:     Address: 0x00000000002201CC  Data is: 0x00010001
CHECK:     Address: 0x00000000002201D0  Data is: 0x15F16899
CHECK:     Address: 0x00000000002201D4  Data is: 0xD8400195
CHECK:     Address: 0x00000000002201D8  Data is: 0x0001C000
CHECK:     Address: 0x00000000002201DC  Data is: 0x00010001
CHECK:     Address: 0x00000000002201E0  Data is: 0x782F0001
CHECK:     Address: 0x00000000002201E4  Data is: 0x00380000
CHECK:     Address: 0x00000000002201E8  Data is: 0x00000040
CHECK:     Address: 0x00000000002201EC  Data is: 0x00000000
CHECK:     Address: 0x00000000002201F0  Data is: 0x07FE42D9
CHECK:     Address: 0x00000000002201F4  Data is: 0x07FEEE59
CHECK:     Address: 0x00000000002201F8  Data is: 0x07FFEC59
CHECK:     Address: 0x00000000002201FC  Data is: 0x07FF7ED9
CHECK:     Address: 0x0000000000220200  Data is: 0x00010001
CHECK:     Address: 0x0000000000220204  Data is: 0x18190001
CHECK:     Address: 0x0000000000220208  Data is: 0x00011000
CHECK:     Address: 0x000000000022020C  Data is: 0x00010001
CHECK:     Address: 0x0000000000220210  Data is: 0x007F0001
CHECK:     Address: 0x0000000000220214  Data is: 0x00710000
CHECK:     Address: 0x0000000000220218  Data is: 0xE0000000
CHECK:     Address: 0x000000000022021C  Data is: 0x000007FF
CHECK:     Address: 0x0000000000220220  Data is: 0x1A1210BB
CHECK:     Address: 0x0000000000220224  Data is: 0x480001C0
CHECK:     Address: 0x0000000000220228  Data is: 0x10BB0100
CHECK:     Address: 0x000000000022022C  Data is: 0x01C04202
CHECK:     Address: 0x0000000000220230  Data is: 0x0050C800
CHECK:     Address: 0x0000000000220234  Data is: 0x249B293B
CHECK:     Address: 0x0000000000220238  Data is: 0x403E17AA
CHECK:     Address: 0x000000000022023C  Data is: 0x36590050
CHECK:     Address: 0x0000000000220240  Data is: 0xF6591C06
CHECK:     Address: 0x0000000000220244  Data is: 0xC0551C84
CHECK:     Address: 0x0000000000220248  Data is: 0x00000C64
CHECK:     Address: 0x000000000022024C  Data is: 0x00010001
CHECK:     Address: 0x0000000000220250  Data is: 0x18C1F679
CHECK:     Address: 0x0000000000220254  Data is: 0x10C42099
CHECK:     Address: 0x0000000000220258  Data is: 0x988B2843
CHECK:     Address: 0x000000000022025C  Data is: 0x000730F7
CHECK:     Address: 0x0000000000220260  Data is: 0x07038ED9
CHECK:     Address: 0x0000000000220264  Data is: 0x07FB86D9
CHECK:     Address: 0x0000000000220268  Data is: 0x00010001
CHECK:     Address: 0x000000000022026C  Data is: 0x00010001
CHECK:     Address: 0x0000000000220270  Data is: 0x54190001
CHECK:     Address: 0x0000000000220274  Data is: 0x00011000
CHECK:     Address: 0x0000000000220278  Data is: 0x00010001
CHECK:     Address: 0x000000000022027C  Data is: 0x00000019
CHECK:     Address: 0x0000000000220280  Data is: 0x280003C0
CHECK:     Address: 0x0000000000220284  Data is: 0x0002800B
CHECK:     Address: 0x0000000000220288  Data is: 0x00000000
CHECK:     Address: 0x000000000022028C  Data is: 0x00000000
CHECK:     Address: 0x0000000000220290  Data is: 0x14E78C19
CHECK:     Address: 0x0000000000220294  Data is: 0x180A1659
CHECK:     Address: 0x0000000000220298  Data is: 0x00010001
CHECK:     Address: 0x000000000022029C  Data is: 0x00010001
CHECK:     Address: 0x00000000002202A0  Data is: 0x1A791659
CHECK:     Address: 0x00000000002202A4  Data is: 0x10001819
CHECK:     Address: 0x00000000002202A8  Data is: 0x8C0B2843
CHECK:     Address: 0x00000000002202AC  Data is: 0x00232101
CHECK:     Address: 0x00000000002202B0  Data is: 0x00010001
CHECK:     Address: 0x00000000002202B4  Data is: 0x8EBB0001
CHECK:     Address: 0x00000000002202B8  Data is: 0x00000003
CHECK:     Address: 0x00000000002202BC  Data is: 0x00000000

CHECK: (MaskWrite64): Address: 0x0000000000232000  Mask: 0x00000002  Data: 0x00000002
CHECK: (MaskWrite64): Address: 0x0000000000232000  Mask: 0x00000002  Data: 0x00000000
CHECK: (Write64): Address:  0x000000000021F000 Data:  0x00000001
CHECK: (Write64): Address:  0x000000000021F010 Data:  0x00000000
CHECK: (NOP Command): Payload Length: 2
CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x000000000021D000  Size: 6
CHECK:     Address: 0x000000000021D000  Data is: 0x00400001
CHECK:     Address: 0x000000000021D004  Data is: 0x00000000
CHECK:     Address: 0x000000000021D008  Data is: 0x00000000
CHECK:     Address: 0x000000000021D00C  Data is: 0x00000000
CHECK:     Address: 0x000000000021D010  Data is: 0x00000000
CHECK:     Address: 0x000000000021D014  Data is: 0x02041FE1

CHECK: (Write64): Address:  0x000000000021DE14 Data:  0x00000000
CHECK: (MaskWrite64): Address: 0x000000000021DE10  Mask: 0x00000000  Data: 0x00000001
CHECK: (Write64): Address:  0x000000000003F008 Data:  0x80000000
CHECK: (Write64): Address:  0x000000000003F100 Data:  0x80000000
CHECK: (Write64): Address:  0x000000000003F010 Data:  0x8000000E
CHECK: (Write64): Address:  0x000000000003F138 Data:  0x80000000
CHECK: (Write64): Address:  0x00000000001B001C Data:  0x8000000D
CHECK: (Write64): Address:  0x00000000001B0134 Data:  0x80000000
CHECK: (Write64): Address:  0x000000000023F014 Data:  0x80000001
CHECK: (Write64): Address:  0x000000000023F104 Data:  0x80000000
CHECK: (MaskWrite64): Address: 0x000000000001F004  Mask: 0x00000030  Data: 0x00000010
CHECK: (MaskWrite64): Address: 0x0000000000232000  Mask: 0x00000001  Data: 0x00000001

clang-format off
*/

#include <string>

#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"

int main(int argc, char** argv) {
  std::string elfPath(argv[1]);

  uint8_t col = 0;
  uint8_t partitionStartCol = 1;
  uint8_t partitionNumCols = 4;
  mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel::AMDAIEDeviceConfig
      deviceConfig;
  deviceConfig.streamSwitchCoreArbiterMax = XAIE2IPU_SS_ARBITER_MAX;
  deviceConfig.streamSwitchCoreMSelMax = XAIE2IPU_SS_MSEL_MAX;
  deviceConfig.streamSwitchMemTileArbiterMax = XAIE2IPU_SS_ARBITER_MAX;
  deviceConfig.streamSwitchMemTileMSelMax = XAIE2IPU_SS_MSEL_MAX;
  deviceConfig.streamSwitchShimArbiterMax = XAIE2IPU_SS_ARBITER_MAX;
  deviceConfig.streamSwitchShimMSelMax = XAIE2IPU_SS_MSEL_MAX;
  mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel deviceModel(
      XAIE_DEV_GEN_AIE2IPU, XAIE2IPU_BASE_ADDR, XAIE2IPU_COL_SHIFT,
      XAIE2IPU_ROW_SHIFT, XAIE2IPU_NUM_COLS, XAIE2IPU_NUM_ROWS,
      XAIE2IPU_MEM_TILE_ROW_START, XAIE2IPU_MEM_TILE_NUM_ROWS,
      XAIE2IPU_SHIM_NUM_ROWS, partitionNumCols, partitionStartCol,
      /*partBaseAddr*/ XAIE2IPU_PARTITION_BASE_ADDR,
      /*npiAddr*/ XAIE2IPU_NPI_BASEADDR,
      /*aieSim*/ false, /*xaieDebug*/ false,
      /*device*/ mlir::iree_compiler::AMDAIE::AMDAIEDevice::npu1_4col,
      /*deviceConfig*/ std::move(deviceConfig));
  XAie_LocType tile00 = {/*Row*/ 0, /*Col*/ col};
  XAie_LocType tile01 = {/*Row*/ 1, /*Col*/ col};
  XAie_LocType tile02 = {/*Row*/ 2, /*Col*/ col};
  XAie_Lock lock01 = {/*LockId*/ 0, /*LockVal*/ 1};
  XAie_Lock lock10 = {/*LockId*/ 1, /*LockVal*/ 0};

  EnAXIdebug();
  setEndianness(Little_Endian);
  startCDOFileStream("pi.cdo");
  FileHeader();

  XAie_LoadElf(&deviceModel.devInst, tile02, elfPath.c_str(),
               /*LoadSym*/ false);

  XAie_CoreReset(&deviceModel.devInst, tile02);
  XAie_CoreUnreset(&deviceModel.devInst, tile02);
  XAie_LockSetValue(&deviceModel.devInst, tile02, lock01);
  XAie_LockSetValue(&deviceModel.devInst, tile02, lock10);

  XAie_DmaDesc dmaDesc;
  XAie_DmaDescInit(&deviceModel.devInst, &dmaDesc, tile02);
  lock10 = XAie_Lock{/*LockId*/ 1, /*LockVal*/ -1};
  dmaDesc.DmaMod->SetLock(&dmaDesc, lock10, lock01, /*AcqEn*/ 1,
                          /*RelEn*/ 0);
  // address 1024 is the beginning of the core's stack
  XAie_DmaSetAddrLen(&dmaDesc, /*Addr*/ 1024, /*Len*/ 4);
  XAie_DmaEnableBd(&dmaDesc);
  uint8_t bdNum = 0, chNum = 0;
  XAie_DmaWriteBd(&deviceModel.devInst, &dmaDesc, tile02, bdNum);
  XAie_DmaChannelSetStartQueue(&deviceModel.devInst, tile02, chNum,
                               XAie_DmaDirection::DMA_MM2S, bdNum,
                               /*RepeatCount*/ 1, /*EnTokenIssue*/ 0);
  XAie_DmaChannelEnable(&deviceModel.devInst, tile02, chNum,
                        XAie_DmaDirection::DMA_MM2S);

  // Slave == source, Master == destination
  // Note, these are internal connections in the switch
  // (so src port -> dest port within the switch itself)
  XAie_StrmConnCctEnable(&deviceModel.devInst, tile00,
                         /*Slave*/ StrmSwPortType::CTRL,
                         /*SlvPortNum*/ 0, /*Master*/ StrmSwPortType::SOUTH,
                         /*MstrPortNum*/ 0);
  XAie_StrmConnCctEnable(&deviceModel.devInst, tile00,
                         /*Slave*/ StrmSwPortType::NORTH,
                         /*SlvPortNum*/ 0,
                         /*Master*/ StrmSwPortType::SOUTH, /*MstrPortNum*/ 2);
  XAie_StrmConnCctEnable(&deviceModel.devInst, tile01,
                         /*Slave*/ StrmSwPortType::NORTH,
                         /*SlvPortNum*/ 0,
                         /*Master*/ StrmSwPortType::SOUTH, /*MstrPortNum*/ 0);
  XAie_StrmConnCctEnable(&deviceModel.devInst, tile02,
                         /*Slave*/ StrmSwPortType::DMA,
                         /*SlvPortNum*/ 0,
                         /*Master*/ StrmSwPortType::SOUTH, /*MstrPortNum*/ 0);
  XAie_EnableAieToShimDmaStrmPort(&deviceModel.devInst, tile00, /*PortNum*/ 2);
  XAie_CoreEnable(&deviceModel.devInst, tile02);

  configureHeader();
  endCurrentCDOFileStream();
}
