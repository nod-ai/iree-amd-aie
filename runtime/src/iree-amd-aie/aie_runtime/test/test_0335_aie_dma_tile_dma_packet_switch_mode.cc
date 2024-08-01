// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*
clang-format off

RUN: test_0335_aie_dma_tile_dma_packet_switch_mode | FileCheck %s

CHECK: Header version 0.1
CHECK: Device Generation: 2
CHECK: Cols, Rows, NumMemRows : (5, 6, 1)
CHECK: TransactionSize: 1168
CHECK: NumOps: 31
CHECK: XAie_BlockWrite32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_BLOCKWRITE, Col: 2, Row: 0), Col: 0, Row: 0, RegOff: 35667968, Size: 144)
CHECK:    0x42204000, 0x1
CHECK:    0x42204004, 0x2
CHECK:    0x42204008, 0x3
CHECK:    0x4220400c, 0x4
CHECK:    0x42204010, 0x5
CHECK:    0x42204014, 0x6
CHECK:    0x42204018, 0x7
CHECK:    0x4220401c, 0x8
CHECK:    0x42204020, 0x9
CHECK:    0x42204024, 0xa
CHECK:    0x42204028, 0xb
CHECK:    0x4220402c, 0xc
CHECK:    0x42204030, 0xd
CHECK:    0x42204034, 0xe
CHECK:    0x42204038, 0xf
CHECK:    0x4220403c, 0x10
CHECK:    0x42204040, 0x11
CHECK:    0x42204044, 0x12
CHECK:    0x42204048, 0x13
CHECK:    0x4220404c, 0x14
CHECK:    0x42204050, 0x15
CHECK:    0x42204054, 0x16
CHECK:    0x42204058, 0x17
CHECK:    0x4220405c, 0x18
CHECK:    0x42204060, 0x19
CHECK:    0x42204064, 0x1a
CHECK:    0x42204068, 0x1b
CHECK:    0x4220406c, 0x1c
CHECK:    0x42204070, 0x1d
CHECK:    0x42204074, 0x1e
CHECK:    0x42204078, 0x1f
CHECK:    0x4220407c, 0x20
CHECK: XAie_BlockWrite32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_BLOCKWRITE, Col: 3, Row: 0), Col: 0, Row: 0, RegOff: 36716544, Size: 144)
CHECK:    0x42304000, 0x21
CHECK:    0x42304004, 0x22
CHECK:    0x42304008, 0x23
CHECK:    0x4230400c, 0x24
CHECK:    0x42304010, 0x25
CHECK:    0x42304014, 0x26
CHECK:    0x42304018, 0x27
CHECK:    0x4230401c, 0x28
CHECK:    0x42304020, 0x29
CHECK:    0x42304024, 0x2a
CHECK:    0x42304028, 0x2b
CHECK:    0x4230402c, 0x2c
CHECK:    0x42304030, 0x2d
CHECK:    0x42304034, 0x2e
CHECK:    0x42304038, 0x2f
CHECK:    0x4230403c, 0x30
CHECK:    0x42304040, 0x31
CHECK:    0x42304044, 0x32
CHECK:    0x42304048, 0x33
CHECK:    0x4230404c, 0x34
CHECK:    0x42304050, 0x35
CHECK:    0x42304054, 0x36
CHECK:    0x42304058, 0x37
CHECK:    0x4230405c, 0x38
CHECK:    0x42304060, 0x39
CHECK:    0x42304064, 0x3a
CHECK:    0x42304068, 0x3b
CHECK:    0x4230406c, 0x3c
CHECK:    0x42304070, 0x3d
CHECK:    0x42304074, 0x3e
CHECK:    0x42304078, 0x3f
CHECK:    0x4230407c, 0x40
CHECK: XAie_Write32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_WRITE, Col: 2, Row: 4), RegOff: 35909892, Value: 3221225472, Size: 24)
CHECK: XAie_Write32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_WRITE, Col: 3, Row: 20), RegOff: 36958484, Value: 3221225472, Size: 24)
CHECK: XAie_Write32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_WRITE, Col: 2, Row: 16), RegOff: 35910160, Value: 2031872, Size: 24)
CHECK: XAie_Write32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_WRITE, Col: 3, Row: 80), RegOff: 36958800, Value: 2031872, Size: 24)
CHECK: XAie_Write32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_WRITE, Col: 2, Row: 52), RegOff: 35909684, Value: 3221225480, Size: 24)
CHECK: XAie_Write32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_WRITE, Col: 3, Row: 4), RegOff: 36958212, Value: 3221225608, Size: 24)
CHECK: XAie_MaskWrite32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_MASKWRITE, Col: 2, Row: 16), RegOff: 35773968, Value: 2, Mask: 2, Size: 32)
CHECK: XAie_MaskWrite32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_MASKWRITE, Col: 2, Row: 24), RegOff: 35773976, Value: 2, Mask: 2, Size: 32)
CHECK: XAie_MaskWrite32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_MASKWRITE, Col: 2, Row: 0), RegOff: 35773952, Value: 2, Mask: 2, Size: 32)
CHECK: XAie_MaskWrite32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_MASKWRITE, Col: 2, Row: 8), RegOff: 35773960, Value: 2, Mask: 2, Size: 32)
CHECK: XAie_MaskWrite32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_MASKWRITE, Col: 2, Row: 16), RegOff: 35773968, Value: 0, Mask: 2, Size: 32)
CHECK: XAie_MaskWrite32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_MASKWRITE, Col: 2, Row: 24), RegOff: 35773976, Value: 0, Mask: 2, Size: 32)
CHECK: XAie_MaskWrite32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_MASKWRITE, Col: 2, Row: 0), RegOff: 35773952, Value: 0, Mask: 2, Size: 32)
CHECK: XAie_MaskWrite32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_MASKWRITE, Col: 2, Row: 8), RegOff: 35773960, Value: 0, Mask: 2, Size: 32)
CHECK: XAie_MaskWrite32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_MASKWRITE, Col: 3, Row: 16), RegOff: 36822544, Value: 2, Mask: 2, Size: 32)
CHECK: XAie_MaskWrite32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_MASKWRITE, Col: 3, Row: 24), RegOff: 36822552, Value: 2, Mask: 2, Size: 32)
CHECK: XAie_MaskWrite32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_MASKWRITE, Col: 3, Row: 0), RegOff: 36822528, Value: 2, Mask: 2, Size: 32)
CHECK: XAie_MaskWrite32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_MASKWRITE, Col: 3, Row: 8), RegOff: 36822536, Value: 2, Mask: 2, Size: 32)
CHECK: XAie_MaskWrite32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_MASKWRITE, Col: 3, Row: 16), RegOff: 36822544, Value: 0, Mask: 2, Size: 32)
CHECK: XAie_MaskWrite32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_MASKWRITE, Col: 3, Row: 24), RegOff: 36822552, Value: 0, Mask: 2, Size: 32)
CHECK: XAie_MaskWrite32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_MASKWRITE, Col: 3, Row: 0), RegOff: 36822528, Value: 0, Mask: 2, Size: 32)
CHECK: XAie_MaskWrite32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_MASKWRITE, Col: 3, Row: 8), RegOff: 36822536, Value: 0, Mask: 2, Size: 32)
CHECK: XAie_BlockWrite32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_BLOCKWRITE, Col: 2, Row: 32), Col: 0, Row: 0, RegOff: 35770400, Size: 40)
CHECK:    0x4221d020, 0x4000020
CHECK:    0x4221d024, 0x40030000
CHECK:    0x4221d028, 0x0
CHECK:    0x4221d02c, 0x0
CHECK:    0x4221d030, 0x0
CHECK:    0x4221d034, 0x0
CHECK: XAie_BlockWrite32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_BLOCKWRITE, Col: 3, Row: 32), Col: 0, Row: 0, RegOff: 36819232, Size: 40)
CHECK:    0x4231d120, 0x4000020
CHECK:    0x4231d124, 0x0
CHECK:    0x4231d128, 0x0
CHECK:    0x4231d12c, 0x0
CHECK:    0x4231d130, 0x0
CHECK:    0x4231d134, 0x0
CHECK: XAie_Write32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_WRITE, Col: 2, Row: 20), RegOff: 35773972, Value: 1, Size: 24)
CHECK: XAie_Write32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_WRITE, Col: 3, Row: 4), RegOff: 36822532, Value: 9, Size: 24)
CHECK: XAie_Write32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_WRITE, Col: 2, Row: 16), RegOff: 35773968, Value: 1, Size: 24)
CHECK: XAie_Write32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_WRITE, Col: 3, Row: 0), RegOff: 36822528, Value: 1, Size: 24)
CHECK: XAie_MaskPoll32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_MASKPOLL, Col: 3, Row: 0), RegOff: 36822784, Value: 0, Size: 32)

clang-format on
*/

#include <cstdint>
#include <cstdio>

#include "interpreter_op_impl.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"

using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::AMDAIE;

#define PACKET_SWITCH_MODE_TEST \
  1  // 1 - Paket switch mode test, 0 - Circuit switch mode test

/************************** Constant Definitions *****************************/
/* Input and output address in aie data memory */
#define DATA_MEM_INOUT_ADDR 0x4000

#define NUM_ELEMS 32

#define MM2S_BD_NUM_1 1
#define S2MM_BD_NUM_9 9

/************************** Function Definitions *****************************/
int main(int argc, char **argv) {
  // setup aie-rt
  XAie_SetupConfig(ConfigPtr, XAIE_DEV_GEN_AIEML, XAIE2IPU_BASE_ADDR,
                   XAIE2IPU_COL_SHIFT, XAIE2IPU_ROW_SHIFT, XAIE2IPU_NUM_COLS,
                   XAIE2IPU_NUM_ROWS, XAIE2IPU_SHIM_ROW,
                   XAIE2IPU_MEM_TILE_ROW_START, XAIE2IPU_MEM_TILE_NUM_ROWS,
                   XAIE2IPU_AIE_TILE_ROW_START, XAIE2IPU_AIE_TILE_NUM_ROWS);

  XAie_InstDeclare(DevInst, &ConfigPtr);
  AieRC RC = XAie_CfgInitialize(&DevInst, &ConfigPtr);
  XAie_TurnEccOff(&DevInst);

  // config code

  XAie_StartTransaction(&DevInst, XAIE_TRANSACTION_DISABLE_AUTO_FLUSH);

  uint32_t data[NUM_ELEMS];
  uint32_t buffer[NUM_ELEMS];
  XAie_LocType Tile_1, Tile_2;
  XAie_DmaDesc Tile_1_MM2S, Tile_2_S2MM;

  Tile_1 = XAie_TileLoc(1, XAIE2IPU_AIE_TILE_ROW_START);
  Tile_2 = XAie_TileLoc(1, XAIE2IPU_AIE_TILE_ROW_START + 1);

  /* Initialize array with random integers */
  for (uint8_t i = 0U; i < NUM_ELEMS; i++) {
    data[i] = i + 1;
    buffer[i] = NUM_ELEMS + i + 1;
  }

  /* Write data to aie tile data memory */
  RC = XAie_DataMemBlockWrite(&DevInst, Tile_1, DATA_MEM_INOUT_ADDR,
                              (void *)data, sizeof(uint32_t) * NUM_ELEMS);
  RC = XAie_DataMemBlockWrite(&DevInst, Tile_2, DATA_MEM_INOUT_ADDR,
                              (void *)buffer, sizeof(uint32_t) * NUM_ELEMS);
  if (RC != XAIE_OK) {
    fprintf(stderr, "Writing data to aie data memory failed.\n");
    return -1;
  }

  /* Configure stream switch ports to move data from Tile_1 to Tile_2 */
#if PACKET_SWITCH_MODE_TEST
  XAie_Packet Pkt;
  Pkt.PktId = 0;
  Pkt.PktType = 0;

  RC = XAie_StrmPktSwSlavePortEnable(&DevInst, Tile_1, DMA, 0);
  RC = XAie_StrmPktSwSlavePortEnable(&DevInst, Tile_2, SOUTH, 0);
  RC = XAie_StrmPktSwSlaveSlotEnable(
      &DevInst, Tile_1, DMA, 0, 0, Pkt, 0x1F, 0,
      0);  // Slot-0, pkt, Mask-1F, MSel-0, Arbitor-0
  RC = XAie_StrmPktSwSlaveSlotEnable(&DevInst, Tile_2, SOUTH, 0, 0, Pkt, 0x1F,
                                     0, 0);

  RC = XAie_StrmPktSwMstrPortEnable(&DevInst, Tile_1, NORTH, 0,
                                    XAIE_SS_PKT_DONOT_DROP_HEADER, 0,
                                    0x1);  // Arbitor-0, MSelEn-0x1
  RC = XAie_StrmPktSwMstrPortEnable(&DevInst, Tile_2, DMA, 0,
                                    XAIE_SS_PKT_DROP_HEADER, 0, 0x1);
#else
  RC = XAie_StrmConnCctEnable(&DevInst, Tile_1, DMA, 0, NORTH, 0);
  RC = XAie_StrmConnCctEnable(&DevInst, Tile_2, SOUTH, 0, DMA, 0);
#endif
  if (RC != XAIE_OK) {
    fprintf(stderr, "Failed to configure stream switches.\n");
    return -1;
  }

  /* Reset All Channels of DMA to make sure DMA channels at idle state */
  RC = XAie_DmaChannelResetAll(&DevInst, Tile_1, DMA_CHANNEL_RESET);
  RC = XAie_DmaChannelResetAll(&DevInst, Tile_1, DMA_CHANNEL_UNRESET);
  RC = XAie_DmaChannelResetAll(&DevInst, Tile_2, DMA_CHANNEL_RESET);
  RC = XAie_DmaChannelResetAll(&DevInst, Tile_2, DMA_CHANNEL_UNRESET);
  if (RC != XAIE_OK) {
    fprintf(stderr, "XAie_DmaChannelResetAll Failed.\n");
    return -1;
  }

  /* Initialize software descriptors for aie dma */
  RC = XAie_DmaDescInit(&DevInst, &Tile_1_MM2S, Tile_1);
  RC = XAie_DmaDescInit(&DevInst, &Tile_2_S2MM, Tile_2);

  /* Configure address and length in dma software descriptors */
#if PACKET_SWITCH_MODE_TEST
  Pkt.PktId = 0;
  Pkt.PktType = 3;
  RC = XAie_DmaSetPkt(&Tile_1_MM2S, Pkt);
#endif
  RC = XAie_DmaSetAddrLen(&Tile_1_MM2S, DATA_MEM_INOUT_ADDR,
                          NUM_ELEMS * sizeof(uint32_t));
  RC = XAie_DmaSetAddrLen(&Tile_2_S2MM, DATA_MEM_INOUT_ADDR,
                          NUM_ELEMS * sizeof(uint32_t));

  /*
   * Configure aie dma hardware using software descriptors. Use buffer
   * descriptor 1 for MM2S and 9 for S2MM on both tiles.
   */
  RC = XAie_DmaWriteBd(&DevInst, &Tile_1_MM2S, Tile_1, MM2S_BD_NUM_1);
  RC = XAie_DmaWriteBd(&DevInst, &Tile_2_S2MM, Tile_2, S2MM_BD_NUM_9);

  /* Push Bd numbers to aie dma channel queues and enable the channels */
  RC = XAie_DmaChannelPushBdToQueue(&DevInst, Tile_1, 0U, DMA_MM2S,
                                    MM2S_BD_NUM_1);
  RC = XAie_DmaChannelPushBdToQueue(&DevInst, Tile_2, 0U, DMA_S2MM,
                                    S2MM_BD_NUM_9);

  /* Enable the buffer descriptors in software dma descriptors */
  RC = XAie_DmaEnableBd(&Tile_1_MM2S);
  RC = XAie_DmaEnableBd(&Tile_2_S2MM);
  if (RC != XAIE_OK) {
    fprintf(stderr, "Failed to setup software dma descriptors.\n");
    return -1;
  }

  RC = XAie_DmaChannelEnable(&DevInst, Tile_1, 0U, DMA_MM2S);
  RC = XAie_DmaChannelEnable(&DevInst, Tile_2, 0U, DMA_S2MM);

  if (RC != XAIE_OK) {
    fprintf(
        stderr,
        "Failed to configure aie dma hardware and start dma tranactions.\n");
    return -1;
  }

  u8 TimeOut = 5;
  while (TimeOut && XAie_DmaWaitForDone(&DevInst, Tile_2, 0, DMA_S2MM, 1000)) {
    TimeOut--;
  }

  SubmitSerializedTransaction(DevInst, /*startColIdx*/ 0);

  XAie_Finish(&DevInst);

  return 0;
}
