// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <unistd.h>

#include <cstdio>

#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"

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
  AieRC RC = XAIE_OK;

  // initialize cdo-driver
  EnAXIdebug();
  setEndianness(Little_Endian);
  startCDOFileStream("test_0335_aie_dma_tile_dma_packet_switch_mode.cdo");
  FileHeader();

  // setup aie-rt
  XAie_SetupConfig(ConfigPtr, XAIE_DEV_GEN_AIEML, XAIE2IPU_BASE_ADDR,
                   XAIE2IPU_COL_SHIFT, XAIE2IPU_ROW_SHIFT, XAIE2IPU_NUM_COLS,
                   XAIE2IPU_NUM_ROWS, XAIE2IPU_SHIM_ROW,
                   XAIE2IPU_MEM_TILE_ROW_START, XAIE2IPU_MEM_TILE_NUM_ROWS,
                   XAIE2IPU_AIE_TILE_ROW_START, XAIE2IPU_AIE_TILE_NUM_ROWS);

  XAie_InstDeclare(DevInst, &ConfigPtr);
  RC = XAie_CfgInitialize(&DevInst, &ConfigPtr);
  fprintf(stdout, "Device_Configure_Intialization Done.\n");
  RC = XAie_PartitionInitialize(&DevInst, NULL);
  if (RC != XAIE_OK) {
    fprintf(stderr, "Partition initialization failed.\n");
    return -1;
  }
  XAie_SetIOBackend(&DevInst, XAIE_IO_BACKEND_CDO);
  XAie_UpdateNpiAddr(&DevInst, XAIE2IPU_NPI_BASEADDR);
  XAie_TurnEccOff(&DevInst);

  // config code

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
  sleep(1);
  while (TimeOut && XAie_DmaWaitForDone(&DevInst, Tile_2, 0, DMA_S2MM, 1000)) {
    sleep(1);
    TimeOut--;
  }
  /*
   * Read data from aie data memory at DATA_MEM_INOUT_ADDR to compare
   * with input data.
   */
  RC = XAie_DataMemBlockRead(&DevInst, Tile_2, DATA_MEM_INOUT_ADDR,
                             (void *)buffer, NUM_ELEMS * sizeof(uint32_t));
  if (RC != XAIE_OK) {
    fprintf(stderr, "Failed to read from aie data memory.\n");
    return -1;
  }

  /* Check for correctness */
  for (uint8_t i = 0; i < NUM_ELEMS; i++) {
    if (data[i] != buffer[i]) {
      fprintf(stderr, "Data mismatch at index %d.\n", i);
      fprintf(stderr, "AIE DMA FoT failed.\n");
      return -1;
    }
  }
  return 0;

  fprintf(stdout, "AIE DMA FoT success.\n");

  RC = XAie_PartitionTeardown(&DevInst);
  if (RC != XAIE_OK) {
    fprintf(stderr, "Partition teardown failed.\n");
    return -1;
  }

  configureHeader();
  endCurrentCDOFileStream();

  return 0;
}
