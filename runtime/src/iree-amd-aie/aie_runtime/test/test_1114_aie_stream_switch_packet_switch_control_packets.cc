// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <unistd.h>

#include <cstdio>

#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"

#define DATA_SIZE 17
#define WRRD_DATA_SIZE 0x02
#define CTR_PKT_OP_WR 0
#define MM2S_CHNUM 0
#define MM2S_BD_ID 1
#define MASK_FIELD 0x1F

/*
 * Control packet can be sent in both circuit switched and packet switche
 * dmodes, enable or disable this macro in which mode you want to test
 */
#define CIRCUIT_SWITCH_MODE 0

// from aie-rt/driver/src/global/xaiemlgbl_params.h
#define XAIEMLGBL_MEMORY_MODULE_DATAMEMORY 0x00000000
#define XAIE_TEST_AIE_TILE_MEMORY_MODULE_DATAMEMORY \
  XAIEMLGBL_MEMORY_MODULE_DATAMEMORY

/*****************************************************************************/
/*
 * This file contains the test application named
 *test_080_aie_stream_switch_packet_strm_one_to_one. This test case mainly
 *writtern to test the packet stream switch is working properly  or not. In this
 *test case I am sending the data present in data memory (input_add = 0x0000 )
 *as the packets from DMA slave port via MM2S by configuring in the BD . This
 *packets received from DMA master port and send via S2MM channel to the data
 *memory at ouput address (output_add = 0x4000) .After I am comparing the input
 *and output buffer.
 *
 * @param	None.
 *
 * @return	0 on success and error code on failure.
 *
 * @note		None.
 ******************************************************************************/
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

  uint32_t input_add = 0x0000;
  uint32_t output_add = 0x020;

  XAie_LocType Loc = {2, 0};
  XAie_DmaDesc DmaWrDesc_MM2S;
  XAie_PartInitOpts Opts;
  u32 input[DATA_SIZE];
  u32 output[DATA_SIZE];
  u8 flag = 1;
  u8 TimeOut = 10;
  XAie_Packet pkt;
  pkt.PktId = 0;
  pkt.PktType = 0;

  fprintf(stdout, "input addr %x and output adder %x\n", input_add, output_add);

  Opts.Locs = &Loc;
  Opts.NumUseTiles = 1;
  Opts.InitOpts = XAIE_PART_INIT_OPT_DEFAULT;
  RC = XAie_PartitionInitialize(&DevInst, &Opts);
  if (RC != XAIE_OK) {
    fprintf(stderr, "Partition initialization failed.\n");
    return -1;
  }

  /************* First Lets Check WRITE operation **************/
  /* Create route through streamswitch connections */
#if CIRCUIT_SWITCH_MODE
  RC = XAie_StrmConnCctEnable(&DevInst, Loc, DMA, MM2S_CHNUM, CTRL, 0);
#else
  RC = XAie_StrmPktSwSlavePortEnable(&DevInst, Loc, DMA, MM2S_CHNUM);
  RC = XAie_StrmPktSwSlaveSlotEnable(&DevInst, Loc, DMA, 0, 0, pkt, MASK_FIELD,
                                     0, 0);
  RC = XAie_StrmPktSwMstrPortEnable(&DevInst, Loc, CTRL, 0,
                                    XAIE_SS_PKT_DROP_HEADER, 0, 1);
  if (RC != XAIE_OK) {
    fprintf(stderr, "Failed to configure stream switch route\n");
    return -1;
  }
#endif

  /* Reset All Channels of DMA to make sure DMA channels at idle state */
  RC = XAie_DmaChannelResetAll(&DevInst, Loc, DMA_CHANNEL_RESET);
  RC = XAie_DmaChannelResetAll(&DevInst, Loc, DMA_CHANNEL_UNRESET);
  if (RC != XAIE_OK) {
    fprintf(stderr, "XAie_DmaChannelResetAll Failed.\n");
    return -1;
  }
  /* Initialize DMA */
  RC = XAie_DmaDescInit(&DevInst, &DmaWrDesc_MM2S, Loc);
  if (RC != XAIE_OK) {
    fprintf(stderr, "XAie_DmaDescInit Failed.\n");
    return -1;
  }

#if !CIRCUIT_SWITCH_MODE
  /*Set the packet id and packet type in MM2S dma Descriptor*/
  RC = XAie_DmaSetPkt(&DmaWrDesc_MM2S, pkt);
  if (RC != XAIE_OK) {
    fprintf(stderr, "XAie_DmaSetPkt Failed.\n");
    return -1;
  }
#endif

  /* Putting some data into Local data memory (input buffer) */
  /* Put control_word in data buffer */
  u32 parity = 0;
  input[0] = 0;
  input[0] = output_add;              // bits[19:0] , Local 32-bit Word Address
  input[0] = (WRRD_DATA_SIZE << 20);  // bits[21:20], Number of data words
  input[0] = (CTR_PKT_OP_WR << 22);   // bits[23:22], Operation
  input[0] = (0x1 << 24);  // bits[28:24], Stream ID for return packet
  /* calculate odd parity on bits[30:0] */
  for (u32 i = 0; i < 31; i++)
    if (input[0] & (1 << i)) parity++;
  if (!(parity % 2)) input[0] = (0x1 << 31);  // bits[31], odd parity
  for (u32 i = 0; i < (WRRD_DATA_SIZE + 1); i++) {
    if (i != 0) input[i] = i;
    XAie_DataMemWrWord(
        &DevInst, Loc,
        XAIE_TEST_AIE_TILE_MEMORY_MODULE_DATAMEMORY + input_add + i * 4,
        input[i]);
  }

  /* Set address and length for descriptor */
  RC = XAie_DmaSetAddrLen(
      &DmaWrDesc_MM2S,
      (XAIE_TEST_AIE_TILE_MEMORY_MODULE_DATAMEMORY + input_add),
      (WRRD_DATA_SIZE + 1) * sizeof(int));
  if (RC != XAIE_OK) {
    fprintf(stderr, "XAie_DmaSetAddrLen Failed.\n");
    return -1;
  }

  /* From AIE4 ValidBd is removed and cosidered BD is
   * always valid if its pushed into channel queue.
   * So no need of calling below APIs
   */
  /* Enable BDs */
  RC = XAie_DmaEnableBd(&DmaWrDesc_MM2S);
  if (RC != XAIE_OK) {
    fprintf(stderr, "XAie_DmaEnableBd Failed.\n");
    return -1;
  }

  /* Write BDs to hardware */
  RC = XAie_DmaWriteBd(&DevInst, &DmaWrDesc_MM2S, Loc, MM2S_BD_ID);
  if (RC != XAIE_OK) {
    fprintf(stderr, "XAie_DmaWriteBd Failed.\n");
    return -1;
  }
  /* Push to channel queue */
  RC = XAie_DmaChannelPushBdToQueue(&DevInst, Loc, MM2S_CHNUM, DMA_MM2S,
                                    MM2S_BD_ID);
  if (RC != XAIE_OK) {
    fprintf(stderr, "XAie_DmaChannelPushBdToQueue Failed.\n");
    return -1;
  }

  /* There is no channel enabling support in AIE4, Pushing BD to queue
   * is enough to start the channel
   */
  RC = XAie_DmaChannelEnable(&DevInst, Loc, 0, DMA_MM2S);
  if (RC != XAIE_OK) {
    fprintf(stderr, "XAie_DmaChannelEnable Failed.\n");
    return -1;
  }

  while (TimeOut &&
         XAie_DmaWaitForDone(&DevInst, Loc, MM2S_CHNUM, DMA_MM2S, 1000)) {
    sleep(1);
    TimeOut--;
  }

  /* Copy output data from tile to ddr */
  for (uint32_t i = 0; i < WRRD_DATA_SIZE; i++) {
    output[i] = 0;
    XAie_DataMemRdWord(
        &DevInst, Loc,
        XAIE_TEST_AIE_TILE_MEMORY_MODULE_DATAMEMORY + output_add + i * 4,
        (uint32_t *)&output + i);
  }

  fprintf(stdout, "Comparing output data with input data\n");
  for (uint32_t i = 0; i < WRRD_DATA_SIZE; i++) {
    if (input[i + 1] != output[i]) {
      fprintf(stderr, "Data mismatch at %d\n", i);
      flag = 0;
      break;
    }
  }

  XAie_Finish(&(DevInst));

  if (!flag || RC) {
    for (uint32_t i = 0; i < WRRD_DATA_SIZE; i++)
      fprintf(stdout, "input[%d] = 0x%08X  |  output[%d] = 0x%08X \n", i,
              input[i], i, output[i]);
    return -1;
  }

  return 0;
}
