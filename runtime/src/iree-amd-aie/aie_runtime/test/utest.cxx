// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"

#define NPI_ADDR 0x0
#define NPU_NUM_COLS 6
#define NPU_NUM_MEM_TILE_ROWS 1
#define NPU_NUM_ROWS 6
#define XAIE_BASE_ADDR 0x40000000
#define XAIE_COL_SHIFT 25
#define XAIE_MEM_TILE_ROW_START 1
#define XAIE_NUM_SHIM_TILE_ROWS 1
#define XAIE_PARTITION_BASE_ADDR 0x0
#define XAIE_ROW_SHIFT 20
#define XAIE_SHIM_ROW 0

int main(int argc, char** argv) {
  std::string elfPath(argv[1]);

  uint8_t col = 0;
  XAie_LocType tile00 = {.Row = 0, .Col = col};
  XAie_LocType tile01 = {.Row = 1, .Col = col};
  XAie_LocType tile02 = {.Row = 2, .Col = col};
  XAie_Lock lock01 = {.LockId = 0, .LockVal = 1};
  XAie_Lock lock10 = {.LockId = 1, .LockVal = 0};

  XAie_PartitionProp partitionProp;

  XAie_Config config = {
      .AieGen = XAIE_DEV_GEN_AIEML,
      .BaseAddr = XAIE_BASE_ADDR,
      .ColShift = XAIE_COL_SHIFT,
      .RowShift = XAIE_ROW_SHIFT,
      .NumRows = NPU_NUM_ROWS,
      .NumCols = NPU_NUM_COLS,
      .ShimRowNum = XAIE_SHIM_ROW,
      .MemTileRowStart = XAIE_MEM_TILE_ROW_START,
      .MemTileNumRows = NPU_NUM_MEM_TILE_ROWS,
      .AieTileRowStart = XAIE_MEM_TILE_ROW_START + NPU_NUM_MEM_TILE_ROWS,
      .AieTileNumRows =
          NPU_NUM_ROWS - NPU_NUM_MEM_TILE_ROWS - XAIE_NUM_SHIM_TILE_ROWS,
      .PartProp = partitionProp};

  uint8_t partitionStartCol = 1;
  uint8_t partitionNumCols = 1;

  XAie_DevInst devInst = {};
  XAie_SetupPartitionConfig(&devInst, XAIE_PARTITION_BASE_ADDR,
                            partitionStartCol, partitionNumCols);
  XAie_CfgInitialize(&devInst, &config);
  XAie_UpdateNpiAddr(&devInst, NPI_ADDR);

  EnAXIdebug();
  setEndianness(Little_Endian);
  startCDOFileStream("pi.cdo");
  FileHeader();

  XAie_LoadElf(&devInst, tile02, elfPath.c_str(), /*LoadSym*/ false);

  XAie_CoreReset(&devInst, tile02);
  XAie_CoreUnreset(&devInst, tile02);
  XAie_LockSetValue(&devInst, tile02, lock01);
  XAie_LockSetValue(&devInst, tile02, lock10);

  XAie_DmaDesc dmaTileBd;
  XAie_DmaDescInit(&devInst, &dmaTileBd, tile02);
  lock10 = XAie_Lock{.LockId = 1, .LockVal = -1};
  dmaTileBd.DmaMod->SetLock(&dmaTileBd, lock10, lock01, /*AcqEn*/ 1,
                            /*RelEn*/ 0);
  // address 1024 is the beginning of the core's stack
  XAie_DmaSetAddrLen(&dmaTileBd, /*Addr*/ 1024, /*Len*/ 4);
  XAie_DmaEnableBd(&dmaTileBd);
  uint8_t bdNum = 0, chNum = 0;
  XAie_DmaWriteBd(&devInst, &dmaTileBd, tile02, bdNum);
  XAie_DmaChannelSetStartQueue(&devInst, tile02, chNum,
                               XAie_DmaDirection::DMA_MM2S, bdNum,
                               /*RepeatCount*/ 1, /*EnTokenIssue*/ 0);
  XAie_DmaChannelEnable(&devInst, tile02, chNum, XAie_DmaDirection::DMA_MM2S);

  // Slave == source, Master == destination
  // Note, these are internal connections in the switch
  // (so src port -> dest port within the switch itself)
  XAie_StrmConnCctEnable(&devInst, tile00, /*Slave*/ StrmSwPortType::CTRL,
                         /*SlvPortNum*/ 0, /*Master*/ StrmSwPortType::SOUTH,
                         /*MstrPortNum*/ 0);
  XAie_StrmConnCctEnable(&devInst, tile00, /*Slave*/ StrmSwPortType::NORTH,
                         /*SlvPortNum*/ 0,
                         /*Master*/ StrmSwPortType::SOUTH, /*MstrPortNum*/ 2);
  XAie_StrmConnCctEnable(&devInst, tile01, /*Slave*/ StrmSwPortType::NORTH,
                         /*SlvPortNum*/ 0,
                         /*Master*/ StrmSwPortType::SOUTH, /*MstrPortNum*/ 0);
  XAie_StrmConnCctEnable(&devInst, tile02, /*Slave*/ StrmSwPortType::DMA,
                         /*SlvPortNum*/ 0,
                         /*Master*/ StrmSwPortType::SOUTH, /*MstrPortNum*/ 0);
  XAie_EnableAieToShimDmaStrmPort(&devInst, tile00, /*PortNum*/ 2);
  XAie_CoreEnable(&devInst, tile02);

  configureHeader();
  endCurrentCDOFileStream();
}
