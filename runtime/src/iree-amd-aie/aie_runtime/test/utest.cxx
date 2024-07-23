// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"

int main(int argc, char** argv) {
  std::string elfPath(argv[1]);

  uint8_t col = 0;
  uint8_t partitionStartCol = 1;
  uint8_t partitionNumCols = 4;

  mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel deviceModel(
      XAIE_DEV_GEN_AIE2IPU, XAIE2IPU_BASE_ADDR, XAIE2IPU_COL_SHIFT,
      XAIE2IPU_ROW_SHIFT, XAIE2IPU_NUM_COLS, XAIE2IPU_NUM_ROWS,
      XAIE2IPU_MEM_TILE_ROW_START, XAIE2IPU_MEM_TILE_NUM_ROWS,
      XAIE2IPU_SHIM_NUM_ROWS, partitionNumCols, partitionStartCol,
      /*aieSim*/ false, /*xaieDebug*/ false,
      mlir::iree_compiler::AMDAIE::AMDAIEDevice::npu1_4col);
  XAie_LocType tile00 = {.Row = 0, .Col = col};
  XAie_LocType tile01 = {.Row = 1, .Col = col};
  XAie_LocType tile02 = {.Row = 2, .Col = col};
  XAie_Lock lock01 = {.LockId = 0, .LockVal = 1};
  XAie_Lock lock10 = {.LockId = 1, .LockVal = 0};

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

  XAie_DmaDesc dmaTileBd;
  XAie_DmaDescInit(&deviceModel.devInst, &dmaTileBd, tile02);
  lock10 = XAie_Lock{.LockId = 1, .LockVal = -1};
  dmaTileBd.DmaMod->SetLock(&dmaTileBd, lock10, lock01, /*AcqEn*/ 1,
                            /*RelEn*/ 0);
  // address 1024 is the beginning of the core's stack
  XAie_DmaSetAddrLen(&dmaTileBd, /*Addr*/ 1024, /*Len*/ 4);
  XAie_DmaEnableBd(&dmaTileBd);
  uint8_t bdNum = 0, chNum = 0;
  XAie_DmaWriteBd(&deviceModel.devInst, &dmaTileBd, tile02, bdNum);
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
