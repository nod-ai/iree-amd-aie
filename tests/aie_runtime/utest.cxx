// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree-amd-aie/runtime/iree_aie_runtime.h"

// TODO(max): find these actually in aie-rt
#define XAIE_BASE_ADDR 0x40000000
#define XAIE_COL_SHIFT 25
#define XAIE_ROW_SHIFT 20
#define XAIE_SHIM_ROW 0
#define XAIE_MEM_TILE_ROW_START 1

int main(int argc, char** argv) {
  std::string elfPath(argv[1]);

  int col = 0;
  // XAie_LocType is row, col but we always think col, row
  XAie_LocType tile_0_0 = {0, static_cast<u8>(col)};
  XAie_LocType tile_0_1 = {1, static_cast<u8>(col)};
  XAie_LocType tile_0_2 = {2, static_cast<u8>(col)};
  XAie_Lock lock_0_1 = XAie_Lock{0, 1};
  XAie_Lock lock_1_0 = XAie_Lock{1, 0};

  XAie_PartitionProp partitionProp;

  XAie_Config config = {
      XAIE_DEV_GEN_AIEML,
      XAIE_BASE_ADDR,
      XAIE_COL_SHIFT,
      XAIE_ROW_SHIFT,
      6,
      5,
      XAIE_SHIM_ROW,
      XAIE_MEM_TILE_ROW_START,
      1,
      (XAIE_MEM_TILE_ROW_START + 1),
      (6 - 1 - 1),
      partitionProp,
      XAIE_IO_BACKEND_CDO,
  };

  XAie_DevInst devInst;
  XAie_SetupPartitionConfig(&devInst, 0, 1, 1);
  XAie_CfgInitialize(&devInst, &config);
  XAie_UpdateNpiAddr(&devInst, 0);

  EnAXIdebug();
  setEndianness(Little_Endian);
  startCDOFileStream("pi.cdo");
  FileHeader();

  XAie_LoadElf(&devInst, tile_0_2, elfPath.c_str(), false);

  XAie_CoreReset(&devInst, tile_0_2);
  XAie_CoreUnreset(&devInst, tile_0_2);
  XAie_LockSetValue(&devInst, tile_0_2, lock_0_1);
  XAie_LockSetValue(&devInst, tile_0_2, lock_1_0);

  XAie_DmaDesc dmaTileBd;
  XAie_DmaDescInit(&devInst, &dmaTileBd, tile_0_2);
  lock_1_0 = XAie_Lock{1, -1};
  dmaTileBd.DmaMod->SetLock(&dmaTileBd, lock_1_0, lock_0_1, 1, 0);
  XAie_DmaSetAddrLen(&dmaTileBd, 1024, 4);
  XAie_DmaEnableBd(&dmaTileBd);
  XAie_DmaWriteBd(&devInst, &dmaTileBd, tile_0_2, 0);
  XAie_DmaChannelSetStartQueue(&devInst, tile_0_2, 0,
                               XAie_DmaDirection::DMA_MM2S, 0, 1, 0);
  XAie_DmaChannelEnable(&devInst, tile_0_2, 0, XAie_DmaDirection::DMA_MM2S);

  XAie_StrmConnCctEnable(&devInst, tile_0_0, StrmSwPortType::CTRL, 0,
                         StrmSwPortType::SOUTH, 0);
  XAie_StrmConnCctEnable(&devInst, tile_0_0, StrmSwPortType::NORTH, 0,
                         StrmSwPortType::SOUTH, 2);
  XAie_StrmConnCctEnable(&devInst, tile_0_1, StrmSwPortType::NORTH, 0,
                         StrmSwPortType::SOUTH, 0);
  XAie_StrmConnCctEnable(&devInst, tile_0_2, StrmSwPortType::DMA, 0,
                         StrmSwPortType::SOUTH, 0);
  XAie_EnableAieToShimDmaStrmPort(&devInst, tile_0_0, 2);
  XAie_CoreEnable(&devInst, tile_0_2);

  configureHeader();
  endCurrentCDOFileStream();
}