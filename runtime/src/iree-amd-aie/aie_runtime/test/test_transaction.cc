// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstdio>
#include <iostream>

#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"

using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::AMDAIE;

int main(int argc, char **argv) {
  AieRC RC = XAIE_OK;

  // setup aie-rt
  XAie_SetupConfig(ConfigPtr, XAIE_DEV_GEN_AIEML, XAIE2IPU_BASE_ADDR,
                   XAIE2IPU_COL_SHIFT, XAIE2IPU_ROW_SHIFT, XAIE2IPU_NUM_COLS,
                   XAIE2IPU_NUM_ROWS, XAIE2IPU_SHIM_ROW,
                   XAIE2IPU_MEM_TILE_ROW_START, XAIE2IPU_MEM_TILE_NUM_ROWS,
                   XAIE2IPU_AIE_TILE_ROW_START, XAIE2IPU_AIE_TILE_NUM_ROWS);

  uint8_t start_col_idx = 0;
  XAie_InstDeclare(DevInst, &ConfigPtr);
  RC = XAie_CfgInitialize(&DevInst, &ConfigPtr);
  XAie_SetIOBackend(&DevInst, XAIE_IO_BACKEND_CDO);
  std::cout << "Device_Configure_Intialization Done.\n";

  // these calls hit RunOp which even txn flow actually tries to run
  //  RC = XAie_PartitionInitialize(&DevInst, nullptr);
  //  if (RC != XAIE_OK) {
  //    std::cout << "Partition initialization failed.\n";
  //    return -1;
  //  }
  //  XAie_UpdateNpiAddr(&DevInst, XAIE2IPU_NPI_BASEADDR);
  XAie_TurnEccOff(&DevInst);

  // config code

  auto tile_0_2 = XAie_TileLoc(0, 2);
  auto tile_1_2 = XAie_TileLoc(1, 2);
  auto tile_2_2 = XAie_TileLoc(2, 2);

  XAie_StartTransaction(&DevInst, XAIE_TRANSACTION_DISABLE_AUTO_FLUSH);

  XAie_CoreReset(&DevInst, tile_0_2);
  XAie_CoreUnreset(&DevInst, tile_0_2);
  XAie_LockSetValue(&DevInst, tile_0_2, XAie_Lock{0, 1});
  XAie_LockSetValue(&DevInst, tile_0_2, XAie_Lock{1, 0});

  XAie_CoreReset(&DevInst, tile_1_2);
  XAie_CoreUnreset(&DevInst, tile_1_2);
  XAie_LockSetValue(&DevInst, tile_1_2, XAie_Lock{0, 1});
  XAie_LockSetValue(&DevInst, tile_1_2, XAie_Lock{1, 0});

  XAie_CoreReset(&DevInst, tile_2_2);
  XAie_CoreUnreset(&DevInst, tile_2_2);
  XAie_LockSetValue(&DevInst, tile_2_2, XAie_Lock{0, 1});
  XAie_LockSetValue(&DevInst, tile_2_2, XAie_Lock{1, 0});

  auto *tmpInst = XAie_ExportTransactionInstance(&DevInst);

  std::cout << tmpInst->NumCmds << "\n";
  for (int i = 0; i < tmpInst->NumCmds; ++i) {
    XAie_TxnCmd *Cmd = &tmpInst->CmdBuf[i];
    std::cout << *Cmd << "\n";
  }

  XAie_TxnHeader *txn_header = reinterpret_cast<XAie_TxnHeader *>(
      XAie_ExportSerializedTransaction(&DevInst, /*NumConsumers=*/1,
                                       /*Flags=*/0));

  printf("Header version %d.%d\n", txn_header->Major, txn_header->Minor);
  printf("Device Generation: %d\n", txn_header->DevGen);
  printf("Cols, Rows, NumMemRows : (%d, %d, %d)\n", txn_header->NumCols,
         txn_header->NumRows, txn_header->NumMemTileRows);
  printf("TransactionSize: %u\n", txn_header->TxnSize);
  uint32_t NumOps = txn_header->NumOps;
  printf("NumOps: %u\n", NumOps);

  uint8_t *ptr = reinterpret_cast<uint8_t *>(txn_header);
  ptr += sizeof(XAie_TxnHeader);
  for (uint32_t i = 0; i < NumOps; i++) {
    XAie_OpHdr *op_header = (XAie_OpHdr *)ptr;
    auto opCode = static_cast<AMDAIE::XAie_TxnOpcode>(op_header->Op);
    switch (opCode) {
      case AMDAIE::XAie_TxnOpcode::XAIE_IO_WRITE: {
        XAie_Write32Hdr *w_header = (XAie_Write32Hdr *)ptr;
        std::cout << (*w_header) << "\n";
        ptr += w_header->Size;
        break;
      }
      case AMDAIE::XAie_TxnOpcode::XAIE_IO_BLOCKWRITE: {
        XAie_BlockWrite32Hdr *bw_header = (XAie_BlockWrite32Hdr *)ptr;
        std::cout << *bw_header << "\n";
        uint32_t *payload = (uint32_t *)(ptr + sizeof(XAie_BlockWrite32Hdr));
        u32 size = (bw_header->Size - sizeof(*bw_header)) / 4;
        for (uint32_t ii = 0; ii < size; ii++) {
          uint64_t addr = bw_header->RegOff + DevInst.BaseAddr + ii * 4U;
          printf("   0x%lx, 0x%x\n", addr, payload[ii]);
        }
        ptr += bw_header->Size;
        break;
      }
      case AMDAIE::XAie_TxnOpcode::XAIE_IO_MASKWRITE: {
        XAie_MaskWrite32Hdr *mw_header = (XAie_MaskWrite32Hdr *)ptr;
        std::cout << *mw_header << "\n";
        ptr += mw_header->Size;
        break;
      }
      case AMDAIE::XAie_TxnOpcode::XAIE_IO_MASKPOLL: {
        XAie_MaskPoll32Hdr *mp_header = (XAie_MaskPoll32Hdr *)ptr;
        std::cout << mp_header << "\n";
        ptr += mp_header->Size;
        break;
      }
      case AMDAIE::XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_TCT: {
        XAie_CustomOpHdr *co_header = (XAie_CustomOpHdr *)ptr;
        tct_op_t *iptr = (tct_op_t *)(ptr + sizeof(*co_header));
        printf("CustomOp TCT: %d\n", iptr->word);
        u32 word = iptr->word;
        u8 Col = ((word & 0x00FF0000) >> 16) + start_col_idx;
        u8 Row = ((word & 0x0000FF00) >> 8);
        u8 dir = ((word) & 0x000000FF);
        XAie_DmaDirection Dir = (dir == 0) ? DMA_S2MM : DMA_MM2S;
        u32 config = iptr->config;
        u8 ChNum = ((config & 0xFF000000) >> 24);
        u8 ColNum = ((config & 0x00FF0000) >> 16);
        u8 RowNum = ((config & 0x0000FF00) >> 8);
        printf(
            "SyncTaskCompleteToken: {col, row, chl, dir} = {%d+%d, %d+%d, %d, "
            "%d}\n",
            Col, ColNum, Row, RowNum, ChNum, Dir);
        ptr += co_header->Size;
        break;
      }
      case AMDAIE::XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_DDR_PATCH: {
        XAie_CustomOpHdr *hdr = (XAie_CustomOpHdr *)ptr;
        patch_op_t *op = (patch_op_t *)(ptr + sizeof(*hdr));
        printf("CustomOp PatchBD argidx %lu\n", op->argidx);
        printf("CustomOp PatchBD regaddr %lx\n",
               op->regaddr + DevInst.BaseAddr);
        ptr += hdr->Size;
        break;
      }
      case AMDAIE::XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_READ_REGS: {
        // Dump Registers opcode
        // Do nothing in sim
        break;
      }
      case AMDAIE::XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_RECORD_TIMER: {
        // Record Timestamp opcode
        // Do nothing in sim
        break;
      }
      case AMDAIE::XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_MERGE_SYNC: {
        XAie_CustomOpHdr *co_header = (XAie_CustomOpHdr *)ptr;
        printf("co_header->Size = %d\n", co_header->Size);
        tct_op_t *iptr = (tct_op_t *)(ptr + sizeof(*co_header));
        u32 word = iptr->word;
        printf("CustomOp MergeSync TCT: 0x%x\n", word);
        u8 num_tokens = ((word) & 0x000000FF);
        u8 num_cols = ((word) & 0x0000FF00) >> 8;
        ptr = ptr + co_header->Size;
        printf("MergeSyncTaskCompleteToken over\n");
        break;
      }
      default:
        return -1;
    }
  }

  return 0;
}
