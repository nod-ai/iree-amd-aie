// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*
clang-format off

RUN: test_transaction | FileCheck %s

CHECK: XAie_TxnCmd(Opcode: 0, Mask: 2, RegOff: 2301952, Value: 2, Size: 0)
CHECK: XAie_TxnCmd(Opcode: 0, Mask: 2, RegOff: 2301952, Value: 0, Size: 0)
CHECK: XAie_TxnCmd(Opcode: 0, Mask: 0, RegOff: 2224128, Value: 1, Size: 0)
CHECK: XAie_TxnCmd(Opcode: 0, Mask: 0, RegOff: 2224144, Value: 0, Size: 0)
CHECK: XAie_TxnCmd(Opcode: 0, Mask: 2, RegOff: 35856384, Value: 2, Size: 0)
CHECK: XAie_TxnCmd(Opcode: 0, Mask: 2, RegOff: 35856384, Value: 0, Size: 0)
CHECK: XAie_TxnCmd(Opcode: 0, Mask: 0, RegOff: 35778560, Value: 1, Size: 0)
CHECK: XAie_TxnCmd(Opcode: 0, Mask: 0, RegOff: 35778576, Value: 0, Size: 0)
CHECK: XAie_TxnCmd(Opcode: 0, Mask: 2, RegOff: 69410816, Value: 2, Size: 0)
CHECK: XAie_TxnCmd(Opcode: 0, Mask: 2, RegOff: 69410816, Value: 0, Size: 0)
CHECK: XAie_TxnCmd(Opcode: 0, Mask: 0, RegOff: 69332992, Value: 1, Size: 0)
CHECK: XAie_TxnCmd(Opcode: 0, Mask: 0, RegOff: 69333008, Value: 0, Size: 0)
CHECK: Header version 0.1
CHECK: Device Generation: 2
CHECK: Cols, Rows, NumMemRows : (5, 6, 1)
CHECK: TransactionSize: 352
CHECK: NumOps: 12
CHECK: XAie_MaskWrite32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_MASKWRITE, Col: 2, Row: 0), RegOff: 2301952, Value: 2, Mask: 2, Size: 32)
CHECK: XAie_MaskWrite32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_MASKWRITE, Col: 2, Row: 0), RegOff: 2301952, Value: 0, Mask: 2, Size: 32)
CHECK: XAie_Write32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_WRITE, Col: 2, Row: 0), RegOff: 2224128, Value: 1, Size: 24)
CHECK: XAie_Write32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_WRITE, Col: 2, Row: 16), RegOff: 2224144, Value: 0, Size: 24)
CHECK: XAie_MaskWrite32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_MASKWRITE, Col: 2, Row: 0), RegOff: 35856384, Value: 2, Mask: 2, Size: 32)
CHECK: XAie_MaskWrite32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_MASKWRITE, Col: 2, Row: 0), RegOff: 35856384, Value: 0, Mask: 2, Size: 32)
CHECK: XAie_Write32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_WRITE, Col: 2, Row: 0), RegOff: 35778560, Value: 1, Size: 24)
CHECK: XAie_Write32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_WRITE, Col: 2, Row: 16), RegOff: 35778576, Value: 0, Size: 24)
CHECK: XAie_MaskWrite32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_MASKWRITE, Col: 2, Row: 0), RegOff: 69410816, Value: 2, Mask: 2, Size: 32)
CHECK: XAie_MaskWrite32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_MASKWRITE, Col: 2, Row: 0), RegOff: 69410816, Value: 0, Mask: 2, Size: 32)
CHECK: XAie_Write32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_WRITE, Col: 2, Row: 0), RegOff: 69332992, Value: 1, Size: 24)
CHECK: XAie_Write32Hdr(OpHdr: XAie_OpHdr(Op: XAie_TxnOpcode::XAIE_IO_WRITE, Col: 2, Row: 16), RegOff: 69333008, Value: 0, Size: 24)

clang-format on
 */

#include <iostream>

#include "interpreter_op_impl.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"

using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::AMDAIE;

int main(int argc, char **argv) {
  // setup aie-rt
  XAie_SetupConfig(ConfigPtr, XAIE_DEV_GEN_AIEML, XAIE2IPU_BASE_ADDR,
                   XAIE2IPU_COL_SHIFT, XAIE2IPU_ROW_SHIFT, XAIE2IPU_NUM_COLS,
                   XAIE2IPU_NUM_ROWS, XAIE2IPU_SHIM_ROW,
                   XAIE2IPU_MEM_TILE_ROW_START, XAIE2IPU_MEM_TILE_NUM_ROWS,
                   XAIE2IPU_AIE_TILE_ROW_START, XAIE2IPU_AIE_TILE_NUM_ROWS);

  XAie_InstDeclare(DevInst, &ConfigPtr);
  XAie_CfgInitialize(&DevInst, &ConfigPtr);
  XAie_SetIOBackend(&DevInst, XAIE_IO_BACKEND_CDO);

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

  SubmitSerializedTransaction(DevInst, /*startColIdx*/ 0);

  XAie_Finish(&DevInst);

  return 0;
}
