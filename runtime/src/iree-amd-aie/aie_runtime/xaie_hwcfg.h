// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions. See
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: # Apache-2.0 WITH LLVM-exception

#ifndef XAIE_HWCFG_H
#define XAIE_HWCFG_H

#include "export.h"

// Gen AIE
extern int XAIE1_AIE_TILE_NUM_ROWS;
extern int XAIE1_AIE_TILE_ROW_START;
extern int XAIE1_COL_SHIFT;
extern int XAIE1_MEM_TILE_DMA_NUM_CH;
extern int XAIE1_MEM_TILE_NUM_LOCKS;
extern int XAIE1_MEM_TILE_NUM_ROWS;
extern int XAIE1_MEM_TILE_ROW_START;
extern int XAIE1_NUM_COLS;
extern int XAIE1_NUM_NOC_INTR_OFFSET;
extern int XAIE1_NUM_ROWS;
extern int XAIE1_ROW_SHIFT;
extern int XAIE1_SHIM_DMA_NUM_CH;
extern int XAIE1_SHIM_NUM_LOCKS;
extern int XAIE1_SHIM_NUM_ROWS;
extern int XAIE1_SHIM_ROW;
extern int XAIE1_TILE_DMA_NUM_CH;
extern int XAIE1_TILE_NUM_LOCKS;
extern uint64_t XAIE1_BASE_ADDR;
extern uint64_t XAIE1_NPI_BASEADDR;
extern uint64_t XAIE1_PARTITION_BASE_ADDR;

// Gen ML
extern int XAIEML_AIE_TILE_NUM_ROWS;
extern int XAIEML_AIE_TILE_ROW_START;
extern int XAIEML_COL_SHIFT;
extern int XAIEML_MEM_TILE_DMA_NUM_CH;
extern int XAIEML_MEM_TILE_NUM_LOCKS;
extern int XAIEML_MEM_TILE_NUM_ROWS;
extern int XAIEML_MEM_TILE_ROW_START;
extern int XAIEML_NUM_COLS;
extern int XAIEML_NUM_NOC_INTR_OFFSET;
extern int XAIEML_NUM_ROWS;
extern int XAIEML_ROW_SHIFT;
extern int XAIEML_SHIM_DMA_NUM_CH;
extern int XAIEML_SHIM_NUM_LOCKS;
extern int XAIEML_SHIM_NUM_ROWS;
extern int XAIEML_SHIM_ROW;
extern int XAIEML_TILE_DMA_NUM_CH;
extern int XAIEML_TILE_NUM_LOCKS;
extern uint64_t XAIEML_BASE_ADDR;
extern uint64_t XAIEML_NPI_BASEADDR;
extern uint64_t XAIEML_PARTITION_BASE_ADDR;

// Gen IPU

extern int XAIE2IPU_AIE_TILE_NUM_ROWS;
extern int XAIE2IPU_AIE_TILE_ROW_START;
extern int XAIE2IPU_COL_SHIFT;
extern int XAIE2IPU_MEM_TILE_DMA_NUM_CH;
extern int XAIE2IPU_MEM_TILE_NUM_LOCKS;
extern int XAIE2IPU_MEM_TILE_NUM_ROWS;
extern int XAIE2IPU_MEM_TILE_ROW_START;
extern int XAIE2IPU_NUM_COLS;
extern int XAIE2IPU_NUM_NOC_INTR_OFFSET;
extern int XAIE2IPU_NUM_ROWS;
extern int XAIE2IPU_ROW_SHIFT;
extern int XAIE2IPU_SHIM_DMA_NUM_CH;
extern int XAIE2IPU_SHIM_NUM_LOCKS;
extern int XAIE2IPU_SHIM_NUM_ROWS;
extern int XAIE2IPU_SHIM_ROW;
extern int XAIE2IPU_TILE_DMA_NUM_CH;
extern int XAIE2IPU_TILE_NUM_LOCKS;
extern uint64_t XAIE2IPU_BASE_ADDR;
extern uint64_t XAIE2IPU_NPI_BASEADDR;
extern uint64_t XAIE2IPU_PARTITION_BASE_ADDR;

extern int XAIE2IPU_MEM_TILE_LOCK_ID_INCR;
extern uint64_t XAIE2IPU_ADDR_ARRAY_OFF;

#endif
