// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions. See
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: # Apache-2.0 WITH LLVM-exception

#ifndef XAIE_HWCFG_H
#define XAIE_HWCFG_H

#include "macros.h"

// Gen AIE
extern uint8_t XAIE1_AIE_TILE_NUM_ROWS;
extern uint8_t XAIE1_AIE_TILE_ROW_START;
extern uint8_t XAIE1_COL_SHIFT;
extern uint8_t XAIE1_MEM_TILE_DMA_NUM_CH;
extern uint8_t XAIE1_MEM_TILE_NUM_LOCKS;
extern uint8_t XAIE1_MEM_TILE_NUM_ROWS;
extern uint8_t XAIE1_MEM_TILE_ROW_START;
extern uint8_t XAIE1_NUM_COLS;
extern uint8_t XAIE1_NUM_NOC_INTR_OFFSET;
extern uint8_t XAIE1_NUM_ROWS;
extern uint8_t XAIE1_ROW_SHIFT;
extern uint8_t XAIE1_SHIM_DMA_NUM_CH;
extern uint8_t XAIE1_SHIM_NUM_LOCKS;
extern uint8_t XAIE1_SHIM_NUM_ROWS;
extern uint8_t XAIE1_SHIM_ROW;
extern uint8_t XAIE1_SS_ARBITER_MAX;
extern uint8_t XAIE1_SS_MSEL_MAX;
extern uint8_t XAIE1_TILE_DMA_NUM_CH;
extern uint8_t XAIE1_TILE_NUM_LOCKS;
extern uint64_t XAIE1_BASE_ADDR;
extern uint64_t XAIE1_NPI_BASEADDR;
extern uint64_t XAIE1_PARTITION_BASE_ADDR;

// Gen ML

extern uint8_t XAIEML_AIE_TILE_NUM_ROWS;
extern uint8_t XAIEML_AIE_TILE_ROW_START;
extern uint8_t XAIEML_COL_SHIFT;
extern uint8_t XAIEML_MEM_TILE_DMA_NUM_CH;
extern uint8_t XAIEML_MEM_TILE_NUM_LOCKS;
extern uint8_t XAIEML_MEM_TILE_NUM_ROWS;
extern uint8_t XAIEML_MEM_TILE_ROW_START;
extern uint8_t XAIEML_NUM_COLS;
extern uint8_t XAIEML_NUM_NOC_INTR_OFFSET;
extern uint8_t XAIEML_NUM_ROWS;
extern uint8_t XAIEML_ROW_SHIFT;
extern uint8_t XAIEML_SHIM_DMA_NUM_CH;
extern uint8_t XAIEML_SHIM_NUM_LOCKS;
extern uint8_t XAIEML_SHIM_NUM_ROWS;
extern uint8_t XAIEML_SS_ARBITER_MAX;
extern uint8_t XAIEML_SS_MSEL_MAX;
extern uint8_t XAIEML_SHIM_ROW;
extern uint8_t XAIEML_TILE_DMA_NUM_CH;
extern uint8_t XAIEML_TILE_NUM_LOCKS;
extern uint64_t XAIEML_BASE_ADDR;
extern uint64_t XAIEML_NPI_BASEADDR;
extern uint64_t XAIEML_PARTITION_BASE_ADDR;

// Gen IPU

extern uint8_t XAIE2IPU_AIE_TILE_NUM_ROWS;
extern uint8_t XAIE2IPU_AIE_TILE_ROW_START;
extern uint8_t XAIE2IPU_COL_SHIFT;
extern uint8_t XAIE2IPU_MEM_TILE_DMA_NUM_CH;
extern uint8_t XAIE2IPU_MEM_TILE_NUM_LOCKS;
extern uint8_t XAIE2IPU_MEM_TILE_NUM_ROWS;
extern uint8_t XAIE2IPU_MEM_TILE_ROW_START;
extern uint8_t XAIE2IPU_NUM_COLS;
extern uint8_t XAIE2IPU_NUM_NOC_INTR_OFFSET;
extern uint8_t XAIE2IPU_NUM_ROWS;
extern uint8_t XAIE2IPU_ROW_SHIFT;
extern uint8_t XAIE2IPU_SHIM_DMA_NUM_CH;
extern uint8_t XAIE2IPU_SHIM_NUM_LOCKS;
extern uint8_t XAIE2IPU_SHIM_NUM_ROWS;
extern uint8_t XAIE2IPU_SHIM_ROW;
extern uint8_t XAIE2IPU_SS_ARBITER_MAX;
extern uint8_t XAIE2IPU_SS_MSEL_MAX;
extern uint8_t XAIE2IPU_TILE_DMA_NUM_CH;
extern uint8_t XAIE2IPU_TILE_NUM_LOCKS;
extern uint64_t XAIE2IPU_BASE_ADDR;
extern uint64_t XAIE2IPU_NPI_BASEADDR;
extern uint64_t XAIE2IPU_PARTITION_BASE_ADDR;

// TODO(max): these are hardcoded in the router - should be moved to the device
// model
extern int XAIE2IPU_MEM_TILE_LOCK_ID_INCR;
extern uint64_t XAIE2IPU_ADDR_ARRAY_OFF;

// Gen Strix B0

extern uint8_t XAIE_STRIXB0_AIE_TILE_NUM_ROWS;
extern uint8_t XAIE_STRIXB0_AIE_TILE_ROW_START;
extern uint8_t XAIE_STRIXB0_COL_SHIFT;
extern uint8_t XAIE_STRIXB0_MEM_TILE_DMA_NUM_CH;
extern uint8_t XAIE_STRIXB0_MEM_TILE_NUM_LOCKS;
extern uint8_t XAIE_STRIXB0_MEM_TILE_NUM_ROWS;
extern uint8_t XAIE_STRIXB0_MEM_TILE_ROW_START;
extern uint8_t XAIE_STRIXB0_NUM_COLS;
extern uint8_t XAIE_STRIXB0_NUM_NOC_INTR_OFFSET;
extern uint8_t XAIE_STRIXB0_NUM_ROWS;
extern uint8_t XAIE_STRIXB0_ROW_SHIFT;
extern uint8_t XAIE_STRIXB0_SHIM_DMA_NUM_CH;
extern uint8_t XAIE_STRIXB0_SHIM_NUM_LOCKS;
extern uint8_t XAIE_STRIXB0_SHIM_NUM_ROWS;
extern uint8_t XAIE_STRIXB0_SHIM_ROW;
extern uint8_t XAIE_STRIXB0_SS_ARBITER_MAX;
extern uint8_t XAIE_STRIXB0_SS_MSEL_MAX;
extern uint8_t XAIE_STRIXB0_TILE_DMA_NUM_CH;
extern uint8_t XAIE_STRIXB0_TILE_NUM_LOCKS;
extern uint64_t XAIE_STRIXB0_BASE_ADDR;
extern uint64_t XAIE_STRIXB0_NPI_BASEADDR;
extern uint64_t XAIE_STRIXB0_PARTITION_BASE_ADDR;

#endif
