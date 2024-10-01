// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_HSA_COMMAND_BUFFER_H_
#define IREE_AMD_AIE_DRIVER_HSA_COMMAND_BUFFER_H_

#include "iree-amd-aie/driver/hsa/dynamic_symbols.h"
#include "iree-amd-aie/driver/hsa/hsa_headers.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

iree_status_t iree_hal_hsa_command_buffer_create(
    iree_hal_device_t* device,
    const iree_hal_hsa_dynamic_symbols_t* hsa_symbols,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity, hsa::hsa_queue_t* queue,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_allocator_t* device_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_AMD_AIE_DRIVER_HSA_COMMAND_BUFFER_H_
