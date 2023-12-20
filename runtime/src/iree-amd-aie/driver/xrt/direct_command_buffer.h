// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_XRT_XRT_COMMAND_BUFFER_H_
#define IREE_AMD_AIE_DRIVER_XRT_XRT_COMMAND_BUFFER_H_

#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates command buffer.
//
// |block_pool| will be used for internal allocations and retaining copies of
// input data until reset.
//
// |out_command_buffer| must be released by the caller (see
// iree_hal_command_buffer_release).
iree_status_t iree_hal_xrt_direct_command_buffer_create(
    iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns true if |command_buffer| is a direct XRT command buffer.
bool iree_hal_xrt_direct_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_AMD_AIE_DRIVER_XRT_XRT_COMMAND_BUFFER_H_
