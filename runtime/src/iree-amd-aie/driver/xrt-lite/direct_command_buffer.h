// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_XRT_LITE_XRT_LITE_COMMAND_BUFFER_H_
#define IREE_AMD_AIE_DRIVER_XRT_LITE_XRT_LITE_COMMAND_BUFFER_H_

#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/device.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// |out_command_buffer| must be released by the caller (see
// iree_hal_command_buffer_release).
iree_status_t iree_hal_xrt_lite_direct_command_buffer_create(
    shim_xdna::device* shim_device, iree_hal_allocator_t* device_allocator,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

#ifdef __cplusplus
}       // extern "C"
#endif  // __cplusplus

#endif  // IREE_AMD_AIE_DRIVER_XRT_LITE_XRT_LITE_COMMAND_BUFFER_H_
