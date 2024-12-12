// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_XRT_LITE_BUFFER_H_
#define IREE_HAL_DRIVERS_XRT_LITE_BUFFER_H_

#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/bo.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

iree_status_t iree_hal_xrt_lite_buffer_wrap(
    shim_xdna::bo* bo, iree_hal_buffer_placement_t placement,
    iree_hal_memory_type_t memory_type, iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

shim_xdna::bo* iree_hal_xrt_lite_buffer_handle(iree_hal_buffer_t* base_buffer);

#endif  // IREE_HAL_DRIVERS_XRT_LITE_BUFFER_H_
