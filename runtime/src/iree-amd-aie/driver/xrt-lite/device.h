// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_XRT_LITE_XRT_LITE_DEVICE_H_
#define IREE_AMD_AIE_DRIVER_XRT_LITE_XRT_LITE_DEVICE_H_

#include "iree-amd-aie/driver/xrt-lite/api.h"
#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/device.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"

struct iree_async_proactor_pool_t;
struct iree_async_proactor_t;

struct iree_hal_xrt_lite_device {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  // TODO(max): not used because "device allocations" are performed through
  // device
  iree_hal_allocator_t* device_allocator;
  iree_hal_channel_provider_t* channel_provider;
  iree_hal_device_topology_info_t topology_info;

  // Proactor pool is retained for the device lifetime (same contract as
  // iree_hal_device_create_params_t).
  iree_async_proactor_pool_t* proactor_pool;
  // Borrowed from the pool while the pool is retained.
  iree_async_proactor_t* proactor;

  // block pool used for command buffer allocations, uses a larger block size
  // since command buffers can contain inlined data
  iree_arena_block_pool_t block_pool;
  shim_xdna::device* shim_device;
  // should come last; see the definition of total_size below in
  // iree_hal_xrt_lite_device_create
  iree_string_view_t identifier;
  iree_string_view_t power_mode;

  iree_hal_xrt_lite_device(const iree_hal_xrt_lite_device_params* options,
                           iree_allocator_t host_allocator);
};

#endif  // IREE_AMD_AIE_DRIVER_XRT_LITE_XRT_LITE_DEVICE_H_
