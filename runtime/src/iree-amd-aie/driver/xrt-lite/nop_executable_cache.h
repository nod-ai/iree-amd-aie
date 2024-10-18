// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_XRT_LITE_NOP_EXECUTABLE_CACHE_H_
#define IREE_AMD_AIE_DRIVER_XRT_LITE_NOP_EXECUTABLE_CACHE_H_

#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/device.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

// `out_executable_cache` must be released by the caller (see
// iree_hal_executable_cache_release).
iree_status_t iree_hal_xrt_lite_nop_executable_cache_create(
    shim_xdna::device* shim_device, iree_string_view_t identifier,
    iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache);

#endif  // IREE_AMD_AIE_DRIVER_XRT_LITE_NOP_EXECUTABLE_CACHE_H_
