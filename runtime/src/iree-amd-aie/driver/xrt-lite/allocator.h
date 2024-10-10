// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_XRT_LITE_ALLOCATOR_H_
#define IREE_HAL_DRIVERS_XRT_LITE_ALLOCATOR_H_

#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/device.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

// Creates a buffer allocator used for persistent allocations.
iree_status_t iree_hal_xrt_lite_allocator_create(
    iree_allocator_t host_allocator, std::shared_ptr<shim_xdna::device> device,
    iree_hal_allocator_t** out_allocator);

#endif  // IREE_HAL_DRIVERS_XRT_LITE_ALLOCATOR_H_
