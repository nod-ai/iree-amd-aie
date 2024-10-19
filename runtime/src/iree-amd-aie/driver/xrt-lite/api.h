// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_XRT_LITE_API_H_
#define IREE_AMD_AIE_DRIVER_XRT_LITE_API_H_

#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/amdxdna_accel.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

struct iree_hal_xrt_lite_device_params {
  int32_t n_core_rows;
  int32_t n_core_cols;
  iree_string_view_t power_mode;
};

IREE_API_EXPORT void iree_hal_xrt_lite_device_options_initialize(
    struct iree_hal_xrt_lite_device_params* out_params);

struct iree_hal_xrt_lite_driver_options {
  struct iree_hal_xrt_lite_device_params device_params;
};

IREE_API_EXPORT void iree_hal_xrt_lite_driver_options_initialize(
    struct iree_hal_xrt_lite_driver_options* out_options);

// The provided `identifier` will be used by programs to distinguish the device
// type from other HAL implementations. If compiling programs with the IREE
// compiler this must match the value used by IREE::HAL::TargetDevice.
//
// `out_driver` must be released by the caller (see iree_hal_driver_release).
IREE_API_EXPORT iree_status_t iree_hal_xrt_lite_driver_create(
    iree_string_view_t identifier,
    const struct iree_hal_xrt_lite_driver_options* options,
    const struct iree_hal_xrt_lite_device_params* device_params,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver);

IREE_API_EXPORT iree_status_t iree_hal_xrt_lite_device_create(
    iree_string_view_t identifier,
    const struct iree_hal_xrt_lite_device_params* params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

#endif  // IREE_AMD_AIE_DRIVER_XRT_LITE_API_H_
