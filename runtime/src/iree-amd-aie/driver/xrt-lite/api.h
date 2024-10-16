// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_XRT_LITE_API_H_
#define IREE_AMD_AIE_DRIVER_XRT_LITE_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct iree_hal_xrt_lite_device_options_t {};

IREE_API_EXPORT void iree_hal_xrt_lite_device_options_initialize(
    struct iree_hal_xrt_lite_device_options_t* out_params);

struct iree_hal_xrt_lite_driver_options_t {
  struct iree_hal_xrt_lite_device_options_t default_device_options;
};

IREE_API_EXPORT void iree_hal_xrt_lite_driver_options_initialize(
    struct iree_hal_xrt_lite_driver_options_t* out_options);

// The provided |identifier| will be used by programs to distinguish the device
// type from other HAL implementations. If compiling programs with the IREE
// compiler this must match the value used by IREE::HAL::TargetDevice.
//
// |out_driver| must be released by the caller (see iree_hal_driver_release).
IREE_API_EXPORT iree_status_t iree_hal_xrt_lite_driver_create(
    iree_string_view_t identifier,
    const struct iree_hal_xrt_lite_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver);

IREE_API_EXPORT iree_status_t iree_hal_xrt_lite_device_create(
    iree_string_view_t identifier,
    const struct iree_hal_xrt_lite_device_options_t* options,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

#ifdef __cplusplus
}       // extern "C"
#endif  // __cplusplus

#endif  // IREE_AMD_AIE_DRIVER_XRT_LITE_API_H_
