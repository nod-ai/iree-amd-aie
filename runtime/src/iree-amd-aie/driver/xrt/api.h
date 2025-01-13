// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// See iree/base/api.h for documentation on the API conventions used.

#ifndef IREE_EXPERIMENTAL_XRT_API_H_
#define IREE_EXPERIMENTAL_XRT_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_xrt_device_params_t
//===----------------------------------------------------------------------===//

// Parameters configuring an iree_hal_xrt_device_t.
// Must be initialized with iree_hal_xrt_device_params_initialize prior to
// use.
typedef struct iree_hal_xrt_device_params_t {
  // Total size of each block in the device shared block pool.
  // Larger sizes will lower overhead and ensure the heap isn't hit for
  // transient allocations while also increasing memory consumption.
  iree_host_size_t arena_block_size;
} iree_hal_xrt_device_params_t;

// Initializes |out_params| to default values.
void iree_hal_xrt_device_params_initialize(
    iree_hal_xrt_device_params_t* out_params);

//===----------------------------------------------------------------------===//
// iree_hal_xrt_driver_t
//===----------------------------------------------------------------------===//

// Creates a XRT HAL driver, from which devices can be created with the given
// |device_params|.
//
// |out_driver| must be released by the caller (see iree_hal_driver_release).
IREE_API_EXPORT iree_status_t iree_hal_xrt_driver_create(
    iree_string_view_t identifier,
    const iree_hal_xrt_device_params_t* device_params,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_EXPERIMENTAL_XRT_API_H_
