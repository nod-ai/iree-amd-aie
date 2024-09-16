// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_HSA_API_H_
#define IREE_AMD_AIE_DRIVER_HSA_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct iree_hal_hsa_device_params_t {
  iree_host_size_t arena_block_size;
};

struct iree_hal_hsa_driver_options_t {
  int default_device_index;
};

IREE_API_EXPORT void iree_hal_hsa_device_params_initialize(
    struct iree_hal_hsa_device_params_t* out_params);

IREE_API_EXPORT void iree_hal_hsa_driver_options_initialize(
    struct iree_hal_hsa_driver_options_t* out_options);

IREE_API_EXPORT iree_status_t iree_hal_hsa_driver_create(
    iree_string_view_t identifier,
    const struct iree_hal_hsa_driver_options_t* options,
    const struct iree_hal_hsa_device_params_t* default_params,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_AMD_AIE_DRIVER_HSA_API_H_
