// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_AMDXDNA_EXECUTABLE_H_
#define IREE_AMD_AIE_DRIVER_AMDXDNA_EXECUTABLE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

struct iree_hal_amdxdna_executable;
struct iree_hal_amdxdna_native_context_t;
struct iree_hal_amdxdna_native_device_t;

// `out_executable` must be released by the caller (see
// iree_hal_executable_release).
iree_status_t iree_hal_amdxdna_native_executable_create(
    iree_hal_amdxdna_native_device_t* native_device,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

iree_status_t iree_hal_amdxdna_native_executable_infer_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size);

iree_hal_amdxdna_executable* iree_hal_amdxdna_executable_cast(
    iree_hal_executable_t* base_executable);

// Returns the executable's cached control-packet native context, if one has
// been resolved. Borrowed; only valid while the executable/device remain alive.
iree_hal_amdxdna_native_context_t*
iree_hal_amdxdna_executable_control_context_borrow(
    iree_hal_executable_t* base_executable);

#endif  // IREE_AMD_AIE_DRIVER_AMDXDNA_EXECUTABLE_H_
