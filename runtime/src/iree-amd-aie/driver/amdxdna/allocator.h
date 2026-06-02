// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_AMDXDNA_ALLOCATOR_H_
#define IREE_AMD_AIE_DRIVER_AMDXDNA_ALLOCATOR_H_

#include "iree-amd-aie/driver/amdxdna/native.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

// Creates a buffer allocator used for persistent allocations.
iree_status_t iree_hal_amdxdna_allocator_create(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_native_device_t* native_device,
    iree_hal_allocator_t** out_allocator);

#endif  // IREE_AMD_AIE_DRIVER_AMDXDNA_ALLOCATOR_H_
