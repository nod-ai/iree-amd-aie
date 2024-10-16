// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_XRT_LITE_NOP_SEMAPHORE_H_
#define IREE_AMD_AIE_DRIVER_XRT_LITE_NOP_SEMAPHORE_H_

#include <cstdint>

#include "iree/base/api.h"
#include "iree/hal/api.h"

iree_status_t iree_hal_xrt_lite_semaphore_create(
    iree_allocator_t host_allocator, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore);

#endif  // IREE_AMD_AIE_DRIVER_XRT_LITE_NOP_SEMAPHORE_H_
