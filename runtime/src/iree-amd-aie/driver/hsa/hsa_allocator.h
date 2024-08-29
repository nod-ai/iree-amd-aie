// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_HIP_ALLOCATOR_H_
#define IREE_EXPERIMENTAL_HIP_ALLOCATOR_H_

#include "experimental/hsa/memory_pools.h"
#include "experimental/hsa/status_util.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

iree_status_t iree_hal_hsa_allocator_create(
    const iree_hal_hsa_dynamic_symbols_t* hsa_symbols, hsa_agent_t agent,
    hsa_device_type_t agent_type, iree_hal_hsa_memory_pools_t* pools,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_EXPERIMENTAL_HIP_ALLOCATOR_H_
