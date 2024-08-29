// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_HSA_ALLOCATOR_H_
#define IREE_EXPERIMENTAL_HSA_ALLOCATOR_H_

#include "iree-amd-aie/driver/hsa/dynamic_symbols.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

iree_status_t iree_hal_hsa_allocator_create(
    const iree_hal_hsa_dynamic_symbols_t* hsa_symbols, hsa_agent_t agent,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator);

struct iree_hal_hsa_allocator_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource{};
  hsa_agent_t aie_agent{};
  // Memory pool for allocating device-mapped memory. Used for PDI/DPU
  // instructions.
  hsa_amd_memory_pool_t global_dev_mem_pool{0};
  // System memory pool. Used for allocating kernel argument data.
  hsa_amd_memory_pool_t global_kernarg_mem_pool{0};
  const iree_hal_hsa_dynamic_symbols_t* symbols{};
  iree_allocator_t host_allocator{};
};

iree_hal_hsa_allocator_t* iree_hal_hsa_allocator_cast(
    iree_hal_allocator_t* base_value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_EXPERIMENTAL_HSA_ALLOCATOR_H_
