// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_HSA_ALLOCATOR_H_
#define IREE_EXPERIMENTAL_HSA_ALLOCATOR_H_

#include "iree-amd-aie/driver/hsa/status_util.h"
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
  iree_hal_resource_t resource;

  hsa_agent_t hsa_agent;
  hsa_agent_t cpu_agent;
  hsa_agent_t aie_agent;
  hsa_amd_memory_pool_t cpu_pool;

  // One memory pool and region for now
  hsa_amd_memory_pool_t buffers_pool;
  hsa_region_t kernel_argument_pool;

  const iree_hal_hsa_dynamic_symbols_t* symbols;

  iree_allocator_t host_allocator;

  // Whether the GPU and CPU can concurrently access HSA managed data in a
  // coherent way. We would need to explicitly perform flushing and invalidation
  // between GPU and CPU if not.
  bool supports_concurrent_managed_access;

  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)
};

iree_hal_hsa_allocator_t* iree_hal_hsa_allocator_cast(
    iree_hal_allocator_t* base_value);

#ifdef __cplusplus
}       // extern "C"
#endif  // __cplusplus

#endif  // IREE_EXPERIMENTAL_HSA_ALLOCATOR_H_
