// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_HSA_DEVICE_H_
#define IREE_EXPERIMENTAL_HSA_DEVICE_H_

#include "iree-amd-aie/driver/hsa/api.h"
#include "iree-amd-aie/driver/hsa/dynamic_symbols.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct iree_hal_hsa_device_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;
  iree_string_view_t identifier;
  iree_hal_driver_t* driver;
  const iree_hal_hsa_dynamic_symbols_t* hsa_symbols;
  iree_hal_hsa_device_params_t params;
  hsa_agent_t hsa_agent;
  hsa_queue_t* hsa_dispatch_queue;
  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;
  // TODO(max): just to satisfy APIs
  iree_arena_block_pool_t block_pool;
};

iree_status_t iree_hal_hsa_device_create(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_hsa_device_params_t* params,
    const iree_hal_hsa_dynamic_symbols_t* symbols, hsa_agent_t agent,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_EXPERIMENTAL_HSA_DEVICE_H_
