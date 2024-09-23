// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/hsa/hsa_allocator.h"

#include "iree-amd-aie/driver/hsa/dynamic_symbols.h"
#include "iree-amd-aie/driver/hsa/hsa_buffer.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "status_util.h"

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
static const char* IREE_HAL_AIE_HSA_ALLOCATOR_ID = "AIE-HSA";
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

namespace {
extern const iree_hal_allocator_vtable_t iree_hal_hsa_allocator_vtable;
}

struct hsa_amd_agent_iterate_memory_pools_package_t {
  const iree_hal_hsa_dynamic_symbols_t* hsa_symbols;
  hsa::hsa_amd_memory_pool_t* pool;
};

iree_hal_hsa_allocator_t* iree_hal_hsa_allocator_cast(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hsa_allocator_vtable);
  return (iree_hal_hsa_allocator_t*)base_value;
}

hsa::hsa_status_t get_coarse_global_mem_pool(
    hsa::hsa_amd_memory_pool_t pool,
    hsa_amd_agent_iterate_memory_pools_package_t* package, bool kernarg) {
  hsa::hsa_amd_segment_t segment_type;
  hsa::hsa_status_t ret = package->hsa_symbols->hsa_amd_memory_pool_get_info(
      pool, hsa::HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment_type);
  if (ret != hsa::HSA_STATUS_SUCCESS) {
    return ret;
  }

  if (segment_type == hsa::HSA_AMD_SEGMENT_GLOBAL) {
    hsa::hsa_amd_memory_pool_global_flag_t global_pool_flags;
    ret = package->hsa_symbols->hsa_amd_memory_pool_get_info(
        pool, hsa::HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &global_pool_flags);
    if (ret != hsa::HSA_STATUS_SUCCESS) {
      return ret;
    }

    if (kernarg) {
      if ((global_pool_flags &
           hsa::HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) &&
          (global_pool_flags & hsa::HSA_REGION_GLOBAL_FLAG_KERNARG)) {
        *package->pool = pool;
      }
    } else {
      if ((global_pool_flags &
           hsa::HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) &&
          !(global_pool_flags & hsa::HSA_REGION_GLOBAL_FLAG_KERNARG)) {
        *package->pool = pool;
      }
    }
  }

  return hsa::HSA_STATUS_SUCCESS;
}

hsa::hsa_status_t get_coarse_global_dev_mem_pool(
    hsa::hsa_amd_memory_pool_t pool, void* package) {
  return get_coarse_global_mem_pool(
      pool, static_cast<hsa_amd_agent_iterate_memory_pools_package_t*>(package),
      false);
}

hsa::hsa_status_t get_coarse_global_kernarg_mem_pool(
    hsa::hsa_amd_memory_pool_t pool, void* package) {
  return get_coarse_global_mem_pool(
      pool, static_cast<hsa_amd_agent_iterate_memory_pools_package_t*>(package),
      true);
}

iree_status_t iree_hal_hsa_allocator_create(
    const iree_hal_hsa_dynamic_symbols_t* hsa_symbols, hsa::hsa_agent_t agent,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(hsa_symbols);
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hsa_allocator_t* allocator = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*allocator),
                                (void**)&allocator));
  iree_hal_resource_initialize(&iree_hal_hsa_allocator_vtable,
                               &allocator->resource);
  allocator->aie_agent = agent;
  allocator->symbols = hsa_symbols;
  allocator->host_allocator = host_allocator;

  // Find a pool for DEV BOs. This is a global system memory pool that is
  // mapped to the device. Will be used for PDIs and DPU instructions.
  auto dev_mem_pool_package = hsa_amd_agent_iterate_memory_pools_package_t{
      hsa_symbols, &allocator->global_dev_mem_pool};
  IREE_HSA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, hsa_symbols,
      hsa_amd_agent_iterate_memory_pools(agent, get_coarse_global_dev_mem_pool,
                                         &dev_mem_pool_package),
      "hsa_amd_agent_iterate_memory_pools");
  IREE_ASSERT(allocator->global_dev_mem_pool.handle);

  // Find a pool that supports kernel args. This is just normal system memory.
  // It will be used for commands and input data.
  auto kernarg_mem_pool_package = hsa_amd_agent_iterate_memory_pools_package_t{
      hsa_symbols, &allocator->global_kernarg_mem_pool};
  IREE_HSA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, hsa_symbols,
      hsa_amd_agent_iterate_memory_pools(
          agent, get_coarse_global_kernarg_mem_pool, &kernarg_mem_pool_package),
      "hsa_amd_agent_iterate_memory_pools");
  IREE_ASSERT(allocator->global_kernarg_mem_pool.handle);

  *out_allocator = (iree_hal_allocator_t*)allocator;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_hsa_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_hsa_allocator_t* allocator =
      iree_hal_hsa_allocator_cast(base_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(allocator->host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_hal_buffer_compatibility_t
iree_hal_hsa_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t* IREE_RESTRICT allocation_size) {
  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE;

  if (iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
    compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
  }

  if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
    if (iree_any_bit_set(params->usage,
                         IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE)) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
    }
  }

  params->type &= ~IREE_HAL_MEMORY_TYPE_OPTIMAL;
  if (*allocation_size == 0) *allocation_size = 4;
  *allocation_size = iree_host_align(*allocation_size, 4);

  return compatibility;
}

static iree_status_t iree_hal_hsa_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_hsa_allocator_t* allocator =
      iree_hal_hsa_allocator_cast(base_allocator);

  iree_hal_buffer_params_t compat_params = *params;
  iree_status_t status = iree_ok_status();
  iree_hal_hsa_buffer_type_t buffer_type = IREE_HAL_HSA_BUFFER_TYPE_DEVICE;
  void* host_ptr = nullptr;
  hsa_device_pointer_t device_ptr = nullptr;
  IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_hal_hsa_buffer_allocate");
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, allocation_size);

  if (iree_all_bits_set(iree_hal_hsa_allocator_query_buffer_compatibility(
                            base_allocator, &compat_params, &allocation_size),
                        IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE)) {
    IREE_HSA_RETURN_IF_ERROR(
        allocator->symbols,
        hsa_amd_memory_pool_allocate(allocator->global_kernarg_mem_pool,
                                     allocation_size, 0, &host_ptr),
        "hsa_amd_memory_pool_allocate");
    buffer_type = IREE_HAL_HSA_BUFFER_TYPE_KERNEL_ARG;
    device_ptr = host_ptr;
  } else {
    buffer_type = IREE_HAL_HSA_BUFFER_TYPE_HOST;
    status = IREE_HSA_RESULT_TO_STATUS(
        allocator->symbols, hsa_amd_memory_pool_allocate(
                                allocator->global_dev_mem_pool, allocation_size,
                                /*flags=*/0, &host_ptr));
    device_ptr = host_ptr;
  }
  IREE_TRACE_ZONE_END(z0);

  iree_hal_buffer_t* buffer = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_hsa_buffer_wrap(
        base_allocator, compat_params.type, compat_params.access,
        compat_params.usage, allocation_size,
        /*byte_offset=*/0,
        /*byte_length=*/allocation_size, buffer_type, device_ptr, host_ptr,
        iree_hal_buffer_release_callback_null(),
        iree_hal_allocator_host_allocator(base_allocator), &buffer);
  }

  if (iree_status_is_ok(status)) {
    IREE_TRACE_ALLOC_NAMED(IREE_HAL_AIE_HSA_ALLOCATOR_ID,
                           (void*)iree_hal_hsa_buffer_device_pointer(buffer),
                           allocation_size);
    *out_buffer = buffer;
  } else {
    if (!buffer && (device_ptr || host_ptr)) {
      iree_hal_hsa_buffer_free(allocator->symbols, buffer_type, device_ptr,
                               host_ptr);
    } else {
      iree_hal_buffer_release(buffer);
    }
  }
  return status;
}

static iree_allocator_t iree_hal_hsa_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  auto* allocator = (iree_hal_hsa_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static void iree_hal_hsa_allocator_deallocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT base_buffer) {
  iree_hal_hsa_allocator_t* allocator =
      iree_hal_hsa_allocator_cast(base_allocator);

  const iree_hal_hsa_buffer_type_t buffer_type =
      iree_hal_hsa_buffer_type(base_buffer);

  iree_hal_hsa_buffer_free(allocator->symbols, buffer_type,
                           iree_hal_hsa_buffer_device_pointer(base_buffer),
                           iree_hal_hsa_buffer_host_pointer(base_buffer));

  switch (buffer_type) {
    case IREE_HAL_HSA_BUFFER_TYPE_DEVICE:
    case IREE_HAL_HSA_BUFFER_TYPE_HOST: {
      IREE_TRACE_FREE_NAMED(
          IREE_HAL_AIE_HSA_ALLOCATOR_ID,
          (void*)iree_hal_hsa_buffer_device_pointer(base_buffer));
      break;
    }
    default:
      // Buffer type not tracked.
      break;
  }

  iree_hal_buffer_destroy(base_buffer);
}

namespace {
const iree_hal_allocator_vtable_t iree_hal_hsa_allocator_vtable = {
    /*destroy=*/iree_hal_hsa_allocator_destroy,
    /*host_allocator=*/iree_hal_hsa_allocator_host_allocator,
    /*trim=*/nullptr,
    /*query_statistics=*/nullptr,
    /*query_memory_heaps=*/nullptr,
    /*query_buffer_compatibility=*/
    iree_hal_hsa_allocator_query_buffer_compatibility,
    /*allocate_buffer=*/iree_hal_hsa_allocator_allocate_buffer,
    /*deallocate_buffer=*/iree_hal_hsa_allocator_deallocate_buffer,
    /*import_buffer=*/nullptr, /*export_buffer=*/nullptr};
}
