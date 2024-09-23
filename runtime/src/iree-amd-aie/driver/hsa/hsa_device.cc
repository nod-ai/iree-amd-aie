// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/hsa/hsa_device.h"

#include "iree-amd-aie/driver/hsa/command_buffer.h"
#include "iree-amd-aie/driver/hsa/hsa_allocator.h"
#include "iree-amd-aie/driver/hsa/hsa_headers.h"
#include "iree-amd-aie/driver/hsa/nop_executable_cache.h"
#include "iree-amd-aie/driver/hsa/nop_semaphore.h"
#include "iree-amd-aie/driver/hsa/status_util.h"
#include "iree-amd-aie/driver/hsa/util.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/deferred_command_buffer.h"

struct iree_hal_hsa_device_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;
  iree_string_view_t identifier;
  iree_hal_driver_t* driver;
  const iree_hal_hsa_dynamic_symbols_t* hsa_symbols;
  hsa::hsa_agent_t hsa_agent;
  hsa::hsa_queue_t* hsa_dispatch_queue;
  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;
  iree_arena_block_pool_t block_pool;
};

namespace {
extern const iree_hal_device_vtable_t iree_hal_hsa_device_vtable;
}  // namespace

#define ARENA_BLOCK_SIZE (32 * 1024)

static iree_hal_hsa_device_t* iree_hal_hsa_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hsa_device_vtable);
  return (iree_hal_hsa_device_t*)base_value;
}

iree_status_t iree_hal_hsa_device_create(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_hsa_dynamic_symbols_t* symbols, hsa::hsa_agent_t agent,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(driver);
  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  hsa::hsa_queue_t* dispatch_queue;

  // 64 queue packets is just a max; we only use a single packet currently
  IREE_HSA_RETURN_IF_ERROR(
      symbols,
      hsa_queue_create(agent, /*num_queue_packets*/ 64,
                       /*queue_type*/ hsa::HSA_QUEUE_TYPE_SINGLE,
                       /*callback*/ nullptr, /*data*/ nullptr,
                       /*private_segment_size*/ 0, /*group_segment_size*/ 0,
                       &dispatch_queue),
      "hsa_queue_create");
  IREE_ASSERT(dispatch_queue->base_address);

  iree_hal_hsa_device_t* device = nullptr;
  iree_host_size_t total_size = iree_sizeof_struct(*device) + identifier.size;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, total_size, reinterpret_cast<void**>(&device)));

  iree_status_t status = iree_hal_hsa_allocator_create(
      symbols, agent, host_allocator, &device->device_allocator);

  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_hsa_device_vtable,
                                 &device->resource);
    iree_string_view_append_to_buffer(
        identifier, &device->identifier,
        (char*)device + iree_sizeof_struct(*device));
    iree_arena_block_pool_initialize(ARENA_BLOCK_SIZE, host_allocator,
                                     &device->block_pool);
    device->driver = driver;
    iree_hal_driver_retain(device->driver);
    device->hsa_symbols = symbols;
    device->hsa_agent = agent;
    device->hsa_dispatch_queue = dispatch_queue;
    device->host_allocator = host_allocator;

    *out_device = (iree_hal_device_t*)device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)device);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_device_release(*out_device);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_string_view_t iree_hal_hsa_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return device->identifier;
}

static iree_hal_allocator_t* iree_hal_hsa_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return device->device_allocator;
}

static iree_allocator_t iree_hal_hsa_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return device->host_allocator;
}

static void iree_hal_hsa_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_allocator_release(device->device_allocator);
  iree_arena_block_pool_deinitialize(&device->block_pool);
  iree_allocator_free(host_allocator, device);
  device->hsa_symbols->hsa_queue_destroy(device->hsa_dispatch_queue);
  iree_hal_driver_release(device->driver);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_hsa_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_loop_t loop, iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return iree_hal_hsa_nop_executable_cache_create(
      identifier, device->hsa_symbols, device->hsa_agent,
      device->host_allocator, device->device_allocator, out_executable_cache);
}

static iree_status_t iree_hal_hsa_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return iree_hal_deferred_command_buffer_create(
      iree_hal_device_allocator(base_device), mode, command_categories,
      binding_capacity, &device->block_pool,
      iree_hal_device_host_allocator(base_device), out_command_buffer);
}

static iree_status_t iree_hal_hsa_device_create_semaphore(
    iree_hal_device_t* base_device, uint64_t initial_value,
    iree_hal_semaphore_flags_t flags, iree_hal_semaphore_t** out_semaphore) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return iree_hal_hsa_semaphore_create(device->host_allocator, initial_value,
                                       out_semaphore);
}

static iree_status_t iree_hal_hsa_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers,
    iree_hal_buffer_binding_table_t const* binding_tables) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  for (iree_host_size_t i = 0; i < command_buffer_count; i++) {
    iree_hal_command_buffer_t* hsa_command_buffer = nullptr;
    iree_hal_command_buffer_mode_t mode =
        IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
        IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION |
        IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_hsa_command_buffer_create(
                base_device, device->hsa_symbols, mode,
                IREE_HAL_COMMAND_CATEGORY_ANY,
                /*binding_capacity=*/0, device->hsa_dispatch_queue,
                &device->block_pool, device->host_allocator,
                device->device_allocator, &hsa_command_buffer));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_deferred_command_buffer_apply(
                command_buffers[i], hsa_command_buffer,
                iree_hal_buffer_binding_table_empty()));
  }
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_hsa_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}

static iree_status_t iree_hal_hsa_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  *out_value = 0;

  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    *out_value =
        iree_string_view_match_pattern(device->identifier, key) ? 1 : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    *out_value = iree_string_view_equal(key, IREE_SV("amdaie-pdi-fb")) ? 1 : 0;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unsupported query");
}

static iree_status_t iree_hal_hsa_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(wait_semaphore_list,
                                                    iree_infinite_timeout()));
  IREE_RETURN_IF_ERROR(
      iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(base_device),
                                         params, allocation_size, out_buffer));
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_signal(signal_semaphore_list));
  return iree_ok_status();
}

template <typename... Params>
iree_hal_semaphore_compatibility_t unimplemented(Params...) {
  IREE_ASSERT(false && "unimplemented");
}
namespace {
const iree_hal_device_vtable_t iree_hal_hsa_device_vtable = {
    /*destroy=*/iree_hal_hsa_device_destroy,
    /*id=*/iree_hal_hsa_device_id,
    /*host_allocator=*/iree_hal_hsa_device_host_allocator,
    /*device_allocator=*/iree_hal_hsa_device_allocator,
    /*replace_device_allocator=*/iree_hal_hsa_replace_device_allocator,
    /*replace_channel_provider=*/unimplemented,
    /*trim=*/unimplemented,
    /*query_i64=*/iree_hal_hsa_device_query_i64,
    /*create_channel=*/unimplemented,
    /*create_command_buffer=*/iree_hal_hsa_device_create_command_buffer,
    /*create_event=*/unimplemented,
    /*create_executable_cache=*/iree_hal_hsa_device_create_executable_cache,
    /*import_file=*/unimplemented,
    /*create_semaphore=*/iree_hal_hsa_device_create_semaphore,
    /*query_semaphore_compatibility*/
    unimplemented,
    /*queue_alloca=*/iree_hal_hsa_device_queue_alloca,
    /*queue_dealloca=*/unimplemented,
    /*queue_read=*/unimplemented,
    /*queue_write=*/unimplemented,
    /*queue_execute=*/iree_hal_hsa_device_queue_execute,
    /*queue_flush=*/unimplemented,
    /*wait_semaphores=*/unimplemented,
    /*profiling_begin=*/unimplemented,
    /*profiling_flush=*/unimplemented,
    /*profiling_end=*/unimplemented,
};
}  // namespace
