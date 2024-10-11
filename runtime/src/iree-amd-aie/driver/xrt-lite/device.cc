// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/device.h"

#include "iree-amd-aie/driver/xrt-lite/allocator.h"
#include "iree-amd-aie/driver/xrt-lite/api.h"
#include "iree-amd-aie/driver/xrt-lite/direct_command_buffer.h"
#include "iree-amd-aie/driver/xrt-lite/nop_executable_cache.h"
#include "iree-amd-aie/driver/xrt-lite/nop_semaphore.h"
#include "iree-amd-aie/driver/xrt-lite/util.h"
#include "iree/hal/utils/deferred_command_buffer.h"
#include "iree/hal/utils/deferred_work_queue.h"

#define ARENA_BLOCK_SIZE (32 * 1024)

struct iree_hal_xrt_lite_device {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;
  iree_allocator_t host_allocator;
  // not used
  iree_hal_allocator_t* device_allocator;
  // Block pool used for command buffers with a larger block size (as command
  // buffers can contain inlined data uploads).
  iree_arena_block_pool_t block_pool;
  std::shared_ptr<shim_xdna::device> shim_device;

  iree_status_t create_executable_cache(
      iree_string_view_t identifier, iree_loop_t loop,
      iree_hal_executable_cache_t** out_executable_cache) {
    return iree_hal_xrt_lite_nop_executable_cache_create(
        shim_device, identifier, host_allocator, out_executable_cache);
  }

  iree_status_t create_command_buffer(
      iree_hal_command_buffer_mode_t mode,
      iree_hal_command_category_t command_categories,
      iree_hal_queue_affinity_t queue_affinity,
      iree_host_size_t binding_capacity,
      iree_hal_command_buffer_t** out_command_buffer) {
    // TODO(null): pass any additional resources required to create the command
    // buffer. The implementation could pool command buffers here.
    if (!iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unimplmented multi-shot command buffer");
    }
    return iree_hal_deferred_command_buffer_create(
        device_allocator, mode, command_categories, binding_capacity,
        &block_pool, host_allocator, out_command_buffer);
  }

  iree_status_t create_semaphore(uint64_t initial_value,
                                 iree_hal_semaphore_flags_t flags,
                                 iree_hal_semaphore_t** out_semaphore) {
    return iree_hal_xrt_lite_semaphore_create(host_allocator, initial_value,
                                              out_semaphore);
  }

  iree_status_t queue_execute(
      iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_host_size_t command_buffer_count,
      iree_hal_command_buffer_t* const* command_buffers,
      iree_hal_buffer_binding_table_t const* binding_tables) {
    IREE_TRACE_ZONE_BEGIN(z0);

    for (iree_host_size_t i = 0; i < command_buffer_count; i++) {
      iree_hal_command_buffer_t* xrt_command_buffer = nullptr;
      iree_hal_command_buffer_mode_t mode =
          IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION |
          IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED;
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_xrt_lite_direct_command_buffer_create(
                  shim_device, device_allocator, mode,
                  IREE_HAL_COMMAND_CATEGORY_ANY,
                  /*binding_capacity=*/0, &block_pool, host_allocator,
                  &xrt_command_buffer));
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_deferred_command_buffer_apply(
                  command_buffers[i], xrt_command_buffer,
                  iree_hal_buffer_binding_table_empty()));
    }
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  void replace_device_allocator(iree_hal_allocator_t* new_allocator) {
    iree_hal_allocator_retain(new_allocator);
    iree_hal_allocator_release(this->device_allocator);
    this->device_allocator = new_allocator;
  }

  iree_status_t query_i64(iree_string_view_t category, iree_string_view_t key,
                          int64_t* out_value) {
    *out_value = 0;
    if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
      *out_value =
          iree_string_view_match_pattern(this->identifier, key) ? 1 : 0;
      return iree_ok_status();
    }

    if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
      *out_value =
          iree_string_view_equal(key, IREE_SV("amdaie-xclbin-fb")) ? 1 : 0;
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unsupported query");
  }

  iree_status_t queue_alloca(
      iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
      iree_device_size_t allocation_size,
      iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
    // TODO: queue-ordered allocations.
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(wait_semaphore_list,
                                                      iree_infinite_timeout()));
    IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
        device_allocator, params, allocation_size, out_buffer));
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_signal(signal_semaphore_list));
    return iree_ok_status();
  }
};

namespace {
extern const iree_hal_device_vtable_t iree_hal_xrt_lite_device_vtable;
}

void iree_hal_xrt_lite_device_options_initialize(
    iree_hal_xrt_lite_device_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  // TODO(null): set defaults based on compiler configuration. Flags should not
  // be used as multiple devices may be configured within the process or the
  // hosting application may be authored in python/etc that does not use a flags
  // mechanism accessible here.
}

iree_status_t iree_hal_xrt_lite_device_create(
    iree_string_view_t identifier,
    const iree_hal_xrt_lite_device_options_t* options,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = nullptr;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_device* device = nullptr;
  iree_host_size_t total_size = sizeof(*device) + identifier.size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size,
                                reinterpret_cast<void**>(&device)));
  iree_hal_resource_initialize(&iree_hal_xrt_lite_device_vtable,
                               &device->resource);
  iree_string_view_append_to_buffer(
      identifier, &device->identifier,
      reinterpret_cast<char*>(device) + total_size - identifier.size);
  device->host_allocator = host_allocator;
  device->shim_device = std::make_shared<shim_xdna::device>();

  // TODO(null): pass device handles and pool configuration to the allocator.
  // Some implementations may share allocators across multiple devices created
  // from the same driver.
  iree_status_t status = iree_hal_xrt_lite_allocator_create(
      host_allocator, device->shim_device, &device->device_allocator);
  iree_arena_block_pool_initialize(ARENA_BLOCK_SIZE, host_allocator,
                                   &device->block_pool);
  // TODO(max): device id
  *out_device = reinterpret_cast<iree_hal_device_t*>(device);
  if (iree_status_is_ok(status)) {
  } else {
    iree_hal_device_release(reinterpret_cast<iree_hal_device_t*>(device));
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_hal_xrt_lite_device* iree_hal_xrt_lite_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_xrt_lite_device_vtable);
  return reinterpret_cast<iree_hal_xrt_lite_device*>(base_value);
}

static iree_string_view_t iree_hal_xrt_lite_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_xrt_lite_device* device = iree_hal_xrt_lite_device_cast(base_device);
  return device->identifier;
}

static void iree_hal_xrt_lite_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_xrt_lite_device* device = iree_hal_xrt_lite_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(null): release all implementation resources here. It's expected that
  // this is only called once all outstanding resources created with this device
  // have been released by the application and no work is outstanding. If the
  // implementation performs internal async operations those should be shutdown
  // and joined first.

  iree_hal_allocator_release(device->device_allocator);
  device->shim_device.reset();
  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
};

static iree_allocator_t iree_hal_xrt_lite_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_xrt_lite_device* device = iree_hal_xrt_lite_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_xrt_lite_device_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_xrt_lite_device* device = iree_hal_xrt_lite_device_cast(base_device);
  return device->device_allocator;
}

#define DEVICE_MEMBER(member, return_t) \
  MEMBER_WRAPPER(iree_hal_device_t, iree_hal_xrt_lite_device, member, return_t)
#define DEVICE_MEMBER_STATUS(member) \
  MEMBER_WRAPPER_STATUS(iree_hal_device_t, iree_hal_xrt_lite_device, member)
#define DEVICE_MEMBER_VOID(member) \
  MEMBER_WRAPPER_VOID(iree_hal_device_t, iree_hal_xrt_lite_device, member)

DEVICE_MEMBER_STATUS(create_executable_cache);
DEVICE_MEMBER_STATUS(create_command_buffer);
DEVICE_MEMBER_STATUS(create_semaphore);
DEVICE_MEMBER_STATUS(queue_execute);
DEVICE_MEMBER_STATUS(query_i64);
DEVICE_MEMBER_STATUS(queue_alloca);
DEVICE_MEMBER_VOID(replace_device_allocator);

namespace {
const iree_hal_device_vtable_t iree_hal_xrt_lite_device_vtable = {
    .destroy = iree_hal_xrt_lite_device_destroy,
    .id = iree_hal_xrt_lite_device_id,
    .host_allocator = iree_hal_xrt_lite_device_host_allocator,
    .device_allocator = iree_hal_xrt_lite_device_device_allocator,
    .replace_device_allocator =
        iree_hal_xrt_lite_device_replace_device_allocator,
    .query_i64 = iree_hal_xrt_lite_device_query_i64,
    .create_command_buffer = iree_hal_xrt_lite_device_create_command_buffer,
    .create_executable_cache = iree_hal_xrt_lite_device_create_executable_cache,
    .create_semaphore = iree_hal_xrt_lite_device_create_semaphore,
    .queue_alloca = iree_hal_xrt_lite_device_queue_alloca,
    .queue_execute = iree_hal_xrt_lite_device_queue_execute,
};
}
