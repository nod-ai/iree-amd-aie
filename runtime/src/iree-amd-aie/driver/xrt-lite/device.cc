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

namespace {
extern const iree_hal_device_vtable_t iree_hal_xrt_lite_device_vtable;
}

struct iree_hal_xrt_lite_device {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;
  iree_allocator_t host_allocator_;
  // not used
  iree_hal_allocator_t* device_allocator_;
  // Block pool used for command buffers with a larger block size (as command
  // buffers can contain inlined data uploads).
  iree_arena_block_pool_t block_pool;
  shim_xdna::device* shim_device;

  iree_hal_xrt_lite_device(const iree_hal_xrt_lite_device_options_t* options,
                           iree_allocator_t host_allocator) {
    IREE_ASSERT_ARGUMENT(options);
    IREE_TRACE_ZONE_BEGIN(z0);

    iree_hal_resource_initialize(&iree_hal_xrt_lite_device_vtable, &resource);
    this->host_allocator_ = host_allocator;
    shim_device = new shim_xdna::device;

    iree_status_t status = iree_hal_xrt_lite_allocator_create(
        host_allocator, shim_device, &device_allocator_);
    IREE_ASSERT(iree_status_is_ok(status));
    iree_arena_block_pool_initialize(ARENA_BLOCK_SIZE, host_allocator,
                                     &block_pool);
    IREE_TRACE_ZONE_END(z0);
  }

  iree_status_t create_executable_cache(
      iree_string_view_t identifier, iree_loop_t loop,
      iree_hal_executable_cache_t** out_executable_cache) {
    return iree_hal_xrt_lite_nop_executable_cache_create(
        shim_device, identifier, host_allocator_, out_executable_cache);
  }

  iree_status_t create_command_buffer(
      iree_hal_command_buffer_mode_t mode,
      iree_hal_command_category_t command_categories,
      iree_hal_queue_affinity_t queue_affinity,
      iree_host_size_t binding_capacity,
      iree_hal_command_buffer_t** out_command_buffer) {
    if (!iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unimplmented multi-shot command buffer");
    }
    return iree_hal_deferred_command_buffer_create(
        device_allocator_, mode, command_categories, binding_capacity,
        &block_pool, host_allocator_, out_command_buffer);
  }

  iree_status_t create_semaphore(uint64_t initial_value,
                                 iree_hal_semaphore_flags_t flags,
                                 iree_hal_semaphore_t** out_semaphore) {
    return iree_hal_xrt_lite_semaphore_create(host_allocator_, initial_value,
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
                  shim_device, device_allocator_, mode,
                  IREE_HAL_COMMAND_CATEGORY_ANY,
                  /*binding_capacity=*/0, &block_pool, host_allocator_,
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
    iree_hal_allocator_release(this->device_allocator_);
    this->device_allocator_ = new_allocator;
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
          iree_string_view_equal(key, IREE_SV("amdaie-pdi-fb")) ? 1 : 0;
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
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(wait_semaphore_list,
                                                      iree_infinite_timeout()));
    IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
        device_allocator_, params, allocation_size, out_buffer));
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_signal(signal_semaphore_list));
    return iree_ok_status();
  }

  iree_string_view_t id() { return this->identifier; }

  void destroy() {
    IREE_TRACE_ZONE_BEGIN(z0);

    iree_hal_allocator_release(this->device_allocator_);
    delete this->shim_device;
    iree_allocator_free(host_allocator_, this);

    IREE_TRACE_ZONE_END(z0);
  };

  iree_allocator_t host_allocator() { return this->host_allocator_; }
  iree_hal_allocator_t* device_allocator() { return this->device_allocator_; }
};

void iree_hal_xrt_lite_device_options_initialize(
    iree_hal_xrt_lite_device_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
}

iree_status_t iree_hal_xrt_lite_device_create(
    iree_string_view_t identifier,
    const iree_hal_xrt_lite_device_options_t* options,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = nullptr;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_device* device =
      new iree_hal_xrt_lite_device(options, host_allocator);
  iree_status_t status = iree_ok_status();
  iree_host_size_t total_size = sizeof(*device) + identifier.size;
  iree_string_view_append_to_buffer(
      identifier, &device->identifier,
      reinterpret_cast<char*>(device) + total_size - identifier.size);
  // TODO(max): device id
  *out_device = reinterpret_cast<iree_hal_device_t*>(device);

  if (iree_status_is_ok(status)) {
  } else {
    iree_hal_device_release(reinterpret_cast<iree_hal_device_t*>(device));
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

#define DEVICE_MEMBER(member, return_t) \
  MEMBER_WRAPPER(iree_hal_device_t, iree_hal_xrt_lite_device, member, return_t)
#define DEVICE_MEMBER_STATUS(member) \
  MEMBER_WRAPPER_STATUS(iree_hal_device_t, iree_hal_xrt_lite_device, member)
#define DEVICE_MEMBER_VOID(member) \
  MEMBER_WRAPPER_VOID(iree_hal_device_t, iree_hal_xrt_lite_device, member)

DEVICE_MEMBER(host_allocator, iree_allocator_t);
DEVICE_MEMBER(device_allocator, iree_hal_allocator_t*);
DEVICE_MEMBER(id, iree_string_view_t);
DEVICE_MEMBER_VOID(destroy);
DEVICE_MEMBER_STATUS(create_command_buffer);
DEVICE_MEMBER_STATUS(create_executable_cache);
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
    .profiling_begin = unimplemented_ok_status,
    .profiling_flush = unimplemented_ok_status,
    .profiling_end = unimplemented_ok_status,
};
}
