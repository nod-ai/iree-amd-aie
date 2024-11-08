// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt/xrt_device.h"

#include "experimental/xrt_system.h"
#include "iree-amd-aie/driver/xrt/direct_allocator.h"
#include "iree-amd-aie/driver/xrt/direct_command_buffer.h"
#include "iree-amd-aie/driver/xrt/nop_executable_cache.h"
#include "iree-amd-aie/driver/xrt/nop_semaphore.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/deferred_command_buffer.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/memory_file.h"

typedef struct iree_hal_xrt_device_t {
  // Abstract resource used for injecting reference counting and vtable; must be
  // at offset 0.
  iree_hal_resource_t resource;

  iree_string_view_t identifier;

  // Block pool used for command buffers with a larger block size (as command
  // buffers can contain inlined data uploads).
  iree_arena_block_pool_t block_pool;

  iree_hal_xrt_device_params_t params;
  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;

  xrtDeviceHandle device_hdl;
} iree_hal_xrt_device_t;

namespace {
extern const iree_hal_device_vtable_t iree_hal_xrt_device_vtable;
}  // namespace

static iree_hal_xrt_device_t* iree_hal_xrt_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_xrt_device_vtable);
  return (iree_hal_xrt_device_t*)base_value;
}

void iree_hal_xrt_device_params_initialize(
    iree_hal_xrt_device_params_t* out_params) {
  memset(out_params, 0, sizeof(*out_params));
  out_params->arena_block_size = 32 * 1024;
}

static iree_status_t iree_hal_xrt_device_create_internal(
    iree_string_view_t identifier, const iree_hal_xrt_device_params_t* params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_xrt_device_t* device = nullptr;

  iree_host_size_t total_size = iree_sizeof_struct(*device) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&device));

  try {
    if (IREE_UNLIKELY(xrt::system::enumerate_devices() == 0)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "No XRT devices found");
    }
  } catch (std::exception& e) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "xrt::system::enumerate_devices failed: %s",
                            e.what());
  }

  xrtDeviceHandle device_hdl = xrtDeviceOpen(0);
  IREE_ASSERT(device_hdl, "failed to open xrt device");

  iree_status_t status =
      iree_hal_xrt_allocator_create((iree_hal_device_t*)device, device_hdl,
                                    host_allocator, &device->device_allocator);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_xrt_device_vtable,
                                 &device->resource);
    iree_string_view_append_to_buffer(
        identifier, &device->identifier,
        (char*)device + iree_sizeof_struct(*device));
    iree_arena_block_pool_initialize(params->arena_block_size, host_allocator,
                                     &device->block_pool);

    device->host_allocator = host_allocator;
    device->device_hdl = device_hdl;
    device->params = *params;
    *out_device = (iree_hal_device_t*)device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)device);
  }
  return status;
}

iree_status_t iree_hal_xrt_device_create(
    iree_string_view_t identifier, const iree_hal_xrt_device_params_t* params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_xrt_device_create_internal(
      identifier, params, host_allocator, out_device);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_xrt_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_xrt_device_t* device = iree_hal_xrt_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_allocator_release(device->device_allocator);
  iree_arena_block_pool_deinitialize(&device->block_pool);
  xrtDeviceHandle device_hdl = device->device_hdl;
  iree_allocator_free(host_allocator, device);
  (void)xrtDeviceClose(device_hdl);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_xrt_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_xrt_device_t* device = iree_hal_xrt_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_xrt_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_xrt_device_t* device = iree_hal_xrt_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_xrt_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_xrt_device_t* device = iree_hal_xrt_device_cast(base_device);
  return device->device_allocator;
}

static void iree_hal_xrt_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  iree_hal_xrt_device_t* device = iree_hal_xrt_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}

static void iree_hal_xrt_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  (void)iree_status_from_code(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_xrt_device_trim(iree_hal_device_t* base_device) {
  iree_hal_xrt_device_t* device = iree_hal_xrt_device_cast(base_device);
  iree_arena_block_pool_trim(&device->block_pool);
  return iree_hal_allocator_trim(device->device_allocator);
}

static iree_status_t iree_hal_xrt_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  iree_hal_xrt_device_t* device = iree_hal_xrt_device_cast(base_device);
  *out_value = 0;

  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    *out_value =
        iree_string_view_match_pattern(device->identifier, key) ? 1 : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    *out_value =
        iree_string_view_equal(key, IREE_SV("amdaie-xclbin-fb")) ? 1 : 0;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unsupported query");
}

static iree_status_t iree_hal_xrt_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not yet supported");
}

static iree_status_t iree_hal_xrt_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_xrt_device_t* device = iree_hal_xrt_device_cast(base_device);
  if (!iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unimplmented multi-shot command buffer");
  }
  return iree_hal_deferred_command_buffer_create(
      iree_hal_device_allocator(base_device), mode, command_categories,
      binding_capacity, &device->block_pool,
      iree_hal_device_host_allocator(base_device), out_command_buffer);
}

static iree_status_t iree_hal_xrt_device_create_event(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t** out_event) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "unimplmented event create");
}

static iree_status_t iree_hal_xrt_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_loop_t loop, iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_xrt_device_t* device = iree_hal_xrt_device_cast(base_device);
  return iree_hal_xrt_nop_executable_cache_create(
      device->device_hdl, identifier, device->host_allocator,
      out_executable_cache);
}

static iree_status_t iree_hal_xrt_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  if (iree_io_file_handle_type(handle) !=
      IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "implementation does not support the external file type");
  }
  return iree_hal_memory_file_wrap(
      queue_affinity, access, handle, iree_hal_device_allocator(base_device),
      iree_hal_device_host_allocator(base_device), out_file);
}

static iree_status_t iree_hal_xrt_device_create_semaphore(
    iree_hal_device_t* base_device, uint64_t initial_value,
    iree_hal_semaphore_flags_t flags, iree_hal_semaphore_t** out_semaphore) {
  iree_hal_xrt_device_t* device = iree_hal_xrt_device_cast(base_device);
  return iree_hal_xrt_semaphore_create(device->host_allocator, initial_value,
                                       out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_xrt_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_NONE;
}

static iree_status_t iree_hal_xrt_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  // TODO: queue-ordered allocations.
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(wait_semaphore_list,
                                                    iree_infinite_timeout()));
  IREE_RETURN_IF_ERROR(
      iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(base_device),
                                         params, allocation_size, out_buffer));
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_signal(signal_semaphore_list));
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer) {
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(wait_semaphore_list,
                                                    iree_infinite_timeout()));
  iree_status_t status = iree_hal_semaphore_list_signal(signal_semaphore_list);
  return status;
}

static iree_status_t iree_hal_xrt_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint32_t flags) {
  // TODO: expose streaming chunk count/size options.
  iree_status_t loop_status = iree_ok_status();
  iree_hal_file_transfer_options_t options = {
      /*.loop=*/iree_loop_inline(&loop_status),
      /*.chunk_count=*/IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      /*.chunk_size=*/IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_read_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_file, source_offset, target_buffer, target_offset, length, flags,
      options));
  return loop_status;
}

static iree_status_t iree_hal_xrt_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, uint32_t flags) {
  // TODO: expose streaming chunk count/size options.
  iree_status_t loop_status = iree_ok_status();
  iree_hal_file_transfer_options_t options = {
      /*.loop=*/iree_loop_inline(&loop_status),
      /*.chunk_count=*/IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      /*.chunk_size=*/IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_write_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_buffer, source_offset, target_file, target_offset, length, flags,
      options));
  return loop_status;
}

static iree_status_t iree_hal_xrt_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_xrt_device_t* device = iree_hal_xrt_device_cast(base_device);
  if (command_buffer) {
    iree_hal_command_buffer_t* xrt_command_buffer = nullptr;
    iree_hal_command_buffer_mode_t mode =
        IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
        IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION |
        IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_xrt_direct_command_buffer_create(
                iree_hal_device_allocator(base_device), mode,
                IREE_HAL_COMMAND_CATEGORY_ANY,
                /*binding_capacity=*/0, &device->block_pool,
                device->host_allocator, &xrt_command_buffer));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_deferred_command_buffer_apply(
                command_buffer, xrt_command_buffer,
                iree_hal_buffer_binding_table_empty()));
  }
  // Do we need to block here like vulkan HAL? Check if we run into some
  // correctness issue in the future.
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "Unimplemented queue flush");
}

static iree_status_t iree_hal_xrt_device_wait_semaphores(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "Unimplemented semaphore wait");
}

static iree_status_t iree_hal_xrt_device_profiling_begin(
    iree_hal_device_t* device,
    const iree_hal_device_profiling_options_t* options) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_device_profiling_end(
    iree_hal_device_t* device) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

namespace {
const iree_hal_device_vtable_t iree_hal_xrt_device_vtable = {
    /*.destroy = */ iree_hal_xrt_device_destroy,
    /*.id = */ iree_hal_xrt_device_id,
    /*.host_allocator = */ iree_hal_xrt_device_host_allocator,
    /*.device_allocator = */ iree_hal_xrt_device_allocator,
    /*.replace_device_allocator = */ iree_hal_xrt_replace_device_allocator,
    /*.replace_channel_provider = */ iree_hal_xrt_replace_channel_provider,
    /*.trim = */ iree_hal_xrt_device_trim,
    /*.query_i64 = */ iree_hal_xrt_device_query_i64,
    /*.create_channel = */ iree_hal_xrt_device_create_channel,
    /*.create_command_buffer = */ iree_hal_xrt_device_create_command_buffer,
    /*.create_event = */ iree_hal_xrt_device_create_event,
    /*.create_executable_cache = */ iree_hal_xrt_device_create_executable_cache,
    /*.import_file = */ iree_hal_xrt_device_import_file,
    /*.create_semaphore = */ iree_hal_xrt_device_create_semaphore,
    /*.query_semaphore_compatibility = */
    iree_hal_xrt_device_query_semaphore_compatibility,
    /*.queue_alloca = */ iree_hal_xrt_device_queue_alloca,
    /*.queue_dealloca = */ iree_hal_xrt_device_queue_dealloca,
    /*.queue_read=*/iree_hal_xrt_device_queue_read,
    /*.queue_write = */ iree_hal_xrt_device_queue_write,
    /*.queue_execute = */ iree_hal_xrt_device_queue_execute,
    /*.queue_flush = */ iree_hal_xrt_device_queue_flush,
    /*.wait_semaphores = */ iree_hal_xrt_device_wait_semaphores,
    /*.profiling_begin = */ iree_hal_xrt_device_profiling_begin,
    /*.profiling_end = */ iree_hal_xrt_device_profiling_end,
};
}  // namespace
