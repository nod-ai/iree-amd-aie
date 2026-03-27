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
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/deferred_command_buffer.h"
#include "iree/hal/utils/file_registry.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/queue_emulation.h"
#include "iree/hal/utils/queue_host_call_emulation.h"

typedef struct iree_hal_xrt_device_t {
  iree_hal_resource_t resource;

  iree_string_view_t identifier;

  iree_arena_block_pool_t block_pool;

  iree_hal_xrt_device_params_t params;
  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;

  iree_hal_channel_provider_t* channel_provider;
  iree_hal_device_topology_info_t topology_info;

  iree_async_proactor_pool_t* proactor_pool;
  iree_async_proactor_t* proactor;

  xrtDeviceHandle device_hdl;
} iree_hal_xrt_device_t;

namespace {
extern const iree_hal_device_vtable_t iree_hal_xrt_device_vtable;

static iree_status_t iree_hal_xrt_device_profiling_ok(
    iree_hal_device_t* device,
    const iree_hal_device_profiling_options_t* options) {
  (void)device;
  (void)options;
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_device_profiling_flush_ok(
    iree_hal_device_t* device) {
  (void)device;
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_device_profiling_end_ok(
    iree_hal_device_t* device) {
  (void)device;
  return iree_ok_status();
}
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

static iree_status_t iree_hal_xrt_device_initialize_async(
    iree_hal_xrt_device_t* device,
    const iree_hal_device_create_params_t* create_params) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(create_params->proactor_pool);
  device->proactor_pool = create_params->proactor_pool;
  iree_async_proactor_pool_retain(device->proactor_pool);
  return iree_async_proactor_pool_get(device->proactor_pool, 0,
                                      &device->proactor);
}

static iree_status_t iree_hal_xrt_device_create_internal(
    iree_string_view_t identifier, const iree_hal_xrt_device_params_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_xrt_device_t* device = nullptr;

  iree_host_size_t total_size = iree_sizeof_struct(*device) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&device));

  device->device_allocator = nullptr;
  memset(&device->topology_info, 0, sizeof(device->topology_info));
  device->channel_provider = nullptr;
  device->proactor_pool = nullptr;
  device->proactor = nullptr;

  try {
    if (IREE_UNLIKELY(xrt::system::enumerate_devices() == 0)) {
      iree_allocator_free(host_allocator, device);
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "No XRT devices found");
    }
  } catch (std::exception& e) {
    iree_allocator_free(host_allocator, device);
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
    status = iree_hal_xrt_device_initialize_async(device, create_params);
    if (iree_status_is_ok(status)) {
      *out_device = (iree_hal_device_t*)device;
    }
  }

  if (!iree_status_is_ok(status)) {
    if (device->device_allocator) {
      iree_hal_allocator_release(device->device_allocator);
    }
    if (device->proactor_pool) {
      iree_async_proactor_pool_release(device->proactor_pool);
    }
    (void)xrtDeviceClose(device_hdl);
    iree_allocator_free(host_allocator, device);
  }
  return status;
}

iree_status_t iree_hal_xrt_device_create(
    iree_string_view_t identifier, const iree_hal_xrt_device_params_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_xrt_device_create_internal(
      identifier, params, create_params, host_allocator, out_device);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_xrt_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_xrt_device_t* device = iree_hal_xrt_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_channel_provider_release(device->channel_provider);
  iree_hal_allocator_release(device->device_allocator);
  iree_arena_block_pool_deinitialize(&device->block_pool);
  if (device->proactor_pool) {
    iree_async_proactor_pool_release(device->proactor_pool);
  }
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
  iree_hal_allocator_t* old_allocator = device->device_allocator;
  device->device_allocator = new_allocator;
  iree_hal_allocator_release(old_allocator);
}

static void iree_hal_xrt_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_xrt_device_t* device = iree_hal_xrt_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
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

static iree_status_t iree_hal_xrt_device_query_capabilities(
    iree_hal_device_t* base_device,
    iree_hal_device_capabilities_t* out_capabilities) {
  (void)base_device;
  memset(out_capabilities, 0, sizeof(*out_capabilities));
  return iree_ok_status();
}

static const iree_hal_device_topology_info_t*
iree_hal_xrt_device_topology_info(iree_hal_device_t* base_device) {
  iree_hal_xrt_device_t* device = iree_hal_xrt_device_cast(base_device);
  return &device->topology_info;
}

static iree_status_t iree_hal_xrt_device_refine_topology_edge(
    iree_hal_device_t* src_device, iree_hal_device_t* dst_device,
    iree_hal_topology_edge_t* edge) {
  (void)src_device;
  (void)dst_device;
  (void)edge;
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_device_assign_topology_info(
    iree_hal_device_t* base_device,
    const iree_hal_device_topology_info_t* topology_info) {
  iree_hal_xrt_device_t* device = iree_hal_xrt_device_cast(base_device);
  device->topology_info = *topology_info;
  return iree_ok_status();
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
      queue_affinity, binding_capacity, &device->block_pool,
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
    iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_xrt_device_t* device = iree_hal_xrt_device_cast(base_device);
  return iree_hal_xrt_nop_executable_cache_create(
      device->device_hdl, identifier, device->host_allocator,
      out_executable_cache);
}

static iree_status_t iree_hal_xrt_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  return iree_hal_file_from_handle(
      iree_hal_device_allocator(base_device), queue_affinity, access, handle,
      iree_hal_device_host_allocator(base_device), out_file);
}

static iree_status_t iree_hal_xrt_device_create_semaphore(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_xrt_device_t* device = iree_hal_xrt_device_cast(base_device);
  return iree_hal_xrt_semaphore_create(device->proactor, queue_affinity,
                                       initial_value, flags,
                                       device->host_allocator, out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_xrt_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  (void)base_device;
  (void)semaphore;
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_xrt_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  (void)queue_affinity;
  (void)pool;
  (void)flags;
  iree_hal_xrt_device_t* device = iree_hal_xrt_device_cast(base_device);
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(
      wait_semaphore_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      device->device_allocator, params, allocation_size, out_buffer));
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_signal(signal_semaphore_list,
                                                      /*frontier=*/NULL));
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  (void)buffer;
  (void)flags;
  return iree_hal_device_queue_barrier(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_EXECUTE_FLAG_NONE);
}

static iree_status_t iree_hal_xrt_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  iree_hal_file_transfer_options_t options = {
      .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  return iree_hal_device_queue_read_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_file, source_offset, target_buffer, target_offset, length, flags,
      options);
}

static iree_status_t iree_hal_xrt_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  iree_hal_file_transfer_options_t options = {
      .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  return iree_hal_device_queue_write_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_buffer, source_offset, target_file, target_offset, length, flags,
      options);
}

static iree_status_t iree_hal_xrt_device_queue_host_call(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  return iree_hal_device_queue_emulated_host_call(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      call, args, flags);
}

static iree_status_t iree_hal_xrt_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)queue_affinity;
  (void)flags;

  iree_hal_xrt_device_t* device = iree_hal_xrt_device_cast(base_device);

  iree_status_t status = iree_hal_semaphore_list_wait(
      wait_semaphore_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);

  if (iree_status_is_ok(status) && command_buffer) {
    iree_hal_command_buffer_t* xrt_command_buffer = nullptr;
    iree_hal_command_buffer_mode_t mode =
        IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
        IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION |
        IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED;
    status = iree_hal_xrt_direct_command_buffer_create(
        iree_hal_device_allocator(base_device), mode,
        IREE_HAL_COMMAND_CATEGORY_ANY,
        /*binding_capacity=*/0, &device->block_pool,
        device->host_allocator, &xrt_command_buffer);
    if (iree_status_is_ok(status)) {
      status = iree_hal_deferred_command_buffer_apply(
          command_buffer, xrt_command_buffer, binding_table);
    }
    iree_hal_command_buffer_release(xrt_command_buffer);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_signal(signal_semaphore_list,
                                            /*frontier=*/NULL);
  } else {
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_xrt_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  (void)base_device;
  (void)queue_affinity;
  return iree_ok_status();
}

namespace {
const iree_hal_device_vtable_t iree_hal_xrt_device_vtable = {
    .destroy = iree_hal_xrt_device_destroy,
    .id = iree_hal_xrt_device_id,
    .host_allocator = iree_hal_xrt_device_host_allocator,
    .device_allocator = iree_hal_xrt_device_allocator,
    .replace_device_allocator = iree_hal_xrt_replace_device_allocator,
    .replace_channel_provider = iree_hal_xrt_replace_channel_provider,
    .trim = iree_hal_xrt_device_trim,
    .query_i64 = iree_hal_xrt_device_query_i64,
    .query_capabilities = iree_hal_xrt_device_query_capabilities,
    .topology_info = iree_hal_xrt_device_topology_info,
    .refine_topology_edge = iree_hal_xrt_device_refine_topology_edge,
    .assign_topology_info = iree_hal_xrt_device_assign_topology_info,
    .create_channel = iree_hal_xrt_device_create_channel,
    .create_command_buffer = iree_hal_xrt_device_create_command_buffer,
    .create_event = iree_hal_xrt_device_create_event,
    .create_executable_cache = iree_hal_xrt_device_create_executable_cache,
    .import_file = iree_hal_xrt_device_import_file,
    .create_semaphore = iree_hal_xrt_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_xrt_device_query_semaphore_compatibility,
    .queue_alloca = iree_hal_xrt_device_queue_alloca,
    .queue_dealloca = iree_hal_xrt_device_queue_dealloca,
    .queue_fill = iree_hal_device_queue_emulated_fill,
    .queue_update = iree_hal_device_queue_emulated_update,
    .queue_copy = iree_hal_device_queue_emulated_copy,
    .queue_read = iree_hal_xrt_device_queue_read,
    .queue_write = iree_hal_xrt_device_queue_write,
    .queue_host_call = iree_hal_xrt_device_queue_host_call,
    .queue_dispatch = iree_hal_device_queue_emulated_dispatch,
    .queue_execute = iree_hal_xrt_device_queue_execute,
    .queue_flush = iree_hal_xrt_device_queue_flush,
    .profiling_begin = iree_hal_xrt_device_profiling_ok,
    .profiling_flush = iree_hal_xrt_device_profiling_flush_ok,
    .profiling_end = iree_hal_xrt_device_profiling_end_ok,
};
}  // namespace
