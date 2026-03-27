// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/device.h"

#include "iree-amd-aie/driver/xrt-lite/allocator.h"
#include "iree-amd-aie/driver/xrt-lite/api.h"
#include "iree-amd-aie/driver/xrt-lite/device.h"
#include "iree-amd-aie/driver/xrt-lite/direct_command_buffer.h"
#include "iree-amd-aie/driver/xrt-lite/nop_executable_cache.h"
#include "iree-amd-aie/driver/xrt-lite/nop_semaphore.h"
#include "iree-amd-aie/driver/xrt-lite/util.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/utils/deferred_command_buffer.h"
#include "iree/hal/utils/deferred_work_queue.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/queue_emulation.h"
#include "iree/hal/utils/queue_host_call_emulation.h"

#define ARENA_BLOCK_SIZE (32 * 1024)

namespace {
extern const iree_hal_device_vtable_t iree_hal_xrt_lite_device_vtable;
}

iree_hal_xrt_lite_device::iree_hal_xrt_lite_device(
    const iree_hal_xrt_lite_device_params* options,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_TRACE_ZONE_BEGIN(z0);

  channel_provider = nullptr;
  memset(&topology_info, 0, sizeof(topology_info));
  proactor_pool = nullptr;
  proactor = nullptr;

  iree_hal_resource_initialize(&iree_hal_xrt_lite_device_vtable, &resource);
  this->host_allocator = host_allocator;
  this->power_mode = options->power_mode;
  if (iree_string_view_equal(power_mode, IREE_SV("default"))) {
    shim_device = new shim_xdna::device(
        options->n_core_rows, options->n_core_cols, POWER_MODE_DEFAULT);
  } else if (iree_string_view_equal(power_mode, IREE_SV("low"))) {
    shim_device = new shim_xdna::device(options->n_core_rows,
                                        options->n_core_cols, POWER_MODE_LOW);
  } else if (iree_string_view_equal(power_mode, IREE_SV("medium"))) {
    shim_device = new shim_xdna::device(
        options->n_core_rows, options->n_core_cols, POWER_MODE_MEDIUM);
  } else if (iree_string_view_equal(power_mode, IREE_SV("high"))) {
    shim_device = new shim_xdna::device(options->n_core_rows,
                                        options->n_core_cols, POWER_MODE_HIGH);
  } else if (iree_string_view_equal(power_mode, IREE_SV("turbo"))) {
    shim_device = new shim_xdna::device(options->n_core_rows,
                                        options->n_core_cols, POWER_MODE_TURBO);
  } else {
    shim_device =
        new shim_xdna::device(options->n_core_rows, options->n_core_cols);
  }

  iree_status_t status = iree_hal_xrt_lite_allocator_create(
      host_allocator, shim_device, &device_allocator);
  IREE_ASSERT(iree_status_is_ok(status));
  iree_arena_block_pool_initialize(ARENA_BLOCK_SIZE, host_allocator,
                                   &block_pool);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_xrt_lite_device_initialize_async(
    iree_hal_xrt_lite_device* device,
    const iree_hal_device_create_params_t* create_params) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(create_params->proactor_pool);
  device->proactor_pool = create_params->proactor_pool;
  iree_async_proactor_pool_retain(device->proactor_pool);
  return iree_async_proactor_pool_get(device->proactor_pool, 0,
                                      &device->proactor);
}

static iree_status_t iree_hal_xrt_lite_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_hal_executable_cache_t** out_executable_cache) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_device* device = IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
      base_device, iree_hal_xrt_lite_device_vtable, iree_hal_xrt_lite_device);

  IREE_TRACE_ZONE_END(z0);
  return iree_hal_xrt_lite_nop_executable_cache_create(
      device->shim_device, identifier, device->host_allocator,
      out_executable_cache);
}

static iree_status_t iree_hal_xrt_lite_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unimplemented multi-shot command buffer");
  }

  iree_hal_xrt_lite_device* device = IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
      base_device, iree_hal_xrt_lite_device_vtable, iree_hal_xrt_lite_device);

  IREE_TRACE_ZONE_END(z0);
  return iree_hal_deferred_command_buffer_create(
      device->device_allocator, mode, command_categories, queue_affinity,
      binding_capacity, &device->block_pool, device->host_allocator,
      out_command_buffer);
}

static iree_hal_semaphore_compatibility_t
iree_hal_xrt_lite_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  (void)base_device;
  (void)semaphore;
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_xrt_lite_device_create_semaphore(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_device* device = IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
      base_device, iree_hal_xrt_lite_device_vtable, iree_hal_xrt_lite_device);

  IREE_TRACE_ZONE_END(z0);
  return iree_hal_xrt_lite_semaphore_create(device->proactor, queue_affinity,
                                            initial_value, flags,
                                            device->host_allocator, out_semaphore);
}

static iree_status_t iree_hal_xrt_lite_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)queue_affinity;
  (void)flags;

  iree_hal_xrt_lite_device* device = IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
      base_device, iree_hal_xrt_lite_device_vtable, iree_hal_xrt_lite_device);

  iree_status_t status = iree_hal_semaphore_list_wait(
      wait_semaphore_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);

  if (iree_status_is_ok(status) && command_buffer) {
    iree_hal_command_buffer_t* xrt_command_buffer = nullptr;
    iree_hal_command_buffer_mode_t mode =
        IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
        IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION |
        IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED;
    status = iree_hal_xrt_lite_direct_command_buffer_create(
        device, mode, IREE_HAL_COMMAND_CATEGORY_ANY,
        /*binding_capacity=*/0, &device->block_pool, device->host_allocator,
        &xrt_command_buffer);
    if (iree_status_is_ok(status)) {
      status = iree_hal_deferred_command_buffer_apply(command_buffer,
                                                      xrt_command_buffer,
                                                      binding_table);
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

static void iree_hal_xrt_lite_device_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_device* device = IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
      base_device, iree_hal_xrt_lite_device_vtable, iree_hal_xrt_lite_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_t* old_allocator = device->device_allocator;
  device->device_allocator = new_allocator;
  iree_hal_allocator_release(old_allocator);

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_xrt_lite_device_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_xrt_lite_device* device = IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
      base_device, iree_hal_xrt_lite_device_vtable, iree_hal_xrt_lite_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}

static iree_status_t iree_hal_xrt_lite_device_trim(iree_hal_device_t* base_device) {
  (void)base_device;
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_lite_device_query_capabilities(
    iree_hal_device_t* base_device,
    iree_hal_device_capabilities_t* out_capabilities) {
  memset(out_capabilities, 0, sizeof(*out_capabilities));
  return iree_ok_status();
}

static const iree_hal_device_topology_info_t*
iree_hal_xrt_lite_device_topology_info(iree_hal_device_t* base_device) {
  iree_hal_xrt_lite_device* device = IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
      base_device, iree_hal_xrt_lite_device_vtable, iree_hal_xrt_lite_device);
  return &device->topology_info;
}

static iree_status_t iree_hal_xrt_lite_device_refine_topology_edge(
    iree_hal_device_t* src_device, iree_hal_device_t* dst_device,
    iree_hal_topology_edge_t* edge) {
  (void)src_device;
  (void)dst_device;
  (void)edge;
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_lite_device_assign_topology_info(
    iree_hal_device_t* base_device,
    const iree_hal_device_topology_info_t* topology_info) {
  iree_hal_xrt_lite_device* device = IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
      base_device, iree_hal_xrt_lite_device_vtable, iree_hal_xrt_lite_device);
  device->topology_info = *topology_info;
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_lite_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  (void)base_device;
  (void)queue_affinity;
  (void)params;
  (void)out_channel;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not implemented");
}

static iree_status_t iree_hal_xrt_lite_device_create_event(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t** out_event) {
  (void)base_device;
  (void)queue_affinity;
  (void)flags;
  (void)out_event;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "events not implemented");
}

static iree_status_t iree_hal_xrt_lite_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  (void)base_device;
  (void)queue_affinity;
  (void)access;
  (void)handle;
  (void)flags;
  (void)out_file;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "files not implemented");
}

static iree_status_t iree_hal_xrt_lite_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_value = 0;
  iree_hal_xrt_lite_device* device = IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
      base_device, iree_hal_xrt_lite_device_vtable, iree_hal_xrt_lite_device);
  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    *out_value =
        iree_string_view_match_pattern(device->identifier, key) ? 1 : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    *out_value = iree_string_view_equal(key, IREE_SV("amdaie-pdi-fb")) ? 1 : 0;
    return iree_ok_status();
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unsupported query");
}

static iree_status_t iree_hal_xrt_lite_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)queue_affinity;
  (void)pool;
  (void)flags;

  iree_hal_xrt_lite_device* device = IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
      base_device, iree_hal_xrt_lite_device_vtable, iree_hal_xrt_lite_device);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_semaphore_list_wait(wait_semaphore_list, iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_allocator_allocate_buffer(device->device_allocator, params,
                                             allocation_size, out_buffer));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_semaphore_list_signal(signal_semaphore_list,
                                         /*frontier=*/NULL));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_lite_device_queue_dealloca(
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

static iree_status_t iree_hal_xrt_lite_device_queue_fill(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  return iree_hal_device_queue_emulated_fill(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      target_buffer, target_offset, length, pattern, pattern_length, flags);
}

static iree_status_t iree_hal_xrt_lite_device_queue_update(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  return iree_hal_device_queue_emulated_update(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_buffer, source_offset, target_buffer, target_offset, length, flags);
}

static iree_status_t iree_hal_xrt_lite_device_queue_read(
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

static iree_status_t iree_hal_xrt_lite_device_queue_write(
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

static iree_status_t iree_hal_xrt_lite_device_queue_host_call(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  return iree_hal_device_queue_emulated_host_call(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      call, args, flags);
}

static iree_status_t iree_hal_xrt_lite_device_queue_dispatch(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  return iree_hal_device_queue_emulated_dispatch(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      executable, export_ordinal, config, constants, bindings, flags);
}

static iree_status_t iree_hal_xrt_lite_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  (void)base_device;
  (void)queue_affinity;
  return iree_ok_status();
}

static iree_string_view_t iree_hal_xrt_lite_device_id(
    iree_hal_device_t* base_device) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_device* device = IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
      base_device, iree_hal_xrt_lite_device_vtable, iree_hal_xrt_lite_device);

  IREE_TRACE_ZONE_END(z0);
  return device->identifier;
}

static void iree_hal_xrt_lite_device_destroy(iree_hal_device_t* base_device) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_device* device = IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
      base_device, iree_hal_xrt_lite_device_vtable, iree_hal_xrt_lite_device);

  iree_arena_block_pool_deinitialize(&device->block_pool);
  iree_hal_channel_provider_release(device->channel_provider);
  iree_hal_allocator_release(device->device_allocator);
  if (device->proactor_pool) {
    iree_async_proactor_pool_release(device->proactor_pool);
  }
  if (!iree_string_view_is_empty(device->power_mode) &&
      !iree_string_view_equal(device->power_mode, IREE_SV("default"))) {
    device->shim_device->set_power_mode(POWER_MODE_DEFAULT);
  }
  delete device->shim_device;
  iree_allocator_free(device->host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
};

static iree_allocator_t iree_hal_xrt_lite_device_host_allocator(
    iree_hal_device_t* base_device) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_device* device = IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
      base_device, iree_hal_xrt_lite_device_vtable, iree_hal_xrt_lite_device);

  IREE_TRACE_ZONE_END(z0);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_xrt_lite_device_device_allocator(
    iree_hal_device_t* base_device) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_device* device = IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
      base_device, iree_hal_xrt_lite_device_vtable, iree_hal_xrt_lite_device);

  IREE_TRACE_ZONE_END(z0);
  return device->device_allocator;
}

void iree_hal_xrt_lite_device_options_initialize(
    iree_hal_xrt_lite_device_params* out_options) {
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_options, 0, sizeof(*out_options));

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_xrt_lite_device_create(
    iree_string_view_t identifier,
    const iree_hal_xrt_lite_device_params* options,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(create_params->proactor_pool);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = nullptr;

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_device* device = nullptr;
  iree_host_size_t total_size = iree_sizeof_struct(*device) + identifier.size;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, total_size, reinterpret_cast<void**>(&device)));
  device = new (device) iree_hal_xrt_lite_device(options, host_allocator);
  iree_string_view_append_to_buffer(
      identifier, &device->identifier,
      reinterpret_cast<char*>(device) + total_size - identifier.size);

  iree_status_t status = iree_hal_xrt_lite_device_initialize_async(
      device, create_params);
  if (!iree_status_is_ok(status)) {
    iree_arena_block_pool_deinitialize(&device->block_pool);
    iree_hal_allocator_release(device->device_allocator);
    delete device->shim_device;
    if (device->proactor_pool) {
      iree_async_proactor_pool_release(device->proactor_pool);
    }
    iree_allocator_free(host_allocator, device);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  *out_device = reinterpret_cast<iree_hal_device_t*>(device);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

namespace {
const iree_hal_device_vtable_t iree_hal_xrt_lite_device_vtable = {
    .destroy = iree_hal_xrt_lite_device_destroy,
    .id = iree_hal_xrt_lite_device_id,
    .host_allocator = iree_hal_xrt_lite_device_host_allocator,
    .device_allocator = iree_hal_xrt_lite_device_device_allocator,
    .replace_device_allocator =
        iree_hal_xrt_lite_device_replace_device_allocator,
    .replace_channel_provider =
        iree_hal_xrt_lite_device_replace_channel_provider,
    .trim = iree_hal_xrt_lite_device_trim,
    .query_i64 = iree_hal_xrt_lite_device_query_i64,
    .query_capabilities = iree_hal_xrt_lite_device_query_capabilities,
    .topology_info = iree_hal_xrt_lite_device_topology_info,
    .refine_topology_edge = iree_hal_xrt_lite_device_refine_topology_edge,
    .assign_topology_info = iree_hal_xrt_lite_device_assign_topology_info,
    .create_channel = iree_hal_xrt_lite_device_create_channel,
    .create_command_buffer = iree_hal_xrt_lite_device_create_command_buffer,
    .create_event = iree_hal_xrt_lite_device_create_event,
    .create_executable_cache = iree_hal_xrt_lite_device_create_executable_cache,
    .import_file = iree_hal_xrt_lite_device_import_file,
    .create_semaphore = iree_hal_xrt_lite_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_xrt_lite_device_query_semaphore_compatibility,
    .queue_alloca = iree_hal_xrt_lite_device_queue_alloca,
    .queue_dealloca = iree_hal_xrt_lite_device_queue_dealloca,
    .queue_fill = iree_hal_xrt_lite_device_queue_fill,
    .queue_update = iree_hal_xrt_lite_device_queue_update,
    .queue_copy = iree_hal_device_queue_emulated_copy,
    .queue_read = iree_hal_xrt_lite_device_queue_read,
    .queue_write = iree_hal_xrt_lite_device_queue_write,
    .queue_host_call = iree_hal_xrt_lite_device_queue_host_call,
    .queue_dispatch = iree_hal_xrt_lite_device_queue_dispatch,
    .queue_execute = iree_hal_xrt_lite_device_queue_execute,
    .queue_flush = iree_hal_xrt_lite_device_queue_flush,
    .profiling_begin = unimplemented_ok_status,
    .profiling_flush = unimplemented_ok_status,
    .profiling_end = unimplemented_ok_status,
};
}
