// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt-lite/buffer.h"

#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/bo.h"
#include "iree-amd-aie/driver/xrt-lite/util.h"

namespace {
extern const iree_hal_buffer_vtable_t iree_hal_xrt_lite_buffer_vtable;
}

struct iree_hal_xrt_lite_buffer {
  iree_hal_buffer_t base;
  shim_xdna::bo* bo;
  iree_allocator_t host_allocator;
  iree_hal_buffer_release_callback_t release_callback;
};

static iree_status_t iree_hal_xrt_lite_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_buffer* buffer = IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
      base_buffer, iree_hal_xrt_lite_buffer_vtable, iree_hal_xrt_lite_buffer);
  if (IREE_UNLIKELY(!buffer->bo)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "buffer does not have device memory attached and cannot be mapped");
  }
  buffer->bo->sync(shim_xdna::direction::device2host, local_byte_length,
                   local_byte_offset);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_lite_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_buffer* buffer = IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
      base_buffer, iree_hal_xrt_lite_buffer_vtable, iree_hal_xrt_lite_buffer);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_buffer_validate_memory_type(
              iree_hal_buffer_memory_type(
                  reinterpret_cast<const iree_hal_buffer_t*>(buffer)),
              IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_buffer_validate_usage(
              iree_hal_buffer_allowed_usage(
                  reinterpret_cast<const iree_hal_buffer_t*>(buffer)),
              mapping_mode == IREE_HAL_MAPPING_MODE_PERSISTENT
                  ? IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT
                  : IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED));

  void* host_ptr = buffer->bo->map();
  // Should be guaranteed by previous checks.
  IREE_ASSERT(host_ptr != nullptr);
  uint8_t* data_ptr = reinterpret_cast<uint8_t*>(host_ptr) + local_byte_offset;
  iree_status_t status = iree_hal_xrt_lite_buffer_invalidate_range(
      base_buffer, local_byte_offset, local_byte_length);
  // If we mapped for discard, scribble over the bytes. This is not a mandated
  // behavior but it will make debugging issues easier. Alternatively for heap
  // buffers we could reallocate them such that ASAN yells, but that would
  // only work if the entire buffer was discarded.
#ifndef NDEBUG
  if (iree_any_bit_set(memory_access, IREE_HAL_MEMORY_ACCESS_DISCARD)) {
    memset(data_ptr, 0xCD, local_byte_length);
  }
#endif  // !NDEBUG
  mapping->contents = iree_make_byte_span(data_ptr, local_byte_length);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_xrt_lite_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_buffer* buffer = IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
      base_buffer, iree_hal_xrt_lite_buffer_vtable, iree_hal_xrt_lite_buffer);
  if (IREE_UNLIKELY(!buffer->bo)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "buffer does not have device memory attached and cannot be mapped");
  }

  buffer->bo->sync(shim_xdna::direction::host2device, local_byte_length,
                   local_byte_offset);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_lite_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  return iree_hal_xrt_lite_buffer_flush_range(base_buffer, local_byte_offset,
                                              local_byte_length);
}

iree_status_t iree_hal_xrt_lite_buffer_wrap(
    shim_xdna::bo* bo, iree_hal_buffer_placement_t placement,
    iree_hal_memory_type_t memory_type, iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = nullptr;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_buffer* buffer = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*buffer),
                                reinterpret_cast<void**>(&buffer)));
  iree_hal_buffer_initialize(placement, &buffer->base, allocation_size,
                             byte_offset, byte_length, memory_type,
                             allowed_access, allowed_usage,
                             &iree_hal_xrt_lite_buffer_vtable, &buffer->base);
  buffer->release_callback = release_callback;
  buffer->bo = bo;
  *out_buffer = &buffer->base;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_xrt_lite_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_hal_xrt_lite_buffer* buffer = IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
      base_buffer, iree_hal_xrt_lite_buffer_vtable, iree_hal_xrt_lite_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = buffer->host_allocator;
  if (buffer->release_callback.fn) {
    buffer->release_callback.fn(buffer->release_callback.user_data,
                                base_buffer);
  }

  delete buffer->bo;
  iree_allocator_free(host_allocator, buffer);

  IREE_TRACE_ZONE_END(z0);
}

shim_xdna::bo* iree_hal_xrt_lite_buffer_handle(iree_hal_buffer_t* base_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_buffer* buffer = IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
      base_buffer, iree_hal_xrt_lite_buffer_vtable, iree_hal_xrt_lite_buffer);

  IREE_TRACE_ZONE_END(z0);
  return buffer->bo;
}

namespace {
const iree_hal_buffer_vtable_t iree_hal_xrt_lite_buffer_vtable = {
    .recycle = iree_hal_buffer_recycle,
    .destroy = iree_hal_xrt_lite_buffer_destroy,
    .map_range = iree_hal_xrt_lite_buffer_map_range,
    .unmap_range = iree_hal_xrt_lite_buffer_unmap_range,
    .invalidate_range = iree_hal_xrt_lite_buffer_invalidate_range,
    .flush_range = iree_hal_xrt_lite_buffer_flush_range,
};
}
