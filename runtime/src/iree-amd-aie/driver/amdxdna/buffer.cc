// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/amdxdna/buffer.h"

#include <limits>

#include "iree-amd-aie/driver/amdxdna/util.h"

namespace {
extern const iree_hal_buffer_vtable_t iree_hal_amdxdna_buffer_vtable;
}

struct iree_hal_amdxdna_buffer {
  iree_hal_buffer_t base;
  iree_hal_amdxdna_native_buffer_t* native_buffer;
  iree_allocator_t host_allocator;
  iree_hal_buffer_release_callback_t release_callback;
  // Set to 1 by iree_hal_amdxdna_buffer_mark_deallocated when the
  // queue_dealloca task fires. Subsequent queue ops on this buffer will fail.
  iree_atomic_uint32_t deallocated;
};

static iree_status_t iree_hal_amdxdna_buffer_resolve_root_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length,
    iree_device_size_t* out_root_byte_offset,
    iree_device_size_t* out_byte_length) {
  iree_device_size_t buffer_byte_length =
      iree_hal_buffer_byte_length(base_buffer);
  if (local_byte_length == IREE_HAL_WHOLE_BUFFER) {
    if (IREE_UNLIKELY(local_byte_offset > buffer_byte_length)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "buffer range offset %" PRIu64
                              " exceeds buffer length %" PRIu64,
                              local_byte_offset, buffer_byte_length);
    }
    local_byte_length = buffer_byte_length - local_byte_offset;
  }
  if (IREE_UNLIKELY(local_byte_offset > buffer_byte_length ||
                    local_byte_length >
                        buffer_byte_length - local_byte_offset)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "buffer range %" PRIu64 " + %" PRIu64 " exceeds buffer length %" PRIu64,
        local_byte_offset, local_byte_length, buffer_byte_length);
  }

  iree_device_size_t buffer_byte_offset =
      iree_hal_buffer_byte_offset(base_buffer);
  if (IREE_UNLIKELY(local_byte_offset >
                    std::numeric_limits<iree_device_size_t>::max() -
                        buffer_byte_offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "buffer root offset overflow");
  }
  *out_root_byte_offset = buffer_byte_offset + local_byte_offset;
  *out_byte_length = local_byte_length;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_buffer* buffer = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_buffer, iree_hal_amdxdna_buffer_vtable, iree_hal_amdxdna_buffer);
  if (IREE_UNLIKELY(!buffer->native_buffer)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "buffer does not have device memory attached and cannot be mapped");
  }
  iree_device_size_t root_byte_offset = 0;
  iree_device_size_t byte_length = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_buffer_resolve_root_range(
              base_buffer, local_byte_offset, local_byte_length,
              &root_byte_offset, &byte_length));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_sync(
              buffer->native_buffer,
              iree_hal_amdxdna_native_sync_direction_t::device_to_host,
              byte_length, root_byte_offset));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_buffer* buffer = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_buffer, iree_hal_amdxdna_buffer_vtable, iree_hal_amdxdna_buffer);
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

  if (IREE_UNLIKELY(!buffer->native_buffer)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "buffer does not have device memory attached and cannot be mapped");
  }
  void* host_ptr = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_map(buffer->native_buffer, &host_ptr));
  iree_device_size_t root_byte_offset = 0;
  iree_device_size_t byte_length = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_buffer_resolve_root_range(
              base_buffer, local_byte_offset, local_byte_length,
              &root_byte_offset, &byte_length));
  uint8_t* data_ptr = reinterpret_cast<uint8_t*>(host_ptr) + root_byte_offset;
  iree_status_t status = iree_hal_amdxdna_buffer_invalidate_range(
      base_buffer, local_byte_offset, local_byte_length);
  // If we mapped for discard, scribble over the bytes. This is not a mandated
  // behavior but it will make debugging issues easier. Alternatively for heap
  // buffers we could reallocate them such that ASAN yells, but that would
  // only work if the entire buffer was discarded.
#ifndef NDEBUG
  if (iree_any_bit_set(memory_access, IREE_HAL_MEMORY_ACCESS_DISCARD)) {
    memset(data_ptr, 0xCD, byte_length);
  }
#endif  // !NDEBUG
  mapping->contents = iree_make_byte_span(data_ptr, byte_length);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdxdna_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_buffer* buffer = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_buffer, iree_hal_amdxdna_buffer_vtable, iree_hal_amdxdna_buffer);
  if (IREE_UNLIKELY(!buffer->native_buffer)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "buffer does not have device memory attached and cannot be mapped");
  }

  iree_device_size_t root_byte_offset = 0;
  iree_device_size_t byte_length = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_buffer_resolve_root_range(
              base_buffer, local_byte_offset, local_byte_length,
              &root_byte_offset, &byte_length));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_sync(
              buffer->native_buffer,
              iree_hal_amdxdna_native_sync_direction_t::host_to_device,
              byte_length, root_byte_offset));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  return iree_hal_amdxdna_buffer_flush_range(base_buffer, local_byte_offset,
                                             local_byte_length);
}

iree_status_t iree_hal_amdxdna_buffer_wrap(
    iree_hal_amdxdna_native_buffer_t* native_buffer,
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = nullptr;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_buffer* buffer = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*buffer),
                                reinterpret_cast<void**>(&buffer)));
  iree_hal_buffer_initialize(placement, &buffer->base, allocation_size,
                             byte_offset, byte_length, memory_type,
                             allowed_access, allowed_usage,
                             &iree_hal_amdxdna_buffer_vtable, &buffer->base);
  buffer->host_allocator = host_allocator;
  buffer->release_callback = release_callback;
  buffer->native_buffer = native_buffer;
  iree_atomic_store(&buffer->deallocated, (uint32_t)0,
                    iree_memory_order_relaxed);
  *out_buffer = &buffer->base;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_amdxdna_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_hal_amdxdna_buffer* buffer = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_buffer, iree_hal_amdxdna_buffer_vtable, iree_hal_amdxdna_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = buffer->host_allocator;
  if (buffer->release_callback.fn) {
    buffer->release_callback.fn(buffer->release_callback.user_data,
                                base_buffer);
  }

  iree_hal_amdxdna_native_buffer_destroy(buffer->native_buffer);
  iree_allocator_free(host_allocator, buffer);

  IREE_TRACE_ZONE_END(z0);
}

iree_hal_amdxdna_native_buffer_t* iree_hal_amdxdna_buffer_handle(
    iree_hal_buffer_t* base_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_buffer* buffer = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_buffer, iree_hal_amdxdna_buffer_vtable, iree_hal_amdxdna_buffer);

  IREE_TRACE_ZONE_END(z0);
  return buffer->native_buffer;
}

bool iree_hal_amdxdna_buffer_is_deallocated(iree_hal_buffer_t* base_buffer) {
  if (!base_buffer) return false;
  iree_hal_amdxdna_buffer* buffer = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_buffer, iree_hal_amdxdna_buffer_vtable, iree_hal_amdxdna_buffer);
  return iree_atomic_load(&buffer->deallocated, iree_memory_order_acquire) != 0;
}

void iree_hal_amdxdna_buffer_mark_deallocated(iree_hal_buffer_t* base_buffer) {
  if (!base_buffer) return;
  iree_hal_amdxdna_buffer* buffer = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_buffer, iree_hal_amdxdna_buffer_vtable, iree_hal_amdxdna_buffer);
  iree_atomic_store(&buffer->deallocated, (uint32_t)1,
                    iree_memory_order_release);
}

namespace {
const iree_hal_buffer_vtable_t iree_hal_amdxdna_buffer_vtable = {
    .recycle = iree_hal_buffer_recycle,
    .destroy = iree_hal_amdxdna_buffer_destroy,
    .map_range = iree_hal_amdxdna_buffer_map_range,
    .unmap_range = iree_hal_amdxdna_buffer_unmap_range,
    .invalidate_range = iree_hal_amdxdna_buffer_invalidate_range,
    .flush_range = iree_hal_amdxdna_buffer_flush_range,
};
}
