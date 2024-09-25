// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/hsa/hsa_buffer.h"

#include <cstdint>
#include <cstring>

#include "iree-amd-aie/driver/hsa/dynamic_symbols.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "status_util.h"

using iree_hal_hsa_buffer_t = struct iree_hal_hsa_buffer_t {
  iree_hal_buffer_t base;
  iree_hal_hsa_buffer_type_t type;
  void* host_ptr;
  hsa_device_pointer_t device_ptr;
  iree_hal_buffer_release_callback_t release_callback;
};

namespace {
extern const iree_hal_buffer_vtable_t iree_hal_hsa_buffer_vtable;
}

static iree_hal_hsa_buffer_t* iree_hal_hsa_buffer_cast(
    iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hsa_buffer_vtable);
  return (iree_hal_hsa_buffer_t*)base_value;
}

static const iree_hal_hsa_buffer_t* iree_hal_hsa_buffer_const_cast(
    const iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hsa_buffer_vtable);
  return (const iree_hal_hsa_buffer_t*)base_value;
}

iree_status_t iree_hal_hsa_buffer_wrap(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree_hal_hsa_buffer_type_t buffer_type, hsa_device_pointer_t device_ptr,
    void* host_ptr, iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  if (!host_ptr && iree_any_bit_set(allowed_usage,
                                    IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |
                                        IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "mappable buffers require host pointers");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hsa_buffer_t* buffer = nullptr;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_initialize(host_allocator, allocator, &buffer->base,
                               allocation_size, byte_offset, byte_length,
                               memory_type, allowed_access, allowed_usage,
                               &iree_hal_hsa_buffer_vtable, &buffer->base);
    buffer->type = buffer_type;
    buffer->host_ptr = host_ptr;
    buffer->device_ptr = device_ptr;
    buffer->release_callback = release_callback;
    *out_buffer = &buffer->base;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_hsa_buffer_free(const iree_hal_hsa_dynamic_symbols_t* hsa_symbols,
                              iree_hal_hsa_buffer_type_t buffer_type,
                              hsa_device_pointer_t device_ptr, void* host_ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  switch (buffer_type) {
    case IREE_HAL_HSA_BUFFER_TYPE_DEVICE:
    case IREE_HAL_HSA_BUFFER_TYPE_KERNEL_ARG:
    case IREE_HAL_HSA_BUFFER_TYPE_HOST: {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "hsa_amd_memory_pool_free");
      IREE_HSA_IGNORE_ERROR(hsa_symbols, hsa_amd_memory_pool_free(device_ptr));
      break;
    }
    case IREE_HAL_HSA_BUFFER_TYPE_HOST_REGISTERED:
    case IREE_HAL_HSA_BUFFER_TYPE_ASYNC:
    case IREE_HAL_HSA_BUFFER_TYPE_EXTERNAL: {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "ignored");
      break;
    }
  }
  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_hsa_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_hal_hsa_buffer_t* buffer = iree_hal_hsa_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = base_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);
  if (buffer->release_callback.fn) {
    buffer->release_callback.fn(buffer->release_callback.user_data,
                                base_buffer);
  }
  iree_allocator_free(host_allocator, buffer);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_hsa_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  iree_hal_hsa_buffer_t* buffer = iree_hal_hsa_buffer_cast(base_buffer);

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(base_buffer),
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(base_buffer),
      mapping_mode == IREE_HAL_MAPPING_MODE_PERSISTENT
          ? IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT
          : IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED));

  uint8_t* data_ptr = (uint8_t*)(buffer->host_ptr) + local_byte_offset;
#ifndef NDEBUG
  if (iree_any_bit_set(memory_access, IREE_HAL_MEMORY_ACCESS_DISCARD)) {
    memset(data_ptr, 0xCD, local_byte_length);
  }
#endif  // NDEBUG

  mapping->contents = iree_make_byte_span(data_ptr, local_byte_length);
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  return iree_ok_status();
}

iree_hal_hsa_buffer_type_t iree_hal_hsa_buffer_type(
    const iree_hal_buffer_t* base_buffer) {
  const iree_hal_hsa_buffer_t* buffer =
      iree_hal_hsa_buffer_const_cast(base_buffer);
  return buffer->type;
}

hsa_device_pointer_t iree_hal_hsa_buffer_device_pointer(
    const iree_hal_buffer_t* base_buffer) {
  const iree_hal_hsa_buffer_t* buffer =
      iree_hal_hsa_buffer_const_cast(base_buffer);
  return buffer->device_ptr;
}

void* iree_hal_hsa_buffer_host_pointer(const iree_hal_buffer_t* base_buffer) {
  const iree_hal_hsa_buffer_t* buffer =
      iree_hal_hsa_buffer_const_cast(base_buffer);
  return buffer->host_ptr;
}

namespace {
const iree_hal_buffer_vtable_t iree_hal_hsa_buffer_vtable = {
    .recycle = iree_hal_buffer_recycle,
    .destroy = iree_hal_hsa_buffer_destroy,
    .map_range = iree_hal_hsa_buffer_map_range,
    .unmap_range = iree_hal_hsa_buffer_unmap_range,
    .invalidate_range = iree_hal_hsa_buffer_invalidate_range,
    .flush_range = iree_hal_hsa_buffer_flush_range,
};
}
