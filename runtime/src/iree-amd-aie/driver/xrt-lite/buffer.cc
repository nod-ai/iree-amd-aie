// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt-lite/buffer.h"

#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/bo.h"
#include "iree-amd-aie/driver/xrt-lite/util.h"

struct iree_hal_xrt_lite_buffer_t {
  iree_hal_buffer_t base;
  std::unique_ptr<shim_xdna::bo> bo;
  iree_hal_buffer_release_callback_t release_callback;
};

namespace {
extern const iree_hal_buffer_vtable_t iree_hal_xrt_lite_buffer_vtable;
}

iree_status_t iree_hal_xrt_lite_buffer_wrap(
    std::unique_ptr<shim_xdna::bo> bo, iree_hal_allocator_t* allocator,
    iree_hal_memory_type_t memory_type, iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = nullptr;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_buffer_t* buffer = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer));
  iree_hal_buffer_initialize(host_allocator, allocator, &buffer->base,
                             allocation_size, byte_offset, byte_length,
                             memory_type, allowed_access, allowed_usage,
                             &iree_hal_xrt_lite_buffer_vtable, &buffer->base);
  buffer->release_callback = release_callback;
  // TODO(null): retain or take ownership of provided handles/pointers/etc.
  // Implementations may want to pass in an internal buffer type discriminator
  // if there are multiple or use different top-level iree_hal_buffer_t
  // implementations.
  buffer->bo = std::move(bo);
  *out_buffer = &buffer->base;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_xrt_lite_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_hal_xrt_lite_buffer_t* buffer =
      reinterpret_cast<iree_hal_xrt_lite_buffer_t*>(base_buffer);
  iree_allocator_t host_allocator = base_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Optionally call a release callback when the buffer is destroyed. Not all
  // implementations may require this but it's cheap and provides additional
  // flexibility.
  if (buffer->release_callback.fn) {
    buffer->release_callback.fn(buffer->release_callback.user_data,
                                base_buffer);
  }

  buffer->bo.reset();
  iree_allocator_free(host_allocator, buffer);

  IREE_TRACE_ZONE_END(z0);
}

namespace {
const iree_hal_buffer_vtable_t iree_hal_xrt_lite_buffer_vtable = {
    .recycle = iree_hal_buffer_recycle,
    .destroy = iree_hal_xrt_lite_buffer_destroy,
    .map_range = unimplemented,
    .unmap_range = unimplemented,
    .invalidate_range = unimplemented,
    .flush_range = unimplemented,
};
}