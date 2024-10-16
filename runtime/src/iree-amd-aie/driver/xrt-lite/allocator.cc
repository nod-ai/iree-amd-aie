// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt-lite/allocator.h"

#include "iree-amd-aie/driver/xrt-lite/buffer.h"
#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/bo.h"
#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/device.h"
#include "iree-amd-aie/driver/xrt-lite/util.h"

namespace {
extern const iree_hal_allocator_vtable_t iree_hal_xrt_lite_allocator_vtable;
}

struct iree_hal_xrt_lite_allocator {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  shim_xdna::device* shim_device;
  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)

  iree_hal_xrt_lite_allocator(iree_allocator_t host_allocator,
                              shim_xdna::device* shim_device)
      : host_allocator(host_allocator), shim_device(shim_device) {
    IREE_TRACE_ZONE_BEGIN(z0);

    iree_hal_resource_initialize(&iree_hal_xrt_lite_allocator_vtable,
                                 &this->resource);

    IREE_TRACE_ZONE_END(z0);
  }
};

static iree_hal_buffer_compatibility_t
iree_hal_xrt_lite_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_params_t* params,
    iree_device_size_t* allocation_size) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE;

  if (iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
    compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
  }

  // Buffers can only be used on the queue if they are device visible.
  if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
    if (iree_any_bit_set(params->usage,
                         IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE)) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
    }
  }

  params->type &= ~IREE_HAL_MEMORY_TYPE_OPTIMAL;

  // Guard against the corner case where the requested buffer size is 0. The
  // application is unlikely to do anything when requesting a 0-byte buffer;
  // but it can happen in real world use cases. So we should at least not
  // crash.
  if (*allocation_size == 0) *allocation_size = 4;
  // Align allocation sizes to 4 bytes so shaders operating on 32 bit types
  // can act safely even on buffer ranges that are not naturally aligned.
  *allocation_size = iree_host_align(*allocation_size, 4);

  IREE_TRACE_ZONE_END(z0);
  return compatibility;
}

static iree_status_t iree_hal_xrt_lite_allocator_allocate_buffer(
    iree_hal_allocator_t* base_allocator,
    const iree_hal_buffer_params_t* params, iree_device_size_t allocation_size,
    iree_hal_buffer_t** out_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_allocator* allocator =
      reinterpret_cast<iree_hal_xrt_lite_allocator*>(base_allocator);
  iree_hal_buffer_params_t compat_params = *params;
  iree_hal_buffer_compatibility_t compatibility =
      iree_hal_xrt_lite_allocator_query_buffer_compatibility(
          base_allocator, &compat_params, &allocation_size);
  if (!iree_all_bits_set(compatibility,
                         IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot allocate a buffer with the given parameters");
  }

  uint32_t flags = XCL_BO_FLAGS_HOST_ONLY;
  shim_xdna::bo* bo =
      allocator->shim_device->alloc_bo(allocation_size, flags).release();
  iree_hal_buffer_t* buffer = nullptr;
  iree_status_t status = iree_hal_xrt_lite_buffer_wrap(
      bo, reinterpret_cast<iree_hal_allocator_t*>(allocator),
      compat_params.type, compat_params.access, compat_params.usage,
      allocation_size,
      /*byte_offset=*/0, /*byte_length=*/allocation_size,
      iree_hal_buffer_release_callback_null(), allocator->host_allocator,
      &buffer);

  if (iree_status_is_ok(status)) {
    IREE_STATISTICS(iree_hal_allocator_statistics_record_alloc(
        &allocator->statistics, compat_params.type, allocation_size));
    *out_buffer = buffer;
  } else {
    iree_hal_buffer_release(buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_xrt_lite_allocator_deallocate_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* base_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_allocator* allocator =
      reinterpret_cast<iree_hal_xrt_lite_allocator*>(base_allocator);
  bool was_imported = false;
  if (!was_imported) {
    IREE_STATISTICS(iree_hal_allocator_statistics_record_free(
        &allocator->statistics, iree_hal_buffer_memory_type(base_buffer),
        iree_hal_buffer_allocation_size(base_buffer)));
  }
  iree_hal_buffer_destroy(base_buffer);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_xrt_lite_allocator_create(
    iree_allocator_t host_allocator, shim_xdna::device* device,
    iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_allocator* allocator = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*allocator),
                                reinterpret_cast<void**>(&allocator)));
  allocator =
      new (allocator) iree_hal_xrt_lite_allocator(host_allocator, device);
  iree_status_t status = iree_ok_status();

  if (iree_status_is_ok(status)) {
    *out_allocator = reinterpret_cast<iree_hal_allocator_t*>(allocator);
  } else {
    iree_hal_allocator_release(
        reinterpret_cast<iree_hal_allocator_t*>(allocator));
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_xrt_lite_allocator_destroy(
    iree_hal_allocator_t* base_allocator) {
  IREE_ASSERT_ARGUMENT(base_allocator);
  iree_hal_xrt_lite_allocator* allocator =
      reinterpret_cast<iree_hal_xrt_lite_allocator*>(base_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_resource_release(&allocator->resource);
  iree_allocator_free(allocator->host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_xrt_lite_allocator_host_allocator(
    const iree_hal_allocator_t* base_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_xrt_lite_allocator* allocator =
      reinterpret_cast<const iree_hal_xrt_lite_allocator*>(base_allocator);

  IREE_TRACE_ZONE_END(z0);
  return allocator->host_allocator;
}

namespace {
const iree_hal_allocator_vtable_t iree_hal_xrt_lite_allocator_vtable = {
    .destroy = iree_hal_xrt_lite_allocator_destroy,
    .host_allocator = iree_hal_xrt_lite_allocator_host_allocator,
    .trim = unimplemented_ok_status,
    .query_statistics = unimplemented_ok_void,
    .query_buffer_compatibility =
        iree_hal_xrt_lite_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_xrt_lite_allocator_allocate_buffer,
    .deallocate_buffer = iree_hal_xrt_lite_allocator_deallocate_buffer,
};
}
