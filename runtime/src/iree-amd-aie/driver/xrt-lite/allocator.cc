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

// TODO(null): use one ID per address space or pool - each shows as a different
// track in tracing tools.
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
static const char* IREE_HAL_XRT_LITE_ALLOCATOR_ID = "XRT-LITE unpooled";
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

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
    // TODO(null): query device heaps, supported features (concurrent
    // access/etc), and prepare any pools that will be used during allocation.
    // It's expected that most failures that occur after creation are allocation
    // request-specific so preparing here will help keep the errors more
    // localized.
    IREE_TRACE_ZONE_END(z0);
  }

  ~iree_hal_xrt_lite_allocator() = default;

  iree_hal_buffer_compatibility_t query_buffer_compatibility(
      iree_hal_buffer_params_t* params, iree_device_size_t* allocation_size) {
    // TODO(null): set compatibility rules based on the implementation.
    // Note that the user may have requested that the allocator place the
    // allocation based on whatever is optimal for the indicated usage by
    // including the IREE_HAL_MEMORY_TYPE_OPTIMAL flag. It's still required that
    // the implementation meet all the requirements but it is free to place it
    // in either host or device memory so long as the appropriate bits are
    // updated to indicate where it landed.
    (void)this;

    // All buffers can be allocated on the heap.
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

    // We are now optimal.
    params->type &= ~IREE_HAL_MEMORY_TYPE_OPTIMAL;

    // Guard against the corner case where the requested buffer size is 0. The
    // application is unlikely to do anything when requesting a 0-byte buffer;
    // but it can happen in real world use cases. So we should at least not
    // crash.
    if (*allocation_size == 0) *allocation_size = 4;
    // Align allocation sizes to 4 bytes so shaders operating on 32 bit types
    // can act safely even on buffer ranges that are not naturally aligned.
    *allocation_size = iree_host_align(*allocation_size, 4);

    return compatibility;
  }

  iree_status_t allocate_buffer(const iree_hal_buffer_params_t* params,
                                iree_device_size_t allocation_size,
                                iree_hal_buffer_t** out_buffer) {
    // Coerce options into those required by the current device.
    iree_hal_buffer_params_t compat_params = *params;
    iree_hal_buffer_compatibility_t compatibility =
        this->query_buffer_compatibility(&compat_params, &allocation_size);
    if (!iree_all_bits_set(compatibility,
                           IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "allocator cannot allocate a buffer with the given parameters");
    }

    // TODO(null): allocate the underlying device memory.
    // The impl_ptr is just used for accounting and can be an opaque value
    // (handle/etc) so long as it is consistent between the alloc and free and
    // unique to the buffer while it is live. An example
    // iree_hal_xrt_lite_buffer_wrap is provided that can be used for
    // implementations that are managing memory using underlying allocators and
    // just wrapping those device pointers in the HAL buffer type. Other
    // implementations that require more tracking can provide their own buffer
    // types that do such tracking for them.

    uint32_t flags = XCL_BO_FLAGS_HOST_ONLY;
    // if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_HOST_CACHED)) {
    //   flags = XCL_BO_FLAGS_CACHEABLE;
    // } else if (iree_all_bits_set(params->type,
    //                              IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE)) {
    //   // TODO(max): the test here isn't specific enough
    //   flags = XCL_BO_FLAGS_EXECBUF;
    // }

    std::unique_ptr<shim_xdna::bo> bo =
        shim_device->alloc_bo(allocation_size, flags);
    iree_hal_buffer_t* buffer = nullptr;
    iree_status_t status = iree_hal_xrt_lite_buffer_wrap(
        std::move(bo), reinterpret_cast<iree_hal_allocator_t*>(this),
        compat_params.type, compat_params.access, compat_params.usage,
        allocation_size,
        /*byte_offset=*/0, /*byte_length=*/allocation_size,
        iree_hal_buffer_release_callback_null(), this->host_allocator, &buffer);

    if (iree_status_is_ok(status)) {
      // TODO(null): ensure this accounting is balanced in deallocate_buffer.
      //      IREE_TRACE_ALLOC_NAMED(IREE_HAL_XRT_LITE_ALLOCATOR_ID, impl_ptr,
      //                             allocation_size);
      IREE_STATISTICS(iree_hal_allocator_statistics_record_alloc(
          &this->statistics, compat_params.type, allocation_size));
      *out_buffer = buffer;
    } else {
      iree_hal_buffer_release(buffer);
    }
    return status;
  }

  void deallocate_buffer(iree_hal_buffer_t* base_buffer) {
    // TODO(null): free the underlying device memory here. Buffers allocated
    // from this allocator will call this method to handle cleanup. Note that
    // because this method is responsible for doing the base
    // iree_hal_buffer_destroy and the caller assumes the memory has been freed
    // an implementation could pool the buffer handle and return it in the
    // future.

    // TODO(null): if the buffer was imported then this accounting may need to
    // be conditional depending on the implementation.
    bool was_imported = false;
    if (!was_imported) {
      // TODO(max):
      //      IREE_TRACE_FREE_NAMED(IREE_HAL_XRT_LITE_ALLOCATOR_ID, impl_ptr);
      IREE_STATISTICS(iree_hal_allocator_statistics_record_free(
          &this->statistics, iree_hal_buffer_memory_type(base_buffer),
          iree_hal_buffer_allocation_size(base_buffer)));
    }

    iree_hal_buffer_destroy(base_buffer);
  }
};

static iree_hal_xrt_lite_allocator* iree_hal_xrt_lite_allocator_cast(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_xrt_lite_allocator_vtable);
  return reinterpret_cast<iree_hal_xrt_lite_allocator*>(base_value);
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
      iree_hal_xrt_lite_allocator_cast(base_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_resource_release(&allocator->resource);
  iree_allocator_free(allocator->host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_xrt_lite_allocator_host_allocator(
    const iree_hal_allocator_t* base_allocator) {
  const iree_hal_xrt_lite_allocator* allocator =
      reinterpret_cast<const iree_hal_xrt_lite_allocator*>(base_allocator);
  return allocator->host_allocator;
}

#define ALLOCATOR_MEMBER(member, return_t)                                  \
  MEMBER_WRAPPER(iree_hal_allocator_t, iree_hal_xrt_lite_allocator, member, \
                 return_t)
#define ALLOCATOR_MEMBER_STATUS(member)                                    \
  MEMBER_WRAPPER_STATUS(iree_hal_allocator_t, iree_hal_xrt_lite_allocator, \
                        member)
#define ALLOCATOR_MEMBER_VOID(member) \
  MEMBER_WRAPPER_VOID(iree_hal_allocator_t, iree_hal_xrt_lite_allocator, member)

ALLOCATOR_MEMBER(query_buffer_compatibility, iree_hal_buffer_compatibility_t);
ALLOCATOR_MEMBER_STATUS(allocate_buffer);
ALLOCATOR_MEMBER_VOID(deallocate_buffer);

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
