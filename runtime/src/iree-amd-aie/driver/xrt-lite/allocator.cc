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
  std::shared_ptr<shim_xdna::device> shim_device;
  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)

  iree_hal_xrt_lite_allocator(iree_allocator_t host_allocator,
                              std::shared_ptr<shim_xdna::device> shim_device)
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

  iree_status_t trim() {
    // TODO(null): if the allocator is retaining any unused resources they
    // should be dropped here. If the underlying implementation has pools or
    // caches it should be notified that a trim is requested. This is called in
    // low-memory situations or when IREE is not going to be used for awhile
    // (low power modes or suspension).
    (void)this;

    return iree_ok_status();
  }

  void query_statistics(iree_hal_allocator_statistics_t* out_statistics) {
    IREE_STATISTICS({
      memcpy(out_statistics, &this->statistics, sizeof(*out_statistics));
      // TODO(null): update statistics (merge).
    });
  }

  iree_status_t query_memory_heaps(iree_host_size_t capacity,
                                   iree_hal_allocator_memory_heap_t* heaps,
                                   iree_host_size_t* out_count) {
    // TODO(null): return heap information. This is called at least once with a
    // capacity that may be 0 (indicating a query for the total count) and the
    // heaps should only be populated if capacity is sufficient to store all of
    // them.
    (void)this;
    iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                            "heap query not implemented");
    return status;
  }

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
      // TODO(benvanik): make a helper for this.
#if IREE_STATUS_MODE
      iree_bitfield_string_temp_t temp0, temp1, temp2;
      iree_string_view_t memory_type_str =
          iree_hal_memory_type_format(params->type, &temp0);
      iree_string_view_t usage_str =
          iree_hal_buffer_usage_format(params->usage, &temp1);
      iree_string_view_t compatibility_str =
          iree_hal_buffer_compatibility_format(compatibility, &temp2);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "allocator cannot allocate a buffer with the given parameters; "
          "memory_type=%.*s, usage=%.*s, compatibility=%.*s",
          (int)memory_type_str.size, memory_type_str.data, (int)usage_str.size,
          usage_str.data, (int)compatibility_str.size, compatibility_str.data);
#else
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "allocator cannot allocate a buffer with the given parameters");
#endif  // IREE_STATUS_MODE
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
    (void)this;

    iree_hal_buffer_t* buffer = nullptr;
    shim_xcl_bo_flags f = {};
    f.flags = XCL_BO_FLAGS_HOST_ONLY;
    f.extension = 0;
    std::unique_ptr<shim_xdna::bo> bo =
        shim_device->alloc_bo(allocation_size, f);
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

  iree_status_t import_buffer(
      const iree_hal_buffer_params_t* params,
      iree_hal_external_buffer_t* external_buffer,
      iree_hal_buffer_release_callback_t release_callback,
      iree_hal_buffer_t** out_buffer) {
    // Coerce options into those required by the current device.
    iree_hal_buffer_params_t compat_params = *params;
    iree_device_size_t allocation_size = external_buffer->size;
    iree_hal_buffer_compatibility_t compatibility =
        this->query_buffer_compatibility(&compat_params, &allocation_size);
    if (!iree_all_bits_set(compatibility,
                           IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE)) {
      // TODO(benvanik): make a helper for this.
#if IREE_STATUS_MODE
      iree_bitfield_string_temp_t temp0, temp1, temp2;
      iree_string_view_t memory_type_str =
          iree_hal_memory_type_format(params->type, &temp0);
      iree_string_view_t usage_str =
          iree_hal_buffer_usage_format(params->usage, &temp1);
      iree_string_view_t compatibility_str =
          iree_hal_buffer_compatibility_format(compatibility, &temp2);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "allocator cannot import a buffer with the given parameters; "
          "memory_type=%.*s, usage=%.*s, compatibility=%.*s",
          (int)memory_type_str.size, memory_type_str.data, (int)usage_str.size,
          usage_str.data, (int)compatibility_str.size, compatibility_str.data);
#else
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "allocator cannot import a buffer with the given parameters");
#endif  // IREE_STATUS_MODE
    }

    // TODO(null): switch on external_buffer->type and import the buffer. See
    // the headers for more information on semantics. Most implementations can
    // service IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION by just wrapping
    // the underlying device pointer. Those that can service
    // IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION may be able to avoid a lot
    // of additional copies when moving data around between host and device or
    // across devices from different drivers.
    (void)this;
    iree_status_t status = iree_make_status(
        IREE_STATUS_UNIMPLEMENTED, "external buffer type not supported");

    return status;
  }

  iree_status_t export_buffer(iree_hal_buffer_t* buffer,
                              iree_hal_external_buffer_type_t requested_type,
                              iree_hal_external_buffer_flags_t requested_flags,
                              iree_hal_external_buffer_t* out_external_buffer) {
    // TODO(null): switch on requested_type and export as appropriate. Most
    // implementations can service
    // IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION by just exposing the
    // underlying device pointer. Those that can service
    // IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION may be able to avoid a lot
    // of additional copies when moving data around between host and device or
    // across devices from different drivers.
    (void)this;
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "external buffer type not supported");
  }
};

static iree_hal_xrt_lite_allocator* iree_hal_xrt_lite_allocator_cast(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_xrt_lite_allocator_vtable);
  return reinterpret_cast<iree_hal_xrt_lite_allocator*>(base_value);
}

iree_status_t iree_hal_xrt_lite_allocator_create(
    iree_allocator_t host_allocator, std::shared_ptr<shim_xdna::device> device,
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

  // TODO(max): shouldn't this be happening automatically via the refcounting
  // (or just the dtor of device?)
  allocator->shim_device.reset();
  iree_hal_resource_release(&allocator->resource);
  // something's not happening here?
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

ALLOCATOR_MEMBER_STATUS(trim)
ALLOCATOR_MEMBER_VOID(query_statistics)
ALLOCATOR_MEMBER_STATUS(query_memory_heaps)
ALLOCATOR_MEMBER(query_buffer_compatibility, iree_hal_buffer_compatibility_t)
ALLOCATOR_MEMBER_STATUS(allocate_buffer)
ALLOCATOR_MEMBER_VOID(deallocate_buffer)
ALLOCATOR_MEMBER_STATUS(import_buffer)
ALLOCATOR_MEMBER_STATUS(export_buffer)

namespace {
const iree_hal_allocator_vtable_t iree_hal_xrt_lite_allocator_vtable = {
    .destroy = iree_hal_xrt_lite_allocator_destroy,
    .host_allocator = iree_hal_xrt_lite_allocator_host_allocator,
    .trim = iree_hal_xrt_lite_allocator_trim,
    .query_statistics = iree_hal_xrt_lite_allocator_query_statistics,
    .query_memory_heaps = iree_hal_xrt_lite_allocator_query_memory_heaps,
    .query_buffer_compatibility =
        iree_hal_xrt_lite_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_xrt_lite_allocator_allocate_buffer,
    .deallocate_buffer = iree_hal_xrt_lite_allocator_deallocate_buffer,
    .import_buffer = iree_hal_xrt_lite_allocator_import_buffer,
    .export_buffer = iree_hal_xrt_lite_allocator_export_buffer,
};

}
