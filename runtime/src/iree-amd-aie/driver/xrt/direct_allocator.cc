// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt/direct_allocator.h"

#include "iree-amd-aie/driver/xrt/xrt_buffer.h"
#include "iree/base/api.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "xrt.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "experimental/xrt_ext.h"

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
static const char* IREE_HAL_XRT_ALLOCATOR_ID = "XRT";
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

typedef struct iree_hal_xrt_allocator_t {
  // Abstract resource used for injecting reference counting and vtable; must be
  // at offset 0.
  iree_hal_resource_t resource;

  // The device that this allocator is attached to.
  iree_hal_device_t* base_device;

  xrt::device device;

  iree_allocator_t host_allocator;

  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)
} iree_hal_xrt_allocator_t;

namespace {
extern const iree_hal_allocator_vtable_t iree_hal_xrt_allocator_vtable;
}  // namespace

static iree_hal_xrt_allocator_t* iree_hal_xrt_allocator_cast(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_xrt_allocator_vtable);
  return (iree_hal_xrt_allocator_t*)base_value;
}

iree_status_t iree_hal_xrt_allocator_create(
    iree_hal_device_t* base_device, xrt::device device,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(base_device);
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_allocator_t* allocator = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*allocator),
                                (void**)&allocator));

  iree_hal_resource_initialize(&iree_hal_xrt_allocator_vtable,
                               &allocator->resource);
  allocator->base_device = base_device;
  iree_hal_device_retain(base_device);
  allocator->device = device;
  allocator->host_allocator = host_allocator;

  *out_allocator = (iree_hal_allocator_t*)allocator;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_xrt_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_xrt_allocator_t* allocator =
      iree_hal_xrt_allocator_cast(base_allocator);
  iree_allocator_t host_allocator = allocator->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_device_release(allocator->base_device);
  iree_allocator_free(host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_xrt_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_xrt_allocator_t* allocator =
      (iree_hal_xrt_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static iree_status_t iree_hal_xrt_allocator_trim(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  return iree_ok_status();
}

static void iree_hal_xrt_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics) {
  IREE_STATISTICS({
    iree_hal_xrt_allocator_t* allocator =
        iree_hal_xrt_allocator_cast(base_allocator);
    memcpy(out_statistics, &allocator->statistics, sizeof(*out_statistics));
  });
}

static iree_hal_buffer_compatibility_t
iree_hal_xrt_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t* IREE_RESTRICT allocation_size) {
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
  // application is unlikely to do anything when requesting a 0-byte buffer; but
  // it can happen in real world use cases. So we should at least not crash.
  if (*allocation_size == 0) *allocation_size = 4;

  // Align allocation sizes to 4 bytes so shaders operating on 32 bit types can
  // act safely even on buffer ranges that are not naturally aligned.
  *allocation_size = iree_host_align(*allocation_size, 4);

  return compatibility;
}

extern int magic_group_id[3];
static int id = 0;
extern xrt::uuid magic_uuid;
extern std::vector<xrt::hw_context> global_contexts;

static iree_status_t iree_hal_xrt_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_xrt_allocator_t* allocator =
      iree_hal_xrt_allocator_cast(base_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, allocation_size);

  // Coerce options into those required by the current device.
  iree_hal_buffer_params_t compat_params = *params;
  if (!iree_all_bits_set(iree_hal_xrt_allocator_query_buffer_compatibility(
                             base_allocator, &compat_params, &allocation_size),
                         IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot allocate a buffer with the given parameters");
  }

  iree_status_t status = iree_ok_status();

  // Note that for IPU host and device share the same DDR RAM address space. So
  // the `HOST_ONLY` flag below is not strictly correct but present for legacy
  // reasons and it is what is used by XRT to identify that we want to allocate
  // in the DDR RAM. Also, group_id is not of relavence in this use case so we
  // set it to 0.
  int group_id = magic_group_id[id++ % 3];
  std::unique_ptr<xrt::ext::bo> xrt_buffer;

  try {
    fprintf(stderr, "uuid %s\n", magic_uuid.to_string().c_str());
    fprintf(stderr, "bo %d\n", group_id);
    // xrt_buffer = std::make_unique<xrt::bo>(allocator->device, allocation_size,
    //                                        XRT_BO_FLAGS_HOST_ONLY, group_id);
    xrt_buffer = std::make_unique<xrt::ext::bo>(global_contexts.front(), allocation_size,
                                                xrt::ext::bo::access_mode::read_write);
  } catch (std::exception &e) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "could not allocate memory for buffer %s", e.what());
  }
  IREE_TRACE_ZONE_END(z0);
  if (!xrt_buffer) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "unable to allocate buffer");
  }

  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_xrt_buffer_wrap(
        xrt_buffer.release(), base_allocator, compat_params.type,
        compat_params.access, compat_params.usage, allocation_size,
        /*byte_offset=*/0, /*byte_length=*/allocation_size,
        iree_hal_buffer_release_callback_null(), &buffer);
  }

  if (iree_status_is_ok(status)) {
    IREE_TRACE_ALLOC_NAMED(IREE_HAL_XRT_ALLOCATOR_ID,
                           (void*)iree_hal_xrt_buffer_handle(buffer),
                           allocation_size);
    IREE_STATISTICS(iree_hal_allocator_statistics_record_alloc(
        &allocator->statistics, compat_params.type, allocation_size));
    *out_buffer = buffer;
  } else {
    if (buffer) iree_hal_buffer_release(buffer);
  }
  return status;
}

static void iree_hal_xrt_allocator_deallocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT base_buffer) {
  iree_hal_xrt_allocator_t* allocator =
      iree_hal_xrt_allocator_cast(base_allocator);

  try {
    delete iree_hal_xrt_buffer_handle(base_buffer);
  } catch (...) {
    (void)iree_status_from_code(IREE_STATUS_DATA_LOSS);
    return;
  }
  IREE_TRACE_FREE_NAMED(IREE_HAL_XRT_ALLOCATOR_ID,
                        (void*)iree_hal_xrt_buffer_handle(base_buffer));
  IREE_STATISTICS(iree_hal_allocator_statistics_record_free(
      &allocator->statistics, iree_hal_buffer_memory_type(base_buffer),
      iree_hal_buffer_allocation_size(base_buffer)));

  iree_hal_buffer_destroy(base_buffer);
}

static iree_status_t iree_hal_xrt_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "external buffer type import not implemented");
}

static iree_status_t iree_hal_xrt_allocator_export_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* IREE_RESTRICT out_external_buffer) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "unsupported exporting to external buffer");
}

static iree_status_t iree_hal_xrt_allocator_query_memory_heaps(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count) {
  return iree_status_from_code(IREE_STATUS_UNIMPLEMENTED);
  ;
}

namespace {
const iree_hal_allocator_vtable_t iree_hal_xrt_allocator_vtable = {
    /*.destroy = */ iree_hal_xrt_allocator_destroy,
    /*.host_allocator = */ iree_hal_xrt_allocator_host_allocator,
    /*.trim = */ iree_hal_xrt_allocator_trim,
    /*.query_statistics = */ iree_hal_xrt_allocator_query_statistics,
    /*.query_memory_heaps=*/iree_hal_xrt_allocator_query_memory_heaps,
    /*.query_buffer_compatibility = */
    iree_hal_xrt_allocator_query_buffer_compatibility,
    /*.allocate_buffer = */ iree_hal_xrt_allocator_allocate_buffer,
    /*.deallocate_buffer = */ iree_hal_xrt_allocator_deallocate_buffer,
    /*.import_buffer = */ iree_hal_xrt_allocator_import_buffer,
    /*.export_buffer = */ iree_hal_xrt_allocator_export_buffer,
};
}  // namespace
