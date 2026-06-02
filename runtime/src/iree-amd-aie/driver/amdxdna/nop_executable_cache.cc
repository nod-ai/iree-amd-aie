// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/amdxdna/nop_executable_cache.h"

#include "iree-amd-aie/driver/amdxdna/executable.h"
#include "iree-amd-aie/driver/amdxdna/util.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"

namespace {
extern const iree_hal_executable_cache_vtable_t
    iree_hal_amdxdna_nop_executable_cache_vtable;

static iree_string_view_t iree_hal_amdxdna_executable_format() {
  return iree_make_cstring_view("amdaie-pdi-fb");
}

static bool iree_hal_amdxdna_executable_format_supported(
    iree_string_view_t executable_format) {
  return iree_string_view_equal(executable_format,
                                iree_hal_amdxdna_executable_format());
}
}  // namespace

struct iree_hal_amdxdna_nop_executable_cache {
  // Abstract resource used for injecting reference counting and vtable; must be
  // at offset 0.
  iree_hal_resource_t resource;
  iree_hal_amdxdna_native_device_t* native_device;
  iree_allocator_t host_allocator;

  iree_hal_amdxdna_nop_executable_cache(
      iree_hal_amdxdna_native_device_t* native_device,
      iree_allocator_t host_allocator)
      : native_device(native_device), host_allocator(host_allocator) {
    IREE_TRACE_ZONE_BEGIN(z0);

    iree_hal_resource_initialize(&iree_hal_amdxdna_nop_executable_cache_vtable,
                                 &resource);

    IREE_TRACE_ZONE_END(z0);
  }
};

iree_status_t iree_hal_amdxdna_nop_executable_cache_create(
    iree_hal_amdxdna_native_device_t* native_device,
    iree_string_view_t identifier, iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache) {
  IREE_ASSERT_ARGUMENT(out_executable_cache);
  *out_executable_cache = nullptr;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_nop_executable_cache* executable_cache = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*executable_cache),
                                reinterpret_cast<void**>(&executable_cache)));
  executable_cache = new (executable_cache)
      iree_hal_amdxdna_nop_executable_cache(native_device, host_allocator);
  *out_executable_cache =
      reinterpret_cast<iree_hal_executable_cache_t*>(executable_cache);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_amdxdna_nop_executable_cache_destroy(
    iree_hal_executable_cache_t* base_executable_cache) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_nop_executable_cache* executable_cache =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
          base_executable_cache, iree_hal_amdxdna_nop_executable_cache_vtable,
          iree_hal_amdxdna_nop_executable_cache);
  iree_allocator_free(executable_cache->host_allocator, executable_cache);

  IREE_TRACE_ZONE_END(z0);
}

static bool iree_hal_amdxdna_nop_executable_cache_can_prepare_format(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  return iree_hal_amdxdna_executable_format_supported(executable_format);
}

static iree_status_t iree_hal_amdxdna_nop_executable_cache_infer_format(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_amdxdna_native_executable_infer_format(
      executable_data, executable_format_capacity, executable_format,
      out_inferred_size);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdxdna_nop_executable_cache_prepare_executable(
    iree_hal_executable_cache_t* base_executable_cache,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_nop_executable_cache* executable_cache =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
          base_executable_cache, iree_hal_amdxdna_nop_executable_cache_vtable,
          iree_hal_amdxdna_nop_executable_cache);
  if (!iree_hal_amdxdna_nop_executable_cache_can_prepare_format(
          base_executable_cache, executable_params->caching_mode,
          executable_params->executable_format)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "no amdxdna executable implementation registered for the given "
        "executable format '%.*s'",
        (int)executable_params->executable_format.size,
        executable_params->executable_format.data);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_hal_amdxdna_native_executable_create(
      executable_cache->native_device, executable_params,
      executable_cache->host_allocator, out_executable);
}

namespace {
const iree_hal_executable_cache_vtable_t
    iree_hal_amdxdna_nop_executable_cache_vtable = {
        .destroy = iree_hal_amdxdna_nop_executable_cache_destroy,
        .infer_format = iree_hal_amdxdna_nop_executable_cache_infer_format,
        .can_prepare_format =
            iree_hal_amdxdna_nop_executable_cache_can_prepare_format,
        .prepare_executable =
            iree_hal_amdxdna_nop_executable_cache_prepare_executable,
};
}  // namespace
