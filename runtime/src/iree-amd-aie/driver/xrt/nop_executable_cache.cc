// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt/nop_executable_cache.h"

#include <stdbool.h>
#include <stddef.h>

#include "iree-amd-aie/driver/xrt/native_executable.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"

typedef struct iree_hal_xrt_nop_executable_cache_t {
  // Abstract resource used for injecting reference counting and vtable; must be
  // at offset 0.
  iree_hal_resource_t resource;

  xrt::device device;

  iree_allocator_t host_allocator;
} iree_hal_xrt_nop_executable_cache_t;

namespace {
extern const iree_hal_executable_cache_vtable_t
    iree_hal_xrt_nop_executable_cache_vtable;
}  // namespace

static iree_hal_xrt_nop_executable_cache_t*
iree_hal_xrt_nop_executable_cache_cast(
    iree_hal_executable_cache_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_xrt_nop_executable_cache_vtable);
  return (iree_hal_xrt_nop_executable_cache_t*)base_value;
}

iree_status_t iree_hal_xrt_nop_executable_cache_create(
    xrt::device device, iree_string_view_t identifier,
    iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache) {
  IREE_ASSERT_ARGUMENT(out_executable_cache);
  *out_executable_cache = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_nop_executable_cache_t* executable_cache = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*executable_cache),
                                (void**)&executable_cache));
  iree_hal_resource_initialize(&iree_hal_xrt_nop_executable_cache_vtable,
                               &executable_cache->resource);
  executable_cache->host_allocator = host_allocator;
  executable_cache->device = device;

  *out_executable_cache = (iree_hal_executable_cache_t*)executable_cache;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_xrt_nop_executable_cache_destroy(
    iree_hal_executable_cache_t* base_executable_cache) {
  iree_hal_xrt_nop_executable_cache_t* executable_cache =
      iree_hal_xrt_nop_executable_cache_cast(base_executable_cache);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(executable_cache->host_allocator, executable_cache);

  IREE_TRACE_ZONE_END(z0);
}

static bool iree_hal_xrt_nop_executable_cache_can_prepare_format(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  return iree_string_view_equal(executable_format,
                                iree_make_cstring_view("XRT"));
}

static iree_status_t iree_hal_xrt_nop_executable_cache_prepare_executable(
    iree_hal_executable_cache_t* base_executable_cache,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable) {
  iree_hal_xrt_nop_executable_cache_t* executable_cache =
      iree_hal_xrt_nop_executable_cache_cast(base_executable_cache);
  return iree_hal_xrt_native_executable_create(
      executable_cache->device, executable_params,
      executable_cache->host_allocator, out_executable);
}

namespace {
const iree_hal_executable_cache_vtable_t
    iree_hal_xrt_nop_executable_cache_vtable = {
        /*.destroy = */ iree_hal_xrt_nop_executable_cache_destroy,
        /*.can_prepare_format = */
        iree_hal_xrt_nop_executable_cache_can_prepare_format,
        /*.prepare_executable = */
        iree_hal_xrt_nop_executable_cache_prepare_executable,
};
}  // namespace
