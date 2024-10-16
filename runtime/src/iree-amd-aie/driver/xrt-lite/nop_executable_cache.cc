// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt-lite/nop_executable_cache.h"

#include <cstddef>

#include "iree-amd-aie/driver/xrt-lite/executable.h"
#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/device.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"

namespace {
extern const iree_hal_executable_cache_vtable_t
    iree_hal_xrt_lite_nop_executable_cache_vtable;
}  // namespace

struct iree_hal_xrt_lite_nop_executable_cache {
  // Abstract resource used for injecting reference counting and vtable; must be
  // at offset 0.
  iree_hal_resource_t resource;
  shim_xdna::device* shim_device;
  iree_allocator_t host_allocator;

  iree_hal_xrt_lite_nop_executable_cache(shim_xdna::device* shim_device,
                                         iree_allocator_t host_allocator)
      : shim_device(shim_device), host_allocator(host_allocator) {
    IREE_TRACE_ZONE_BEGIN(z0);

    iree_hal_resource_initialize(&iree_hal_xrt_lite_nop_executable_cache_vtable,
                                 &resource);

    IREE_TRACE_ZONE_END(z0);
  }
};

iree_status_t iree_hal_xrt_lite_nop_executable_cache_create(
    shim_xdna::device* shim_device, iree_string_view_t identifier,
    iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache) {
  IREE_ASSERT_ARGUMENT(out_executable_cache);
  *out_executable_cache = nullptr;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_nop_executable_cache* executable_cache = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*executable_cache),
                                (void**)&executable_cache));
  executable_cache = new (executable_cache)
      iree_hal_xrt_lite_nop_executable_cache(shim_device, host_allocator);
  *out_executable_cache =
      reinterpret_cast<iree_hal_executable_cache_t*>(executable_cache);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_xrt_lite_nop_executable_cache_destroy(
    iree_hal_executable_cache_t* base_executable_cache) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_nop_executable_cache* executable_cache =
      reinterpret_cast<iree_hal_xrt_lite_nop_executable_cache*>(
          base_executable_cache);
  iree_allocator_free(executable_cache->host_allocator, executable_cache);

  IREE_TRACE_ZONE_END(z0);
}

static bool iree_hal_xrt_lite_nop_executable_cache_can_prepare_format(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  return iree_string_view_equal(executable_format,
                                iree_make_cstring_view("PDIR"));
}

static iree_status_t iree_hal_xrt_lite_nop_executable_cache_prepare_executable(
    iree_hal_executable_cache_t* base_executable_cache,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_nop_executable_cache* executable_cache =
      reinterpret_cast<iree_hal_xrt_lite_nop_executable_cache*>(
          base_executable_cache);

  IREE_TRACE_ZONE_END(z0);
  return iree_hal_xrt_lite_native_executable_create(
      executable_cache->shim_device, executable_params,
      executable_cache->host_allocator, out_executable);
}

namespace {
const iree_hal_executable_cache_vtable_t
    iree_hal_xrt_lite_nop_executable_cache_vtable = {
        .destroy = iree_hal_xrt_lite_nop_executable_cache_destroy,
        .can_prepare_format =
            iree_hal_xrt_lite_nop_executable_cache_can_prepare_format,
        .prepare_executable =
            iree_hal_xrt_lite_nop_executable_cache_prepare_executable,
};
}  // namespace
