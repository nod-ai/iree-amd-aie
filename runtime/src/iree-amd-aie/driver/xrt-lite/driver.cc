// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt-lite/api.h"
#include "iree-amd-aie/driver/xrt-lite/util.h"

#define IREE_HAL_XRT_LITE_DEVICE_ID_DEFAULT 0

struct iree_hal_xrt_lite_driver {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_xrt_lite_driver_options options;
  // + trailing identifier string storage
  iree_string_view_t identifier;
};

namespace {
extern const iree_hal_driver_vtable_t iree_hal_xrt_lite_driver_vtable;
}

void iree_hal_xrt_lite_driver_options_initialize(
    iree_hal_xrt_lite_driver_options* out_options) {
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_options, 0, sizeof(*out_options));
  iree_hal_xrt_lite_device_options_initialize(&out_options->device_params);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t iree_hal_xrt_lite_driver_create(
    iree_string_view_t identifier,
    const iree_hal_xrt_lite_driver_options* options,
    const iree_hal_xrt_lite_device_params* device_params,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_driver);
  *out_driver = nullptr;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_driver* driver = nullptr;
  iree_host_size_t total_size = sizeof(*driver) + identifier.size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size,
                                reinterpret_cast<void**>(&driver)));
  iree_hal_resource_initialize(&iree_hal_xrt_lite_driver_vtable,
                               &driver->resource);
  driver->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(
      identifier, &driver->identifier,
      reinterpret_cast<char*>(driver) + total_size - identifier.size);
  memcpy(&driver->options, options, sizeof(*options));
  memcpy(&driver->options.device_params, device_params, sizeof(*device_params));
  *out_driver = reinterpret_cast<iree_hal_driver_t*>(driver);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_xrt_lite_driver_destroy(iree_hal_driver_t* base_driver) {
  iree_hal_xrt_lite_driver* driver = IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
      base_driver, iree_hal_xrt_lite_driver_vtable, iree_hal_xrt_lite_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_xrt_lite_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
  IREE_TRACE_ZONE_BEGIN(z0);

  static const iree_hal_device_info_t device_infos[1] = {
      {
          .device_id = IREE_HAL_XRT_LITE_DEVICE_ID_DEFAULT,
          .name = iree_string_view_literal("default"),
      },
  };
  *out_device_info_count = IREE_ARRAYSIZE(device_infos);

  IREE_TRACE_ZONE_END(z0);
  return iree_allocator_clone(
      host_allocator,
      iree_make_const_byte_span(device_infos, sizeof(device_infos)),
      reinterpret_cast<void**>(out_device_infos));
}

static iree_status_t iree_hal_xrt_lite_driver_create_device_by_id(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_driver* driver = IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
      base_driver, iree_hal_xrt_lite_driver_vtable, iree_hal_xrt_lite_driver);
  iree_hal_xrt_lite_device_params options = driver->options.device_params;

  IREE_TRACE_ZONE_END(z0);
  return iree_hal_xrt_lite_device_create(driver->identifier, &options,
                                         host_allocator, out_device);
}

static iree_status_t iree_hal_xrt_lite_driver_create_device_by_path(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_driver* driver = IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
      base_driver, iree_hal_xrt_lite_driver_vtable, iree_hal_xrt_lite_driver);
  iree_hal_xrt_lite_device_params options = driver->options.device_params;

  IREE_TRACE_ZONE_END(z0);
  return iree_hal_xrt_lite_device_create(driver->identifier, &options,
                                         host_allocator, out_device);
}

namespace {
const iree_hal_driver_vtable_t iree_hal_xrt_lite_driver_vtable = {
    .destroy = iree_hal_xrt_lite_driver_destroy,
    .query_available_devices = iree_hal_xrt_lite_driver_query_available_devices,
    .dump_device_info = unimplemented_ok_status,
    .create_device_by_id = iree_hal_xrt_lite_driver_create_device_by_id,
    .create_device_by_path = iree_hal_xrt_lite_driver_create_device_by_path,
};
}
