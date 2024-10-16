// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt-lite/api.h"
#include "util.h"

#define IREE_HAL_XRT_LITE_DEVICE_ID_DEFAULT 0

typedef struct iree_hal_xrt_lite_driver_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_string_view_t identifier;
  iree_hal_xrt_lite_driver_options_t options;
  // + trailing identifier string storage
} iree_hal_xrt_lite_driver_t;

namespace {
extern const iree_hal_driver_vtable_t iree_hal_xrt_lite_driver_vtable;
}

static iree_hal_xrt_lite_driver_t* iree_hal_xrt_lite_driver_cast(
    iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_xrt_lite_driver_vtable);
  return reinterpret_cast<iree_hal_xrt_lite_driver_t*>(base_value);
}

void iree_hal_xrt_lite_driver_options_initialize(
    iree_hal_xrt_lite_driver_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  iree_hal_xrt_lite_device_options_initialize(
      &out_options->default_device_options);
}

IREE_API_EXPORT iree_status_t iree_hal_xrt_lite_driver_create(
    iree_string_view_t identifier,
    const iree_hal_xrt_lite_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_driver);
  *out_driver = nullptr;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_driver_t* driver = nullptr;
  iree_host_size_t total_size = sizeof(*driver) + identifier.size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&driver));
  iree_hal_resource_initialize(&iree_hal_xrt_lite_driver_vtable,
                               &driver->resource);
  driver->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(
      identifier, &driver->identifier,
      (char*)driver + total_size - identifier.size);
  memcpy(&driver->options, options, sizeof(*options));
  *out_driver = (iree_hal_driver_t*)driver;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_xrt_lite_driver_destroy(iree_hal_driver_t* base_driver) {
  iree_hal_xrt_lite_driver_t* driver =
      iree_hal_xrt_lite_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_xrt_lite_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
  static const iree_hal_device_info_t device_infos[1] = {
      {
          .device_id = IREE_HAL_XRT_LITE_DEVICE_ID_DEFAULT,
          .name = iree_string_view_literal("default"),
      },
  };
  *out_device_info_count = IREE_ARRAYSIZE(device_infos);
  return iree_allocator_clone(
      host_allocator,
      iree_make_const_byte_span(device_infos, sizeof(device_infos)),
      (void**)out_device_infos);
}

static iree_status_t iree_hal_xrt_lite_driver_create_device_by_id(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_xrt_lite_driver_t* driver =
      iree_hal_xrt_lite_driver_cast(base_driver);
  iree_hal_xrt_lite_device_options_t options =
      driver->options.default_device_options;
  return iree_hal_xrt_lite_device_create(driver->identifier, &options,
                                         host_allocator, out_device);
}

static iree_status_t iree_hal_xrt_lite_driver_create_device_by_path(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  iree_hal_xrt_lite_driver_t* driver =
      iree_hal_xrt_lite_driver_cast(base_driver);
  iree_hal_xrt_lite_device_options_t options =
      driver->options.default_device_options;
  return iree_hal_xrt_lite_device_create(driver->identifier, &options,
                                         host_allocator, out_device);
}

namespace {
const iree_hal_driver_vtable_t iree_hal_xrt_lite_driver_vtable = {
    .destroy = iree_hal_xrt_lite_driver_destroy,
    .query_available_devices = iree_hal_xrt_lite_driver_query_available_devices,
    .create_device_by_id = iree_hal_xrt_lite_driver_create_device_by_id,
    .create_device_by_path = iree_hal_xrt_lite_driver_create_device_by_path,
};
}
