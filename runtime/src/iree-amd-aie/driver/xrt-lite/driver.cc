// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt-lite/api.h"
#include "util.h"

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
  return (iree_hal_xrt_lite_driver_t*)base_value;
}

void iree_hal_xrt_lite_driver_options_initialize(
    iree_hal_xrt_lite_driver_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));

  // TODO(null): set defaults based on compiler configuration. Flags should not
  // be used as multiple devices may be configured within the process or the
  // hosting application may be authored in python/etc that does not use a flags
  // mechanism accessible here.
  iree_hal_xrt_lite_device_options_initialize(
      &out_options->default_device_options);
}

static iree_status_t iree_hal_xrt_lite_driver_options_verify(
    const iree_hal_xrt_lite_driver_options_t* options) {
  // TODO(null): verify that the parameters are within expected ranges and any
  // requested features are supported.
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_xrt_lite_driver_create(
    iree_string_view_t identifier,
    const iree_hal_xrt_lite_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_driver);
  *out_driver = nullptr;
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(null): verify options; this may be moved after any libraries are
  // loaded so the verification can use underlying implementation queries.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_xrt_lite_driver_options_verify(options));

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

  // TODO(null): if there are any string fields then they will need to be
  // retained as well (similar to the identifier they can be tagged on to the
  // end of the driver struct).
  memcpy(&driver->options, options, sizeof(*options));

  // TODO(null): load libraries and query driver support from the system.
  // Devices need not be enumerated here if doing so is expensive; the
  // application may create drivers just to see if they are present but defer
  // device enumeration until the user requests one. Underlying implementations
  // can sometimes do bonkers static init stuff as soon as they are touched and
  // this code may want to do that on-demand instead.
  iree_status_t status = iree_ok_status();

  if (iree_status_is_ok(status)) {
    *out_driver = (iree_hal_driver_t*)driver;
  } else {
    iree_hal_driver_release((iree_hal_driver_t*)driver);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_xrt_lite_driver_destroy(iree_hal_driver_t* base_driver) {
  iree_hal_xrt_lite_driver_t* driver =
      iree_hal_xrt_lite_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(null): if the driver loaded any libraries they should be closed here.

  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

namespace {
const iree_hal_driver_vtable_t iree_hal_xrt_lite_driver_vtable = {
    .destroy = iree_hal_xrt_lite_driver_destroy,
    .query_available_devices = unimplemented,
};
}
