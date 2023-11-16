// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt/api.h"
#include "iree/base/api.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"

// XRT includes
#include "experimental/xrt_system.h"
#include "xrt.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// Maximum device path length we support. The path is always a 16 character hex
// string.
#define IREE_HAL_XRT_MAX_DEVICE_PATH_LENGTH 32
// Maximum device name length we support.
#define IREE_HAL_XRT_MAX_DEVICE_NAME_LENGTH 64

// Utility macros to convert between xrt::device and iree_hal_device_id_t.
// #define XRT_DEVICE_TO_DEVICE_ID(device) (iree_hal_device_id_t)((void*)device)
// #define DEVICE_ID_TO_XRT_DEVICE(device_id) (xrt::device)(device_id)
// using namespace iree::hal::xrt;

typedef struct iree_hal_xrt_driver_t {
  // Abstract resource used for injecting reference counting and vtable; must be
  // at offset 0.
  iree_hal_resource_t resource;

  iree_allocator_t host_allocator;

  // Identifier used for the driver in the IREE driver registry..
  iree_string_view_t identifier;

  xrt::device device;

} iree_hal_xrt_driver_t;

namespace {
extern const iree_hal_driver_vtable_t iree_hal_xrt_driver_vtable;
}  // namespace

static iree_hal_xrt_driver_t* iree_hal_xrt_driver_cast(
    iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_xrt_driver_vtable);
  return (iree_hal_xrt_driver_t*)base_value;
}

iree_status_t iree_hal_xrt_driver_create_internal(
    iree_string_view_t identifier, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  iree_hal_xrt_driver_t* driver = NULL;
  iree_host_size_t total_size = iree_sizeof_struct(*driver) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&driver));

  iree_hal_resource_initialize(&iree_hal_xrt_driver_vtable, &driver->resource);
  driver->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(
      identifier, &driver->identifier,
      (char*)driver + iree_sizeof_struct(*driver));

  int device_count = xrt::system::enumerate_devices();
  if (IREE_UNLIKELY(device_count == 0)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "No XRT devices found");
  }
  // Get handle to xrt device
  driver->device = xrt::device(0);
  *out_driver = (iree_hal_driver_t*)driver;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_xrt_driver_create(
    iree_string_view_t identifier, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(out_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_xrt_driver_create_internal(
      identifier, host_allocator, out_driver);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_xrt_driver_destroy(iree_hal_driver_t* base_driver) {
  iree_hal_xrt_driver_t* driver = iree_hal_xrt_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
  return;
}
static iree_status_t iree_hal_xrt_driver_dump_device_info(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_string_builder_t* builder) {
  return iree_status_from_code(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_xrt_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
  return iree_status_from_code(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_xrt_driver_create_device_by_id(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  return iree_status_from_code(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_xrt_driver_create_device_by_path(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  return iree_status_from_code(IREE_STATUS_UNIMPLEMENTED);
}

namespace {
const iree_hal_driver_vtable_t iree_hal_xrt_driver_vtable = {
    /*.destroy = */ iree_hal_xrt_driver_destroy,
    /*.query_available_devices = */ iree_hal_xrt_driver_query_available_devices,
    /*.dump_device_info = */ iree_hal_xrt_driver_dump_device_info,
    /*.create_device_by_id = */ iree_hal_xrt_driver_create_device_by_id,
    /*.create_device_by_path = */ iree_hal_xrt_driver_create_device_by_path,
};
}  // namespace
