// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "util.h"

// Maximum device path length we support. The path is always a 16 character hex
// string.
#define IREE_HAL_XRT_LITE_MAX_DEVICE_PATH_LENGTH 32
// Maximum device name length we support.
#define IREE_HAL_XRT_LITE_MAX_DEVICE_NAME_LENGTH 64

struct iree_hal_xrt_lite_driver_t {
  // Abstract resource used for injecting reference counting and vtable; must be
  // at offset 0.
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_string_view_t identifier;
  uint64_t device_hdl;
};

static void iree_hal_xrt_lite_driver_destroy(iree_hal_driver_t* base_driver) {
  iree_hal_xrt_lite_driver_t* driver =
      reinterpret_cast<iree_hal_xrt_lite_driver_t*>(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_xrt_lite_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
  iree_hal_xrt_lite_driver_t* driver =
      reinterpret_cast<iree_hal_xrt_lite_driver_t*>(base_driver);
  uint64_t device_hdl = driver->device_hdl;
  // Allocate the return infos and populate with the devices.
  iree_hal_device_info_t* device_infos = nullptr;
  iree_host_size_t single_info_size =
      sizeof(iree_hal_device_info_t) +
      (IREE_HAL_XRT_LITE_MAX_DEVICE_PATH_LENGTH +
       IREE_HAL_XRT_LITE_MAX_DEVICE_NAME_LENGTH) *
          sizeof(char);
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, single_info_size,
                                             (void**)&device_infos));

  uint8_t* buffer_ptr = (uint8_t*)device_infos + sizeof(iree_hal_device_info_t);
  memset(device_infos, 0, sizeof(*device_infos));

  //  device_infos->device_id = 0;
  //  std::string device_name = "aie2";
  //  const size_t name_len = strlen(device_name.c_str());
  //  if (name_len >= IREE_HAL_XRT_LITE_MAX_DEVICE_NAME_LENGTH) {
  //    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
  //                            "device name out of range");
  //  }
  //  buffer_ptr += iree_string_view_append_to_buffer(
  //      iree_make_string_view(device_name.c_str(), name_len),
  //      &device_infos->name, (char*)buffer_ptr);
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED);

  *out_device_info_count = 1;
  *out_device_infos = device_infos;
  return status;
}

static iree_status_t iree_hal_xrt_lite_driver_create_device_by_id(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_xrt_lite_driver_t* driver =
      reinterpret_cast<iree_hal_xrt_lite_driver_t*>(base_driver);
  iree_string_view_t device_name = iree_make_cstring_view("xrt-lite");

  //  iree_status_t status = iree_hal_xrt_lite_device_create(
  //      device_name, &driver->device_params, host_allocator, out_device);

  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_xrt_lite_driver_create_device_by_path(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_xrt_lite_driver_t* driver =
      reinterpret_cast<iree_hal_xrt_lite_driver_t*>(base_driver);
  iree_string_view_t device_name = iree_make_cstring_view("xrt");

  //  iree_status_t status = iree_hal_xrt_lite_device_create(
  //      device_name, &driver->device_params, host_allocator, out_device);

  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

namespace {
const iree_hal_driver_vtable_t iree_hal_xrt_lite_driver_vtable = {
    /*.destroy = */ iree_hal_xrt_lite_driver_destroy,
    /*.query_available_devices = */
    iree_hal_xrt_lite_driver_query_available_devices,
    /*.dump_device_info = */ unimplemented,
    /*.create_device_by_id = */ iree_hal_xrt_lite_driver_create_device_by_id,
    /*.create_device_by_path = */
    iree_hal_xrt_lite_driver_create_device_by_path,
};
}  // namespace

IREE_API_EXPORT iree_status_t iree_hal_xrt_lite_driver_create(
    iree_string_view_t identifier, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(out_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_driver_t* driver = nullptr;
  iree_host_size_t total_size = iree_sizeof_struct(*driver) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&driver));

  iree_hal_resource_initialize(&iree_hal_xrt_lite_driver_vtable,
                               &driver->resource);

  driver->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(
      identifier, &driver->identifier,
      (char*)driver + iree_sizeof_struct(*driver));

  *out_driver = reinterpret_cast<iree_hal_driver_t*>(driver);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
