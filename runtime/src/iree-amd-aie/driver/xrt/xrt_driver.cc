// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt/xrt_device.h"
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

  // Parameters used to control device behavior.
  iree_hal_xrt_device_params_t device_params;

  xrt::device* device;

} iree_hal_xrt_driver_t;

namespace {
extern const iree_hal_driver_vtable_t iree_hal_xrt_driver_vtable;
}  // namespace

static iree_hal_xrt_driver_t* iree_hal_xrt_driver_cast(
    iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_xrt_driver_vtable);
  return (iree_hal_xrt_driver_t*)base_value;
}

static const iree_hal_xrt_driver_t* iree_hal_xrt_driver_const_cast(
    const iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_xrt_driver_vtable);
  return (const iree_hal_xrt_driver_t*)base_value;
}

static iree_status_t iree_hal_xrt_device_check_params(
    const iree_hal_xrt_device_params_t* params) {
  if (params->arena_block_size < 4096) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "arena block size too small (< 4096 bytes)");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_xrt_driver_create_internal(
    iree_string_view_t identifier,
    const iree_hal_xrt_device_params_t* device_params,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  iree_hal_xrt_driver_t* driver = NULL;
  iree_host_size_t total_size = iree_sizeof_struct(*driver) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&driver));

  iree_hal_resource_initialize(&iree_hal_xrt_driver_vtable, &driver->resource);
  driver->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(
      identifier, &driver->identifier,
      (char*)driver + iree_sizeof_struct(*driver));
  driver->device_params = *device_params;

  int device_count = xrt::system::enumerate_devices();
  if (IREE_UNLIKELY(device_count == 0)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "No XRT devices found");
  }
  // Get handle to xrt device
  std::cerr << xrt::system::enumerate_devices() << "\n";
  try {
    global_device = xrt::device(0);
  } catch (std::runtime_error& e) {
    return iree_make_status(IREE_STATUS_INTERNAL, "xrt::device(0) failed: %s",
                            e.what());
  }
  driver->device = &global_device;
  *out_driver = (iree_hal_driver_t*)driver;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_xrt_driver_create(
    iree_string_view_t identifier,
    const iree_hal_xrt_device_params_t* device_params,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(out_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_xrt_device_check_params(device_params));
  iree_status_t status = iree_hal_xrt_driver_create_internal(
      identifier, device_params, host_allocator, out_driver);

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
  iree_hal_xrt_driver_t* driver = iree_hal_xrt_driver_cast(base_driver);
  xrt::device* device = driver->device;
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "\n- Platform:"));

  std::string platform_info = device->get_info<xrt::info::device::platform>();
  const char* platform_info_str = platform_info.c_str();
  if (platform_info_str) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, " "));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, platform_info_str));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  return iree_ok_status();
}

// Populates device information from the given XRT physical device handle.
// |out_device_info| must point to valid memory and additional data will be
// appended to |buffer_ptr| and the new pointer is returned.
static iree_status_t iree_hal_xrt_populate_device_info(
    xrt::device* device, uint8_t* buffer_ptr, uint8_t** out_buffer_ptr,
    iree_hal_device_info_t* out_device_info) {
  *out_buffer_ptr = buffer_ptr;

  memset(out_device_info, 0, sizeof(*out_device_info));

  // We currenly only work with one XRT device and its device id is 0.
  out_device_info->device_id = 0;
  // TODO (nirvedhmeshram) : Add device path, initial attempt below to use the
  // info api for this gave an error.
  /*std::string device_path =
      device.get_info<xrt::info::device::interface_uuid>().to_string();
  const size_t path_len = strlen(device_path.c_str());
  buffer_ptr += iree_string_view_append_to_buffer(
      iree_make_string_view(device_path.c_str(), path_len),
      &out_device_info->path, (char*)buffer_ptr);*/
  std::string device_name = device->get_info<xrt::info::device::name>();
  const size_t name_len = strlen(device_name.c_str());
  if (name_len >= IREE_HAL_XRT_MAX_DEVICE_NAME_LENGTH) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "device name out of range");
  }
  buffer_ptr += iree_string_view_append_to_buffer(
      iree_make_string_view(device_name.c_str(), name_len),
      &out_device_info->name, (char*)buffer_ptr);

  *out_buffer_ptr = buffer_ptr;

  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
  iree_hal_xrt_driver_t* driver = iree_hal_xrt_driver_cast(base_driver);
  xrt::device* device = driver->device;
  // Allocate the return infos and populate with the devices.
  iree_hal_device_info_t* device_infos = NULL;
  iree_host_size_t single_info_size =
      sizeof(iree_hal_device_info_t) + (IREE_HAL_XRT_MAX_DEVICE_PATH_LENGTH +
                                        IREE_HAL_XRT_MAX_DEVICE_NAME_LENGTH) *
                                           sizeof(char);
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, single_info_size,
                                             (void**)&device_infos));

  // Append all path and name strings at the end of the struct.
  uint8_t* buffer_ptr = (uint8_t*)device_infos + sizeof(iree_hal_device_info_t);
  iree_status_t status = iree_hal_xrt_populate_device_info(
      device, buffer_ptr, &buffer_ptr, device_infos);
  if (iree_status_is_ok(status)) {
    // We currenly only work with one XRT device.
    *out_device_info_count = 1;
    *out_device_infos = device_infos;
  } else {
    iree_allocator_free(host_allocator, device_infos);
  }
  return status;
}

static iree_status_t iree_hal_xrt_driver_create_device_by_id(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_xrt_driver_t* driver = iree_hal_xrt_driver_cast(base_driver);
  iree_string_view_t device_name = iree_make_cstring_view("xrt");

  iree_status_t status =
      iree_hal_xrt_device_create(device_name, &driver->device_params,
                                 driver->device, host_allocator, out_device);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_xrt_driver_create_device_by_path(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_xrt_driver_t* driver = iree_hal_xrt_driver_cast(base_driver);
  iree_string_view_t device_name = iree_make_cstring_view("xrt");

  iree_status_t status =
      iree_hal_xrt_device_create(device_name, &driver->device_params,
                                 driver->device, host_allocator, out_device);

  IREE_TRACE_ZONE_END(z0);
  return status;
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
