// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt-lite/device.h"

#include "iree-amd-aie/driver/xrt-lite/allocator.h"
#include "iree-amd-aie/driver/xrt-lite/api.h"
#include "iree-amd-aie/driver/xrt-lite/command_buffer.h"
#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/device.h"
#include "iree-amd-aie/driver/xrt-lite/util.h"
#include "nop_executable_cache.h"

struct iree_hal_xrt_lite_device {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;
  iree_allocator_t host_allocator;
  // not used
  iree_hal_allocator_t* device_allocator;
  std::shared_ptr<shim_xdna::device> shim_device;

  iree_status_t create_executable_cache(
      iree_string_view_t identifier, iree_loop_t loop,
      iree_hal_executable_cache_t** out_executable_cache) {
    return iree_hal_xrt_lite_nop_executable_cache_create(
        shim_device, identifier, host_allocator, out_executable_cache);
  }

  iree_status_t create_command_buffer(
      iree_hal_command_buffer_mode_t mode,
      iree_hal_command_category_t command_categories,
      iree_hal_queue_affinity_t queue_affinity,
      iree_host_size_t binding_capacity,
      iree_hal_command_buffer_t** out_command_buffer) {
    // TODO(null): pass any additional resources required to create the command
    // buffer. The implementation could pool command buffers here.
    return iree_hal_xrt_lite_command_buffer_create(
        device_allocator, mode, command_categories, queue_affinity,
        binding_capacity, host_allocator, out_command_buffer);
  }
};

namespace {
extern const iree_hal_device_vtable_t iree_hal_xrt_lite_device_vtable;
}

void iree_hal_xrt_lite_device_options_initialize(
    iree_hal_xrt_lite_device_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  // TODO(null): set defaults based on compiler configuration. Flags should not
  // be used as multiple devices may be configured within the process or the
  // hosting application may be authored in python/etc that does not use a flags
  // mechanism accessible here.
}

iree_status_t iree_hal_xrt_lite_device_create(
    iree_string_view_t identifier,
    const iree_hal_xrt_lite_device_options_t* options,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = nullptr;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_device* device = nullptr;
  iree_host_size_t total_size = sizeof(*device) + identifier.size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size,
                                reinterpret_cast<void**>(&device)));
  iree_hal_resource_initialize(&iree_hal_xrt_lite_device_vtable,
                               &device->resource);
  iree_string_view_append_to_buffer(
      identifier, &device->identifier,
      reinterpret_cast<char*>(device) + total_size - identifier.size);
  device->host_allocator = host_allocator;
  device->shim_device = std::make_shared<shim_xdna::device>();

  // TODO(null): pass device handles and pool configuration to the allocator.
  // Some implementations may share allocators across multiple devices created
  // from the same driver.
  iree_status_t status = iree_hal_xrt_lite_allocator_create(
      host_allocator, device->shim_device, &device->device_allocator);
  // TODO(max): device id
  *out_device = reinterpret_cast<iree_hal_device_t*>(device);
  if (iree_status_is_ok(status)) {
  } else {
    iree_hal_device_release(reinterpret_cast<iree_hal_device_t*>(device));
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_hal_xrt_lite_device* iree_hal_xrt_lite_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_xrt_lite_device_vtable);
  return reinterpret_cast<iree_hal_xrt_lite_device*>(base_value);
}

static iree_string_view_t iree_hal_xrt_lite_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_xrt_lite_device* device = iree_hal_xrt_lite_device_cast(base_device);
  return device->identifier;
}

static void iree_hal_xrt_lite_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_xrt_lite_device* device = iree_hal_xrt_lite_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(null): release all implementation resources here. It's expected that
  // this is only called once all outstanding resources created with this device
  // have been released by the application and no work is outstanding. If the
  // implementation performs internal async operations those should be shutdown
  // and joined first.

  iree_hal_allocator_release(device->device_allocator);
  device->shim_device.reset();
  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
};

static iree_allocator_t iree_hal_xrt_lite_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_xrt_lite_device* device = iree_hal_xrt_lite_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_xrt_lite_device_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_xrt_lite_device* device = iree_hal_xrt_lite_device_cast(base_device);
  return device->device_allocator;
}

#define DEVICE_MEMBER(member, return_t) \
  MEMBER_WRAPPER(iree_hal_device_t, iree_hal_xrt_lite_device, member, return_t)
#define DEVICE_MEMBER_STATUS(member) \
  MEMBER_WRAPPER_STATUS(iree_hal_device_t, iree_hal_xrt_lite_device, member)
#define DEVICE_MEMBER_VOID(member) \
  MEMBER_WRAPPER_VOID(iree_hal_device_t, iree_hal_xrt_lite_device, member)

DEVICE_MEMBER_STATUS(create_executable_cache);
DEVICE_MEMBER_STATUS(create_command_buffer);

namespace {
const iree_hal_device_vtable_t iree_hal_xrt_lite_device_vtable = {
    .destroy = iree_hal_xrt_lite_device_destroy,
    .id = iree_hal_xrt_lite_device_id,
    .host_allocator = iree_hal_xrt_lite_device_host_allocator,
    .device_allocator = iree_hal_xrt_lite_device_device_allocator,
    .create_executable_cache = iree_hal_xrt_lite_device_create_executable_cache,
    .create_command_buffer = iree_hal_xrt_lite_device_create_command_buffer};
}
