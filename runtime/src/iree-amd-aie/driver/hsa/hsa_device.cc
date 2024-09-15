// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/hsa/hsa_device.h"

#include "iree-amd-aie/driver/hsa/api.h"
#include "iree-amd-aie/driver/hsa/hsa_allocator.h"
#include "iree-amd-aie/driver/hsa/hsa_headers.h"
#include "iree-amd-aie/driver/hsa/status_util.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/deferred_command_buffer.h"

namespace {
extern const iree_hal_device_vtable_t iree_hal_hsa_device_vtable;
}  // namespace

IREE_API_EXPORT void iree_hal_hsa_device_params_initialize(
    iree_hal_hsa_device_params_t* out_params) {
  memset(out_params, 0, sizeof(*out_params));
}

static iree_hal_hsa_device_t* iree_hal_hsa_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hsa_device_vtable);
  return (iree_hal_hsa_device_t*)base_value;
}

iree_status_t iree_hal_hsa_device_create(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_hsa_device_params_t* params,
    const iree_hal_hsa_dynamic_symbols_t* symbols, hsa::hsa_agent_t agent,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(driver);
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  size_t num_queue_packets = 64;
  hsa::hsa_queue_type_t queue_type = hsa::HSA_QUEUE_TYPE_SINGLE;
  void (*callback)(hsa::hsa_status_t, hsa::hsa_queue_t*, void*) = nullptr;
  void* data = nullptr;
  uint32_t private_segment_size = 0;
  uint32_t group_segment_size = 0;
  hsa::hsa_queue_t* dispatch_queue;

  IREE_HSA_RETURN_IF_ERROR(
      symbols,
      hsa_queue_create(agent, num_queue_packets, queue_type, callback, data,
                       private_segment_size, group_segment_size,
                       &dispatch_queue),
      "hsa_queue_create");
  IREE_ASSERT(dispatch_queue->base_address);

  iree_hal_hsa_device_t* device = nullptr;
  iree_host_size_t total_size = iree_sizeof_struct(*device) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&device));

  iree_hal_resource_initialize(&iree_hal_hsa_device_vtable, &device->resource);
  iree_string_view_append_to_buffer(
      identifier, &device->identifier,
      (char*)device + iree_sizeof_struct(*device));
  device->driver = driver;
  iree_hal_driver_retain(device->driver);
  device->hsa_symbols = symbols;
  device->params = *params;
  device->hsa_agent = agent;
  device->hsa_dispatch_queue = dispatch_queue;
  device->host_allocator = host_allocator;

  iree_status_t status = iree_hal_hsa_allocator_create(
      symbols, agent, host_allocator, &device->device_allocator);

  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)device);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_device_release(*out_device);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_string_view_t iree_hal_hsa_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return device->identifier;
}

static iree_hal_allocator_t* iree_hal_hsa_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return device->device_allocator;
}

static iree_allocator_t iree_hal_hsa_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return device->host_allocator;
}

static void iree_hal_hsa_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_allocator_release(device->device_allocator);
  iree_hal_driver_release(device->driver);
  iree_allocator_free(host_allocator, device);
  device->hsa_symbols->hsa_queue_destroy(device->hsa_dispatch_queue);

  IREE_TRACE_ZONE_END(z0);
}

namespace {
const iree_hal_device_vtable_t iree_hal_hsa_device_vtable = {
    /*destroy=*/iree_hal_hsa_device_destroy,
    /*id=*/iree_hal_hsa_device_id,
    /*host_allocator=*/iree_hal_hsa_device_host_allocator,
    /*device_allocator=*/iree_hal_hsa_device_allocator,
};
}  // namespace
