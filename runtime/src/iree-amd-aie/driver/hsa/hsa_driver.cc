// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree-amd-aie/driver/hsa/api.h"
#include "iree-amd-aie/driver/hsa/dynamic_symbols.h"
#include "iree-amd-aie/driver/hsa/hsa_device.h"
#include "iree-amd-aie/driver/hsa/hsa_headers.h"
#include "iree/base/api.h"
#include "iree/base/assert.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "status_util.h"

#define IREE_HAL_HSA_MAX_DEVICE_NAME_LENGTH 64
#define IREE_HAL_HSA_MAX_DEVICES 64
#define IREE_HAL_HSA_DEVICE_NOT_FOUND IREE_HAL_HSA_MAX_DEVICES

#define IREE_DEVICE_ID_TO_HSADEVICE(device_id) (int)((device_id) - 1)

struct iree_hal_hsa_driver_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource{};
  iree_allocator_t host_allocator{};
  iree_string_view_t identifier{};
  iree_hal_hsa_dynamic_symbols_t hsa_symbols{};
  iree_hal_hsa_device_params_t device_params;
  int default_device_index{};
  int num_aie_agents{};
  hsa::hsa_agent_t agents[IREE_HAL_HSA_MAX_DEVICES]{};
};

struct iree_hal_hsa_callback_package_t {
  iree_hal_hsa_driver_t* driver;
  size_t* index;
  void* return_value;
};

namespace {
extern const iree_hal_driver_vtable_t iree_hal_hsa_driver_vtable;
}

static iree_hal_hsa_driver_t* iree_hal_hsa_driver_cast(
    iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hsa_driver_vtable);
  return (iree_hal_hsa_driver_t*)base_value;
}

IREE_API_EXPORT void iree_hal_hsa_driver_options_initialize(
    iree_hal_hsa_driver_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  memset(out_options, 0, sizeof(*out_options));
}

hsa::hsa_status_t iterate_populate_aie_agents_callback(hsa::hsa_agent_t agent,
                                                       void* base_driver) {
  auto* package = (iree_hal_hsa_callback_package_t*)(base_driver);
  iree_hal_hsa_driver_t* driver = package->driver;
  size_t* index_ptr = package->index;
  auto* agents_ptr = (hsa::hsa_agent_t*)package->return_value;

  hsa::hsa_device_type_t type;
  hsa::hsa_status_t status =
      (&(driver->hsa_symbols))
          ->hsa_agent_get_info(agent, hsa::HSA_AGENT_INFO_DEVICE, &type);
  if (status != hsa::HSA_STATUS_SUCCESS) {
    return status;
  }

  if (type == hsa::HSA_DEVICE_TYPE_AIE) {
    size_t current_index = *index_ptr;
    agents_ptr[current_index] = agent;
    *index_ptr = current_index + 1;
    driver->num_aie_agents += 1;
  }
  return hsa::HSA_STATUS_SUCCESS;
}

iree_status_t iree_hal_hsa_init(iree_hal_hsa_driver_t* driver) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status =
      IREE_HSA_RESULT_TO_STATUS(&driver->hsa_symbols, hsa_init(), "hsa_init");

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_hsa_driver_create(
    iree_string_view_t identifier, const iree_hal_hsa_driver_options_t* options,
    const iree_hal_hsa_device_params_t* device_params,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(device_params);
  IREE_ASSERT_ARGUMENT(out_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hsa_driver_t* driver = nullptr;
  iree_host_size_t total_size = iree_sizeof_struct(*driver) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&driver));

  iree_hal_resource_initialize(&iree_hal_hsa_driver_vtable, &driver->resource);
  driver->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(
      identifier, &driver->identifier,
      (char*)driver + iree_sizeof_struct(*driver));
  driver->default_device_index = options->default_device_index;
  iree_status_t status = iree_hal_hsa_dynamic_symbols_initialize(
      host_allocator, &driver->hsa_symbols);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_hal_hsa_init(driver));

  std::memcpy(&driver->device_params, device_params,
              sizeof(driver->device_params));
  size_t agent_index = 0;
  iree_hal_hsa_callback_package_t symbols_and_agents = {
      .driver = driver, .index = &agent_index, .return_value = driver->agents};

  IREE_HSA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, &driver->hsa_symbols,
      hsa_iterate_agents(&iterate_populate_aie_agents_callback,
                         &symbols_and_agents),
      "hsa_iterate_agents");

  if (iree_status_is_ok(status)) {
    *out_driver = (iree_hal_driver_t*)driver;
  } else {
    iree_hal_driver_release((iree_hal_driver_t*)driver);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_hsa_driver_destroy(iree_hal_driver_t* base_driver) {
  IREE_ASSERT_ARGUMENT(base_driver);

  iree_hal_hsa_driver_t* driver = iree_hal_hsa_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(max): UNKNOWN; HSA driver error 'HSA_STATUS_ERROR_OUT_OF_RESOURCES'
  // (4104): HSA_STATUS_ERROR_OUT_OF_RESOURCES: The runtime failed to allocate
  // the necessary resources. This error may also occur when the core runtime
  // library needs to spawn threads or create internal OS-specific events.
  driver->hsa_symbols.hsa_shut_down();
  //  iree_hal_hsa_dynamic_symbols_deinitialize(&driver->hsa_symbols);

  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

static iree_hal_device_id_t iree_hsa_device_to_device_id(
    iree_hal_hsa_driver_t* driver, hsa::hsa_agent_t agent) {
  iree_hal_device_id_t device_id = 0;
  while (device_id != IREE_HAL_HSA_MAX_DEVICES &&
         driver->agents[device_id++].handle != agent.handle)
    ;

  return device_id;
}

static hsa::hsa_agent_t iree_device_id_to_hsadevice(
    iree_hal_hsa_driver_t* driver, iree_hal_device_id_t device_id) {
  return driver->agents[device_id];
}

static iree_status_t get_hsa_agent_uuid(iree_hal_hsa_dynamic_symbols_t* syms,
                                        hsa::hsa_agent_t agent,
                                        char* out_device_uuid) {
  // `HSA_AMD_AGENT_INFO_UUID` is part of the `hsa_amd_agent_info_t`
  // However, hsa_agent_get_info expects a hsa_agent_info_t.
  // TODO(max): this should be updated to use hsa_amd_agent_info_s
  auto uuid_info =
      static_cast<hsa::hsa_agent_info_t>(hsa::HSA_AMD_AGENT_INFO_UUID);
  IREE_HSA_RETURN_IF_ERROR(
      syms, hsa_agent_get_info(agent, uuid_info, out_device_uuid),
      "hsa_agent_get_info");

  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_populate_device_info(
    iree_hal_hsa_driver_t* driver, hsa::hsa_agent_t agent,
    iree_hal_hsa_dynamic_symbols_t* syms, uint8_t* buffer_ptr,
    uint8_t** out_buffer_ptr, iree_hal_device_info_t* out_device_info) {
  *out_buffer_ptr = buffer_ptr;

  char device_name[IREE_HAL_HSA_MAX_DEVICE_NAME_LENGTH];

  IREE_HSA_RETURN_IF_ERROR(
      syms, hsa_agent_get_info(agent, hsa::HSA_AGENT_INFO_NAME, device_name),
      "hsa_agent_get_info");
  memset(out_device_info, 0, sizeof(*out_device_info));

  out_device_info->device_id = iree_hsa_device_to_device_id(driver, agent);

  // Maximum UUID is 21
  char device_uuid[21] = {0};
  get_hsa_agent_uuid(syms, agent, device_uuid);

  char device_path_str[4 + 36 + 1] = {0};
  snprintf(device_path_str, sizeof(device_path_str),
           "%c%c%c-"
           "%02x%02x%02x%02x-"
           "%02x%02x-"
           "%02x%02x-"
           "%02x%02x-"
           "%02x%02x%02x%02x%02x%02x",
           device_uuid[0], device_uuid[1], device_uuid[2],
           (uint8_t)device_uuid[4], (uint8_t)device_uuid[5],
           (uint8_t)device_uuid[6], (uint8_t)device_uuid[7],
           (uint8_t)device_uuid[8], (uint8_t)device_uuid[9],
           (uint8_t)device_uuid[10], (uint8_t)device_uuid[11],
           (uint8_t)device_uuid[12], (uint8_t)device_uuid[13],
           (uint8_t)device_uuid[14], (uint8_t)device_uuid[15],
           (uint8_t)device_uuid[16], (uint8_t)device_uuid[17],
           (uint8_t)device_uuid[18], (uint8_t)device_uuid[19]);

  buffer_ptr += iree_string_view_append_to_buffer(
      iree_make_string_view(device_path_str,
                            IREE_ARRAYSIZE(device_path_str) - 1),
      &out_device_info->path, (char*)buffer_ptr);

  iree_string_view_t device_name_str =
      iree_make_string_view(device_name, strlen(device_name));
  buffer_ptr += iree_string_view_append_to_buffer(
      device_name_str, &out_device_info->name, (char*)buffer_ptr);

  *out_buffer_ptr = buffer_ptr;
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
  IREE_ASSERT_ARGUMENT(base_driver);
  IREE_ASSERT_ARGUMENT(out_device_info_count);
  IREE_ASSERT_ARGUMENT(out_device_infos);
  iree_hal_hsa_driver_t* driver = iree_hal_hsa_driver_cast(base_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  int device_count = driver->num_aie_agents;
  iree_hal_device_info_t* device_infos = nullptr;
  iree_host_size_t total_size =
      device_count * (sizeof(iree_hal_device_info_t) +
                      IREE_HAL_HSA_MAX_DEVICE_NAME_LENGTH * sizeof(char));
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&device_infos);
  hsa::hsa_agent_t* agents = driver->agents;
  int valid_device_count = 0;
  if (iree_status_is_ok(status)) {
    uint8_t* buffer_ptr =
        (uint8_t*)device_infos + device_count * sizeof(iree_hal_device_info_t);
    for (iree_host_size_t i = 0; i < device_count; ++i) {
      hsa::hsa_agent_t device = agents[i];

      status = iree_hal_hsa_populate_device_info(
          driver, device, &driver->hsa_symbols, buffer_ptr, &buffer_ptr,
          &device_infos[valid_device_count]);
      if (!iree_status_is_ok(status)) break;
      valid_device_count++;
    }
  }

  if (iree_status_is_ok(status)) {
    *out_device_info_count = valid_device_count;
    *out_device_infos = device_infos;
  } else {
    iree_allocator_free(host_allocator, device_infos);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hsa_driver_dump_device_info(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_string_builder_t* builder) {
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_driver_select_default_device(
    iree_hal_driver_t* base_driver, iree_hal_hsa_dynamic_symbols_t* syms,
    int default_device_index, iree_allocator_t host_allocator,
    hsa::hsa_agent_t* out_device) {
  iree_hal_device_info_t* device_infos = nullptr;
  iree_host_size_t device_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_hsa_driver_query_available_devices(
      base_driver, host_allocator, &device_count, &device_infos));

  iree_hal_hsa_driver_t* driver = iree_hal_hsa_driver_cast(base_driver);

  iree_status_t status = iree_ok_status();
  if (device_count == 0) {
    status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "no compatible HSA devices were found");
  } else if (default_device_index >= device_count) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "default device %d not found (of %" PRIhsz
                              " enumerated)",
                              default_device_index, device_count);
  } else {
    *out_device = iree_device_id_to_hsadevice(driver, default_device_index);
  }
  iree_allocator_free(host_allocator, device_infos);

  return status;
}

static iree_status_t iree_hal_hsa_driver_create_device_by_id(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(base_driver);
  IREE_ASSERT_ARGUMENT(out_device);

  iree_hal_hsa_driver_t* driver = iree_hal_hsa_driver_cast(base_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  hsa::hsa_agent_t agent;
  if (device_id == IREE_HAL_DEVICE_ID_DEFAULT) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_hsa_driver_select_default_device(
                base_driver, &driver->hsa_symbols, driver->default_device_index,
                host_allocator, &agent));
  } else {
    agent = iree_device_id_to_hsadevice(driver,
                                        IREE_DEVICE_ID_TO_HSADEVICE(device_id));
  }

  iree_string_view_t device_name = iree_make_cstring_view("amd-aie-hsa");
  iree_status_t status = iree_hal_hsa_device_create(
      base_driver, device_name, &driver->device_params, &driver->hsa_symbols,
      agent, host_allocator, out_device);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hsa_driver_create_device_by_path(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(base_driver);
  IREE_ASSERT_ARGUMENT(out_device);

  iree_string_view_t zeroth_device = iree_make_cstring_view("0");
  if (iree_string_view_is_empty(device_path) ||
      iree_string_view_equal(device_path, zeroth_device)) {
    return iree_hal_hsa_driver_create_device_by_id(
        base_driver, IREE_HAL_DEVICE_ID_DEFAULT, param_count, params,
        host_allocator, out_device);
  }

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unsupported device path");
}

namespace {
const iree_hal_driver_vtable_t iree_hal_hsa_driver_vtable = {
    .destroy = iree_hal_hsa_driver_destroy,
    .query_available_devices = iree_hal_hsa_driver_query_available_devices,
    .dump_device_info = iree_hal_hsa_driver_dump_device_info,
    .create_device_by_id = iree_hal_hsa_driver_create_device_by_id,
    .create_device_by_path = iree_hal_hsa_driver_create_device_by_path,
};
}

#undef IREE_HAL_HSA_MAX_DEVICE_NAME_LENGTH
#undef IREE_HAL_HSA_MAX_DEVICES
#undef IREE_HAL_HSA_DEVICE_NOT_FOUND
