// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>

#include "iree-amd-aie/driver/amdxdna/api.h"
#include "iree-amd-aie/driver/amdxdna/util.h"

#define IREE_HAL_AMDXDNA_DEVICE_ID_DEFAULT 0

struct iree_hal_amdxdna_driver {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_amdxdna_driver_options options;
  // + trailing identifier string storage
  iree_string_view_t identifier;
};

namespace {
extern const iree_hal_driver_vtable_t iree_hal_amdxdna_driver_vtable;
}

static bool iree_hal_amdxdna_power_mode_is_valid(iree_string_view_t value) {
  return iree_string_view_equal(value, IREE_SV("default")) ||
         iree_string_view_equal(value, IREE_SV("low")) ||
         iree_string_view_equal(value, IREE_SV("medium")) ||
         iree_string_view_equal(value, IREE_SV("high")) ||
         iree_string_view_equal(value, IREE_SV("turbo"));
}

static iree_status_t iree_hal_amdxdna_parse_non_negative_int32_option(
    iree_string_view_t key, iree_string_view_t value, int32_t* out_value) {
  if (!iree_string_view_atoi_int32(value, out_value)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Option '%.*s' expected to be int. Got: '%.*s'",
                            (int)key.size, key.data, (int)value.size,
                            value.data);
  }
  if (*out_value < 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Option '%.*s' expected to be >= 0. Got: '%.*s'",
                            (int)key.size, key.data, (int)value.size,
                            value.data);
  }
  return iree_ok_status();
}

void iree_hal_amdxdna_driver_options_initialize(
    iree_hal_amdxdna_driver_options* out_options) {
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_options, 0, sizeof(*out_options));
  iree_hal_amdxdna_device_options_initialize(&out_options->device_params);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amdxdna_device_options_parse(
    iree_hal_amdxdna_device_params* params, iree_host_size_t pairs_size,
    const iree_string_pair_t* pairs) {
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(!pairs_size || pairs);

  for (iree_host_size_t i = 0; i < pairs_size; ++i) {
    iree_string_view_t key = pairs[i].key;
    iree_string_view_t value = pairs[i].value;
    int32_t ivalue = 0;

    if (iree_string_view_equal(key, IREE_SV("amdxdna_n_core_rows"))) {
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_parse_non_negative_int32_option(
          key, value, &ivalue));
      params->n_core_rows = ivalue;
    } else if (iree_string_view_equal(key, IREE_SV("amdxdna_n_core_cols"))) {
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_parse_non_negative_int32_option(
          key, value, &ivalue));
      params->n_core_cols = ivalue;
    } else if (iree_string_view_equal(key, IREE_SV("amdxdna_cmd_chain"))) {
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_parse_non_negative_int32_option(
          key, value, &ivalue));
      params->cmd_chain = ivalue;
    } else if (iree_string_view_equal(key, IREE_SV("amdxdna_device_path"))) {
      params->device_path = value;
    } else if (iree_string_view_equal(key, IREE_SV("amdxdna_power_mode"))) {
      if (!iree_hal_amdxdna_power_mode_is_valid(value)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "Option 'amdxdna_power_mode' expected to be default | low | "
            "medium | high | turbo. Got: '%.*s'",
            (int)value.size, value.data);
      }
      params->power_mode = value;
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Unrecognized option: %.*s", (int)key.size,
                              key.data);
    }
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_amdxdna_driver_create(
    iree_string_view_t identifier,
    const iree_hal_amdxdna_driver_options* options,
    const iree_hal_amdxdna_device_params* device_params,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(device_params);
  IREE_ASSERT_ARGUMENT(out_driver);
  *out_driver = nullptr;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_driver* driver = nullptr;
  iree_host_size_t total_size = sizeof(*driver);
  if (IREE_UNLIKELY(
          !iree_host_size_checked_add(total_size, identifier.size,
                                      &total_size) ||
          !iree_host_size_checked_add(
              total_size, device_params->device_path.size, &total_size) ||
          !iree_host_size_checked_add(
              total_size, device_params->power_mode.size, &total_size))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "amdxdna driver option strings are too large");
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size,
                                reinterpret_cast<void**>(&driver)));
  iree_hal_resource_initialize(&iree_hal_amdxdna_driver_vtable,
                               &driver->resource);
  driver->host_allocator = host_allocator;
  char* string_storage = reinterpret_cast<char*>(driver) + sizeof(*driver);
  iree_string_view_append_to_buffer(identifier, &driver->identifier,
                                    string_storage);
  string_storage += identifier.size;
  memcpy(&driver->options, options, sizeof(*options));
  memcpy(&driver->options.device_params, device_params, sizeof(*device_params));
  string_storage += iree_string_view_append_to_buffer(
      device_params->device_path, &driver->options.device_params.device_path,
      string_storage);
  string_storage += iree_string_view_append_to_buffer(
      device_params->power_mode, &driver->options.device_params.power_mode,
      string_storage);
  *out_driver = reinterpret_cast<iree_hal_driver_t*>(driver);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_amdxdna_driver_destroy(iree_hal_driver_t* base_driver) {
  iree_hal_amdxdna_driver* driver = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_driver, iree_hal_amdxdna_driver_vtable, iree_hal_amdxdna_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_amdxdna_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
  IREE_TRACE_ZONE_BEGIN(z0);

  static const iree_hal_device_info_t device_infos[1] = {
      {
          .device_id = IREE_HAL_AMDXDNA_DEVICE_ID_DEFAULT,
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

static iree_status_t iree_hal_amdxdna_driver_dump_device_info(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_string_builder_t* builder) {
  iree_hal_amdxdna_driver* driver = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_driver, iree_hal_amdxdna_driver_vtable, iree_hal_amdxdna_driver);
  const iree_hal_amdxdna_device_params* params = &driver->options.device_params;
  const iree_string_view_t device_path =
      params->device_path.size ? params->device_path : IREE_SV("<auto>");
  const iree_string_view_t power_mode =
      params->power_mode.size ? params->power_mode : IREE_SV("<unchanged>");

  if (device_id != IREE_HAL_AMDXDNA_DEVICE_ID_DEFAULT) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "no amdxdna device with id %" PRIu64,
                            static_cast<uint64_t>(device_id));
  }

  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- Driver: %.*s\n", static_cast<int>(driver->identifier.size),
      driver->identifier.data));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "  device_path: %.*s\n", static_cast<int>(device_path.size),
      device_path.data));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "  core_grid: %dx%d\n", params->n_core_rows,
      params->n_core_cols));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "  power_mode: %.*s\n", static_cast<int>(power_mode.size),
      power_mode.data));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "  cmd_chain: %s\n", params->cmd_chain ? "true" : "false"));
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_driver_create_device_by_id(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, device_id);
  if (device_id != IREE_HAL_AMDXDNA_DEVICE_ID_DEFAULT) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "no amdxdna device with id %" PRIu64,
                            (uint64_t)device_id);
  }

  iree_hal_amdxdna_driver* driver = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_driver, iree_hal_amdxdna_driver_vtable, iree_hal_amdxdna_driver);
  iree_hal_amdxdna_device_params options = driver->options.device_params;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_device_options_parse(&options, param_count, params));

  IREE_TRACE_ZONE_END(z0);
  return iree_hal_amdxdna_device_create(
      driver->identifier, &options, create_params, host_allocator, out_device);
}

static iree_status_t iree_hal_amdxdna_driver_create_device_by_path(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)driver_name;

  iree_hal_amdxdna_driver* driver = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_driver, iree_hal_amdxdna_driver_vtable, iree_hal_amdxdna_driver);
  iree_hal_amdxdna_device_params options = driver->options.device_params;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_device_options_parse(&options, param_count, params));
  if (!iree_string_view_is_empty(device_path)) {
    options.device_path = device_path;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_hal_amdxdna_device_create(
      driver->identifier, &options, create_params, host_allocator, out_device);
}

namespace {
const iree_hal_driver_vtable_t iree_hal_amdxdna_driver_vtable = {
    .destroy = iree_hal_amdxdna_driver_destroy,
    .query_available_devices = iree_hal_amdxdna_driver_query_available_devices,
    .dump_device_info = iree_hal_amdxdna_driver_dump_device_info,
    .create_device_by_id = iree_hal_amdxdna_driver_create_device_by_id,
    .create_device_by_path = iree_hal_amdxdna_driver_create_device_by_path,
};
}
