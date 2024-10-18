// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt-lite/registration/driver_module.h"

#include "iree-amd-aie/driver/xrt-lite/api.h"
#include "iree/base/api.h"
#include "iree/base/internal/flags.h"

IREE_FLAG(int32_t, xrt_lite_n_core_rows, 0,
          "Number of core rows to use on NPU.");
IREE_FLAG(int32_t, xrt_lite_n_core_cols, 0,
          "Number of core cols to use on NPU.");

static const iree_string_view_t key_xrt_lite_n_core_rows =
    iree_string_view_literal("xrt_lite_n_core_rows");
static const iree_string_view_t key_xrt_lite_n_core_cols =
    iree_string_view_literal("xrt_lite_n_core_cols");

static iree_status_t iree_hal_xrt_lite_driver_factory_enumerate(
    void* self, iree_host_size_t* out_driver_info_count,
    const iree_hal_driver_info_t** out_driver_infos) {
  IREE_TRACE_ZONE_BEGIN(z0);

  static const iree_hal_driver_info_t default_driver_info = {
      .driver_name = IREE_SVL("xrt-lite"),
      .full_name = IREE_SVL("XRT-LITE driver (for AIE)"),
  };
  *out_driver_info_count = 1;
  *out_driver_infos = &default_driver_info;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_lite_driver_parse_flags(
    iree_string_pair_builder_t* builder) {
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_string_pair_builder_add_int32(builder, key_xrt_lite_n_core_rows,
                                             FLAG_xrt_lite_n_core_rows));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_string_pair_builder_add_int32(builder, key_xrt_lite_n_core_cols,
                                             FLAG_xrt_lite_n_core_cols));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_lite_driver_populate_options(
    iree_allocator_t host_allocator,
    struct iree_hal_xrt_lite_driver_options* driver_options,
    struct iree_hal_xrt_lite_device_params* device_params,
    iree_host_size_t pairs_size, iree_string_pair_t* pairs) {
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < pairs_size; ++i) {
    iree_string_view_t key = pairs[i].key;
    iree_string_view_t value = pairs[i].value;
    int32_t ivalue;

    if (iree_string_view_equal(key, key_xrt_lite_n_core_rows)) {
      if (!iree_string_view_atoi_int32(value, &ivalue)) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "Option 'key_xrt_lite_n_core_rows' expected to be int. Got: '%.*s'",
            (int)value.size, value.data);
      }
      if (ivalue <= 0) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "Option 'key_xrt_lite_n_core_rows' expected to be > 0. Got: '%.*s'",
            (int)value.size, value.data);
      }
      device_params->n_core_rows = ivalue;
    } else if (iree_string_view_equal(key, key_xrt_lite_n_core_cols)) {
      if (!iree_string_view_atoi_int32(value, &ivalue)) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "Option 'key_xrt_lite_n_core_cols' expected to be int. Got: '%.*s'",
            (int)value.size, value.data);
      }
      if (ivalue <= 0) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "Option 'key_xrt_lite_n_core_cols' expected to be > 0. Got: '%.*s'",
            (int)value.size, value.data);
      }
      device_params->n_core_cols = ivalue;
    } else {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "Unrecognized options: %.*s", (int)key.size,
                              key.data);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_lite_driver_factory_try_create(
    void* self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!iree_string_view_equal(driver_name, IREE_SV("xrt-lite"))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver '%.*s' is provided by this factory",
                            (int)driver_name.size, driver_name.data);
  }

  struct iree_hal_xrt_lite_driver_options driver_options;
  iree_hal_xrt_lite_driver_options_initialize(&driver_options);
  struct iree_hal_xrt_lite_device_params device_params;
  iree_hal_xrt_lite_device_options_initialize(&device_params);

  iree_string_pair_builder_t flag_option_builder;
  iree_string_pair_builder_initialize(host_allocator, &flag_option_builder);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_xrt_lite_driver_parse_flags(&flag_option_builder));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_xrt_lite_driver_populate_options(
              host_allocator, &driver_options, &device_params,
              iree_string_pair_builder_size(&flag_option_builder),
              iree_string_pair_builder_pairs(&flag_option_builder)));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_xrt_lite_driver_create(driver_name, &driver_options,
                                          &device_params, host_allocator,
                                          out_driver));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_hal_xrt_lite_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_xrt_lite_driver_factory_enumerate,
      .try_create = iree_hal_xrt_lite_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
