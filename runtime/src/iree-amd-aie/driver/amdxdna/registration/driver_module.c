// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/amdxdna/registration/driver_module.h"

#include "iree-amd-aie/driver/amdxdna/api.h"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"

IREE_FLAG(int32_t, amdxdna_n_core_rows, 0,
          "Number of core rows to use on NPU. 0 discovers from hardware.");
IREE_FLAG(int32_t, amdxdna_n_core_cols, 0,
          "Number of core cols to use on NPU. 0 discovers from hardware.");
IREE_FLAG(string, amdxdna_device_path, "",
          "DRM accel device path to open (for example /dev/accel/accel0). "
          "Empty discovers the first /dev/accel/accel* node.");
IREE_FLAG(string, amdxdna_power_mode, "", "Set the power mode of the NPU.");
IREE_FLAG(int32_t, amdxdna_cmd_chain, 0,
          "Batch each dispatch's commands into a single ERT_CMD_CHAIN "
          "(removes the per-command host round-trip). 0 = off (default).");

static const iree_string_view_t key_amdxdna_n_core_rows =
    iree_string_view_literal("amdxdna_n_core_rows");
static const iree_string_view_t key_amdxdna_n_core_cols =
    iree_string_view_literal("amdxdna_n_core_cols");
static const iree_string_view_t key_amdxdna_device_path =
    iree_string_view_literal("amdxdna_device_path");
static const iree_string_view_t key_amdxdna_power_mode =
    iree_string_view_literal("amdxdna_power_mode");
static const iree_string_view_t key_amdxdna_cmd_chain =
    iree_string_view_literal("amdxdna_cmd_chain");

static iree_status_t iree_hal_amdxdna_driver_factory_enumerate(
    void* self, iree_host_size_t* out_driver_info_count,
    const iree_hal_driver_info_t** out_driver_infos) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)self;

  static const iree_hal_driver_info_t default_driver_info = {
      .driver_name = IREE_SVL("amdxdna"),
      .full_name = IREE_SVL("amdxdna driver (for AIE)"),
  };
  *out_driver_info_count = 1;
  *out_driver_infos = &default_driver_info;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_driver_parse_flags(
    iree_string_pair_builder_t* builder) {
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_string_pair_builder_add_int32(builder, key_amdxdna_n_core_rows,
                                             FLAG_amdxdna_n_core_rows));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_string_pair_builder_add_int32(builder, key_amdxdna_n_core_cols,
                                             FLAG_amdxdna_n_core_cols));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_string_pair_builder_add_int32(builder, key_amdxdna_cmd_chain,
                                             FLAG_amdxdna_cmd_chain));
  iree_string_view_t device_path = IREE_SV(FLAG_amdxdna_device_path);
  if (!iree_string_view_is_empty(device_path)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_string_pair_builder_add(
                builder,
                iree_make_string_pair(key_amdxdna_device_path, device_path)));
  }
  iree_string_view_t power_mode = IREE_SV(FLAG_amdxdna_power_mode);
  if (!iree_string_view_is_empty(power_mode)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_string_pair_builder_add(
                builder,
                iree_make_string_pair(key_amdxdna_power_mode, power_mode)));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_driver_populate_options(
    iree_allocator_t host_allocator,
    struct iree_hal_amdxdna_driver_options* driver_options,
    struct iree_hal_amdxdna_device_params* device_params,
    iree_host_size_t pairs_size, iree_string_pair_t* pairs) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)host_allocator;
  (void)driver_options;

  iree_status_t status =
      iree_hal_amdxdna_device_options_parse(device_params, pairs_size, pairs);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdxdna_driver_factory_try_create(
    void* self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!iree_string_view_equal(driver_name, IREE_SV("amdxdna"))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver '%.*s' is provided by this factory",
                            (int)driver_name.size, driver_name.data);
  }

  struct iree_hal_amdxdna_driver_options driver_options;
  iree_hal_amdxdna_driver_options_initialize(&driver_options);
  struct iree_hal_amdxdna_device_params device_params;
  iree_hal_amdxdna_device_options_initialize(&device_params);

  iree_string_pair_builder_t flag_option_builder;
  iree_string_pair_builder_initialize(host_allocator, &flag_option_builder);
  iree_status_t status =
      iree_hal_amdxdna_driver_parse_flags(&flag_option_builder);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_driver_populate_options(
        host_allocator, &driver_options, &device_params,
        iree_string_pair_builder_size(&flag_option_builder),
        iree_string_pair_builder_pairs(&flag_option_builder));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_driver_create(driver_name, &driver_options,
                                            &device_params, host_allocator,
                                            out_driver);
  }
  iree_string_pair_builder_deinitialize(&flag_option_builder);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_amdxdna_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_amdxdna_driver_factory_enumerate,
      .try_create = iree_hal_amdxdna_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
