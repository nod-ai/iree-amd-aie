// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/amdxdna/api.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(DriverTest, DeviceOptionsParseOverridesDefaults) {
  iree_hal_amdxdna_device_params params;
  iree_hal_amdxdna_device_options_initialize(&params);

  iree_string_pair_t pairs[] = {
      iree_make_string_pair(IREE_SV("amdxdna_n_core_rows"), IREE_SV("4")),
      iree_make_string_pair(IREE_SV("amdxdna_n_core_cols"), IREE_SV("5")),
      iree_make_string_pair(IREE_SV("amdxdna_device_path"),
                            IREE_SV("/dev/accel/accel2")),
      iree_make_string_pair(IREE_SV("amdxdna_power_mode"), IREE_SV("turbo")),
      iree_make_string_pair(IREE_SV("amdxdna_cmd_chain"), IREE_SV("1")),
  };
  IREE_ASSERT_OK(iree_hal_amdxdna_device_options_parse(
      &params, IREE_ARRAYSIZE(pairs), pairs));

  EXPECT_EQ(params.n_core_rows, 4);
  EXPECT_EQ(params.n_core_cols, 5);
  EXPECT_TRUE(
      iree_string_view_equal(params.device_path, IREE_SV("/dev/accel/accel2")));
  EXPECT_TRUE(iree_string_view_equal(params.power_mode, IREE_SV("turbo")));
  EXPECT_EQ(params.cmd_chain, 1);
}

TEST(DriverTest, DeviceOptionsParseRejectsInvalidValues) {
  iree_hal_amdxdna_device_params params;
  iree_hal_amdxdna_device_options_initialize(&params);

  iree_string_pair_t negative_rows[] = {
      iree_make_string_pair(IREE_SV("amdxdna_n_core_rows"), IREE_SV("-1")),
  };
  iree_status_t status = iree_hal_amdxdna_device_options_parse(
      &params, IREE_ARRAYSIZE(negative_rows), negative_rows);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_FAILED_PRECONDITION);
  iree_status_free(status);

  iree_string_pair_t bad_power_mode[] = {
      iree_make_string_pair(IREE_SV("amdxdna_power_mode"), IREE_SV("warp")),
  };
  status = iree_hal_amdxdna_device_options_parse(
      &params, IREE_ARRAYSIZE(bad_power_mode), bad_power_mode);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_FAILED_PRECONDITION);
  iree_status_free(status);

  iree_string_pair_t unknown_key[] = {
      iree_make_string_pair(IREE_SV("amdxdna_missing"), IREE_SV("1")),
  };
  status = iree_hal_amdxdna_device_options_parse(
      &params, IREE_ARRAYSIZE(unknown_key), unknown_key);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_free(status);
}

TEST(DriverTest, CreateDeviceByIdRejectsUnknownIdBeforeOpeningDevice) {
  iree_hal_amdxdna_driver_options driver_options;
  iree_hal_amdxdna_driver_options_initialize(&driver_options);
  iree_hal_amdxdna_device_params device_params;
  iree_hal_amdxdna_device_options_initialize(&device_params);

  iree_hal_driver_t* driver = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_driver_create(
      IREE_SV("amdxdna"), &driver_options, &device_params,
      iree_allocator_system(), &driver));

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_driver_create_device_by_id(
      driver, /*device_id=*/42, /*param_count=*/0, /*params=*/nullptr,
      &create_params, iree_allocator_system(), &device);

  EXPECT_EQ(iree_status_code(status), IREE_STATUS_NOT_FOUND);
  iree_status_free(status);
  EXPECT_EQ(device, nullptr);

  iree_hal_driver_release(driver);
}

TEST(DriverTest, CreateDeviceByPathAcceptsAdvertisedDeviceName) {
  iree_hal_amdxdna_driver_options driver_options;
  iree_hal_amdxdna_driver_options_initialize(&driver_options);
  iree_hal_amdxdna_device_params device_params;
  iree_hal_amdxdna_device_options_initialize(&device_params);

  iree_hal_driver_t* driver = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_driver_create(
      IREE_SV("amdxdna"), &driver_options, &device_params,
      iree_allocator_system(), &driver));

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  iree_string_pair_t bad_rows[] = {
      iree_make_string_pair(IREE_SV("amdxdna_n_core_rows"), IREE_SV("-1")),
  };
  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_driver_create_device_by_path(
      driver, IREE_SV("default"), iree_string_view_empty(),
      IREE_ARRAYSIZE(bad_rows), bad_rows, &create_params,
      iree_allocator_system(), &device);

  EXPECT_EQ(iree_status_code(status), IREE_STATUS_FAILED_PRECONDITION);
  iree_status_free(status);
  EXPECT_EQ(device, nullptr);

  iree_hal_driver_release(driver);
}

TEST(DriverTest, DeviceCreateRejectsInvalidRawOptionsBeforeOpeningDevice) {
  uint32_t node_id = 0;
  iree_async_proactor_pool_t* proactor_pool = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      /*node_count=*/1, &node_id, iree_async_proactor_pool_options_default(),
      iree_allocator_system(), &proactor_pool));

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;

  iree_hal_amdxdna_device_params device_params;
  iree_hal_amdxdna_device_options_initialize(&device_params);
  device_params.n_core_rows = -1;

  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_amdxdna_device_create(
      IREE_SV("amdxdna"), &device_params, &create_params,
      iree_allocator_system(), &device);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_FAILED_PRECONDITION);
  iree_status_free(status);
  EXPECT_EQ(device, nullptr);

  iree_hal_amdxdna_device_options_initialize(&device_params);
  device_params.power_mode = IREE_SV("warp");
  status = iree_hal_amdxdna_device_create(IREE_SV("amdxdna"), &device_params,
                                          &create_params,
                                          iree_allocator_system(), &device);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_FAILED_PRECONDITION);
  iree_status_free(status);
  EXPECT_EQ(device, nullptr);

  iree_async_proactor_pool_release(proactor_pool);
}

TEST(DriverTest, CreateDeviceByPathReturnsStatusForMissingDevice) {
  iree_hal_amdxdna_driver_options driver_options;
  iree_hal_amdxdna_driver_options_initialize(&driver_options);
  iree_hal_amdxdna_device_params device_params;
  iree_hal_amdxdna_device_options_initialize(&device_params);

  iree_hal_driver_t* driver = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_driver_create(
      IREE_SV("amdxdna"), &driver_options, &device_params,
      iree_allocator_system(), &driver));

  uint32_t node_id = 0;
  iree_async_proactor_pool_t* proactor_pool = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      /*node_count=*/1, &node_id, iree_async_proactor_pool_options_default(),
      iree_allocator_system(), &proactor_pool));

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;
  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_driver_create_device_by_path(
      driver, IREE_SV("amdxdna"),
      IREE_SV("/dev/accel/iree-amdxdna-test-missing"), /*param_count=*/0,
      /*params=*/nullptr, &create_params, iree_allocator_system(), &device);

  EXPECT_EQ(iree_status_code(status), IREE_STATUS_NOT_FOUND);
  iree_status_free(status);
  EXPECT_EQ(device, nullptr);

  iree_async_proactor_pool_release(proactor_pool);
  iree_hal_driver_release(driver);
}

}  // namespace
