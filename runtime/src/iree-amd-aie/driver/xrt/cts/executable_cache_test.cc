// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt/registration/driver_module.h"
#include "iree/base/api.h"
#include "iree/base/string_view.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "xrt_executables_c.h"

namespace iree::hal::cts {

static iree_status_t CreateXrtDevice(iree_hal_driver_t** out_driver,
                                     iree_hal_device_t** out_device) {
  iree_status_t status =
      iree_hal_xrt_driver_module_register(iree_hal_driver_registry_default());
  if (iree_status_is_already_exists(status)) {
    iree_status_ignore(status);
    status = iree_ok_status();
  }
  iree_hal_driver_t* driver = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(), iree_make_cstring_view("xrt"),
        iree_allocator_system(), &driver);
  }
  iree_hal_device_t* device = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_create_default_device(
        driver, iree_allocator_system(), &device);
  }
  if (iree_status_is_ok(status)) {
    *out_driver = driver;
    *out_device = device;
  } else {
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
  }
  return status;
}

static iree_const_byte_span_t GetTestExecutableData(
    iree_string_view_t file_name) {
  const struct iree_file_toc_t* toc =
      iree_cts_testdata_executables_aie_xrt_create();
  const auto& file = toc[0];
  return iree_make_const_byte_span(file.data, file.size);
}

class ExecutableCacheTest : public CtsTestBase<> {};

TEST_P(ExecutableCacheTest, Create) {
  iree_status_t loop_status = iree_ok_status();
  iree_hal_executable_cache_t* executable_cache = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_cache_create(
      device_, iree_make_cstring_view("default"),
      iree_loop_inline(&loop_status), &executable_cache));

  iree_hal_executable_cache_release(executable_cache);
  IREE_ASSERT_OK(loop_status);
}

TEST_P(ExecutableCacheTest, CantPrepareUnknownFormat) {
  iree_status_t loop_status = iree_ok_status();
  iree_hal_executable_cache_t* executable_cache = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_cache_create(
      device_, iree_make_cstring_view("default"),
      iree_loop_inline(&loop_status), &executable_cache));

  EXPECT_FALSE(iree_hal_executable_cache_can_prepare_format(
      executable_cache, /*caching_mode=*/0, iree_make_cstring_view("FOO?")));

  iree_hal_executable_cache_release(executable_cache);
  IREE_ASSERT_OK(loop_status);
}

TEST_P(ExecutableCacheTest, PrepareExecutable) {
  iree_status_t loop_status = iree_ok_status();
  iree_hal_executable_cache_t* executable_cache = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_cache_create(
      device_, iree_make_cstring_view("default"),
      iree_loop_inline(&loop_status), &executable_cache));

  iree_hal_executable_params_t executable_params;
  iree_hal_executable_params_initialize(&executable_params);
  executable_params.caching_mode =
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
  executable_params.executable_format =
      iree_make_cstring_view(executable_format());
  executable_params.executable_data =
      executable_data(iree_make_cstring_view("executable_cache_test.bin"));

  iree_hal_executable_t* executable = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_cache_prepare_executable(
      executable_cache, &executable_params, &executable));

  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(executable_cache);
  IREE_ASSERT_OK(loop_status);
}

INSTANTIATE_TEST_SUITE_P(XRT, ExecutableCacheTest,
                         ::testing::Values(BackendInfo{
                             "xrt",
                             CreateXrtDevice,
                             /*executable_format=*/"amdaie-xclbin-fb",
                             /*executable_data=*/GetTestExecutableData,
                             RecordingMode::kDirect,
                             /*unsupported_tests=*/{},
                         }),
                         [](const ::testing::TestParamInfo<BackendInfo>& info) {
                           return info.param.name;
                         });

}  // namespace iree::hal::cts
