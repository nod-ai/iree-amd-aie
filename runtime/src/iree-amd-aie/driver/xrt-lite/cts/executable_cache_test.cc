// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt-lite/registration/driver_module.h"
#include "iree/base/api.h"
#include "iree/base/string_view.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "xrt_lite_executables_c.h"

namespace iree::hal::cts {

const char* get_test_driver_name() { return "xrt-lite"; }

iree_status_t register_test_driver(iree_hal_driver_registry_t* registry) {
  return iree_hal_xrt_lite_driver_module_register(registry);
}

const char* get_test_executable_format() { return "amdaie-pdi-fb"; }

iree_const_byte_span_t get_test_executable_data(iree_string_view_t file_name) {
  const struct iree_file_toc_t* toc =
      iree_cts_testdata_executables_aie_xrt_lite_create();
  const auto& file = toc[0];
  return iree_make_const_byte_span(file.data, file.size);
}

class ExecutableCacheTest : public CTSTestBase<> {};

TEST_F(ExecutableCacheTest, Create) {
  iree_status_t loop_status = iree_ok_status();
  iree_hal_executable_cache_t* executable_cache = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_cache_create(
      device_, iree_make_cstring_view("default"),
      iree_loop_inline(&loop_status), &executable_cache));

  iree_hal_executable_cache_release(executable_cache);
  IREE_ASSERT_OK(loop_status);
}

TEST_F(ExecutableCacheTest, CantPrepareUnknownFormat) {
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

TEST_F(ExecutableCacheTest, PrepareExecutable) {
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
      iree_make_cstring_view(get_test_executable_format());
  executable_params.executable_data = get_test_executable_data(
      iree_make_cstring_view("executable_cache_test.bin"));

  iree_hal_executable_t* executable = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_cache_prepare_executable(
      executable_cache, &executable_params, &executable));

  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(executable_cache);
  IREE_ASSERT_OK(loop_status);
}

}  // namespace iree::hal::cts
