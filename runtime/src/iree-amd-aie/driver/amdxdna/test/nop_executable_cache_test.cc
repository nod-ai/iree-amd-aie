// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/amdxdna/nop_executable_cache.h"

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(NopExecutableCacheTest, CanPreparePublicAmdxdnaFormat) {
  iree_hal_executable_cache_t* executable_cache = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_nop_executable_cache_create(
      /*native_device=*/nullptr, iree_make_cstring_view("default"),
      iree_allocator_system(), &executable_cache));

  EXPECT_TRUE(iree_hal_executable_cache_can_prepare_format(
      executable_cache, /*caching_mode=*/0,
      iree_make_cstring_view("amdaie-pdi-fb")));
  EXPECT_FALSE(iree_hal_executable_cache_can_prepare_format(
      executable_cache, /*caching_mode=*/0, iree_make_cstring_view("PDIR")));
  EXPECT_FALSE(iree_hal_executable_cache_can_prepare_format(
      executable_cache, /*caching_mode=*/0, iree_make_cstring_view("FOO?")));

  iree_hal_executable_cache_release(executable_cache);
}

TEST(NopExecutableCacheTest, PrepareRejectsUnknownFormatBeforeParsing) {
  iree_hal_executable_cache_t* executable_cache = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_nop_executable_cache_create(
      /*native_device=*/nullptr, iree_make_cstring_view("default"),
      iree_allocator_system(), &executable_cache));

  iree_hal_executable_params_t executable_params;
  iree_hal_executable_params_initialize(&executable_params);
  executable_params.executable_format = iree_make_cstring_view("PDIR");
  executable_params.executable_data = iree_const_byte_span_empty();

  iree_hal_executable_t* executable = nullptr;
  iree_status_t status = iree_hal_executable_cache_prepare_executable(
      executable_cache, &executable_params, &executable);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_NOT_FOUND);
  iree_status_free(status);
  EXPECT_EQ(executable, nullptr);

  iree_hal_executable_cache_release(executable_cache);
}

}  // namespace
