// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/amdxdna/allocator.h"

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(AllocatorTest, CreateAndRelease) {
  iree_hal_allocator_t* allocator = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_allocator_create(
      iree_allocator_system(), /*native_device=*/nullptr, &allocator));
  ASSERT_NE(allocator, nullptr);
  iree_hal_allocator_release(allocator);
}

}  // namespace
