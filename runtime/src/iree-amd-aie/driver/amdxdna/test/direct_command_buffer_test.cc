// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/amdxdna/direct_command_buffer.h"

#include "iree-amd-aie/driver/amdxdna/api.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class DirectCommandBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_hal_amdxdna_device_options_initialize(&device_params_);
    device_ =
        new iree_hal_amdxdna_device(&device_params_, iree_allocator_system());
    iree_arena_block_pool_initialize(/*total_block_size=*/8 * 1024,
                                     iree_allocator_system(), &block_pool_);
  }

  void TearDown() override {
    iree_arena_block_pool_deinitialize(&block_pool_);
    iree_arena_block_pool_deinitialize(&device_->block_pool);
    delete device_;
  }

  iree_hal_amdxdna_device_params device_params_;
  iree_hal_amdxdna_device* device_ = nullptr;
  iree_arena_block_pool_t block_pool_;
};

TEST_F(DirectCommandBufferTest, CreateRejectsUnretainedMode) {
  iree_hal_command_buffer_t* command_buffer = nullptr;
  iree_status_t status = iree_hal_amdxdna_direct_command_buffer_create(
      device_,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED |
          IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*binding_capacity=*/0, &block_pool_,
      iree_allocator_system(), &command_buffer);

  EXPECT_EQ(iree_status_code(status), IREE_STATUS_UNIMPLEMENTED);
  iree_status_free(status);
  EXPECT_EQ(command_buffer, nullptr);
}

TEST_F(DirectCommandBufferTest, CreateAllowsRetainedMode) {
  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_direct_command_buffer_create(
      device_,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*binding_capacity=*/0, &block_pool_,
      iree_allocator_system(), &command_buffer));

  ASSERT_NE(command_buffer, nullptr);
  iree_hal_command_buffer_release(command_buffer);
}

}  // namespace
