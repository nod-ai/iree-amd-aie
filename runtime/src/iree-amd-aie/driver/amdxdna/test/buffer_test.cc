// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/amdxdna/buffer.h"

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct CountingAllocatorState {
  int alloc_count = 0;
  int free_count = 0;
};

static iree_status_t CountingAllocatorCtl(void* self,
                                          iree_allocator_command_t command,
                                          const void* params,
                                          void** inout_ptr) {
  auto* state = reinterpret_cast<CountingAllocatorState*>(self);
  switch (command) {
    case IREE_ALLOCATOR_COMMAND_MALLOC:
      ++state->alloc_count;
      return iree_allocator_malloc_uninitialized(
          iree_allocator_system(),
          reinterpret_cast<const iree_allocator_alloc_params_t*>(params)
              ->byte_length,
          inout_ptr);
    case IREE_ALLOCATOR_COMMAND_CALLOC:
      ++state->alloc_count;
      return iree_allocator_malloc(
          iree_allocator_system(),
          reinterpret_cast<const iree_allocator_alloc_params_t*>(params)
              ->byte_length,
          inout_ptr);
    case IREE_ALLOCATOR_COMMAND_REALLOC:
      return iree_allocator_realloc(
          iree_allocator_system(),
          reinterpret_cast<const iree_allocator_alloc_params_t*>(params)
              ->byte_length,
          inout_ptr);
    case IREE_ALLOCATOR_COMMAND_FREE:
      ++state->free_count;
      iree_allocator_free(iree_allocator_system(), *inout_ptr);
      *inout_ptr = nullptr;
      return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported allocator command");
}

TEST(BufferTest, WrapStoresHostAllocatorForDestroy) {
  CountingAllocatorState state;
  iree_allocator_t allocator = {
      .self = &state,
      .ctl = CountingAllocatorCtl,
  };

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_buffer_wrap(
      /*native_buffer=*/nullptr, iree_hal_buffer_placement_undefined(),
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_USAGE_DEFAULT, /*allocation_size=*/64,
      /*byte_offset=*/0, /*byte_length=*/64,
      iree_hal_buffer_release_callback_null(), allocator, &buffer));

  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(state.alloc_count, 1);
  EXPECT_EQ(state.free_count, 0);

  iree_hal_buffer_release(buffer);
  EXPECT_EQ(state.free_count, 1);
}

}  // namespace
