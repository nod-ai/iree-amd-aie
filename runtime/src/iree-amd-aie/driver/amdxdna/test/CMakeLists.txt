# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_cc_test(
  NAME
    allocator_test
  SRCS
    "allocator_test.cc"
  DEPS
    iree-amd-aie::driver::amdxdna::amdxdna
    iree::base
    iree::hal
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_test(
  NAME
    async_queue_test
  SRCS
    "async_queue_test.cc"
  DEPS
    iree-amd-aie::driver::amdxdna::amdxdna
    iree::async
    iree::base
    iree::base::internal::arena
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_test(
  NAME
    buffer_test
  SRCS
    "buffer_test.cc"
  DEPS
    iree-amd-aie::driver::amdxdna::amdxdna
    iree::base
    iree::hal
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_test(
  NAME
    direct_command_buffer_test
  SRCS
    "direct_command_buffer_test.cc"
  DEPS
    iree-amd-aie::driver::amdxdna::amdxdna
    iree::base
    iree::base::internal::arena
    iree::hal
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_test(
  NAME
    driver_test
  SRCS
    "driver_test.cc"
  DEPS
    iree-amd-aie::driver::amdxdna::amdxdna
    iree::async::util::proactor_pool
    iree::base
    iree::hal
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_test(
  NAME
    nop_executable_cache_test
  SRCS
    "nop_executable_cache_test.cc"
  DEPS
    iree-amd-aie::driver::amdxdna::amdxdna
    iree::base
    iree::hal
    iree::testing::gtest
    iree::testing::gtest_main
)
