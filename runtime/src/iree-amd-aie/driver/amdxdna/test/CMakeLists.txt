# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
