# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_cc_test(
  NAME
    ChannelBdIdGeneratorTest
  SRCS
    "ChannelBdIdGeneratorTest.cpp"
  DEPS
    gtest
    iree-amd-aie::aie_runtime::Utils::Utils
)

iree_cc_test(
  NAME
    ChannelGeneratorTest
  SRCS
    "ChannelGeneratorTest.cpp"
  DEPS
    gtest
    iree-amd-aie::aie_runtime::Utils::Utils
)

iree_cc_test(
  NAME
    LockIdGeneratorTest
  SRCS
    "LockIdGeneratorTest.cpp"
  DEPS
    gtest
    iree-amd-aie::aie_runtime::Utils::Utils
)
