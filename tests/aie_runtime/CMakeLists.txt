# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cc_binary(
  NAME
    aie_runtime_utest
  SRCS
    "utest.cxx"
  DEPS
    iree-amd-aie::runtime::iree_aie_runtime_static
)

iree_native_test(
  NAME
    aie_runtime_utest_test
  SRC
    ::aie_runtime_utest
  DATA
    ${CMAKE_CURRENT_SOURCE_DIR}/pi.elf
  ARGS
    "${CMAKE_CURRENT_SOURCE_DIR}/pi.elf"
)
