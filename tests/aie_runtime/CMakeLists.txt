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

iree_lit_test(
  NAME
    aie_runtime_utest_test
  TEST_FILE
    "cdo_dump.txt"
  TOOLS
    ::aie_runtime_utest
    FileCheck
  LABELS
    "hostonly"
)