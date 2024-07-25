# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# The purpose of this test is nothing more than to exercise the aie-rt and
# cdo_driver dependencies. If you're really curious the code represents a
# configuration that fetches a single float value from core_tile (0,2) via the
# mem_tile(0,1) and shim_tile(0,0). That value is produced by the program in
# the adjacent pi.elf (the value being 3.14159).

iree_cc_binary(
  NAME
    aie_runtime_utest
  SRCS
    "utest.cxx"
  DEPS
    iree-amd-aie::aie_runtime::iree_aie_runtime_static
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

iree_cc_test(
  NAME
    DeviceModelTestCppTest
  SRCS
    DeviceModelTest.cpp
  DEPS
    gtest
    iree::target::amd-aie::aie::AIEDialectIR
    iree-amd-aie::aie_runtime::iree_aie_runtime_static
)

target_include_directories(
  iree-amd-aie_aie_runtime_test_DeviceModelTestCppTest
  PRIVATE "${IREE_MLIR_AIE_SOURCE_DIR}/include"
)

add_subdirectory(cdo)
