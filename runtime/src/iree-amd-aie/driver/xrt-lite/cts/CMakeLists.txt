# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeDependentOption)

set(PEANO_INSTALL_DIR "" CACHE PATH "")
set(VITIS_DIR "" CACHE PATH "")
if((NOT PEANO_INSTALL_DIR) AND (NOT VITIS_DIR))
  message(FATAL_ERROR "either PEANO_INSTALL_DIR or VITIS_DIR must be set")
endif()
cmake_dependent_option(USE_CHESS "" "1" "VITIS_DIR" "0")
set(TARGET_DEVICE "npu1_4col" CACHE STRING "")

iree_bytecode_module(
  NAME
    xrt_lite_executable_cache_test_module
  MODULE_FILE_NAME
    xrt_lite_executable_cache_test.bin
  SRC
    "${CMAKE_CURRENT_LIST_DIR}/executable_cache_test.mlir"
  FLAGS
    --compile-mode=hal-executable
    --iree-hal-dump-executable-files-to=${CMAKE_CURRENT_BINARY_DIR}
    --iree-hal-target-backends=amd-aie
    --iree-amdaie-lower-to-aie-pipeline=air
    --iree-amdaie-target-device=${TARGET_DEVICE}
    --iree-amd-aie-peano-install-dir=${PEANO_INSTALL_DIR}
    --iree-amd-aie-vitis-install-dir=${VITIS_DIR}
    --iree-amd-aie-enable-chess=$<BOOL:${USE_CHESS}>
    --iree-amd-aie-show-invoked-commands
  PUBLIC
  TESTONLY
)

iree_c_embed_data(
  NAME
    xrt_lite_executables_c
  SRCS
    xrt_lite_executable_cache_test.bin
  C_FILE_OUTPUT
    xrt_lite_executables_c.c
  H_FILE_OUTPUT
    xrt_lite_executables_c.h
  IDENTIFIER
    iree_cts_testdata_executables_aie_xrt_lite
  STRIP_PREFIX
    xrt_lite_
  DEPENDS
    ::xrt_lite_executable_cache_test_module
  FLATTEN
  PUBLIC
  TESTONLY
)

iree_hal_cts_test_suite(
  DRIVER_NAME
    xrt-lite
  DRIVER_REGISTRATION_HDR
    "iree-amd-aie/driver/xrt-lite/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_xrt_lite_driver_module_register"
  COMPILER_TARGET_BACKEND
    "amd-aie"
  EXECUTABLE_FORMAT
    "\"amdaie-pdi-fb\""
  DEPS
    iree-amd-aie::driver::xrt-lite::registration
  INCLUDED_TESTS
#    "allocator"
#    "buffer_mapping"
#    "command_buffer"
    "driver"
)

iree_cc_test(
  NAME
    xrt_lite_command_buffer_dispatch_test
  SRCS
    xrt_lite_command_buffer_dispatch_test.cc
  DEPS
    ::xrt_lite_executables_c
    iree-amd-aie::driver::xrt-lite::registration
    iree::base
    iree::hal
    iree::hal::cts::cts_test_base
    iree::testing::gtest_main
    iree::tools::testing::e2e::e2e_test_util
)

target_include_directories(iree-amd-aie_driver_xrt-lite_cts_xrt_lite_command_buffer_dispatch_test PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")