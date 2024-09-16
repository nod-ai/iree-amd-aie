# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# because we're an "external" plugin, these need to be initialized/set by us
# whereas for in-tree plugins they're set in iree/runtime/src/iree/hal/cts/CMakeLists.txt#L28
set(IREE_EXECUTABLE_CTS_TESTS
  "executable_cache"
)

set(PEANO_INSTALL_DIR "" CACHE PATH "")
if(NOT EXISTS ${PEANO_INSTALL_DIR})
  message(FATAL_ERROR "PEANO_INSTALL_DIR=${PEANO_INSTALL_DIR} does not exist.")
endif()

iree_bytecode_module(
  NAME
    amd-aie-hsa_executable_cache_test_module
  MODULE_FILE_NAME
    amd-aie-hsa_executable_cache_test.bin
  SRC
    "${CMAKE_CURRENT_LIST_DIR}/executable_cache_test.mlir"
  FLAGS
    --compile-mode=hal-executable
    --iree-hal-dump-executable-files-to=${CMAKE_CURRENT_BINARY_DIR}
    --iree-hal-target-backends=amd-aie-hsa
    --iree-amdaie-lower-to-aie-pipeline=air
    --iree-amd-aie-peano-install-dir=${PEANO_INSTALL_DIR}
  PUBLIC
  TESTONLY
)

iree_c_embed_data(
  NAME
    amd-aie-hsa_executables_c
  SRCS
    amd-aie-hsa_executable_cache_test.bin
  C_FILE_OUTPUT
    amd-aie-hsa_executables_c.c
  H_FILE_OUTPUT
    amd-aie-hsa_executables_c.h
  IDENTIFIER
    iree_cts_testdata_executables
  STRIP_PREFIX
    amd-aie-hsa_
  FLATTEN
  PUBLIC
  TESTONLY
)

iree_hal_cts_test_suite(
  DRIVER_NAME
    amd-aie-hsa
  DRIVER_REGISTRATION_HDR
    "iree-amd-aie/driver/hsa/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_hsa_driver_module_register"
  COMPILER_TARGET_BACKEND
    "amd-aie-hsa"
  EXECUTABLE_FORMAT
    "\"amdaie-hsa-fb\""
  DEPS
    iree-amd-aie::driver::hsa::registration
  INCLUDED_TESTS
    "allocator"
    "buffer_mapping"
    "command_buffer"
#    "command_buffer_copy_buffer"
#    "command_buffer_dispatch"
#    "command_buffer_dispatch_constants"
#    "command_buffer_fill_buffer"
#    "command_buffer_update_buffer"
    "driver"
#    "event"
    "executable_cache"
#    "file"
#    "semaphore"
#    "semaphore_submission"
)

iree_cc_test(
  NAME
    amd_aie_hsa_command_buffer_dispatch_test
  SRCS
    amd_aie_hsa_command_buffer_dispatch_test.cc
  DEPS
    ::amd-aie-hsa_executables_c
    iree-amd-aie::driver::hsa::registration
    iree::base
    iree::hal
    iree::hal::cts::cts_test_base
    iree::testing::gtest_main
)

target_include_directories(iree-amd-aie_driver_hsa_cts_amd_aie_hsa_command_buffer_dispatch_test PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")