# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(ENABLE_AMDXDNA_AIE_EXECUTABLE_TESTS OFF CACHE BOOL "")
if(NOT ENABLE_AMDXDNA_AIE_EXECUTABLE_TESTS)
  return()
endif()

set(PEANO_INSTALL_DIR "" CACHE PATH "")
if(NOT PEANO_INSTALL_DIR)
  message(FATAL_ERROR "PEANO_INSTALL_DIR must be set")
endif()
set(TARGET_DEVICE "npu1_4col" CACHE STRING "")

iree_bytecode_module(
  NAME
    amdxdna_executable_cache_test_module
  MODULE_FILE_NAME
    amdxdna_executable_cache_test.bin
  SRC
    "${CMAKE_CURRENT_LIST_DIR}/executable_cache_test.mlir"
  FLAGS
    --compile-mode=hal-executable
    --iree-hal-dump-executable-files-to=${CMAKE_CURRENT_BINARY_DIR}
    --iree-hal-target-backends=amd-aie
    --iree-amdaie-lower-to-aie-pipeline=objectFifo
    --iree-amdaie-target-device=${TARGET_DEVICE}
    --iree-amd-aie-peano-install-dir=${PEANO_INSTALL_DIR}
    --iree-amdaie-device-hal=amdxdna
    --iree-hal-memoization=false
    --iree-hal-indirect-command-buffers=false
  PUBLIC
  TESTONLY
)

iree_c_embed_data(
  NAME
    amdxdna_executables_c
  SRCS
    amdxdna_executable_cache_test.bin
  C_FILE_OUTPUT
    amdxdna_executables_c.c
  H_FILE_OUTPUT
    amdxdna_executables_c.h
  IDENTIFIER
    iree_amdxdna_testdata_executables_aie
  STRIP_PREFIX
    amdxdna_
  DEPENDS
    ::amdxdna_executable_cache_test_module
  FLATTEN
  PUBLIC
  TESTONLY
)

iree_cc_test(
  NAME
    amdxdna_executable_cache_test
  SRCS
    executable_cache_test.cc
  DEPS
    ::amdxdna_executables_c
    iree-amd-aie::driver::amdxdna::registration
    iree::async::util::proactor_pool
    iree::base
    iree::base::threading
    iree::hal
    iree::testing::gtest_main
  LABELS
    requires-npu
    requires-peano
)

iree_cc_test(
  NAME
    amdxdna_dispatch_test
  SRCS
    matmul_dispatch_test.cc
  DEPS
    ::amdxdna_executables_c
    iree-amd-aie::driver::amdxdna::registration
    iree::async::util::proactor_pool
    iree::base
    iree::base::threading
    iree::hal
    iree::testing::gtest_main
  LABELS
    requires-npu
    requires-peano
)

target_include_directories(iree-amd-aie_driver_amdxdna_test_executable_amdxdna_executable_cache_test PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(iree-amd-aie_driver_amdxdna_test_executable_amdxdna_dispatch_test PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

# Cross-executable ERT_CMD_CHAIN test. Compiles a small control-packet matmul at
# build time (npu4-only, since ERT_CMD_CHAIN is firmware-gated to npu4) and
# embeds the resulting amdaie-pdi-fb so the test can load it twice as two
# distinct hal.executable objects.
iree_bytecode_module(
  NAME
    xexec_chain_test_module
  MODULE_FILE_NAME
    xexec_chain_test.bin
  SRC
    "${CMAKE_CURRENT_LIST_DIR}/xexec_chain_test.mlir"
  FLAGS
    --compile-mode=hal-executable
    --iree-hal-target-backends=amd-aie
    --iree-amdaie-lower-to-aie-pipeline=objectFifo
    --iree-amdaie-target-device=npu4
    --iree-amd-aie-peano-install-dir=${PEANO_INSTALL_DIR}
    --iree-amdaie-device-hal=amdxdna
    --iree-amdaie-enable-control-packet=true
    --iree-amdaie-packet-flow-strategy=auto
    --iree-amdaie-num-rows=1
    --iree-amdaie-num-cols=1
    --iree-amdaie-stack-size=1024
    --iree-hal-memoization=false
    --iree-hal-indirect-command-buffers=false
  PUBLIC
  TESTONLY
)

iree_c_embed_data(
  NAME
    xexec_chain_executables_c
  SRCS
    xexec_chain_test.bin
  C_FILE_OUTPUT
    xexec_chain_executables_c.c
  H_FILE_OUTPUT
    xexec_chain_executables_c.h
  IDENTIFIER
    iree_amdxdna_testdata_executables_xexec_chain
  STRIP_PREFIX
    xexec_chain_
  DEPENDS
    ::xexec_chain_test_module
  FLATTEN
  PUBLIC
  TESTONLY
)

iree_cc_test(
  NAME
    xexec_chain_test
  SRCS
    xexec_chain_test.cc
  DEPS
    ::xexec_chain_executables_c
    iree-amd-aie::driver::amdxdna::amdxdna
    iree::async::util::proactor_pool
    iree::base
    iree::base::tooling::flags
    iree::hal
    iree::testing::gtest_main
  LABELS
    requires-npu
    requires-peano
)
target_include_directories(iree-amd-aie_driver_amdxdna_test_executable_xexec_chain_test PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
