# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(ENABLE_AMDXDNA_CTS_TESTS OFF CACHE BOOL "")
if(NOT ENABLE_AMDXDNA_CTS_TESTS)
  return()
endif()

iree_cc_library(
  NAME
    backends
  SRCS
    "backends.cc"
  DEPS
    iree::async::util::proactor_pool
    iree::base::threading
    iree::hal
    iree::hal::cts::util::registry
    iree::base::tooling::flags
    iree-amd-aie::driver::amdxdna::registration
  TESTONLY
  ALWAYSLINK
  PUBLIC
)

iree_hal_cts_test_suite(
  BACKENDS_LIB
    ::backends
)
