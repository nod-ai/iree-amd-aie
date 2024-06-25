# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree::target::amd-aie::Utils")
iree_add_all_subdirs()

iree_cc_library(
  NAME
    Utils
  HDRS
    "Utils.h"
  SRCS
    "Utils.cpp"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRParser
    MLIRSupport
  PUBLIC
)
