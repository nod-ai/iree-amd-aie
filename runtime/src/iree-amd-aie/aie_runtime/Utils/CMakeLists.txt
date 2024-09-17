# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree-amd-aie::aie_runtime::Utils")
iree_add_all_subdirs()

iree_cc_library(
  NAME
    Utils
  HDRS
    "ChannelBdIdGenerator.h"
    "ChannelGenerator.h"
    "LockIdGenerator.h"
  SRCS
    "ChannelBdIdGenerator.cpp"
    "LockIdGenerator.cpp"
  DEPS
    iree-amd-aie::aie_runtime::iree_aie_runtime_static
    LLVMSupport
    MLIRIR
    MLIRParser
    MLIRSupport
  PUBLIC
)
