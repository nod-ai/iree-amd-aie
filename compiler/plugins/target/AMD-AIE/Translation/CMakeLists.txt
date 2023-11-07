# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree::target::amd-aie::Translation")

iree_cc_library(
    NAME
      AIESerializer
    HDRS
      "AIESerializer.h"
    SRCS
      "AIESerializer.cpp"
    DEPS
      iree::compiler::Dialect::HAL::IR::HALDialect
      iree::compiler::Dialect::Stream::IR
      MLIRGPUDialect
      MLIRSCFDialect
      MLIRTranslateLib
      MLIRSupport
      IREELinalgExtDialect
    PUBLIC
)
