# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_cc_library(
  NAME
    Target
  HDRS
    "AIETarget.h"
    "AIETargetDirect.h"
  SRCS
    "AIETarget.cpp"
    "AIETargetDirect.cpp"
  DEPS
    iree::compiler::Dialect::HAL::Target
    iree::target::amd-aie::IR::AMDAIEDialect
    iree::target::amd-aie::Transforms
    iree::target::amd-aie::air::AIRDialectIR
    iree::base::internal::flatcc::building
    iree::base::internal::flatcc::parsing
    iree-amd-aie::schemas::xrt_executable_def_c_fbs
  PUBLIC
)
