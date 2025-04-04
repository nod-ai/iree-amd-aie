# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree::target::amd-aie::Transforms::Utils")
iree_add_all_subdirs()

iree_cc_library(
  NAME
    Utils
  HDRS
    "AMDAIEDmaUtils.h"
    "AMDAIELogicalObjFifoSplittingUtils.h"
    "AMDAIEOpUtils.h"
    "AMDAIETileSizeSelectionUtils.h"
    "AMDAIEUtils.h"
  SRCS
    "AMDAIEDmaUtils.cpp"
    "AMDAIELogicalObjFifoSplittingUtils.cpp"
    "AMDAIETileSizeSelectionUtils.cpp"
    "AMDAIEUtils.cpp"
  DEPS
    iree::compiler::Utils
    iree-amd-aie::aie_runtime::iree_aie_runtime_static
    iree::target::amd-aie::IR::AMDAIEDialect
  PUBLIC
)
