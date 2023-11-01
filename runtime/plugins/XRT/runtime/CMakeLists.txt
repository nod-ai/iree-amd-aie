# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "")

find_package(XRT REQUIRED)

set(xrt_core_LIBRARY XRT::xrt_core)
set(xrt_coreutil_LIBRARY XRT::xrt_coreutil)
set(xrt_xilinxopencl_LIBRARY XRT::xilinxopencl)

iree_cc_library(
  NAME
    iree-test-xrt
  SRCS
    "iree-test-xrt.cc"
  INCLUDES
    "${XRT_INCLUDE_DIRS}"
  DEPS
   ${xrt_core_LIBRARY}
   ${xrt_coreutil_LIBRARY}
   ${xrt_xilinxopencl_LIBRARY}
)
