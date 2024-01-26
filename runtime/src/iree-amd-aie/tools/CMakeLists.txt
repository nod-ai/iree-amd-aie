# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cc_library(
  NAME
    iree-test-xrt
  SRCS
    "iree-test-xrt.cc"
  INCLUDES
    # "${XRT_INCLUDE_DIRS}"
  DEPS
  #  ${xrt_core_LIBRARY}
  #  ${xrt_coreutil_LIBRARY}
  #  ${xrt_xilinxopencl_LIBRARY}
)
