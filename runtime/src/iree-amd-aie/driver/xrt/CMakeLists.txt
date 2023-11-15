# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#-------------------------------------------------------------------------------
# Experimental XRT HAL driver
# Enable with: -DIREE_EXTERNAL_HAL_DRIVERS=xrt
#-------------------------------------------------------------------------------

iree_register_external_hal_driver(
  NAME
    xrt
  DRIVER_TARGET
    iree-amd-aie::driver::xrt::registration
  REGISTER_FN
    iree_hal_xrt_driver_module_register
)

iree_add_all_subdirs()

find_package(XRT REQUIRED)

iree_cc_library(
  NAME
    xrt
  HDRS
    "api.h"
  SRCS
    "api.h"
    "direct_allocator.h"
    "direct_allocator.cc"
    "xrt_buffer.h"
    "xrt_buffer.cc"
    "xrt_device.cc"
    "xrt_driver.cc"
    "nop_semaphore.cc"
  INCLUDES
    "${XRT_INCLUDE_DIRS}"
  DEPS
    iree::base
    iree::base::core_headers
    iree::hal
    iree::hal::utils::buffer_transfer
    # TODO (nirvedhmeshram) : Figure out the flatbuffer format for this runtime
    # iree::schemas::xrt_executable_def_c_fbs
    XRT::xrt_coreutil
  COPTS
  # currenly this is needed, it can be removed after this issue is resolved
  # https://github.com/Xilinx/XRT/issues/7810
   -frtti
  PUBLIC
)
