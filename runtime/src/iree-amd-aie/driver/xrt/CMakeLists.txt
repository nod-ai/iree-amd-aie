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

# find_package(XRT REQUIRED)

# set(xrt_core_LIBRARY XRT::xrt_core)
# set(xrt_coreutil_LIBRARY XRT::xrt_coreutil)
# set(xrt_xilinxopencl_LIBRARY XRT::xilinxopencl)

iree_cc_library(
  NAME
    xrt
  HDRS
    "api.h"
  SRCS
    "api.h"
    "xrt_driver.c"
  INCLUDES
    # "${XRT_INCLUDE_DIRS}"
  DEPS
    iree::base
    iree::base::core_headers
    iree::hal
    # TODO (nirvedhmeshram) : Figure out the flatbuffer format for this runtime
    # iree::schemas::xrt_executable_def_c_fbs
    # ${xrt_core_LIBRARY}
    # ${xrt_coreutil_LIBRARY}
    # ${xrt_xilinxopencl_LIBRARY}
  PUBLIC
)
