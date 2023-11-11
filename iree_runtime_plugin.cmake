# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_AMD_AIE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/runtime/src AMD-AIE)

iree_register_external_hal_driver(
  NAME
    xrt
  DRIVER_TARGET
    iree-amd-aie::driver::xrt::registration
  REGISTER_FN
    iree_hal_xrt_driver_module_register
)
