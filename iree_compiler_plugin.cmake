# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(IREE_AMD_AIE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_AMD_AIE_RUNTIME_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/runtime/src)
set(IREE_MLIR_AIR_SOURCE_DIR "${IREE_AMD_AIE_SOURCE_DIR}/third_party/mlir-air/mlir")

set(IREE_AMD_AIE_ENABLE_XRT_DRIVER OFF)
if("xrt" IN_LIST IREE_EXTERNAL_HAL_DRIVERS)
  message(STATUS "Enabling XRT build because it is an enabled HAL driver")
  set(IREE_AMD_AIE_ENABLE_XRT_DRIVER ON)
endif()

if(IREE_AMD_AIE_ENABLE_XRT_DRIVER)
  include(iree_aie_xrt)
endif()
include(iree_aie_bootgen)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/compiler/plugins/target/AMD-AIE target/AMD-AIE)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/compiler/plugins/preprocessing/XDNA-OPLIB preprocessing/XDNA-OPLIB)
