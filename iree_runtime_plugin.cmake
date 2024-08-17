# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# for BUILD_LOCAL_INTERFACE for xrt_coreutil
# (see runtime/src/iree-amd-aie/driver/xrt/CMakeLists.txt)
cmake_minimum_required(VERSION 3.26)

include(FetchContent)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/build_tools/cmake")
include(iree_aie_utils)

set(IREE_AMD_AIE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_MLIR_AIR_SOURCE_DIR "${IREE_AMD_AIE_SOURCE_DIR}/third_party/mlir-air/mlir")

set(IREE_AMD_AIE_ENABLE_XRT_DRIVER OFF)
if("xrt" IN_LIST IREE_EXTERNAL_HAL_DRIVERS)
  message(STATUS "Enabling XRT build because it is an enabled HAL driver")
  set(IREE_AMD_AIE_ENABLE_XRT_DRIVER ON)
endif()

if(IREE_AMD_AIE_ENABLE_XRT_DRIVER)
  set(Boost_USE_STATIC_LIBS ON CACHE BOOL "" FORCE)
  find_package(Threads REQUIRED)
  find_package(Boost REQUIRED COMPONENTS filesystem program_options system)
  include(iree_aie_xrt)
  include(iree_aie_bootgen)
  message(STATUS "Boost include directories:" ${Boost_INCLUDE_DIRS})
endif()

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/runtime/src AMD-AIE)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/experimental AMD-AIE-experimental)
