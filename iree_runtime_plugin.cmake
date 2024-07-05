# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(FetchContent)

set(IREE_AMD_AIE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")

set(IREE_AMD_AIE_ENABLE_XRT_DRIVER OFF)
if("xrt" IN_LIST IREE_EXTERNAL_HAL_DRIVERS)
  message(STATUS "Enabling XRT build because it is an enabled HAL driver")
  set(IREE_AMD_AIE_ENABLE_XRT_DRIVER ON)
endif()

if(IREE_AMD_AIE_ENABLE_XRT_DRIVER)
  set(Boost_USE_STATIC_LIBS ON CACHE BOOL "" FORCE)
  find_package(XRT REQUIRED)
  find_package(Threads REQUIRED)
  find_package(Boost REQUIRED COMPONENTS system program_options filesystem)
  message(STATUS "Boost include directories:" ${Boost_INCLUDE_DIRS})

  if(NOT WIN32)
    find_package(RapidJSON REQUIRED)
  endif()
endif()

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/runtime/src AMD-AIE)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/experimental AMD-AIE-experimental)
