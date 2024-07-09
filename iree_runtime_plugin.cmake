# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(FetchContent)
include(ExternalProject)

set(IREE_AMD_AIE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")

set(IREE_AMD_AIE_ENABLE_XRT_DRIVER OFF)
if("xrt" IN_LIST IREE_EXTERNAL_HAL_DRIVERS)
  message(STATUS "Enabling XRT build because it is an enabled HAL driver")
  set(IREE_AMD_AIE_ENABLE_XRT_DRIVER ON)
endif()

if(IREE_AMD_AIE_ENABLE_XRT_DRIVER)
  find_package(Threads REQUIRED)

  set(BOOST_INCLUDE_LIBRARIES boost filesystem system program_options)
  set(BOOST_ENABLE_CMAKE ON)
  Set(FETCHCONTENT_QUIET FALSE)
  include(FetchContent)
  FetchContent_Declare(
    Boost
    URL      https://github.com/boostorg/boost/releases/download/boost-1.85.0/boost-1.85.0-cmake.tar.xz
    URL_HASH MD5=badea970931766604d4d5f8f4090b176
    USES_TERMINAL_DOWNLOAD TRUE
    GIT_PROGRESS TRUE
    DOWNLOAD_NO_EXTRACT FALSE
    OVERRIDE_FIND_PACKAGE
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(Boost)
  message(STATUS "Boost include directories:" ${Boost_INCLUDE_DIRS})

  ExternalProject_Add(
    XRT
    SOURCE_DIR ${IREE_AMD_AIE_SOURCE_DIR}/third_party/XRT
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/XRT
    OVERRIDE_FIND_PACKAGE
  )
  find_package(XRT REQUIRED)


  if(NOT WIN32)
    find_package(RapidJSON REQUIRED)
  endif()
endif()

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/runtime/src AMD-AIE)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/experimental AMD-AIE-experimental)
