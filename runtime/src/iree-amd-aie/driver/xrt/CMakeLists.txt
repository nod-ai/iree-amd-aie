# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#-------------------------------------------------------------------------------
# Experimental XRT HAL driver
# Enable with: -DIREE_EXTERNAL_HAL_DRIVERS=xrt
#-------------------------------------------------------------------------------

iree_add_all_subdirs()

function(replace_string_in_file _file _match_string _replace_string)
  if(NOT (EXISTS ${_file}))
    message(FATAL_ERROR "file ${_file} does not exist")
  endif()
  file(READ "${_file}" _file_contents)
  if(_file_contents STREQUAL "")
    message(FATAL_ERROR "empty file contents for ${_file}")
  endif()
  string(REPLACE "${_match_string}" "${_replace_string}" _file_contents "${_file_contents}")
  if(_file_contents STREQUAL "")
    message(FATAL_ERROR "empty replacement contents for ${_file}")
  endif()
  file(WRITE "${_file}" "${_file_contents}")
endfunction()

# TODO(max): make a cmake module dir
# otherwise the various stois that read these will explode...
file(READ ${IREE_XRT_SOURCE_DIR}/CMakeLists.txt _xrt_cmake_file_contents)
string(REGEX MATCH "XRT_VERSION_MAJOR ([0-9]+)" XRT_VERSION_MAJOR ${_xrt_cmake_file_contents})
# note CMAKE_MATCH_0 is the whole match...
set(XRT_VERSION_MAJOR ${CMAKE_MATCH_1})
string(REGEX MATCH "XRT_VERSION_MINOR ([0-9]+)" XRT_VERSION_MINOR ${_xrt_cmake_file_contents})
set(XRT_VERSION_MINOR ${CMAKE_MATCH_1})
string(REGEX MATCH "XRT_VERSION_PATCH ([0-9]+)" XRT_VERSION_PATCH ${_xrt_cmake_file_contents})
set(XRT_VERSION_PATCH ${CMAKE_MATCH_1})
set(XRT_VERSION_STRING ${XRT_VERSION_MAJOR}.${XRT_VERSION_MINOR}.${XRT_VERSION_PATCH} CACHE INTERNAL "")
set(XRT_SOVERSION ${XRT_VERSION_MAJOR} CACHE INTERNAL "")
set(XRT_HEAD_COMMITS 0xDEADBEEF CACHE INTERNAL "")
set(XRT_BRANCH_COMMITS 0xDEADFACE CACHE INTERNAL "")
configure_file(${IREE_XRT_SOURCE_DIR}/CMake/config/version.h.in
               ${IREE_XRT_SOURCE_DIR}/runtime_src/core/include/xrt/detail/version.h)
file(MAKE_DIRECTORY ${IREE_XRT_SOURCE_DIR}/runtime_src/core/common/gen)
configure_file(${IREE_XRT_SOURCE_DIR}/CMake/config/version.h.in
               ${IREE_XRT_SOURCE_DIR}/runtime_src/core/common/gen/version.h)
configure_file(${IREE_XRT_SOURCE_DIR}/CMake/config/version.h.in
               ${IREE_XRT_SOURCE_DIR}/runtime_src/core/common/api/version.h)
replace_string_in_file(${IREE_XRT_SOURCE_DIR}/runtime_src/core/common/query.h
                       "#include <stdexcept>" "#include <any>")

# remove -Wextra flag
set(GSL_TEST OFF CACHE BOOL "")
include_directories(${IREE_XRT_SOURCE_DIR}/runtime_src/core/common/elf SYSTEM)
# remove requirement for generated version.h
include_directories(${IREE_XRT_SOURCE_DIR}/runtime_src/core/include)
include_directories(${IREE_XRT_SOURCE_DIR}/runtime_src/core/common/gsl/include)
include_directories(${IREE_XRT_SOURCE_DIR}/runtime_src)
include_directories(${Boost_INCLUDE_DIRS})
add_subdirectory(${IREE_XRT_SOURCE_DIR}/runtime_src/core/common iree-aie-xrt-coreutil)
iree_install_targets(
  TARGETS xrt_coreutil
  COMPONENT IREEBundledLibraries
  EXPORT_SET Runtime
)

iree_register_external_hal_driver(
  NAME
    xrt
  DRIVER_TARGET
    iree-amd-aie::driver::xrt::registration
  REGISTER_FN
    iree_hal_xrt_driver_module_register
)

iree_cc_library(
  NAME
    xrt
  HDRS
    "api.h"
  SRCS
    "api.h"
    "direct_allocator.h"
    "direct_allocator.cc"
    "direct_command_buffer.h"
    "direct_command_buffer.cc"
    "xrt_buffer.h"
    "xrt_buffer.cc"
    "xrt_device.cc"
    "xrt_driver.cc"
    "native_executable.h"
    "native_executable.cc"
    "nop_semaphore.cc"
    "nop_executable_cache.h"
    "nop_executable_cache.cc"
    "pipeline_layout.h"
    "pipeline_layout.cc"
  INCLUDES
    "${XRT_INCLUDE_DIRS}"
  DEPS
    iree::base
    iree::base::core_headers
    iree::base::internal::arena
    iree::base::internal::flatcc::building
    iree::base::internal::flatcc::parsing
    iree::hal::utils::deferred_command_buffer
    iree::hal::utils::file_transfer
    iree::hal
    iree-amd-aie::schemas::xrt_executable_def_c_fbs
    xrt_coreutil
    Boost::boost
  COPTS
  # currenly this is needed, it can be removed after this issue is resolved
  # https://github.com/Xilinx/XRT/issues/7810
   -frtti
   -fexceptions
  PUBLIC
)
