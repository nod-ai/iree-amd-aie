# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions. See
# https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# ##############################################################################
# Prerequisites
# ##############################################################################

if(NOT ${CMAKE_SIZEOF_VOID_P} EQUAL 8)
  message(
    FATAL_ERROR
      "Building on 32bit platforms/toolchains is not supported; if you are seeing this on windows, "
      "it's possible you have opened the win32 developer shell rather than the x64 developer shell."
  )
endif()

# We use our own, slightly modified, FindOpenSSL because of issues in
# CMake's distribution of the same for versions prior to 3.29.
# https://gitlab.kitware.com/cmake/cmake/-/issues/25702
find_package(OpenSSL)
if(NOT DEFINED OPENSSL_FOUND OR NOT ${OPENSSL_FOUND})
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
  find_package(OpenSSL)
  if(NOT DEFINED USE_IREE_AMD_AIE_FIND_OPENSSL OR NOT ${USE_IREE_AMD_AIE_FIND_OPENSSL})
    message(FATAL_ERROR "Didn't pickup/use adjacent FindOpenSSL.cmake")
  endif()
  if(NOT DEFINED OPENSSL_FOUND OR NOT ${OPENSSL_FOUND})
    message(FATAL_ERROR "OpenSSL not found")
  endif()
endif()
message(STATUS "OpenSSL include directories:" ${OPENSSL_INCLUDE_DIR})

# ##############################################################################
# Bootgen
# ##############################################################################

set(_bootgen_source_dir ${IREE_AMD_AIE_SOURCE_DIR}/third_party/bootgen)
file(GLOB _bootgen_sources "${_bootgen_source_dir}/*.c"
     "${_bootgen_source_dir}/*.cpp")
list(REMOVE_ITEM _bootgen_sources "${_bootgen_source_dir}/main.cpp")

add_library(bootgen-lib STATIC ${_bootgen_sources})
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  target_compile_definitions(bootgen-lib PUBLIC YY_NO_UNISTD_H)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  set(bootgen_warning_ignores
      -Wno-cast-qual
      -Wno-covered-switch-default
      -Wno-date-time
      -Wno-deprecated-copy
      -Wno-deprecated-declarations
      -Wno-deprecated-register
      -Wno-dynamic-class-memaccess
      -Wno-implicit-fallthrough
      -Wno-incompatible-function-pointer-types
      -Wno-incompatible-pointer-types-discards-qualifiers
      -Wno-misleading-indentation
      -Wno-non-virtual-dtor
      -Wno-overloaded-virtual
      -Wno-pointer-bool-conversion
      -Wno-register
      -Wno-reorder
      -Wno-sign-compare
      -Wno-suggest-override
      -Wno-tautological-overlap-compare
      -Wno-unused)
endif()
target_compile_options(bootgen-lib PRIVATE ${bootgen_warning_ignores})
target_include_directories(bootgen-lib PUBLIC ${_bootgen_source_dir}
                                              ${OPENSSL_INCLUDE_DIR})
target_compile_definitions(bootgen-lib PUBLIC OPENSSL_USE_APPLINK)
target_link_libraries(bootgen-lib PUBLIC OpenSSL::SSL OpenSSL::applink)

# malloc.h is deprecated and should not be used
# https://stackoverflow.com/a/56463133 If you want to use malloc, then include
# stdlib.h
file(READ ${_bootgen_source_dir}/cdo-npi.c FILE_CONTENTS)
string(REPLACE "#include <malloc.h>" "" FILE_CONTENTS "${FILE_CONTENTS}")
file(WRITE ${_bootgen_source_dir}/cdo-npi.c "${FILE_CONTENTS}")

file(READ ${_bootgen_source_dir}/cdo-alloc.c FILE_CONTENTS)
string(REPLACE "#include <malloc.h>" "" FILE_CONTENTS "${FILE_CONTENTS}")
file(WRITE ${_bootgen_source_dir}/cdo-alloc.c "${FILE_CONTENTS}")

# ##############################################################################
# cdo-drver
# ##############################################################################

add_library(cdo_driver STATIC ${_bootgen_source_dir}/cdo-driver/cdo_driver.c)
target_include_directories(cdo_driver PUBLIC ${_bootgen_source_dir}/cdo-driver)
set_target_properties(cdo_driver PROPERTIES LINKER_LANGUAGE C)
iree_install_targets(
  TARGETS cdo_driver
  COMPONENT IREEBundledLibraries
  EXPORT_SET Compiler
)

# ##############################################################################
# aie-rt
# ##############################################################################

set(_aie_rt_source_dir ${IREE_AMD_AIE_SOURCE_DIR}/third_party/aie-rt)
set(_aie_rt_binary_include_dir ${CMAKE_BINARY_DIR}/include)

include(${_aie_rt_source_dir}/fal/cmake/collect.cmake)
# We would prefer not to actually include the aie-rt directory
# (i.e., prefer not to run their CMake) but there are configure/install
# commands in the CMake there that perform the "installs" of the
# build/include/xaiengine headers.
add_subdirectory(${_aie_rt_source_dir}/driver/src iree_aie_runtime)

get_target_property(_aie_runtime_compile_options aienginev2 COMPILE_OPTIONS)
list(REMOVE_ITEM _aie_runtime_compile_options -D__AIEBAREMETAL__)
set_target_properties(
  aienginev2
  PROPERTIES COMPILE_OPTIONS
             "${_aie_runtime_compile_options};${bootgen_warning_ignores}")
target_compile_definitions(aienginev2 PRIVATE ${XAIE_DEBUG} __AIECDO__)
# For <elf.h>
target_include_directories(aienginev2 PRIVATE SYSTEM ${_bootgen_source_dir})
target_link_libraries(aienginev2 PRIVATE cdo_driver)

iree_install_targets(
  TARGETS aienginev2
  COMPONENT IREEBundledLibraries
  EXPORT_SET Compiler
)

# ##############################################################################
# iree-aie-runtime-static
# ##############################################################################

iree_cc_library(
  NAME
    iree_aie_runtime_static
  HDRS
    iree_aie_runtime.h
  SRCS
    iree_aie_runtime.cc
  INCLUDES
    ${_aie_rt_binary_include_dir}
    # For <elf.h>
    ${_bootgen_source_dir}
  PUBLIC
)

target_link_libraries(iree-amd-aie_aie_runtime_iree_aie_runtime_static
                      PRIVATE aienginev2)

add_subdirectory(test)
