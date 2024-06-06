# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions. See
# https://llvm.org/LICENSE.txt for license information. SPDX-License-Identifier:
# Apache-2.0 WITH LLVM-exception

# ##############################################################################
# requirements
# ##############################################################################

if(NOT ${CMAKE_SIZEOF_VOID_P} EQUAL 8)
  # if you open up the wrong developer shell in windows...
  message(
    FATAL_ERROR
      "Something has gone terribly wrong: CMAKE_SIZEOF_VOID_P=${CMAKE_SIZEOF_VOID_P}"
  )
endif()
# can delete after this is all worked out
# https://gitlab.kitware.com/cmake/cmake/-/issues/25702
find_package(OpenSSL)
if(NOT DEFINED OPENSSL_FOUND OR NOT ${OPENSSL_FOUND})
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
  find_package(OpenSSL)
  if(NOT DEFINED USE_OUR_FIND_OPENSSL OR NOT ${USE_OUR_FIND_OPENSSL})
    message(FATAL_ERROR "Didn't pickup our FindOpenSSL.cmake")
  endif()
  if(NOT DEFINED OPENSSL_FOUND OR NOT ${OPENSSL_FOUND})
    message(FATAL_ERROR "OpenSSL not found")
  endif()
endif()
message(STATUS "OpenSSL include directories:" ${OPENSSL_INCLUDE_DIR})

iree_add_all_subdirs()

# ##############################################################################
# bootgen
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

add_library(cdo_driver STATIC ${_bootgen_source_dir}/cdo-driver/cdo_driver.c)
target_include_directories(cdo_driver PUBLIC ${_bootgen_source_dir}/cdo-driver)
set_target_properties(cdo_driver PROPERTIES LINKER_LANGUAGE C)

# malloc.h is deprecated and should not be used
# https://stackoverflow.com/a/56463133 If you want to use malloc, then include
# stdlib.h
file(READ ${_bootgen_source_dir}/cdo-npi.c FILE_CONTENTS)
string(REPLACE "#include <malloc.h>" "" _FILE_CONTENTS "${FILE_CONTENTS}")
file(WRITE ${_bootgen_source_dir}/cdo-npi.c "${_FILE_CONTENTS}")

file(READ ${_bootgen_source_dir}/cdo-alloc.c FILE_CONTENTS)
string(REPLACE "#include <malloc.h>" "" _FILE_CONTENTS "${FILE_CONTENTS}")
file(WRITE ${_bootgen_source_dir}/cdo-alloc.c "${_FILE_CONTENTS}")

# ##############################################################################
# aie-rt
# ##############################################################################

set(_aie_rt_source_dir ${IREE_AMD_AIE_SOURCE_DIR}/third_party/aie-rt)
# TODO(max): there's no choice here without hacks (the cmake uses file(COPY
# ...)) does this need to cleaner (in a subdir)
set(_aie_rt_binary_include_dir ${CMAKE_BINARY_DIR}/include)

include(${_aie_rt_source_dir}/fal/cmake/collect.cmake)
# gotta add the subdirectory so the copies to build/include/xaiengine occur...
add_subdirectory(${_aie_rt_source_dir}/driver/src iree_aie_runtime)

string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if(uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  set(XAIE_DEBUG "XAIE_DEBUG")
endif()

get_target_property(_aie_runtime_compile_options aienginev2 COMPILE_OPTIONS)
list(REMOVE_ITEM _aie_runtime_compile_options -D__AIEBAREMETAL__)
set_target_properties(
  aienginev2
  PROPERTIES COMPILE_OPTIONS
             "${_aie_runtime_compile_options};${bootgen_warning_ignores}")
target_compile_definitions(aienginev2 PRIVATE ${XAIE_DEBUG} __AIECDO__)
# <elf.h>
target_include_directories(aienginev2 PRIVATE SYSTEM ${_bootgen_source_dir})
target_link_libraries(aienginev2 PUBLIC cdo_driver)

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
    # <elf.h>
    ${_bootgen_source_dir}

  PUBLIC
  # TODO(max)
  # includes target "iree-amd-aie_runtime_iree_aie_runtime_static"
  # which requires target "aienginev2" that is not in any export
  TESTONLY
)

target_link_libraries(iree-amd-aie_runtime_iree_aie_runtime_static
                      PRIVATE aienginev2)
