# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions. See
# https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(${CMAKE_CURRENT_LIST_DIR}/iree_aie_utils.cmake)

if(TARGET iree-aie-bootgen)
  return()
endif()

set(_BOOTGEN_SOURCE_DIR ${IREE_AMD_AIE_SOURCE_DIR}/third_party/bootgen)

# malloc.h is deprecated and should not be used
# https://stackoverflow.com/a/56463133 If you want to use malloc, then include stdlib.h
replace_string_in_file(${_BOOTGEN_SOURCE_DIR}/cdo-npi.c "#include <malloc.h>" "#include <stdlib.h>")
replace_string_in_file(${_BOOTGEN_SOURCE_DIR}/cdo-alloc.c "#include <malloc.h>" "#include <stdlib.h>")
replace_string_in_file("${_BOOTGEN_SOURCE_DIR}/main.cpp" "#include \"openssl/ms/applink.c\"" "")
replace_string_in_file("${_BOOTGEN_SOURCE_DIR}/main.cpp" "int main" "int iree_aie_bootgen_main")
replace_string_in_file("${_BOOTGEN_SOURCE_DIR}/main.cpp" "DisplayBanner();" "")
replace_string_in_file("${_BOOTGEN_SOURCE_DIR}/main.cpp" "LOG_MSG(\"\\n[INFO]   : Bootimage generated successfully\\n\");" "")

file(GLOB _bootgen_sources "${_BOOTGEN_SOURCE_DIR}/*.c" "${_BOOTGEN_SOURCE_DIR}/*.cpp")
add_library(iree-aie-bootgen STATIC ${_bootgen_sources})

if(WIN32)
  target_compile_definitions(iree-aie-bootgen PUBLIC YY_NO_UNISTD_H)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  set(_bootgen_c_warning_ignores -w)
  set(_bootgen_cxx_warning_ignores -w -Wno-register)
endif()

target_compile_options(iree-aie-bootgen PRIVATE
                       $<$<COMPILE_LANGUAGE:C>:${_bootgen_c_warning_ignores}>
                       $<$<COMPILE_LANGUAGE:CXX>:${_bootgen_c_warning_ignores};${_bootgen_cxx_warning_ignores}>)

set(OPENSSL_USE_STATIC_LIBS TRUE CACHE BOOL "" FORCE)
set(_openssl_libs "")
if(WIN32)
  if(NOT ${CMAKE_SIZEOF_VOID_P} EQUAL 8)
    message(
      FATAL_ERROR
        "Building on 32bit platforms/toolchains is not supported; if you are seeing this on windows, "
        "it's possible you have opened the win32 developer shell rather than the x64 developer shell."
    )
  endif()
  # We use our own, slightly modified, FindOpenSSL because of issues in CMake's
  # distribution of the same for versions prior to 3.29.
  # https://gitlab.kitware.com/cmake/cmake/-/issues/25702
  find_package(OpenSSL)
  if(NOT DEFINED OPENSSL_FOUND OR NOT ${OPENSSL_FOUND})
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
    find_package(OpenSSL)
    if(NOT DEFINED USE_IREE_AMD_AIE_FIND_OPENSSL
            OR NOT ${USE_IREE_AMD_AIE_FIND_OPENSSL})
      message(FATAL_ERROR "Didn't pickup/use adjacent FindOpenSSL.cmake")
    endif()
    if(NOT DEFINED OPENSSL_FOUND OR NOT ${OPENSSL_FOUND})
      message(FATAL_ERROR "OpenSSL not found")
    endif()
  endif()
  message(STATUS "OpenSSL include directories:" ${OPENSSL_INCLUDE_DIR})
  list(APPEND _openssl_libs OpenSSL::SSL OpenSSL::Crypto)
else()
  set(BUILD_OPENSSL ON)
  # openssl-cmake doesn't have 1.1.1b but this one works
  set(OPENSSL_BUILD_VERSION 1.1.1k)
  # no zlib
  set(OPENSSL_MODULES no-comp)
  set(OPENSSL_ENABLE_TESTS OFF)
  add_subdirectory(${IREE_AMD_AIE_SOURCE_DIR}/third_party/openssl openssl)
  list(APPEND _openssl_libs $<BUILD_LOCAL_INTERFACE:ssl> $<BUILD_LOCAL_INTERFACE:crypto>)
endif()
target_include_directories(iree-aie-bootgen PRIVATE ${_BOOTGEN_SOURCE_DIR} ${OPENSSL_INCLUDE_DIR})
target_link_libraries(iree-aie-bootgen PRIVATE ${_openssl_libs})

iree_install_targets(
  TARGETS iree-aie-bootgen
  COMPONENT IREEBundledLibraries
  EXPORT_SET Compiler
)
