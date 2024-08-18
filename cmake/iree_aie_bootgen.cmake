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
replace_string_in_file("${_BOOTGEN_SOURCE_DIR}/main.cpp"
                       "#include \"openssl/ms/applink.c\"" "//#include \"openssl/ms/applink.c\"")
replace_string_in_file("${_BOOTGEN_SOURCE_DIR}/main.cpp"
                       "int main" "int iree_aie_bootgen_main")
replace_string_in_file("${_BOOTGEN_SOURCE_DIR}/main.cpp"
                       "DisplayBanner();" "//DisplayBanner();")

file(GLOB _bootgen_sources "${_BOOTGEN_SOURCE_DIR}/*.c" "${_BOOTGEN_SOURCE_DIR}/*.cpp")
add_library(iree-aie-bootgen STATIC ${_bootgen_sources})

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  target_compile_definitions(iree-aie-bootgen PUBLIC YY_NO_UNISTD_H)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  set(_bootgen_c_warning_ignores
      -Wno-cast-qual
      -Wno-covered-switch-default
      -Wno-date-time
      -Wno-deprecated-declarations
      -Wno-deprecated-register
      -Wno-dynamic-class-memaccess
      -Wno-format
      -Wno-implicit-fallthrough
      -Wno-incompatible-function-pointer-types
      -Wno-incompatible-pointer-types-discards-qualifiers
      -Wno-misleading-indentation
      -Wno-pointer-bool-conversion
      -Wno-sign-compare
      -Wno-tautological-overlap-compare
      -Wno-unused)
  set(_bootgen_cxx_warning_ignores
      -Wno-deprecated-copy -Wno-non-virtual-dtor -Wno-overloaded-virtual
      -Wno-register -Wno-reorder -Wno-suggest-override)
endif()

target_compile_options(iree-aie-bootgen PRIVATE
                       $<$<COMPILE_LANGUAGE:C>:${_bootgen_c_warning_ignores}>
                       $<$<COMPILE_LANGUAGE:CXX>:${_bootgen_c_warning_ignores};${_bootgen_cxx_warning_ignores}>)

set(OPENSSL_USE_STATIC_LIBS TRUE CACHE BOOL "" FORCE)
if(WIN32)
  set(SYSTEM_OPENSSL ON)
else()
  set(BUILD_OPENSSL ON)
  # openssl-cmake doesn't have 1.1.1b but this one works
  set(OPENSSL_BUILD_VERSION 1.1.1k)
  # no zlib
  set(OPENSSL_MODULES no-comp)
endif()
add_subdirectory(${IREE_AMD_AIE_SOURCE_DIR}/third_party/openssl openssl)
target_include_directories(iree-aie-bootgen PRIVATE ${_BOOTGEN_SOURCE_DIR})
target_link_libraries(iree-aie-bootgen PRIVATE
                                       $<BUILD_LOCAL_INTERFACE:ssl>
                                       $<BUILD_LOCAL_INTERFACE:crypto>)

iree_install_targets(
  TARGETS iree-aie-bootgen
  COMPONENT IREEBundledLibraries
  EXPORT_SET Runtime
)
