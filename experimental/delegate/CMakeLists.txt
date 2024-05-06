# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_TARGET_BACKEND_LLVM_CPU OR
   NOT IREE_HAL_DRIVER_LOCAL_SYNC OR
   NOT IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF OR
   NOT IREE_AMD_AIE_ENABLE_XRT_DRIVER)
  return()
endif()

## Current support is only for x86.
if(NOT IREE_ARCH STREQUAL "x86_64")
  message(STATUS "IREE mlp_pluging sample ignored -- only builds for x86_64 (today)")
  return()
endif()

# system-library plugin mechanism using the system dynamic library loader.
if(IREE_HAL_EXECUTABLE_PLUGIN_SYSTEM_LIBRARY)

iree_add_all_subdirs()

find_package(XRT)

add_library(mlp_bf16_aie_delegate SHARED
  mlp_aie_bf16_plugin.cpp
)
target_include_directories(mlp_bf16_aie_delegate
  PRIVATE
    ${IREE_SOURCE_DIR}/runtime/src/
    ${XRT_INCLUDE_DIRS}
)

# NOTE: this is only required because we want this sample to run on all
# platforms without needing to change the library name (libfoo.so/foo.dll).
set_target_properties(mlp_bf16_aie_delegate
  PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    PREFIX ""
    # OUTPUT_NAME "mlp_bf16_aie_delegate"
)

target_link_libraries(
  mlp_bf16_aie_delegate
  PRIVATE
  XRT::xrt_coreutil
  Boost::boost
)
set_property(TARGET mlp_bf16_aie_delegate PROPERTY CXX_STANDARD 20)

add_dependencies(mlp_bf16_aie_delegate
  aie_delegate_kernels
)
add_dependencies(iree-sample-deps
  mlp_bf16_aie_delegate
)

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "mlp.mlir"
    "matmul.mlir"
    "opt.mlir"
    "tres-leches.mlir"
  DATA
    "linalg.pdl.mlir"
    "opt.pdl.mlir"
    "tres-leches.pdl.mlir"
  TOOLS
    FileCheck
    iree-compile
    iree-opt
)

endif(IREE_HAL_EXECUTABLE_PLUGIN_SYSTEM_LIBRARY)
