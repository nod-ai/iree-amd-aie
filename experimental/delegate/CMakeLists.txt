# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_TARGET_BACKEND_LLVM_CPU OR
   NOT IREE_HAL_DRIVER_LOCAL_SYNC OR
   NOT IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF)
  return()
endif()

## Current support is only for x86.
if(NOT IREE_ARCH STREQUAL "x86_64")
  message(STATUS "IREE mlp_pluging sample ignored -- only builds for x86_64 (today)")
  return()
endif()

# system-library plugin mechanism using the system dynamic library loader.
if(IREE_HAL_EXECUTABLE_PLUGIN_SYSTEM_LIBRARY)

find_package(XRT)

add_library(mlp_bf16_aie_delegate SHARED
  mlp_plugin.c
  mlp_aie_bf16.cpp
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

add_dependencies(iree-sample-deps
  mlp_bf16_aie_delegate
  iree_samples_custom_dispatch_cpu_system_plugin)


iree_lit_test_suite(
  NAME
    mlp_example
  SRCS
    "mlp.mlir"
  TOOLS
    FileCheck
    iree-compile
    iree-run-module
    mlp_bf16_aie_delegate
  LABELS
    "driver=local-sync"
    "hostonly"
)

endif(IREE_HAL_EXECUTABLE_PLUGIN_SYSTEM_LIBRARY)
