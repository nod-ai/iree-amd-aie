# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_register_external_hal_driver(
  NAME
    hsa
  DRIVER_TARGET
    iree-amd-aie::driver::hsa::registration
  REGISTER_FN
    iree_hal_hsa_driver_module_register
)

iree_cc_library(
  NAME
    dynamic_symbols
  HDRS
    dynamic_symbols.h
    status_util.h
  TEXTUAL_HDRS
    dynamic_symbol_tables.h
  SRCS
    dynamic_symbols.cc
    hsa_headers.h
    status_util.cc
  DEPS
    iree::base
    iree::base::core_headers
    iree::base::internal::dynamic_library
)

iree_cc_library(
  NAME
    hsa
  HDRS
    api.h
  SRCS
    api.h
    hsa_buffer.cc
    hsa_buffer.h
    hsa_allocator.cc
    hsa_allocator.h
    hsa_driver.cc
    hsa_device.cc
    hsa_device.h
  DEPS
    ::dynamic_symbols
    iree-amd-aie::schemas::hsa_executable_def_c_fbs
    iree::base
    iree::base::core_headers
    iree::base::internal
    iree::base::internal::arena
    iree::base::internal::atomic_slist
    iree::base::internal::dynamic_library
    iree::base::internal::event_pool
    iree::base::internal::flatcc::parsing
    iree::base::internal::synchronization
    iree::base::internal::threading
    iree::base::internal::wait_handle
    iree::hal
    iree::hal::utils::collective_batch
    iree::hal::utils::deferred_command_buffer
    iree::hal::utils::file_transfer
    iree::hal::utils::memory_file
    iree::hal::utils::resource_set
    iree::hal::utils::semaphore_base
  PUBLIC
)

option(IREE_AIE_HSA_RUNTIME_DIRECT_LINK "" OFF)
if(IREE_AIE_HSA_RUNTIME_DIRECT_LINK)
  find_package(hsa-runtime64 CONFIG REQUIRED NAMES hsa-runtime64 hsa_runtime64)
  target_link_libraries(iree-amd-aie_driver_hsa_dynamic_symbols PUBLIC hsa-runtime64::hsa-runtime64)
  target_compile_definitions(iree-amd-aie_driver_hsa_dynamic_symbols PUBLIC IREE_AIE_HSA_RUNTIME_DIRECT_LINK)
else()
  if(NOT DEFINED HSA_API_HEADERS_ROOT)
    set(HSA_API_HEADERS_ROOT "${IREE_AMD_AIE_SOURCE_DIR}/third_party/ROCR-Runtime/runtime/hsa-runtime/inc")
  endif()
  target_include_directories(iree-amd-aie_driver_hsa_dynamic_symbols PUBLIC "${HSA_API_HEADERS_ROOT}")
endif()