# Copyright 2023 The IREE Authors
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

if (DEFINED IREE_HSA_PATH)
  include_directories(${IREE_HSA_PATH}/include)
elseif(DEFIEND IREE_ROCM_PATH)
  include_directories(${IREE_ROCM_PATH}/include)
endif()

find_package(hip REQUIRED)


iree_cc_library(
  NAME
    hsa
  HDRS
    "api.h"
  SRCS
    "api.h"
    "event_pool.c"
    "event_pool.h"
    "event_semaphore.c"
    "event_semaphore.h"
    "graph_command_buffer.c"
    "graph_command_buffer.h"
    "hsa_allocator.c"
    "hsa_allocator.h"
    "hsa_buffer.c"
    "hsa_buffer.h"
    "hsa_device.c"
    "hsa_device.h"
    "hsa_driver.c"
    "memory_pools.c"
    "memory_pools.h"
    "native_executable.c"
    "native_executable.h"
    "nop_executable_cache.c"
    "nop_executable_cache.h"
    "pending_queue_actions.c"
    "pending_queue_actions.h"
    "pipeline_layout.c"
    "pipeline_layout.h"
    "stream_command_buffer.c"
    "stream_command_buffer.h"
    "timepoint_pool.c"
    "timepoint_pool.h"
    "tracing.c"
    "tracing.h"
  INCLUDES
    "${HIP_API_HEADERS_ROOT}"
    "${hip_INCLUDE_DIRS}"
  DEPS
    ::dynamic_symbols
    iree::base
    iree::base::core_headers
    iree::base::internal
    iree::base::internal::arena
    iree::base::internal::atomic_slist
    iree::base::internal::event_pool
    iree::base::internal::synchronization
    iree::base::internal::threading
    iree::base::internal::wait_handle
    iree::base::internal::flatcc::parsing
    iree::hal
    iree::hal::utils::collective_batch
    iree::hal::utils::deferred_command_buffer
    iree::hal::utils::file_transfer
    iree::hal::utils::memory_file
    iree::hal::utils::resource_set
    iree::hal::utils::semaphore_base
    iree::schemas::rocm_executable_def_c_fbs
    iree::experimental::hsa::schemas::hsa_amd_aie_executable_def_c_fbs
  PUBLIC
    hip::device
    hip::host
)

iree_cc_library(
  NAME
    dynamic_symbols
  HDRS
    "dynamic_symbols.h"
    "status_util.h"
  TEXTUAL_HDRS
    "dynamic_symbol_tables.h"
  SRCS
    "dynamic_symbols.c"
    "hsa_headers.h"
    "status_util.c"
  INCLUDES
    "${HIP_API_HEADERS_ROOT}"
    "${hip_INCLUDE_DIRS}"
  DEPS
    iree::base
    iree::base::core_headers
    iree::base::internal::dynamic_library
  PUBLIC
    hip::device
    hip::host
)

iree_cc_test(
  NAME
    dynamic_symbols_test
  SRCS
    "dynamic_symbols_test.cc"
  DEPS
    ::dynamic_symbols
    iree::base
    iree::testing::gtest
    iree::testing::gtest_main
  LABELS
    "driver=hsa"
)


# link_directories(/opt/rocm/lib)
# target_link_libraries(iree_runtime_unified hsa-runtime64)
