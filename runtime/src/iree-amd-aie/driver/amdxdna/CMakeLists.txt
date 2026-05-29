# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_register_external_hal_driver(
  NAME
    amdxdna
  DRIVER_TARGET
    iree-amd-aie::driver::amdxdna::registration
  REGISTER_FN
    iree_hal_amdxdna_driver_module_register
)

iree_cc_library(
  NAME
    amdxdna
  SRCS
    allocator.cc
    allocator.h
    api.h
    async_queue.cc
    async_queue.h
    buffer.cc
    buffer.h
    direct_command_buffer.cc
    direct_command_buffer.h
    device.cc
    driver.cc
    executable.cc
    executable.h
    nop_event.cc
    nop_event.h
    nop_executable_cache.cc
    nop_executable_cache.h
    nop_semaphore.cc
    nop_semaphore.h
    util.h
  DEPS
    iree::async
    iree::async::util::proactor_pool
    iree::base
    iree::base::core_headers
    iree::hal::memory::cpu_slab_provider
    iree::hal::memory::passthrough_pool
    iree::hal::utils::deferred_command_buffer
    iree::hal::utils::file_transfer
    iree::hal::utils::files
    iree::hal::utils::queue_emulation
    iree::hal::utils::queue_host_call_emulation
    iree::base::internal::flatcc::parsing
    iree-amd-aie::schemas::pdi_executable_def_c_fbs
    iree-amd-aie::driver::amdxdna::shim::linux::kmq::shim-xdna
  PUBLIC
)
