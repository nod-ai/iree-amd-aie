# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree::target::amd-aie")

iree_cc_library(
  NAME
    registration
  SRCS
    "PluginRegistration.cpp"
  DEPS
    iree::compiler::Dialect::HAL::Target
    iree::compiler::PluginAPI
    iree::target::amd-aie::IR::AMDAIEDialect
    iree::target::amd-aie::Target
    iree::target::amd-aie::Transforms
    iree::target::amd-aie::aie::AIEDialectIR
    iree::target::amd-aie::aie::AIEXDialectIR
    iree::target::amd-aie::air::AIRDialectIR
    iree::target::amd-aie::air::AIRPasses
    iree::target::amd-aie::aie::AIEPass
    iree::base::core_headers
    iree::base::internal::flatcc::building
    iree-amd-aie::schemas::xrt_executable_def_c_fbs
  PUBLIC
)

iree_compiler_register_plugin(
  PLUGIN_ID
    amd_aie
  TARGET
    ::registration
)

add_subdirectory(IR)
add_subdirectory(Target)
add_subdirectory(Transforms)
