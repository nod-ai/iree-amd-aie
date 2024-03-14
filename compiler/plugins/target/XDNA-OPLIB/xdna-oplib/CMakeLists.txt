# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree::target::xdna-oplib")

iree_cc_library(
  NAME
    xdnaregistration
  SRCS
    "XDNAOplibRegistration.cpp"
  DEPS
    iree::compiler::PluginAPI
    iree::compiler::Utils
    iree::target::xdna-oplib::Transforms
  PUBLIC
)

iree_compiler_register_plugin(
  PLUGIN_ID
    xdna_oplib
  TARGET
    ::xdnaregistration
)

add_subdirectory(Transforms)
