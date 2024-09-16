# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "canonicalize.mlir"
    "invalid.mlir"
    "roundtrip.mlir"
  TOOLS
    FileCheck
    iree-opt
)

iree_cc_test(
  NAME
    AMDAIEDmaOpInterfaceTest
  SRCS
    "AMDAIEDmaOpInterfaceTest.cpp"
  DEPS
    gtest
    iree::target::amd-aie::IR::AMDAIEDialect
)
