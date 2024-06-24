# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

file(GLOB _mlir_files *.mlir)

iree_lit_test_suite(
  NAME
    lit
  SRCS
    ${_mlir_files}
  TOOLS
    ${IREE_LLD_TARGET}
    FileCheck
    iree-opt
  LABELS
    "hostonly"
)


# iree_cc_test(
#   NAME
#     AMDAIEUtilsCppTest
#   SRCS
#     "AMDAIEUtilsTest.cpp"
#   DEPS
#     gtest
#     iree::target::amd-aie::Transforms
# )
