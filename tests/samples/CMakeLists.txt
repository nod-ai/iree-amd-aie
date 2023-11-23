# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_lit_test_suite(
  NAME
    lit
  SRCS
    # TODO(MaheshRavishankar): This is supposed to
    # be matmul_fill_i32_static.mlir via use of the
    # transform dialect script in
    # matmul_fill_spec_pad.mlir. That is blocked
    # by LLVM #72931.
    "air_to_ipu_expected_input.mlir"
  TOOLS
    ${IREE_LLD_TARGET}
    FileCheck
    iree-opt
  DATA
    matmul_fill_spec_pad.mlir
)
