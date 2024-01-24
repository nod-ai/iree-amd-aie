# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "matmul_fill_static_i32.mlir"
    "matmul_fill_static_i32_config.mlir"
    "matmul_pack_pipeline_e2e_test.mlir"
  TOOLS
    ${IREE_LLD_TARGET}
    FileCheck
    iree-opt
    iree-compile
  DATA
    matmul_fill_spec_pad.mlir
)
