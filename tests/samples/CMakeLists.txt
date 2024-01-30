# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "pack_pipeline_e2e.mlir"
    "pad_pipeline_e2e.mlir"
    "pipeline_select.mlir"
  TOOLS
    ${IREE_LLD_TARGET}
    FileCheck
    iree-opt
    iree-compile
  DATA
    matmul_fill_spec_pad.mlir
)
