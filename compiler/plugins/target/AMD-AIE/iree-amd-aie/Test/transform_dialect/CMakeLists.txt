# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "conv_fill_spec_pad.mlir"
    "matmul_fill_spec_pack_funcIR.mlir"
    "matmul_fill_spec_pack_peel.mlir"
    "matmul_fill_spec_pad.mlir"
    "matmul_fill_spec_pad_pack.mlir"
    "matmul_fill_spec_pad_pack_peel.mlir"
    "matmul_fill_spec_simple_pack.mlir"
  TOOLS
    ${IREE_LLD_TARGET}
    FileCheck
    iree-opt
    iree-compile
  LABELS
    "hostonly"
)
