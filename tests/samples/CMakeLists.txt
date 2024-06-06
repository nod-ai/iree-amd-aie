# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "matmul_peeled_objectfifo.mlir"
    "pack_peel_pipeline_matmul.mlir"
    "pack_peel_pipeline_matmul_elementwise.mlir"
    "pad_pack_pipeline_e2e.mlir"
    "xdna_oplib_plugin.mlir"
  TOOLS
    ${IREE_LLD_TARGET}
    FileCheck
    iree-opt
    iree-compile
)
