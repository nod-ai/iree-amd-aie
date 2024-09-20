# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "conv2d_nhwc_air_e2e.mlir"
    "matmul_elementwise_pack_peel_air_e2e.mlir"
    "matmul_pack_peel_air_e2e.mlir"
    "matmul_pack_peel_objectfifo.mlir"
    "matmul_pack_peel_objectfifo_e2e.mlir"
    "matmul_pad_pack_air_e2e.mlir"
    "xdna_oplib_plugin.mlir"
  TOOLS
    ${IREE_LLD_TARGET}
    FileCheck
    iree-opt
    iree-compile
)
