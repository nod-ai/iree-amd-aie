# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "bufferize_to_allocation_pack.mlir"
    "bufferize_to_allocation_pad.mlir"
    "bufferize_to_allocation_pad_last.mlir"
    "lower_workgroup_count.mlir"
    "pack_and_transpose_level1.mlir"
    "pack_and_transpose_level2.mlir"
    "pad.mlir"
    "peel_for_loop.mlir"
    "tile_and_fuse_using_scf_for.mlir"
    "tile_and_fuse_using_scf_forall.mlir"
  TOOLS
    ${IREE_LLD_TARGET}
    FileCheck
    iree-opt
  LABELS
    "hostonly"
)
