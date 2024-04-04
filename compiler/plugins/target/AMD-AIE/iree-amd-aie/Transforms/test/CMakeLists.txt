# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "bridge_to_air.mlir"
    "bufferize_to_allocation.mlir"
    "canonicalize_dma.mlir"
    "disable_vectorization.mlir"
    "fuse_fill_into_forall.mlir"
    "fuse_pack_into_loop.mlir"
    "insert_loops_for_vectorization.mlir"
    "lower_to_ukernel.mlir"
    "lower_workgroup_count.mlir"
    "lowering_strategy.mlir"
    "map_forall_to_cores.mlir"
    "pack_and_transpose_level1.mlir"
    "pack_and_transpose_level2.mlir"
    "pack_to_air.mlir"
    "pack_to_dma.mlir"
    "pad.mlir"
    "peel_for_loop.mlir"
    "propagate_data_layout.mlir"
    "tile_and_fuse_using_scf_for.mlir"
    "tile_and_fuse_using_scf_forall.mlir"
    "tile_copy_using_scf_for.mlir"
    "vectorization.mlir"
  TOOLS
    ${IREE_LLD_TARGET}
    FileCheck
    iree-opt
  LABELS
    "hostonly"
)


iree_cc_test(
  NAME
    AMDAIEUtilsCppTest
  SRCS
    "AMDAIEUtilsTest.cpp"
  DEPS
    gtest
    iree::target::amd-aie::Transforms
)
