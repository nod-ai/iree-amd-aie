# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "air_dma_to_amdaie_dma.mlir"
    "bridge_to_air.mlir"
    "bufferize_to_allocation.mlir"
    "canonicalize_dma.mlir"
    "canonicalize_doubly_strided_op.mlir"
    "consume_produce_to_acquire_release.mlir"
    "controlcode_loop_unrolling.mlir"
    "create_aie_workgroup.mlir"
    "create_logical_objectfifo_link.mlir"
    "disable_vectorization.mlir"
    "dma_to_circular_dma.mlir"
    "fuse_consumer_into_loop.mlir"
    "fuse_fill_into_forall.mlir"
    "fuse_logical_objectfifo_into_workgroup.mlir"
    "fuse_pack_into_loop.mlir"
    "hoist_for_affine_apply.mlir"
    "insert_aie_workgroup.mlir"
    "insert_loops_for_vectorization.mlir"
    "lower_to_aie.mlir"
    "lower_to_ukernel.mlir"
    "lower_workgroup_count.mlir"
    "lowering_strategy.mlir"
    "lowering_strategy_failures.mlir"
    "map_forall_to_cores.mlir"
    "normalize_loop_bounds.mlir"
    "pack_and_transpose_level1.mlir"
    "pack_and_transpose_level2.mlir"
    "pack_to_air.mlir"
    "pack_to_dma.mlir"
    "pack_to_dma_failures.mlir"
    "pad.mlir"
    "peel_for_loop.mlir"
    "propagate_data_layout.mlir"
    "tile_and_fuse_using_scf_for.mlir"
    "tile_and_fuse_using_scf_forall.mlir"
    "tile_copy_using_scf_for.mlir"
    "unroll_and_distribute_workgroup.mlir"
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
