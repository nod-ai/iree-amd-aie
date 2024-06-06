# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_lit_test_suite(
  NAME
    lit
  SRCS
    basic_dma_transpose.mlir
    basic_matrix_multiplication_matrix_vector.mlir
    basic_matrix_multiplication_single_core.mlir
    basic_matrix_scalar_add.mlir
    basic_passthrough_dmas.mlir
    basic_passthrough_kernel.mlir
    basic_vector_exp.mlir
    basic_vector_reduce_add.mlir
    basic_vector_reduce_max.mlir
    basic_vector_reduce_min.mlir
    basic_vector_scalar_add.mlir
    basic_vector_scalar_mul.mlir
    basic_vector_vector_add.mlir
    basic_vector_vector_mul.mlir
    ml_bottleneck.mlir
    ml_conv2d.mlir
    ml_conv2d_fused_relu.mlir
    ml_eltwise_add.mlir
    ml_eltwise_mul.mlir
    ml_relu.mlir
    ml_resnet_layers_conv2_x.mlir
    ml_softmax.mlir
    vision_vision_passthrough.mlir
  TOOLS
    iree-opt
)
