# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "pad_and_bufferize.mlir"
    "tile_and_fuse_using_scf_for.mlir"
    "tile_and_fuse_using_scf_forall.mlir"
  TOOLS
    ${IREE_LLD_TARGET}
    FileCheck
    iree-opt
  LABELS
    "hostonly"
)
