# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "AIESerializer.mlir"
    "single_core.mlir"
    "large_M.mlir"
  TOOLS
    ${IREE_LLD_TARGET}
    FileCheck
    iree-aie-translate
  LABELS
    "hostonly"
)