# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cc_binary(
  NAME
    aie_cdo_gen_test
  SRCS
    aie_cdo_gen_test.cxx
  DEPS
    iree::target::amd-aie::aie::AIEDialectIR
    iree_target_amd-aie_Target_AIETargets
)

target_include_directories(
  iree_target_amd-aie_Target_tests_cdo_aie_cdo_gen_test
  PRIVATE "${IREE_MLIR_AIE_SOURCE_DIR}/include"
)

file(GLOB _mlir_files *.mlir)

iree_lit_test_suite(
  NAME
    lit
  SRCS
    ${_mlir_files}
  TOOLS
    ::aie_cdo_gen_test
    FileCheck
  LABELS
    "hostonly"
)
