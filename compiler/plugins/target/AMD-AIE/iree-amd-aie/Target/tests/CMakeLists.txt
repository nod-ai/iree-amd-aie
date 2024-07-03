# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cc_binary(
  NAME
    aie_targets_test
  SRCS
    aie_targets_test.cxx
  DEPS
    iree::target::amd-aie::aie::AIEDialectIR
    iree_target_amd-aie_Target_AIETargets
    iree-amd-aie::aie_runtime::iree_aie_runtime_static
)

target_include_directories(
  iree_target_amd-aie_Target_tests_aie_targets_test
  PRIVATE "${IREE_MLIR_AIE_SOURCE_DIR}/include"
)

iree_lit_test(
  NAME
    aie_targets_test_test
  TEST_FILE
    matmul_64x64_64xi8__dispatch_0_matmul_64x6_0.aiecc.mlir
  TOOLS
    ::aie_targets_test
    FileCheck
  LABELS
    "hostonly"
)

iree_lit_test_suite(
  NAME
    lit
  SRCS
    amd_aie_target_backend.mlir
  TOOLS
    ${IREE_LLD_TARGET}
    FileCheck
    iree-compile
)

add_subdirectory(cdo)
