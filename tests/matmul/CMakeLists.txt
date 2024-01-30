# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

message(STATUS "IREE_EXTERNAL_TESTS::tests/matmul/CMakeLists.txt")

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: '${CMAKE_CURRENT_SOURCE_DIR}'")

iree_generated_e2e_matmul_test(
  NAME
    amdaie_e2e_matmul_dt_i32_i32_small
  GENERATOR
    "generate_e2e_matmul_tests.py"
    # "${CMAKE_CURRENT_SOURCE_DIR}/generate_e2e_matmul_tests.py"
    # "${IREE_EXTERNAL_TESTS_SOURCE_DIR}/tests/matmul/generate_e2e_matmul_tests.py"
  GENERATOR_ARGS
    "--lhs_rhs_type=i32"
    "--acc_type=i32"
    "--shapes=small"
  TEST_RUNNER
    iree-e2e-matmul-test
  TARGET_BACKENDS
    "amd-aie"
  DRIVERS
    "xrt"
  COMPILER_FLAGS
    "--iree-amd-aie-peano-install-dir=${PEANO_PATH}"
    "--iree-amd-aie-mlir-aie-install-dir=${MLIRAIE_PATH}"
    "--iree-amd-aie-vitis-install-dir=${VITIS_PATH}"
  LABELS
)

#iree_generated_e2e_matmul_test(
#  NAME
#    e2e_matmul_dt_i32_i32_large
#  GENERATOR
#    "generate_e2e_matmul_tests.py"
#    # "${CMAKE_CURRENT_SOURCE_DIR}/generate_e2e_matmul_tests.py"
#    # "${IREE_EXTERNAL_TESTS_SOURCE_DIR}/tests/matmul/generate_e2e_matmul_tests.py"
#  GENERATOR_ARGS
#    "--lhs_rhs_type=i32"
#    "--acc_type=i32"
#    "--shapes=large"
#  TEST_RUNNER
#    iree-e2e-matmul-test
#  TARGET_BACKENDS
#    "amd-aie"
#  DRIVERS
#    "xrt"
#  COMPILER_FLAGS
#    "--iree-amd-aie-peano-install-dir=${PEANO_PATH}"
#    "--iree-amd-aie-mlir-aie-install-dir=${MLIRAIE_PATH}"
#    "--iree-amd-aie-vitis-install-dir=${VITIS_PATH}"
#  LABELS
#)
