#!/bin/bash
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Manual IREE matmul test script.
#
# This pulls code from
#   * https://github.com/openxla/iree/blob/main/build_tools/cmake/iree_e2e_matmul_test.cmake
#   * https://github.com/openxla/iree/blob/main/tests/e2e/matmul/CMakeLists.txt
#
# Usage:
#   1. Build IREE (or install packages). You'll need `iree-compile` to include
#      support for the compiler target backend you wish to test and
#      `iree-e2e-matmul-test` to include support for the runtime HAL
#      driver/device you wish to test.
#   2. Update the paths in this script or specify them via environment variables
#   3. Run: `./run_matmul_demo.sh <peano_install_path> <mlir_aie_install_path> <vitis_path>`

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

set -uo pipefail

THIS_DIR="$(cd $(dirname $0) && pwd)"
ROOT_DIR="$(cd $THIS_DIR/../.. && pwd)"

OUTPUT_DIR="${IREE_MATMUL_BUILD_DIR:-build-matmul}"
GENERATOR="${ROOT_DIR}/tests/matmul/generate_e2e_matmul_demos.py"
IREE_PYTHON3_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE:-python3}"

IREE_INSTALL_BIN="${IREE_INSTALL_BIN:-${ROOT_DIR}/../iree-build/tools/}"
IREE_COMPILE_EXE="${IREE_INSTALL_BIN}/iree-compile"
TEST_RUNNER="${IREE_INSTALL_BIN}/iree-e2e-matmul-test"

###############################################################################
# Setup and checking for dependencies                                         #
###############################################################################

echo "Python version: $("${IREE_PYTHON3_EXECUTABLE}" --version)"
echo "iree-compile version: $("${IREE_COMPILE_EXE}" --version)"
mkdir -p ${OUTPUT_DIR}

###############################################################################
# Define helper function                                                      #
###############################################################################

# This should be Python, CMake, or ... just... not Bash.
# Reference for named args: https://tecadmin.net/create-bash-functions-with-arguments/

function run_matmul_test() {
  local name=""
  local lhs_rhs_type=""
  local acc_type=""
  local shapes=""
  local target_backend=""
  local device=""
  local peano_install_path=""
  local mlir_aie_install_path=""
  local vitis_path=""

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --name)
        name="$2"
        shift 2
        ;;
      --lhs_rhs_type)
        lhs_rhs_type="$2"
        shift 2
        ;;
      --acc_type)
        acc_type="$2"
        shift 2
        ;;
      --shapes)
        shapes="$2"
        shift 2
        ;;
      --target_backend)
        target_backend="$2"
        shift 2
        ;;
      --device)
        device="$2"
        shift 2
        ;;
      --peano_install_path)
        peano_install_path="$2"
        shift 2
        ;;
      --mlir_aie_install_path)
        mlir_aie_install_path="$2"
        shift 2
        ;;
     --vitis_path)
        vitis_path="$2"
        shift 2
        ;;
      *)
        echo "Unknown option: $1"
        return 1
        ;;
    esac
  done

  echo "**** Generating .mlir files ****"
  ${IREE_PYTHON3_EXECUTABLE} ${GENERATOR} \
      --output_matmuls_mlir="${OUTPUT_DIR}/${name}_matmuls.mlir" \
      --output_calls_mlir="${OUTPUT_DIR}/${name}_calls.mlir" \
      --lhs_rhs_type=${lhs_rhs_type} \
      --acc_type=${acc_type} \
      --shapes=${shapes}

  echo "**** Generating .vmfb files ****"
  ${IREE_COMPILE_EXE} \
      "${OUTPUT_DIR}/${name}_matmuls.mlir" \
      --iree-hal-target-backends=${target_backend} \
      --iree-amd-aie-peano-install-dir=${peano_install_path} \
      --iree-amd-aie-mlir-aie-install-dir=${mlir_aie_install_path} \
      --iree-amd-aie-vitis-install-dir=${vitis_path} \
      -o "${OUTPUT_DIR}/${name}_matmuls.vmfb" &> ${OUTPUT_DIR}/${name}_matmuls_logs.txt
  
  # Capture the exit code
  exit_code=$?

  # Check the exit code and print pass or fail
  if [ $exit_code -eq 0 ]; then
    printf "${GREEN}Compiler Passed${NC}\n"
  else
    printf "${RED}Failed in compiler (Exit code: $exit_code)${NC}\n"
  fi
  ${IREE_COMPILE_EXE} \
      "${OUTPUT_DIR}/${name}_calls.mlir" \
      --iree-hal-target-backends=${target_backend} \
      -o "${OUTPUT_DIR}/${name}_calls.vmfb"

  echo "**** Running '${name}' matmul tests ****"
  echo ""

  ${TEST_RUNNER} \
      --module="${OUTPUT_DIR}/${name}_matmuls.vmfb" \
      --module="${OUTPUT_DIR}/${name}_calls.vmfb" \
      --device=${device}

# Capture the exit code
exit_code=$?

# Check the exit code and print pass or fail
if [ $exit_code -eq 0 ]; then
    printf "${GREEN}Runtime Passed${NC}\n"
else
    printf "${RED}Failed in runtime (Exit code: $exit_code)${NC}\n"
fi

}

###############################################################################
# Run a few tests                                                             #
###############################################################################

run_matmul_test \
    --name "matmul_i32_i32_small1_amd-aie_xrt" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --shapes "small1" \
    --target_backend "amd-aie" \
    --device "xrt" \
    --peano_install_path "$1" \
    --mlir_aie_install_path "$2" \
    --vitis_path  "$3"

run_matmul_test \
    --name "matmul_i32_i32_small2_amd-aie_xrt" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --shapes "small2" \
    --target_backend "amd-aie" \
    --device "xrt" \
    --peano_install_path "$1" \
    --mlir_aie_install_path "$2" \
    --vitis_path  "$3"

run_matmul_test \
    --name "matmul_i32_i32_small3_amd-aie_xrt" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --shapes "small3" \
    --target_backend "amd-aie" \
    --device "xrt" \
    --peano_install_path "$1" \
    --mlir_aie_install_path "$2" \
    --vitis_path  "$3"

run_matmul_test \
    --name "matmul_i32_i32_large1_amd-aie_xrt" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --shapes "large1" \
    --target_backend "amd-aie" \
    --device "xrt" \
    --peano_install_path "$1" \
    --mlir_aie_install_path "$2" \
    --vitis_path  "$3"

run_matmul_test \
    --name "matmul_i32_i32_large2_amd-aie_xrt" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --shapes "large2" \
    --target_backend "amd-aie" \
    --device "xrt" \
    --peano_install_path "$1" \
    --mlir_aie_install_path "$2" \
    --vitis_path  "$3"

run_matmul_test \
    --name "matmul_i32_i32_large3_amd-aie_xrt" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --shapes "large3" \
    --target_backend "amd-aie" \
    --device "xrt" \
    --peano_install_path "$1" \
    --mlir_aie_install_path "$2" \
    --vitis_path  "$3"
