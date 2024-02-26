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
#   3. Run: `./run_matmul_tests.sh <output_dir_path> <iree_install_path>`

set -euox pipefail

MLIR_AIE_INSTALL=`realpath .venv/lib/python3.10/site-packages/mlir_aie`
THIS_DIR="$(cd $(dirname $0) && pwd)"
ROOT_DIR="$(cd $THIS_DIR/../.. && pwd)"

OUTPUT_DIR=`realpath "$1"`
IREE_INSTALL_DIR="$2"
if [ -d "${IREE_INSTALL_DIR}/tools" ]; then
    IREE_INSTALL_TOOLS=`realpath "${IREE_INSTALL_DIR}/tools"`
fi
if [ -d "${IREE_INSTALL_DIR}/bin" ]; then
    IREE_INSTALL_BIN=`realpath "${IREE_INSTALL_DIR}/bin"`
fi

GENERATOR="${ROOT_DIR}/tests/matmul/generate_e2e_matmul_tests.py"
IREE_PYTHON3_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE:-python3}"

IREE_COMPILE_EXE="${IREE_INSTALL_BIN}/iree-compile"
TEST_RUNNER="${IREE_INSTALL_TOOLS}/iree-e2e-matmul-test"

XRT_DIR=/opt/xilinx/xrt
PEANO=/opt/llvm-aie
VITIS=/opt/Xilinx/Vitis/2023.2
# An alternate to a full vitis install, will work here but not for a full build of mlir-aie
# https://riallto.ai/install-riallto.html#install-riallto
# VITIS=/opt/Riallto/Vitis/2023.1

source $XRT_DIR/setup.sh

###############################################################################
# Setup and checking for dependencies                                         #
###############################################################################

echo "Python version: $("${IREE_PYTHON3_EXECUTABLE}" --version)"
echo "iree-compile version: $("${IREE_COMPILE_EXE}" --version)"
mkdir -p ${OUTPUT_DIR}
cd ${OUTPUT_DIR}

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
  local pipeline=""

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
      --pipeline)
        pipeline="$2"
        shift 2
        ;;
      *)
        echo "Unknown option: $1"
        return 1
        ;;
    esac
  done

  set -x

  echo "**** Generating .mlir files ****"
  ${IREE_PYTHON3_EXECUTABLE} ${GENERATOR} \
      --output_matmuls_mlir="${OUTPUT_DIR}/${name}_matmuls.mlir" \
      --output_calls_mlir="${OUTPUT_DIR}/${name}_calls.mlir" \
      --lhs_rhs_type=${lhs_rhs_type} \
      --acc_type=${acc_type} \
      --shapes=${shapes}

  echo "**** Generating .vmfb files for $pipeline pipeline ****"
  ${IREE_COMPILE_EXE} \
      "${OUTPUT_DIR}/${name}_matmuls.mlir" \
      --iree-hal-target-backends=${target_backend} \
      --iree-amdaie-use-pipeline=${pipeline} \
      --iree-amd-aie-peano-install-dir=${peano_install_path} \
      --iree-amd-aie-mlir-aie-install-dir=${mlir_aie_install_path} \
      --iree-amd-aie-vitis-install-dir=${vitis_path} \
      --iree-hal-dump-executable-files-to=$PWD \
      -o "${OUTPUT_DIR}/${name}_matmuls.vmfb"
  ${IREE_COMPILE_EXE} \
      "${OUTPUT_DIR}/${name}_calls.mlir" \
      --iree-hal-target-backends=${target_backend} \
      -o "${OUTPUT_DIR}/${name}_calls.vmfb"

  # Extract function names from the mlir file
  function_names=$(grep -oP '@\K\S+(?=\()' ${OUTPUT_DIR}/${name}_matmuls.mlir)

  # Iterate over each function name and sign the corresponding XCLBIN
  for func_name in $function_names; do
    # Location of XCLBIN files
    XCLBIN_DIR="module_${func_name}_dispatch_0_amdaie_xclbin_fb"
    # Define the XCLBIN variable
    XCLBIN="module_${func_name}_dispatch_0_amdaie_xclbin_fb.xclbin"
    # Ensure unique file name
    XCLBIN_UNIQ="github.${GITHUB_RUN_ID}.${GITHUB_RUN_ATTEMPT}.${XCLBIN}"
    cp "${XCLBIN_DIR}/${XCLBIN}" "${XCLBIN_DIR}/${XCLBIN_UNIQ}"
    # Deploy firmware
    sudo $XRT_DIR/amdxdna/setup_xclbin_firmware.sh -dev Phoenix -xclbin "${XCLBIN_DIR}/${XCLBIN_UNIQ}"
  done

  echo "**** Running '${name}' matmul tests ****"
  echo ""

  ${TEST_RUNNER} \
      --module="${OUTPUT_DIR}/${name}_matmuls.vmfb" \
      --module="${OUTPUT_DIR}/${name}_calls.vmfb" \
      --device=${device}

  set +x
}

###############################################################################
# Run a few tests                                                             #
###############################################################################

run_matmul_test \
    --name "matmul_i32_i32_small_amd-aie_xrt_pad" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --shapes "small" \
    --target_backend "amd-aie" \
    --device "xrt" \
    --peano_install_path "${PEANO}" \
    --mlir_aie_install_path "${MLIR_AIE_INSTALL}" \
    --vitis_path  "${VITIS}" \
    --pipeline "pad"

run_matmul_test \
    --name "matmul_i32_i32_large_amd-aie_xrt_pad" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --shapes "large" \
    --target_backend "amd-aie" \
    --device "xrt" \
    --peano_install_path "${PEANO}" \
    --mlir_aie_install_path "${MLIR_AIE_INSTALL}" \
    --vitis_path  "${VITIS}" \
    --pipeline "pad"

run_matmul_test \
    --name "matmul_i32_i32_small_amd-aie_xrt_simple-pack" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --shapes "small" \
    --target_backend "amd-aie" \
    --device "xrt" \
    --peano_install_path "${PEANO}" \
    --mlir_aie_install_path "${MLIR_AIE_INSTALL}" \
    --vitis_path  "${VITIS}" \
    --pipeline "simple-pack"

run_matmul_test \
    --name "matmul_i32_i32_large_amd-aie_xrt_simple-pack" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --shapes "large" \
    --target_backend "amd-aie" \
    --device "xrt" \
    --peano_install_path "${PEANO}" \
    --mlir_aie_install_path "${MLIR_AIE_INSTALL}" \
    --vitis_path  "${VITIS}" \
    --pipeline "simple-pack"
