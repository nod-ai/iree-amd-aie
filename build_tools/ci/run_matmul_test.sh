#!/bin/bash
#
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
#   3. Run: `./run_matmul_tests.sh <output_dir_path> <iree_install_path> [<mlir_aie_install_path>] [<peano_install_path>] [<xrt_path>] [<vitis_path>]`
#      The directories above in square brackets are optional, the first 2 directories are required.

set -euox pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 6 ]; then

   # The expected parameters are
   #    1) <output-dir>            (required)
   #    2) <iree-install-dir>      (required)
   #    3) <mlir-aie-install-dir>  (optional)
   #    4) <peano-install-dir>     (optional)
   #    5) <xrt-dir>               (optional)
   #    6) <vitis-install-dir>     (optional)
    echo -e "Illegal number of parameters: $#, expected 2,3,4,5, or 6 parameters." \
            "\n The parameters are as follows:" \
            "\n     1) <output-dir>               (required)" \
            "\n     2) <iree-install-dir>         (required)" \
            "\n     3) <mlir-aie-install-dir>     (optional)" \
            "\n     4) <peano-install-dir>        (optional)" \
            "\n     5) <xrt-dir>                  (optional)" \
            "\n     6) <vitis-install-dir>        (optional)" \
            "\n Example, dependent on environment variables:" \
            "\n     ./run_matmul_test.sh  " \
            "results_dir_tmp  \$IREE_INSTALL_DIR  \$MLIR_AIE_INSTALL_DIR  " \
            "\$PEANO_INSTALL_DIR  /opt/xilinx/xrt  \$VITIS_INSTALL_PATH"
    exit 1
fi



OUTPUT_DIR=`realpath "$1"`
mkdir -p ${OUTPUT_DIR}
if [ ! -d "${OUTPUT_DIR}" ]; then
  echo "Failed to locate on construct OUTPUT_DIR '${OUTPUT_DIR}'."
  exit 1
fi

IREE_INSTALL_DIR=`realpath "$2"`
if [ ! -d "${IREE_INSTALL_DIR}" ]; then
  echo "IREE_INSTALL_DIR must be a directory, '${IREE_INSTALL_DIR}' is not."
  exit 1
fi

# Search for iree-compile and iree-e2e-matmul-test in the user provided directory.
IREE_COMPILE_EXE=""
TEST_RUNNER=""
for dir in "${IREE_INSTALL_DIR}" "${IREE_INSTALL_DIR}/bin" "${IREE_INSTALL_DIR}/tools"; do
  if [ -f "${dir}/iree-compile" ]; then
    IREE_COMPILE_EXE="${dir}/iree-compile"
  fi
  if [ -f "${dir}/iree-e2e-matmul-test" ]; then
    TEST_RUNNER="${dir}/iree-e2e-matmul-test"
  fi
done

if [ -z "${IREE_COMPILE_EXE}" ]; then
  echo "No 'iree-compile' found in any of the following directories: " \
       "'${IREE_INSTALL_DIR}', '${IREE_INSTALL_DIR}/bin', '${IREE_INSTALL_DIR}/tools'."
  exit 1
fi
if [ -z "${TEST_RUNNER}" ]; then
  echo "No 'iree-e2e-matmul-test' found in any of the following directories: " \
       "'${IREE_INSTALL_DIR}', '${IREE_INSTALL_DIR}/bin', '${IREE_INSTALL_DIR}/tools'."
  exit 1
fi


# Parameter 3) <mlir-aie-install-dir>
if [ -z "${3-}" ]; then
  MLIR_AIE_INSTALL=`realpath .venv/lib/python3.10/site-packages/mlir_aie`
else
  MLIR_AIE_INSTALL=`realpath "$3"`
fi
if [ ! -d "${MLIR_AIE_INSTALL}" ]; then
  echo "No directory '${MLIR_AIE_INSTALL}' (argument 3) found."
  exit 1
fi

# Parameter 4) <peano-install-dir>
if [ -z "${4-}" ]; then
  PEANO=/opt/llvm-aie
else
  PEANO=`realpath "$4"`
fi
if [ ! -d "${PEANO}" ]; then
  echo "No directory '${PEANO}' (argument 4) found."
  exit 1
fi

# Parameter 5) <xrt-dir>
if [ -z "${5-}" ]; then
  XRT_DIR=/opt/xilinx/xrt
else
  XRT_DIR=`realpath "$5"`
fi
if [ ! -d "${XRT_DIR}" ]; then
  echo "No directory '${XRT_DIR}' (argument 5) found."
  exit 1
fi

# Parameter 6) <vitis-install-dir>
if [ -z "${6-}" ]; then
  # An alternate to a full vitis install, will work
  # here but not for a full build of mlir-aie
  # https://riallto.ai/install-riallto.html#install-riallto
  # VITIS=/opt/Riallto/Vitis/2023.1
  VITIS=/opt/Xilinx/Vitis/2023.2
else
  VITIS=`realpath "$6"`
fi
if [ ! -d "${VITIS}" ]; then
  echo "No directory '${VITIS}' (argument 6) found."
  exit 1
fi

THIS_DIR="$(cd $(dirname $0) && pwd)"
ROOT_DIR="$(cd $THIS_DIR/../.. && pwd)"

GENERATOR="${ROOT_DIR}/tests/matmul/generate_e2e_matmul_tests.py"
# Verify that generator exists
if [ ! -f "${GENERATOR}" ]; then
  echo "Generator script '${GENERATOR}' not found."
  exit 1
fi

IREE_PYTHON3_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE:-python3}"
if [ -z "$IREE_PYTHON3_EXECUTABLE" ]; then
  echo "IREE_PYTHON3_EXECUTABLE is not set."
  exit 1
else
  echo "Python version: $("${IREE_PYTHON3_EXECUTABLE}" --version)"
fi

source $XRT_DIR/setup.sh

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

  # Make a guess as to whether we need to sign the XCLBIN:
  SIGNER=${XRT_DIR}/amdxdna/setup_xclbin_firmware.sh
  # 1) check if $XRT_DIR/amdxdna/setup_xclbin_firmware.sh exists:
  if [ ! -f "$SIGNER" ]; then
    echo "**** Skipping XCLBIN signing: $SIGNER not found ****"
  else
    # Iterate over each function name and sign the corresponding XCLBIN
    for func_name in $function_names; do
      # Location of XCLBIN files
      XCLBIN_DIR="module_${func_name}_dispatch_0_amdaie_xclbin_fb"
      # Define the XCLBIN variable
      XCLBIN="module_${func_name}_dispatch_0_amdaie_xclbin_fb.xclbin"
      # Ensure unique file name
      echo "**** Getting unique id for XCLBIN ****"
      XCLBIN_UNIQ="github.${GITHUB_RUN_ID}.${GITHUB_RUN_ATTEMPT}.${XCLBIN}"
      cp "${XCLBIN_DIR}/${XCLBIN}" "${XCLBIN_DIR}/${XCLBIN_UNIQ}"
      # Deploy firmware
      sudo $SIGNER -dev Phoenix -xclbin "${XCLBIN_DIR}/${XCLBIN_UNIQ}"
    done
  fi

  echo "**** Running '${name}' matmul tests ****"

  COMMAND="${TEST_RUNNER} \
      --module=${OUTPUT_DIR}/${name}_matmuls.vmfb \
      --module=${OUTPUT_DIR}/${name}_calls.vmfb \
      --device=${device}"

  echo "Running command: ${COMMAND}"

  # Execute the command, and print the status:
  eval "${COMMAND}"
  echo "Command returned with status: $?"

  set +x
}

###############################################################################
# Run a few tests                                                             #
###############################################################################

run_matmul_test \
    --name "matmul_i32_i32_small_amd-aie_xrt_pad" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --shapes "small_legacy" \
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
    --shapes "large_legacy" \
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
    --shapes "small_legacy" \
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
    --shapes "large_legacy" \
    --target_backend "amd-aie" \
    --device "xrt" \
    --peano_install_path "${PEANO}" \
    --mlir_aie_install_path "${MLIR_AIE_INSTALL}" \
    --vitis_path  "${VITIS}" \
    --pipeline "simple-pack"

run_matmul_test \
    --name "matmul_bf16_bf16_large_amd-aie_xrt_simple-pack" \
    --lhs_rhs_type "bf16" \
    --acc_type "f32" \
    --shapes "large_legacy" \
    --target_backend "amd-aie" \
    --device "xrt" \
    --peano_install_path "${PEANO}" \
    --mlir_aie_install_path "${MLIR_AIE_INSTALL}" \
    --vitis_path  "${VITIS}" \
    --pipeline "simple-pack"

run_matmul_test \
    --name "matmul_i32_i32_small_amd-aie_xrt_pad-pack" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --shapes "small" \
    --target_backend "amd-aie" \
    --device "xrt" \
    --peano_install_path "${PEANO}" \
    --mlir_aie_install_path "${MLIR_AIE_INSTALL}" \
    --vitis_path  "${VITIS}" \
    --pipeline "pad-pack"

run_matmul_test \
    --name "matmul_i32_i32_large_amd-aie_xrt_pad-pack" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --shapes "large" \
    --target_backend "amd-aie" \
    --device "xrt" \
    --peano_install_path "${PEANO}" \
    --mlir_aie_install_path "${MLIR_AIE_INSTALL}" \
    --vitis_path  "${VITIS}" \
    --pipeline "pad-pack"
