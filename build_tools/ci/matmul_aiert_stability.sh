#!/bin/bash
#
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

OUTPUT_DIR=`realpath "$1"`
if [ -d "${OUTPUT_DIR}" ]; then
  rm -rf "${OUTPUT_DIR}";
fi
mkdir -p ${OUTPUT_DIR}
if [ ! -d "${OUTPUT_DIR}" ]; then
  echo "Failed to locate or construct OUTPUT_DIR '${OUTPUT_DIR}'."
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
  if [ -f "${dir}/testing/e2e/iree-e2e-matmul-test" ]; then
    TEST_RUNNER="${dir}/testing/e2e/iree-e2e-matmul-test"
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

# Parameter 3) <peano-install-dir>
if [ -z "${3-}" ]; then
  PEANO=/opt/llvm-aie
else
  PEANO=`realpath "$3"`
fi
if [ ! -d "${PEANO}" ]; then
  echo "No directory '${PEANO}' (argument 3) found."
  exit 1
fi

# Parameter 4) <xrt-dir>
if [ -z "${4-}" ]; then
  XRT_DIR=/opt/xilinx/xrt
else
  XRT_DIR=`realpath "$4"`
fi
if [ ! -d "${XRT_DIR}" ]; then
  echo "No directory '${XRT_DIR}' (argument 4) found."
  exit 1
fi

# Parameter 5) <vitis-install-dir>
if [ -z "${5-}" ]; then
  VITIS=/opt/Xilinx/Vitis/2024.2
else
  VITIS=`realpath "$5"`
fi
if [ ! -d "${VITIS}" ]; then
  echo "No directory '${VITIS}' (argument 5) found."
  exit 1
fi

THIS_DIR="$(cd $(dirname $0) && pwd)"
ROOT_DIR="$(cd $THIS_DIR/../.. && pwd)"

GENERATOR="${THIS_DIR}/generate_e2e_matmul_tests.py"
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
# Circumvent xclbin security (no longer needed as of April 2024 XDNA driver)
export XRT_HACK_UNSECURE_LOADING_XCLBIN=1

ME_BASIC_URL=https://github.com/nod-ai/iree-amd-aie/releases/download/ukernels/me_basic.o

if [ -d "$PEANO" ]; then
  PEANO_ME_BASIC_FP="$PEANO/lib/aie2-none-unknown-elf/me_basic.o"
  if [ -f "$PEANO_ME_BASIC_FP" ]; then
    echo "File 'me_basic.o' already exists at $PEANO_ME_BASIC_FP"
  else
    echo "Downloading 'me_basic.o' to $PEANO_ME_BASIC_FP"
    wget $ME_BASIC_URL -O "$PEANO_ME_BASIC_FP"
  fi
else
  echo "Peano install not found at $PEANO; not downloading me_basic."
fi

AIERT_COMMIT=$(git submodule status $ROOT_DIR/third_party/aie-rt | cut -d' ' -f2)

if [ -x "${IREE_INSTALL_DIR}/bin/FileCheck" ]; then
  FILECHECK_EXE="${IREE_INSTALL_DIR}/bin/FileCheck"
elif [ -x "$(command -v FileCheck)" ]; then
  FILECHECK_EXE="$(command -v FileCheck)"
else
  echo "FileCheck does not exist or isn't executable in ${IREE_INSTALL_DIR}/bin or on PATH."
  exit 1
fi

VERBOSE=${VERBOSE:-0}
DODIFF=${DODIFF:-0}
GOLDEN_DIR=${GOLDEN_DIR:-$THIS_DIR/golden}

cd ${OUTPUT_DIR}

function run_matmul_test() {

  # Options without defaults
  # ========================
  local lhs_rhs_type=""
  local acc_type=""
  local m=""
  local n=""
  local k=""

  # Options with defaults
  # =====================
  local lower_to_aie_pipeline="air"

  local name_prefix="noprefix"

  local target_backend="amd-aie"

  local target_device="npu1_4col"

  local device="xrt"

  local peano_install_path="${PEANO}"

  local amd_aie_install_path="${IREE_INSTALL_DIR}"

  local vitis_path="${VITIS}"

  local use_chess="false"

  local tile_pipeline="pad-pack"

  local dynamicity="static"

  local accumulate="false"

  local do_transpose_rhs="0"

  local max_elements_to_check="20000"

  # The default is to not use microkernels.
  local use_ukernel="0"

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --lower_to_aie_pipeline)
        lower_to_aie_pipeline="$2"
        shift 2
        ;;
      --max_elements_to_check)
        max_elements_to_check="$2"
        shift 2
        ;;
      --do_transpose_rhs)
        do_transpose_rhs="$2"
        shift 2
        ;;
      --name_prefix)
        name_prefix="$2"
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
      --use_ukernel)
        use_ukernel="$2"
        shift 2
        ;;
      --target_device)
        target_device="$2"
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
      --amd_aie_install_path)
        amd_aie_install_path="$2"
        shift 2
        ;;
      --use_chess)
        use_chess="$2"
        shift 2
        ;;
     --vitis_path)
        vitis_path="$2"
        shift 2
        ;;
      --tile_pipeline)
        tile_pipeline="$2"
        shift 2
        ;;
      --dynamicity)
        dynamicity="$2"
        shift 2
        ;;
      --accumulate)
        accumulate="$2"
        shift 2
        ;;
      --m)
        m="$2"
        shift 2
        ;;
      --n)
        n="$2"
        shift 2
        ;;
      --k)
        k="$2"
        shift 2
        ;;
      *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
  done

  start_time=$(date +%s%3N)

  name="mm_${name_prefix}_${lhs_rhs_type}_${acc_type}_m${m}_n${n}_k${k}"

  set +e
  # Check if the name contains a ',' or ' ', which indicates multiple matmuls.
  nameContainsComma=$(echo $name | grep -c ",")
  nameContainsSpace=$(echo $name | grep -c " ")
  if [ $nameContainsComma -ne 0 ] || [ $nameContainsSpace -ne 0 ]; then
    name="${name_prefix}"
  fi

  # Confirm that the name does not contain a comma or space, now that just
  # the name_prefix is used.
  nameContainsComma=$(echo $name | grep -c ",")
  nameContainsSpace=$(echo $name | grep -c " ")
  if [ $nameContainsComma -ne 0 ] || [ $nameContainsSpace -ne 0 ]; then
    echo "Name contains a comma or space: not allowed."
    exit 1
  fi
  set -e

  matmul_ir="${OUTPUT_DIR}/${name}_ir.mlir"
  calls_ir="${OUTPUT_DIR}/${name}_calls.mlir"
  matmul_vmfb="${OUTPUT_DIR}/${name}.vmfb"
  calls_vmfb="${OUTPUT_DIR}/${name}_calls.vmfb"

  generation_flags="--lhs_rhs_type=${lhs_rhs_type} \
                    --acc_type=${acc_type} \
                    --m=${m} \
                    --n=${n} \
                    --k=${k} \
                    --dynamicity=${dynamicity} \
                    --accumulate=${accumulate} \
                    --output_matmuls_mlir=${matmul_ir} \
                    --output_calls_mlir=${calls_ir}"

  if [ $do_transpose_rhs -ne 0 ]; then
    generation_flags="${generation_flags} --transpose_rhs"
  fi

  echo "**** Generating .mlir file containing matmul function(s) ****"
  ${IREE_PYTHON3_EXECUTABLE} ${GENERATOR} ${generation_flags}

  generated_time=$(date +%s%3N)

  set +e
  compilation_flags="--iree-hal-target-backends=${target_backend} \
                      --iree-amdaie-target-device=${target_device} \
                      --iree-amdaie-lower-to-aie-pipeline=${lower_to_aie_pipeline} \
                      --iree-amdaie-tile-pipeline=${tile_pipeline} \
                      --iree-amd-aie-peano-install-dir=${peano_install_path} \
                      --iree-amd-aie-install-dir=${amd_aie_install_path} \
                      --iree-amd-aie-vitis-install-dir=${vitis_path} \
                      --iree-amd-aie-enable-chess=${use_chess} \
                      --iree-hal-dump-executable-files-to=$PWD \
                      -debug-only=aie-generate-cdo"

  if [ $use_ukernel -ne 0 ]; then
    compilation_flags="${compilation_flags} \
                        --iree-amdaie-enable-ukernels=all"
  fi
  if [ $VERBOSE -ne 0 ]; then
    compilation_flags="${compilation_flags} \
                        --iree-amd-aie-show-invoked-commands"
  fi

  echo "**** Generating matmul .aiert.log file for ${name} ****"
  echo "aie-rt commit: $AIERT_COMMIT" > "${OUTPUT_DIR}/${name}.aiert.log"
  # change to tee >(grep -E "XAIE|cdo-driver" >> "${OUTPUT_DIR}/${name}.aiert.log") if you want to see
  ${IREE_COMPILE_EXE} "${matmul_ir}" ${compilation_flags} -o "${matmul_vmfb}" \
    2>&1 | grep -E "XAIE|cdo-driver" >> "${OUTPUT_DIR}/${name}.aiert.log"

  compileResult=$?
  if [ $compileResult -ne 0 ]; then
    echo "Expected compilation success, got compilation failure."
    exit 1
  fi
  set -e

  if [ $DODIFF -ne 0 ]; then
    diff "${GOLDEN_DIR}/${name}.aiert.log" "${OUTPUT_DIR}/${name}.aiert.log"
  fi
}

########################################################
# Run tests                                            #
########################################################

run_matmul_test \
  --name_prefix "transpose_i8_i32" \
  --lhs_rhs_type "i8" \
  --acc_type "i32" \
  --m "16" --n "32" --k "64" \
  --do_transpose_rhs "1"

run_matmul_test \
  --name_prefix "transpose_bf16" \
  --lhs_rhs_type "bf16" \
  --acc_type "f32" \
  --m "256" --n "256" --k "256" \
  --do_transpose_rhs "1"

run_matmul_test \
    --name_prefix "multiple_matmuls" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --m "512,8,16" \
    --n "512,32,16" \
    --k "256,16,8"

run_matmul_test \
    --name_prefix "small" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --m "16"  --n "16" --k "8"

run_matmul_test \
    --name_prefix "small" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --m "8"  --n "32" --k "16"

run_matmul_test \
    --name_prefix "large" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --m "64"  --n "64" --k "128"

run_matmul_test \
    --name_prefix "large" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --m "512"  --n "512" --k "512"

run_matmul_test \
    --name_prefix "int8" \
    --lhs_rhs_type "i8" \
    --acc_type "i32" \
    --m "64"  --n "64" --k "64"

run_matmul_test \
    --name_prefix "bf16_2304" \
    --lhs_rhs_type "bf16" \
    --acc_type "f32" \
    --m "128"  --n "128" --k "2304"

run_matmul_test \
    --name_prefix "packPeel" \
    --tile_pipeline "pack-peel" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --m "64"  --n "64" --k "128"

run_matmul_test \
    --name_prefix "packPeel" \
    --tile_pipeline "pack-peel" \
    --lhs_rhs_type "bf16" \
    --acc_type "f32" \
    --m "64"  --n "64" --k "128"

run_matmul_test \
    --name_prefix "packPeelLarge" \
    --tile_pipeline "pack-peel" \
    --lhs_rhs_type "bf16" \
    --acc_type "f32" \
    --m "512"  --n "512" --k "512"

run_matmul_test \
    --name_prefix "packPeel2304" \
    --tile_pipeline "pack-peel" \
    --lhs_rhs_type "bf16" \
    --acc_type "f32" \
    --m "128"  --n "128" --k "2304"


run_matmul_test \
  --name_prefix "packPeel_t_bf16" \
  --tile_pipeline "pack-peel" \
  --lhs_rhs_type "bf16" \
  --acc_type "f32" \
  --m "128" --n "256" --k "512" \
  --do_transpose_rhs "1"

###################################################################

run_matmul_test \
    --name_prefix "mm2" \
    --lhs_rhs_type "bf16" \
    --acc_type "f32" \
    --m "308"  --k "9728" --n "2432"

run_matmul_test \
    --name_prefix "mm3" \
    --lhs_rhs_type "bf16" \
    --acc_type "f32" \
    --m "308"  --k "2432" --n "2432"

run_matmul_test \
     --name_prefix "mm4" \
     --lhs_rhs_type "bf16" \
     --acc_type "f32" \
     --m "308"  --k "2432" --n "7296"

run_matmul_test \
     --name_prefix "mm5" \
     --lhs_rhs_type "bf16" \
     --acc_type "f32" \
     --m "8192" --k "2432" --n "9728"

run_matmul_test \
    --name_prefix "mm6" \
    --lhs_rhs_type "bf16" \
    --acc_type "f32" \
    --m "308"  --k "2432" --n "9728"

run_matmul_test \
    --name_prefix "mm7" \
    --lhs_rhs_type "bf16" \
    --acc_type "f32" \
    --m "8192" --k "2432" --n "2432"

run_matmul_test \
     --name_prefix "mm8" \
     --lhs_rhs_type "bf16" \
     --acc_type "f32" \
     --m "8192" --k "9728" --n "2432"

run_matmul_test \
    --name_prefix "mm9" \
    --lhs_rhs_type "bf16" \
    --acc_type "f32" \
    --m "8192" --k "2432" --n "7296"

###################################################################
# ObjectFifo Matmul tests
###################################################################

run_matmul_test \
    --name_prefix "small" \
    --lower_to_aie_pipeline "objectFifo" \
    --tile_pipeline "pack-peel" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --m "32" --k "32" --n "32"

run_matmul_test \
    --name_prefix "small" \
    --lower_to_aie_pipeline "objectFifo" \
    --tile_pipeline "pack-peel" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --m "64" --k "32" --n "128"

run_matmul_test \
    --name_prefix "small" \
    --lower_to_aie_pipeline "objectFifo" \
    --tile_pipeline "pack-peel" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --m "128" --k "32" --n "64"

run_matmul_test \
    --name_prefix "small" \
    --lower_to_aie_pipeline "objectFifo" \
    --tile_pipeline "pack-peel" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --m "128" --k "32" --n "128"

run_matmul_test \
    --name_prefix "small" \
    --lower_to_aie_pipeline "objectFifo" \
    --tile_pipeline "pack-peel" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --m "256" --k "32" --n "256"

run_matmul_test \
    --name_prefix "small" \
    --lower_to_aie_pipeline "objectFifo" \
    --tile_pipeline "pack-peel" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --m "128" --k "256" --n "128"

run_matmul_test \
    --name_prefix "medium" \
    --lower_to_aie_pipeline "objectFifo" \
    --tile_pipeline "pack-peel" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --m "1024" --k "1024" --n "1024"

run_matmul_test \
    --name_prefix "medium" \
    --lower_to_aie_pipeline "objectFifo" \
    --tile_pipeline "pack-peel" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --m "1536" --k "2048" --n "1536"
