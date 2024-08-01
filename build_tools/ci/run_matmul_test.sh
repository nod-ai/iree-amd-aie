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
#   3. Run: `./run_matmul_tests.sh <output_dir_path> <iree_install_path> [<peano_install_path>] [<xrt_path>] [<vitis_path>] [do_signing]`
#      The directories above in square brackets are optional, the first 2 directories are required.

set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 5 ]; then

   # The expected parameters are
   #    1) <output-dir>            (required)
   #    2) <iree-install-dir>      (required)
   #    4) <peano-install-dir>     (optional)
   #    5) <xrt-dir>               (optional)
   #    6) <vitis-install-dir>     (optional)
    echo -e "Illegal number of parameters: $#, expected 2-5 parameters." \
            "\n The parameters are as follows:" \
            "\n     1) <output-dir>               (required)" \
            "\n     2) <iree-install-dir>         (required)" \
            "\n     3) <peano-install-dir>        (optional)" \
            "\n     4) <xrt-dir>                  (optional)" \
            "\n     5) <vitis-install-dir>        (optional)" \
            "\n Example, dependent on environment variables:" \
            "\n     ./run_matmul_test.sh  " \
            "results_dir_tmp  \$IREE_INSTALL_DIR " \
            "\$PEANO_INSTALL_DIR  /opt/xilinx/xrt  \$VITIS_INSTALL_PATH"
    exit 1
fi

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
  VITIS=/opt/Xilinx/Vitis/2024.1
else
  VITIS=`realpath "$5"`
fi
if [ ! -d "${VITIS}" ]; then
  echo "No directory '${VITIS}' (argument 5) found."
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

GITHUB_ACTIONS="${GITHUB_ACTIONS:-false}"

source $XRT_DIR/setup.sh
# Circumvent xclbin security (no longer needed as of April 2024 XDNA driver)
export XRT_HACK_UNSECURE_LOADING_XCLBIN=1

cd ${OUTPUT_DIR}

export MATMUL_TESTS_FAILS=0

###############################################################################
# Define helper function                                                      #
###############################################################################

# This should be Python, CMake, or ... just... not Bash.
# Reference for named args: https://tecadmin.net/create-bash-functions-with-arguments/

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

  # name_prefix: A prefix for the name of the test. The full test name will be
  # extended with m,n,k if they are unique.
  local name_prefix="noprefix"

  local target_backend="amd-aie"

  local target_device="npu1_4col"

  local device="xrt"

  local peano_install_path="${PEANO}"

  local amd_aie_install_path="${IREE_INSTALL_DIR}"

  local vitis_path="${VITIS}"

  local use_chess="false"

  local tile_pipeline="pad-pack"

  # By default, the m,n,k provided are used, and there are no dynamic tensor
  # dimensions.
  local dynamicity="static"

  local accumulate="false"

  # The default is to not expect a compilation failure.
  local expect_compile_failure="0"

  local do_transpose_rhs="0"

  # The maximum number of elements to check for correctness.
  # See https://github.com/iree-org/iree/blob/tools/testing/e2e/test_utils.c#L40-L47
  local max_elements_to_check="20000"

  # The default is to not use microkernels.
  local use_ukernel="0"

  # After compilation, the test with be run 'num_repeat_runs' times. This option (when
  # set greater than 1) is useful for shapes which might be 'flakey' and fail
  # intermittently. It is also useful if a test is know to fail at runtime but
  # should still be checked to compile (set num_repeat_runs=0 in this case).
  local num_repeat_runs="1"

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --lower_to_aie_pipeline)
        lower_to_aie_pipeline="$2"
        shift 2
        ;;
      --num_repeat_runs)
        num_repeat_runs="$2"
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
      --expect_compile_failure)
        expect_compile_failure="$2"
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



  # Record the current time in milliseconds. Record the time at certain
  # checkpoints, and print statistics summarizing how much time is spent in
  # compilation and execution.
  start_time=$(date +%s%3N)


  # Generate a name for the test based on name_prefix and the matmul dimensions,
  # but only if the test has 1  matmul in it. If there are multiple matmuls,
  # then use name_prefix as is. This is to prevent long names when there are
  # many matmuls in a test.
  #
  # Generate a name, assuming m, n, k are just single integers:
  name="mm_${name_prefix}_${lhs_rhs_type}_${acc_type}_m${m}_n${n}_k${k}"

  # Disable exit on failure:
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

  # Re-enable exit on failure:
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


  ## Disable exit on failure:
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
                      --mlir-elide-resource-strings-if-larger=10 \
                      --iree-amd-aie-show-invoked-commands"

  if [ $use_ukernel -ne 0 ]; then

    compilation_flags="${compilation_flags} \
                        --iree-amdaie-enable-ukernels=all"
  fi

  echo "**** Generating matmul .vmfb file for ${name} ****"
  ${IREE_COMPILE_EXE} "${matmul_ir}" \
    ${compilation_flags} -o "${matmul_vmfb}"


  compileResult=$?


  # Handle cases other than when compilation is expected to, and does, succeed:
  if [ $expect_compile_failure -ne 0 ]; then
    if [ $compileResult -ne 0 ]; then
      echo "Expected compilation failure, got compilation failure."
      return 0
    else
      echo "Expected compilation failure, got compilation success."
      exit 1
    fi
  else
    if [ $compileResult -ne 0 ]; then
      echo "Expected compilation success, got compilation failure."
      exit 1
    fi
  fi

  # Renable exit on failure:
  set -e
  echo "**** Generating calls .vmfb file for ${name} ****"
  ${IREE_COMPILE_EXE} "${calls_ir}" \
      --iree-hal-target-backends=${target_backend} \
      -o "${calls_vmfb}"

  compiled_time=$(date +%s%3N)

  echo "**** Running '${name}' matmul tests ****"

  COMMAND="${TEST_RUNNER} \
      --module=${matmul_vmfb} \
      --module=${calls_vmfb} \
      --device=${device} \
      --max_elements_to_check=${max_elements_to_check}"

  set +e

  echo "**** Running '${name}' matmul test ${num_repeat_runs} times (command ${COMMAND}) ****"
  for i in $(seq 1 $num_repeat_runs); do
    # Only reset NPU in CI to facilitate easier local testing without sudo access.
    if [ "${GITHUB_ACTIONS}" = true ]; then
      echo "Reset NPU"
      bash $THIS_DIR/reset_npu.sh
    fi
    echo "Run number ${i} / ${num_repeat_runs}"
    eval "${COMMAND}"
    return_status=$?
    if [ $return_status -ne 0 ]; then
      echo "'${name}' matmul test failed!"
      export MATMUL_TESTS_FAILS=$(( $MATMUL_TESTS_FAILS+1 ))
    fi
  done
  set -e

  end_time=$(date +%s%3N)

  #print the time spent in each stage:
  echo "Time spent in generation: $((generated_time - start_time)) [ms]"
  echo "Time spent in compilation: $((compiled_time - generated_time)) [ms]"
  echo "Time spent in execution and verification: $((end_time - compiled_time)) [ms]"

}

# Helper function to run the same matmul test on an array of shapes with format 'MxKxN'.
function run_matmul_test_on_shapes() {
  shapes=()
  while [[ $1 != --* ]]
  do
    shapes+=($1)
    shift
  done
  for shape in "${shapes[@]}"
  do
    IFS="x" read -r -a elems <<< "${shape}"
    run_matmul_test \
        "$@" \
        --m "${elems[0]}" --k "${elems[1]}" --n "${elems[2]}"
  done
}

###################################################################
# ObjectFifo Matmul tests
###################################################################

i32_shapes_small=(
  '32x32x32'
  '64x32x128'
  '128x32x64'
  '128x32x64'
  '128x32x128'
  '256x32x256'
  '32x64x32'
  '64x64x64'
  '128x256x128'
)

i32_shapes_medium=(
  '1024x1024x1024' 
  '1536x2048x1536'
)

run_matmul_test_on_shapes ${i32_shapes_small[@]} \
    --name_prefix "small" \
    --lower_to_aie_pipeline "objectFifo" \
    --tile_pipeline "pack-peel" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --num_repeat_runs "100"

run_matmul_test_on_shapes ${i32_shapes_medium[@]} \
    --name_prefix "medium" \
    --lower_to_aie_pipeline "objectFifo" \
    --tile_pipeline "pack-peel" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --num_repeat_runs "100"

# # bf16 Matmul tests.

# bf16_i8_shapes_small=(
#   '64x64x64'
#   '128x256x128'
# )

# bf16_i8_shapes_medium=(
#   '1024x1024x1024'
#   '1536x2048x1536'
# )

# bf16_ukernel_shapes_small=(
#   '256x256x256'
# )

# run_matmul_test_on_shapes ${bf16_i8_shapes_small[@]} \
#     --name_prefix "small" \
#     --lower_to_aie_pipeline "objectFifo" \
#     --tile_pipeline "pack-peel" \
#     --lhs_rhs_type "bf16" \
#     --acc_type "f32" \
#     --num_repeat_runs "2"

# run_matmul_test_on_shapes ${bf16_i8_shapes_medium[@]} \
#     --name_prefix "medium" \
#     --lower_to_aie_pipeline "objectFifo" \
#     --tile_pipeline "pack-peel" \
#     --lhs_rhs_type "bf16" \
#     --acc_type "f32" \
#     --num_repeat_runs "2"

# # i8 Matmul tests.
# run_matmul_test_on_shapes ${bf16_i8_shapes_small[@]} \
#     --name_prefix "small" \
#     --lower_to_aie_pipeline "objectFifo" \
#     --tile_pipeline "pack-peel" \
#     --lhs_rhs_type "i8" \
#     --acc_type "i32" \
#     --num_repeat_runs "2"

# run_matmul_test_on_shapes ${bf16_i8_shapes_medium[@]} \
#     --name_prefix "medium" \
#     --lower_to_aie_pipeline "objectFifo" \
#     --tile_pipeline "pack-peel" \
#     --lhs_rhs_type "i8" \
#     --acc_type "i32" \
#     --num_repeat_runs "2"

# run_matmul_test_on_shapes ${bf16_ukernel_shapes_small[@]} \
#     --name_prefix "small" \
#     --lower_to_aie_pipeline "objectFifo" \
#     --tile_pipeline "pack-peel" \
#     --lhs_rhs_type "bf16" \
#     --acc_type "f32" \
#     --num_repeat_runs "2" \
#     --use_ukernel "1"

# ###################################################################
# # Chess tests
# ###################################################################

# run_matmul_test \
#     --name_prefix "chess_i32_matmul" \
#     --lhs_rhs_type "i32" \
#     --acc_type "i32" \
#     --m "32" \
#     --n "32" \
#     --k "32" \
#     --use_chess "1" \
#     --num_repeat_runs "10"

# run_matmul_test \
#     --name_prefix "chess_bf16_ukernel" \
#     --lhs_rhs_type "bf16" \
#     --acc_type "f32" \
#     --m "64" \
#     --n "64" \
#     --k "64" \
#     --use_chess "1" \
#     --num_repeat_runs "10" \
#     --use_ukernel "1"

if [ $MATMUL_TESTS_FAILS -ne 0 ]; then
  echo "$MATMUL_TESTS_FAILS matmul tests failed! Scroll up and look for the 🦄 and 🐞..."
  exit 1
fi
