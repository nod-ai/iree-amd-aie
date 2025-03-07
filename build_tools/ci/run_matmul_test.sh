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
#   3. Run: `./run_matmul_tests.sh <output_dir_path> <iree_install_path> [<peano_install_path>] [<xrt_path>]`
#      The directories above in square brackets are optional, the first 2 directories are required.

set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then

   # The expected parameters are
   #    1) <output-dir>            (required)
   #    2) <iree-install-dir>      (required)
   #    3) <peano-install-dir>     (optional)
   #    4) <xrt-dir>               (optional)
    echo -e "Illegal number of parameters: $#, expected 2-5 parameters." \
            "\n The parameters are as follows:" \
            "\n     1) <output-dir>               (required)" \
            "\n     2) <iree-install-dir>         (required)" \
            "\n     3) <peano-install-dir>        (optional)" \
            "\n     4) <xrt-dir>                  (optional)" \
            "\n Example, dependent on environment variables:" \
            "\n     ./run_matmul_test.sh  " \
            "results_dir_tmp  \$IREE_INSTALL_DIR " \
            "\$PEANO_INSTALL_DIR  /opt/xilinx/xrt "
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
for dir in "${IREE_INSTALL_DIR}" "${IREE_INSTALL_DIR}/bin" "${IREE_INSTALL_DIR}/tools" "${IREE_INSTALL_DIR}/tools/testing/e2e"; do
  echo "Looking in $dir"
  if [ -f "${dir}/iree-compile" ]; then
    IREE_COMPILE_EXE="${dir}/iree-compile"
  fi
  if [ -f "${dir}/iree-compile.exe" ]; then
    IREE_COMPILE_EXE="${dir}/iree-compile.exe"
  fi
  if [ -f "${dir}/iree-e2e-matmul-test" ]; then
    TEST_RUNNER="${dir}/iree-e2e-matmul-test"
  fi
  if [ -f "${dir}/iree-e2e-matmul-test.exe" ]; then
    TEST_RUNNER="${dir}/iree-e2e-matmul-test.exe"
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
if [ -z "${5-}" ]; then
  XRT_DIR=/opt/xilinx/xrt
else
  XRT_DIR=`realpath "$4"`
fi
if [ -f "$XRT_DIR/setup.sh" ]; then
  source $XRT_DIR/setup.sh
fi


THIS_DIR="$(cd $(dirname $0) && pwd)"
ROOT_DIR="$(cd $THIS_DIR/../.. && pwd)"

GENERATOR="${THIS_DIR}/generate_e2e_matmul_tests.py"
# Verify that generator exists
if [ ! -f "${GENERATOR}" ]; then
  echo "Generator script '${GENERATOR}' not found."
  exit 1
fi

IREE_PYTHON3_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE:-python}"
if [ -z "$IREE_PYTHON3_EXECUTABLE" ]; then
  echo "IREE_PYTHON3_EXECUTABLE is not set."
  exit 1
else
  echo "Python version: $("${IREE_PYTHON3_EXECUTABLE}" --version)"
fi

GITHUB_ACTIONS="${GITHUB_ACTIONS:-false}"

# Circumvent xclbin security (no longer needed as of April 2024 XDNA driver)
export XRT_HACK_UNSECURE_LOADING_XCLBIN=1

cd ${OUTPUT_DIR}

export MATMUL_TESTS_RUN=0
export MATMUL_TESTS_FAILS=0

DEVICE_HAL="${DEVICE_HAL:-xrt-lite}"

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
  local lower_to_aie_pipeline="objectFifo"

  # name_prefix: A prefix for the name of the test. The full test name will be
  # extended with m,n,k if they are unique.
  local name_prefix="noprefix"

  local target_backend="amd-aie"

  local target_device="npu1_4col"

  local peano_install_path="${PEANO}"

  local amd_aie_install_path="${IREE_INSTALL_DIR}"

  local tile_pipeline="pack-peel"

  # By default, the m,n,k provided are used, and there are no dynamic tensor
  # dimensions.
  local dynamicity="static"

  local accumulate="false"

  local enable_packet_flow="false";

  # The default is to not expect a compilation failure.
  local expect_compile_failure="0"

  local do_transpose_rhs="0"

  # The maximum number of elements to check for correctness.
  # See https://github.com/iree-org/iree/blob/tools/testing/e2e/test_utils.c#L40-L47
  local max_elements_to_check="20000"

  # After compilation, the test with be run 'num_repeat_runs' times. This option (when
  # set greater than 1) is useful for shapes which might be 'flakey' and fail
  # intermittently. It is also useful if a test is know to fail at runtime but
  # should still be checked to compile (set num_repeat_runs=0 in this case).
  local num_repeat_runs="1"

  # Run the test 'num_corruption_repeat_runs' times without an NPU reset in
  # between. This can be used to check for corruption, i.e. the AIE might be
  # left in a bad state in between runs. Additionally, this increases the speed
  # of the repeated test
  local num_corruption_repeat_runs="1"

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
      --num_corruption_repeat_runs)
        num_corruption_repeat_runs="$2"
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
      --enable_packet_flow)
        enable_packet_flow="$2"
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
      --target_device)
        target_device="$2"
        shift 2
        ;;
      --target_backend)
        target_backend="$2"
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

  export MATMUL_TESTS_RUN=$(( $MATMUL_TESTS_RUN+1 ))

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
                      --iree-amdaie-enable-input-packet-flow=${enable_packet_flow} \
                      --iree-hal-dump-executable-files-to=$PWD \
                      --iree-amdaie-device-hal=${DEVICE_HAL} \
                      --iree-hal-memoization=false \
                      --iree-hal-indirect-command-buffers=false \
                      --mlir-elide-resource-strings-if-larger=10 \
                      --iree-amd-aie-show-invoked-commands"


  set +e

  echo "**** Generating matmul .vmfb file for ${name} ****"
  ${IREE_COMPILE_EXE} "${matmul_ir}" \
    ${compilation_flags} --compile-to=vm -o "${matmul_vmfb}.vm"
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
      export MATMUL_TESTS_FAILS=$(( $MATMUL_TESTS_FAILS+1 ))
      return
    fi
  else
    if [ $compileResult -ne 0 ]; then
      echo "Expected compilation success, got compilation failure."
      export MATMUL_TESTS_FAILS=$(( $MATMUL_TESTS_FAILS+1 ))
      return
    fi
  fi

  # Re-enable exit on failure:
  echo "**** Generating calls .vmfb file for ${name} ****"
  ${IREE_COMPILE_EXE} "${calls_ir}" \
      --iree-hal-target-backends=${target_backend} \
      -o "${calls_vmfb}"

  return_status=$?
  if [ $return_status -ne 0 ]; then
    echo "'${name}' matmul compile failed!"
    export MATMUL_TESTS_FAILS=$(( $MATMUL_TESTS_FAILS+1 ))
    return
  fi

  compiled_time=$(date +%s%3N)

  echo "**** Running '${name}' matmul tests ****"

  COMMAND="${TEST_RUNNER} \
      --module=${matmul_vmfb} \
      --module=${calls_vmfb} \
      --device=${DEVICE_HAL} \
      --max_elements_to_check=${max_elements_to_check}"

  if [ -n "${XRT_LITE_N_CORE_ROWS:-}" ]; then
    COMMAND="${COMMAND} --xrt_lite_n_core_rows=$XRT_LITE_N_CORE_ROWS"
  fi
  if [ -n "${XRT_LITE_N_CORE_COLS:-}" ]; then
    COMMAND="${COMMAND} --xrt_lite_n_core_cols=$XRT_LITE_N_CORE_COLS"
  fi

  total_num_runs=$(( num_repeat_runs * num_corruption_repeat_runs))
  echo "**** Running '${name}' matmul test ${total_num_runs} times (command ${COMMAND}) ****"
  for i in $(seq 1 $num_repeat_runs); do
    # Disabled reset npu temporarily due to instability issues in CI.
    # Only reset NPU in CI to facilitate easier local testing without sudo access.
#    if [[ "$OSTYPE" == "linux-gnu"* ]] && [ "${GITHUB_ACTIONS}" = true ]; then
#      echo "Reset NPU"
#      bash $THIS_DIR/reset_npu.sh
#    fi
    for j in $(seq 1 $num_corruption_repeat_runs); do
      run_number=$(( (i - 1) * num_corruption_repeat_runs + j))
      echo "Run number ${run_number} / ${total_num_runs}"
      eval "${COMMAND}"
      return_status=$?
      if [ $return_status -ne 0 ]; then
        echo "'${name}' matmul test failed!"
        export MATMUL_TESTS_FAILS=$(( $MATMUL_TESTS_FAILS+1 ))
      fi
    done
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

########################################################
# Run tests                                            #
########################################################

# Notes:
# 1. Be conservative in adding more shapes, as it can increase both the
#    build and execution latency of tests. The build latency is nearly the
#    same for all shapes, while execution latency grows cubicly i.e.
#    linearly with m*k*n.

# Example of a run without any defaults arguments.
run_matmul_test \
    --name_prefix "test1" \
    --lhs_rhs_type "bf16" \
    --acc_type "f32" \
    --target_backend "amd-aie" \
    --target_device "npu1_4col" \
    --peano_install_path "${PEANO}" \
    --amd_aie_install_path "${IREE_INSTALL_DIR}" \
    --lower_to_aie_pipeline "air" \
    --tile_pipeline "pack-peel" \
    --m "64" \
    --n "64" \
    --k "64" \
    --dynamicity "static" \
    --accumulate "false" \
    --expect_compile_failure "0" \
    --do_transpose_rhs "0" \
    --max_elements_to_check "0" \
    --num_repeat_runs "2"

run_matmul_test \
    --name_prefix "packPeel_i32" \
    --lower_to_aie_pipeline "air" \
    --tile_pipeline "pack-peel" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --m "64"  --n "64" --k "128"

run_matmul_test \
    --name_prefix "packPeel_bf16" \
    --lower_to_aie_pipeline "air" \
    --tile_pipeline "pack-peel" \
    --lhs_rhs_type "bf16" \
    --acc_type "f32" \
    --m "512"  --n "512" --k "512"

run_matmul_test \
  --name_prefix "packPeel_t_bf16" \
  --lower_to_aie_pipeline "air" \
  --tile_pipeline "pack-peel" \
  --lhs_rhs_type "bf16" \
  --acc_type "f32" \
  --m "128" --n "256" --k "512" \
  --do_transpose_rhs "1"

run_matmul_test \
  --name_prefix "packPeel4LvlBf16" \
  --lower_to_aie_pipeline "air" \
  --tile_pipeline "pack-peel-4-level-tiling" \
  --lhs_rhs_type "bf16" \
  --acc_type "f32" \
  --m "512" --n "512" --k "512" \
  --num_repeat_runs "1"

###################################################################
# ObjectFifo Matmul tests
###################################################################

# Run repeatedly to check for non-deterministic hangs and numerical
# issues.
repeat_shapes=(
  '32x32x32'
)

run_matmul_test_on_shapes ${repeat_shapes[@]} \
    --name_prefix "small" \
    --lower_to_aie_pipeline "objectFifo" \
    --tile_pipeline "pack-peel" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --num_corruption_repeat_runs "1000"

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

run_matmul_test_on_shapes ${i32_shapes_small[@]} \
    --name_prefix "small_i32" \
    --lower_to_aie_pipeline "objectFifo" \
    --tile_pipeline "pack-peel" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --num_repeat_runs "10"

run_matmul_test_on_shapes ${i32_shapes_small[@]} \
    --name_prefix "small" \
    --lower_to_aie_pipeline "objectFifo" \
    --tile_pipeline "pack-peel" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --num_corruption_repeat_runs "10" \
    --enable_packet_flow "true"

i32_shapes_medium=(
  '1024x1024x1024'
)
# TODO(jornt): re-enable `1536x2048x1536`
#if [ "$OSTYPE" != "msys" ]; then
#  i32_shapes_medium+=('1536x2048x1536')
#fi

run_matmul_test_on_shapes ${i32_shapes_medium[@]} \
    --name_prefix "medium_i32" \
    --lower_to_aie_pipeline "objectFifo" \
    --tile_pipeline "pack-peel" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --num_repeat_runs "2"

# bf16 Matmul tests.

bf16_i8_shapes_small=(
  '64x64x64'
  '128x256x128'
)

bf16_i8_shapes_medium=(
  '512x512x512'
  '1024x1024x1024'
  '1536x2048x1536'
  '4096x2048x4096'
)


run_matmul_test_on_shapes ${bf16_i8_shapes_small[@]} \
    --name_prefix "small_bf16" \
    --lower_to_aie_pipeline "objectFifo" \
    --tile_pipeline "pack-peel" \
    --lhs_rhs_type "bf16" \
    --acc_type "f32" \
    --num_repeat_runs "2"

run_matmul_test_on_shapes ${bf16_i8_shapes_medium[@]} \
    --name_prefix "medium_bf16" \
    --lower_to_aie_pipeline "objectFifo" \
    --tile_pipeline "pack-peel" \
    --lhs_rhs_type "bf16" \
    --acc_type "f32" \
    --num_repeat_runs "2"

# i8 Matmul tests.
run_matmul_test_on_shapes ${bf16_i8_shapes_small[@]} \
    --name_prefix "small_i8" \
    --lower_to_aie_pipeline "objectFifo" \
    --tile_pipeline "pack-peel" \
    --lhs_rhs_type "i8" \
    --acc_type "i32" \
    --num_repeat_runs "2"

run_matmul_test_on_shapes ${bf16_i8_shapes_medium[@]} \
    --name_prefix "medium_i8" \
    --lower_to_aie_pipeline "objectFifo" \
    --tile_pipeline "pack-peel" \
    --lhs_rhs_type "i8" \
    --acc_type "i32" \
    --num_repeat_runs "2"


# TODO(jornt): Disabled turbo tests temporarily due to instability issues in CI. Locally, I have been able to crash the NPU in this mode.
# # note this will not actually show any devices because --xrt_lite_n_core_rows --xrt_lite_n_core_cols are not passed
# # which i have omitted to make the conditional slightly more succinct
# if [[ $($IREE_INSTALL_DIR/bin/iree-benchmark-module --dump_devices | grep xrt-lite) ]]; then

#   $IREE_INSTALL_DIR/bin/iree-benchmark-module \
#     --module=$OUTPUT_DIR/mm_test1_bf16_f32_m64_n64_k64.vmfb \
#     --function=matmul_64x64_64xbf16_ \
#     --input=64x64xbf16 \
#     --input=64x64xbf16 \
#     --device=xrt-lite \
#     --benchmark_repetitions=10 \
#     --xrt_lite_n_core_rows=$XRT_LITE_N_CORE_ROWS \
#     --xrt_lite_n_core_cols=$XRT_LITE_N_CORE_COLS \

#   # TURBO POWER!!!!!!!!!!!!!!!!!
#   set +o pipefail
#   sudo -nv 2>&1 && has_sudo="true" || has_sudo="false"
#   set -o pipefail
#   if [ has_sudo == "true" ]; then
#     sudo $IREE_INSTALL_DIR/bin/iree-benchmark-module \
#       --module=$OUTPUT_DIR/mm_test1_bf16_f32_m64_n64_k64.vmfb \
#       --function=matmul_64x64_64xbf16_ \
#       --input=64x64xbf16 \
#       --input=64x64xbf16 \
#       --device=xrt-lite \
#       --benchmark_repetitions=10 \
#       --xrt_lite_n_core_rows=$XRT_LITE_N_CORE_ROWS \
#       --xrt_lite_n_core_cols=$XRT_LITE_N_CORE_COLS \
#       --xrt_lite_power_mode=turbo
#   fi

# fi

echo "$MATMUL_TESTS_RUN matmul tests run!"
if [ $MATMUL_TESTS_FAILS -ne 0 ]; then
  echo "$MATMUL_TESTS_FAILS matmul tests failed! Scroll up and look for the 🦄 and 🐞..."
  exit 1
fi
