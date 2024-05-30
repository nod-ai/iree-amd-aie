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
#   3. Run: `./run_matmul_tests.sh <output_dir_path> <iree_install_path> [<mlir_aie_install_path>] [<peano_install_path>] [<xrt_path>] [<vitis_path>] [do_signing]`
#      The directories above in square brackets are optional, the first 2 directories are required.

set -euox pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 7 ]; then

   # The expected parameters are
   #    1) <output-dir>            (required)
   #    2) <iree-install-dir>      (required)
   #    3) <mlir-aie-install-dir>  (optional)
   #    4) <peano-install-dir>     (optional)
   #    5) <xrt-dir>               (optional)
   #    6) <vitis-install-dir>     (optional)
   #    7) <do-signing>            (optional)
    echo -e "Illegal number of parameters: $#, expected 2-7 parameters." \
            "\n The parameters are as follows:" \
            "\n     1) <output-dir>               (required)" \
            "\n     2) <iree-install-dir>         (required)" \
            "\n     3) <mlir-aie-install-dir>     (optional)" \
            "\n     4) <peano-install-dir>        (optional)" \
            "\n     5) <xrt-dir>                  (optional)" \
            "\n     6) <vitis-install-dir>        (optional)" \
            "\n     7) <do-signing>               (optional)" \
            "\n Example, dependent on environment variables:" \
            "\n     ./run_matmul_test.sh  " \
            "results_dir_tmp  \$IREE_INSTALL_DIR  \$MLIR_AIE_INSTALL_DIR  " \
            "\$PEANO_INSTALL_DIR  /opt/xilinx/xrt  \$VITIS_INSTALL_PATH 0"
    exit 1
fi

OUTPUT_DIR=`realpath "$1"`
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

# Parameter 7) <do-signing>
if [ -z "${7-}" ]; then
  DO_SIGNING=1
else
  DO_SIGNING=$7
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

  # Options without defaults
  # ========================
  local lhs_rhs_type=""
  local acc_type=""
  local m=""
  local n=""
  local k=""

  # Options with defaults
  # =====================
  # name_prefix: A prefix for the name of the test. The full test name will be
  # extended with m,n,k if they are unique.
  local name_prefix="noprefix"

  local target_backend="amd-aie"

  local device="xrt"

  local peano_install_path="${PEANO}"

  local mlir_aie_install_path="${MLIR_AIE_INSTALL}"

  local vitis_path="${VITIS}"

  local pipeline="pad-pack"

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

  set -x


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
                      --iree-amdaie-use-pipeline=${pipeline} \
                      --iree-amd-aie-peano-install-dir=${peano_install_path} \
                      --iree-amd-aie-mlir-aie-install-dir=${mlir_aie_install_path} \
                      --iree-amd-aie-vitis-install-dir=${vitis_path} \
                      --iree-hal-dump-executable-files-to=$PWD \
                      --iree-amd-aie-show-invoked-commands"

  if [ $use_ukernel -ne 0 ]; then

    compilation_flags="${compilation_flags} \
                        --iree-amdaie-enable-ukernels=all"


    # The flag '--iree-amdaie-path-to-ukernels' currently does not work,
    # see for example https://github.com/nod-ai/iree-amd-aie/issues/340
    # Therefore we need to manually copy (or link) the mm.o file to the
    # directory in which iree-compile is run. iree-compile is run in the
    # output directory. Create the softlink only if it is has not already
    # been created.
    if [ -f "${OUTPUT_DIR}/mm.o" ]; then
      echo "File 'mm.o' already exists in ${OUTPUT_DIR}."
    else
      SRC_DIR="${mlir_aie_install_path}/aie_kernels/mm.o"
      ln -s ${SRC_DIR} ${OUTPUT_DIR}/mm.o
    fi
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

  # Extract function names from the mlir file
  function_names=$(grep -oP '@\K\S+(?=\()' ${matmul_ir})

  compiled_time=$(date +%s%3N)


  # Behavior of <do-signing> depends on if the script for
  # signing xclbins is found:
  #
  # do-signing     |  setup_xclbin_firmware.sh found | Behavior
  # -------------- | ------------------------------- | -------------
  # 1              | yes                             | Sign XCLBIN
  # 1              | no                              | Error
  # 0              | no/yes                          | Skip signing
  # -------------- | ------------------------------- | -------------

  if [ $DO_SIGNING -eq 0 ]; then
    echo "**** Skipping XCLBIN signing: DO_SIGNING set to 0 ****"
  else
    # Informed guess where the signing script is.TODO: make this a script param.
    SIGNER=${XRT_DIR}/amdxdna/setup_xclbin_firmware.sh
    if [ ! -f "$SIGNER" ]; then
      echo "**** With DO_SIGNING=1, the script for signing xclbins was not found at $SIGNER ****"
      exit 1
    fi
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
      --module=${matmul_vmfb} \
      --module=${calls_vmfb} \
      --device=${device} \
      --max_elements_to_check=${max_elements_to_check}"



  echo "**** Running '${name}' matmul test ${num_repeat_runs} times ****"
  for i in $(seq 1 $num_repeat_runs); do
    echo "Run number ${i} / ${num_repeat_runs} of command ${COMMAND}"
    eval "${COMMAND}"
    return_status=$?
    if [ $return_status -ne 0 ]; then
      echo "Command returned with status: ${return_status}"
      exit 1
    fi
  done


  end_time=$(date +%s%3N)

  #print the time spent in each stage:
  echo "Time spent in generation: $((generated_time - start_time)) [ms]"
  echo "Time spent in compilation: $((compiled_time - generated_time)) [ms]"
  echo "Time spent in execution and verification: $((end_time - compiled_time)) [ms]"

  set +x
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
    --device "xrt" \
    --peano_install_path "${PEANO}" \
    --mlir_aie_install_path "${MLIR_AIE_INSTALL}" \
    --vitis_path  "${VITIS}" \
    --pipeline "pad-pack" \
    --m "64" \
    --n "64" \
    --k "64" \
    --dynamicity "static" \
    --accumulate "false" \
    --expect_compile_failure "0" \
    --do_transpose_rhs "0" \
    --max_elements_to_check "0" \
    --use_ukernel "0" \
    --num_repeat_runs "2"

run_matmul_test \
    --name_prefix "ukern" \
    --lhs_rhs_type "bf16" \
    --acc_type "f32" \
    --m "256"  --k "256" --n "256" \
    --use_ukernel "1"

run_matmul_test \
  --name_prefix "transpose_int32" \
  --lhs_rhs_type "i32" \
  --acc_type "i32" \
  --m "8" --n "16" --k "32" \
  --do_transpose_rhs "1"

# TODO: fix this.
#: error: 'aie.dma_bd' op Cannot give more than 4 dimensions for step sizes
#  and wraps in this  tile (got 5 dimensions).
run_matmul_test \
  --name_prefix "transpose_bf16" \
  --lhs_rhs_type "bf16" \
  --acc_type "f32" \
  --m "256" --n "256" --k "256" \
  --do_transpose_rhs "1" \
  --expect_compile_failure "1"


# The below matmul case passes with
# tile_sizes = [[1, 1], [0, 0, 250], [1, 1], [0, 0, 2]], packedSizes = [1, 1, 5]
# but fails with tile_sizes = [[1, 1], [0, 0, 200], [1, 1], [0, 0, 1]], packedSizes = [1, 1, 8],
# with the error LLVM ERROR: unable to legalize instruction: %152:_(<2 x s32>) = G_FMUL %148:_, %150:_ (in function: core_0_2)
# The later is what a more vectorization friendly packing looks like so this test is expected failing the test here.
# TODO: check if the test will pass with a more recent llvm-aie and if it doesnt, report it upstream.
run_matmul_test \
   --name_prefix "failure_0" \
   --lhs_rhs_type "i32" \
   --acc_type "i32" \
   --m "1"  --n "1" --k "1000" \
   --expect_compile_failure "1"

# The below matmul case passes with
# tile_sizes = [52, 52], [0, 0, 63], [26, 26], [0, 0, 3], packedSizes = [2, 2, 7]
# but fails with tile_sizes = [[52, 52], [0, 0, 63], [4, 4], [0, 0, 3]], packedSizes = [4, 4, 7],
# in AIRHerdPlacementPass with the error No valid placement found
# The later is what a more vectorization friendly packing looks like so we are expected failing the test here.
# We should fix this failure.
run_matmul_test \
   --name_prefix "failure_0" \
   --lhs_rhs_type "i32" \
   --acc_type "i32" \
   --m "52"  --n "52" --k "63" \
   --expect_compile_failure "1"

# Example of a run with a group of 2+ matmuls. Currently this test is passed
# the flag '--num_repeat_runs 0" as there is currently an issue with the runtime if
# multiple matmuls are run in the same test. TODO(newling/nmeshram): Document
# this issue.
run_matmul_test \
    --name_prefix "multiple_matmuls" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --m "512,8,16,7" \
    --n "512,32,16,15" \
    --k "256,16,8,9" \
    --num_repeat_runs "0"

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
    --name_prefix "small" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --m "9"  --n "7" --k "16"

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
    --pipeline "pack-peel" \
    --lhs_rhs_type "i32" \
    --acc_type "i32" \
    --m "64"  --n "64" --k "128"

# We're seeing intermittent numerical errors in these 3 tests,
# needs investigation. TODO(newling/yzhang93): Add more info.
# Appears to be only pack-peel pipeline with bf16->f32.
# Using 'num_repeat_runs=0' flag to avoid running the numerical test.
#################################################################


# TODO: compilation error with the below test.
#
# error: 'aie.dma_bd' op Cannot give more than 3 dimensions for step sizes and wraps in this  tile (got 4 dimensions).
#
# The config generated with the current strategy is:
#
# packing_config = #amdaie.packing_config<packing_config =
#   [{packedSizes = [64, 64, 64],
#     transposePackIndices = [1],
#     unpackEmpty = [false],
#     innerPerm = [[1, 0]],
#     outerPerm = [[0, 1]]},
#     {
#       packedSizes = [0, 0, 0, 4, 4, 8],
#       transposePackIndices = [0, 1, 2],
#       unpackEmpty = [false, false, true],
#       innerPerm = [[0, 1], [1, 0], [0, 1]],
#       outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
#     }
run_matmul_test \
    --name_prefix "packPeel" \
    --pipeline "pack-peel" \
    --lhs_rhs_type "bf16" \
    --acc_type "f32" \
    --m "64"  --n "64" --k "128" \
    --expect_compile_failure "1"

run_matmul_test \
    --name_prefix "packPeelLarge" \
    --pipeline "pack-peel" \
    --lhs_rhs_type "bf16" \
    --acc_type "f32" \
    --m "512"  --n "512" --k "512" \
    --num_repeat_runs "0"

run_matmul_test \
    --name_prefix "packPeel2304" \
    --pipeline "pack-peel" \
    --lhs_rhs_type "bf16" \
    --acc_type "f32" \
    --m "128"  --n "128" --k "2304" \
    --num_repeat_runs "0"

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
