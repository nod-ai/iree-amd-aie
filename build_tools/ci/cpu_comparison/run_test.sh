#!/bin/bash
#
# Copyright 2024 The IREE Authors

# This script is for running tests on the IREE AIE backend and
# the IREE CPU backend, and comparing the results.
#
# There are a few ways to add tests:
#
# 1) add a single test file in `./test_files` which should follow the same
#    format as the example `./test_files/matmul_int32.mlir`.
#
# 2) use an existing template in `./matmul_template` to generate a test file
#    with a fixed structure. Currently a handful of matmul templates exist in
#    that directory.
#
# 3) create a new matmul template in `./matmul_template`, for example if you
#    want to add a new variant with tranposed operands or unary elementwise
#    operations.
#
# 4) create a new template generator, duplicating the directory structure of
#    ./matmul_template. For example you might want to create ./conv_template
#

set -euox pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 5 ]; then

    echo -e "Illegal number of parameters: $#, expected 2-5 parameters." \
            "\n The parameters are as follows:" \
            "\n     1) <output-dir>               (required)" \
            "\n     2) <iree-install-dir>         (required)" \
            "\n     3) <peano-install-dir>        (optional)" \
            "\n     4) <xrt-dir>                  (optional)" \
            "\n     5) <vitis-install-dir>        (optional)" \
            "\n Example, dependent on environment variables:" \
            "\n     ./run_test.sh  " \
            "results_dir_tmp  \$IREE_INSTALL_DIR " \
            "\$PEANO_INSTALL_DIR  /opt/xilinx/xrt  \$VITIS_INSTALL_PATH"
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


# Search for iree-compile and iree-run-module
IREE_COMPILE_EXE=""
IREE_RUN_EXE=""
for dir in "${IREE_INSTALL_DIR}" "${IREE_INSTALL_DIR}/bin" "${IREE_INSTALL_DIR}/tools"; do
  if [ -f "${dir}/iree-compile" ]; then
    IREE_COMPILE_EXE="${dir}/iree-compile"
  fi
  if [ -f "${dir}/iree-run-module" ]; then
    IREE_RUN_EXE="${dir}/iree-run-module"
  fi
done
if [ -z "${IREE_COMPILE_EXE}" ]; then
  echo "No iree-compile executable found in '${IREE_INSTALL_DIR}' or subdirectories."
  exit 1
fi
if [ -z "${IREE_RUN_EXE}" ]; then
  echo "No iree-run-module executable found in '${IREE_INSTALL_DIR}' or subdirectories."
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

# The name of the test file (will pass as command line argument in the future).
INPUT_GENERATOR="${THIS_DIR}/input_generator.py"
OUTPUT_COMPARER="${THIS_DIR}/output_comparer.py"
MATMUL_GENERATOR="${THIS_DIR}/matmul_template/matmul_generator.py"

# Verify that the input generator, output comparer, and matmul generator
# scripts exist.
if [ ! -f "${INPUT_GENERATOR}" ]; then
  echo "Input generator script not found at ${INPUT_GENERATOR}"
  exit 1
fi
if [ ! -f "${OUTPUT_COMPARER}" ]; then
  echo "Output comparer script not found at ${OUTPUT_COMPARER}"
  exit 1
fi
if [ ! -f "${MATMUL_GENERATOR}" ]; then
  echo "Matmul generator script not found at ${MATMUL_GENERATOR}"
  exit 1
fi


source $XRT_DIR/setup.sh
# Circumvent xclbin security (no longer needed as of April 2024 XDNA driver)
export XRT_HACK_UNSECURE_LOADING_XCLBIN=1

MM_KERNEL_URL=https://github.com/nod-ai/iree-amd-aie/releases/download/ukernels/mm.o

cd ${OUTPUT_DIR}

function generate_matmul_test() {

  # Options without defaults
  # ========================
  local lhs_rhs_type=""
  local acc_type=""
  local m=""
  local n=""
  local k=""
  local output_fn=""
  local input_fn=""

  # Options with defaults
  # =====================

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --output_fn)
        output_fn="$2"
        shift 2
        ;;
      --input_fn)
        input_fn="$2"
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

  python3 ${MATMUL_GENERATOR} ${output_fn} ${input_fn} \
          ${m} ${n} ${k} ${lhs_rhs_type} ${acc_type}

}


function run_test() {

  # Options without defaults
  # ========================
  local test_file=""
  local function=""

  # Options with defaults
  # =====================
  local peano_install_path="${PEANO}"
  local amd_aie_install_path="${IREE_INSTALL_DIR}"
  local vitis_path="${VITIS}"
  local pipeline="pad-pack"
  local rtol="1e-6"
  local atol="1e-6"
  local seed="1"
  local use_ukernel="0"
  local expect_compile_failure="0"

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --rtol)
        rtol="$2"
        shift 2
        ;;
      --seed)
        seed="$2"
        shift 2
        ;;
      --atol)
        atol="$2"
        shift 2
        ;;
      --use_ukernel)
        use_ukernel="$2"
        shift 2
        ;;
      --expect_compile_failure)
        expect_compile_failure="$2"
        shift 2
        ;;
      --test_file)
        test_file="$2"
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
     --vitis_path)
        vitis_path="$2"
        shift 2
        ;;
      --pipeline)
        pipeline="$2"
        shift 2
        ;;
      --name_prefix)
        name_prefix="$2"
        shift 2
        ;;
      --function)
        function="$2"
        shift 2
        ;;
      *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
  done

  # Assert that the test file is exist
  if [ -z "${test_file}" ]; then
    echo "The test file must be provided."
    exit 1
  fi

  # Update test_file to be a complete path to the file
  test_file=$(realpath "${test_file}")

  # Now test_file is a string of the /path/to/name.mlir.
  # Confirm this with clear error messages, and extract 'name' as a variable
  name=$(basename "${test_file}" .mlir)

  # Running the ${INPUT_GENERATOR} script below does 2 things:
  # 1) it creates binary files with the data that will be consumed as inputs
  #    by iree-run-module
  # 2) it prints a line with the names of the binary files, which
  #    iree-run-module will have appended as input and output flags.
  #
  python3 ${INPUT_GENERATOR} ${test_file} ${OUTPUT_DIR} ${seed}

  echo "**** Running test for ${test_file} ****"

  aie_vmfb="${OUTPUT_DIR}/${name}_aie.vmfb"
  cpu_vmfb="${OUTPUT_DIR}/${name}_cpu.vmfb"


  echo "**** Generating AIE .vmfb file for ${name} ****"
  compilation_flags="--iree-hal-target-backends=amd-aie \
      --iree-amdaie-tile-pipeline=${pipeline} \
      --iree-amdaie-matmul-elementwise-fusion \
      --iree-amd-aie-peano-install-dir=${peano_install_path} \
      --iree-amd-aie-install-dir=${amd_aie_install_path} \
      --iree-amd-aie-vitis-install-dir=${vitis_path} \
      --iree-hal-dump-executable-files-to=$PWD \
      --iree-amd-aie-show-invoked-commands \
      --mlir-disable-threading -o ${aie_vmfb}"


  # TODO(newling) The following logic is copied from run_matmul_test.sh,
  # factorize out common code in these 2 test scripts to increase
  # maintainability: https://github.com/nod-ai/iree-amd-aie/issues/532
  if [ $use_ukernel -ne 0 ]; then
    compilation_flags="${compilation_flags} --iree-amdaie-enable-ukernels=all"

    # The flag '--iree-amdaie-path-to-ukernels' currently does not work,
    # see for example https://github.com/nod-ai/iree-amd-aie/issues/340.
    # Therefore we need to manually copy (or link) the mm.o file to the
    # directory in which iree-compile is run. iree-compile is run in the
    # output directory. Create the softlink only if it is has not already
    # been created.
    if [ -f "${OUTPUT_DIR}/mm.o" ]; then
      echo "File 'mm.o' already exists in ${OUTPUT_DIR}."
    else
      wget $MM_KERNEL_URL -O  ${OUTPUT_DIR}/mm.o
    fi
  fi


  # Disable exit on error, so that we can do a custom triage of compilation
  # failures.
  set +e

  ${IREE_COMPILE_EXE} ${test_file} ${compilation_flags}
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

  # Re-enable the exit on error, to stop the script as soon as an error occurs.
  set -e


  echo "**** Generating CPU .vmfb file ****"
  ${IREE_COMPILE_EXE} "${test_file}"  \
      --iree-hal-target-backends=llvm-cpu \
      --iree-llvmcpu-target-cpu-features=host \
      -o "${cpu_vmfb}"

  input_output_line=$(cat ${OUTPUT_DIR}/${name}_input_args.txt)

  function_line="--function='${function}'"

  echo "Running the module through the CPU backend"
  eval "${IREE_RUN_EXE} --module=${cpu_vmfb} ${input_output_line} --output=@${name}_cpu.npy ${function_line}"

  echo "Running the module through the AIE backend"
  eval "${IREE_RUN_EXE} --module=${aie_vmfb} ${input_output_line} --device=xrt --output=@${name}_aie.npy ${function_line}"

  # Check that values in cpu.npy and aie.npy are close enough.
  eval "python3 ${OUTPUT_COMPARER} ${name}_cpu.npy ${name}_aie.npy ${rtol} ${atol}"
}

# Example of running a test directly from an .mlir file with a function.
run_test --test_file ${THIS_DIR}/test_files/matmul_int32.mlir

# An example of an arbitrary graph with three matmuls which form three dispatches.
run_test --test_file ${THIS_DIR}/test_files/three_matmuls.mlir --function 'three_$mm$'

# Example of generating a matmul test from a template, and then running it.
test_name=${OUTPUT_DIR}/test_from_template.mlir
matmul_template_dir=${THIS_DIR}/matmul_template
template_name=${matmul_template_dir}/matmul_MxK_KxN.mlir
generate_matmul_test \
   --output_fn ${test_name} \
   --input_fn ${template_name} \
   --lhs_rhs_type "bf16" \
   --acc_type "f32" \
   --m "32"  --n "32" --k "64"
run_test --test_file ${test_name} --rtol 1e-5 --atol 1e-5

template_name=${matmul_template_dir}/matmul_bias_MxK_KxN_MxN.mlir
generate_matmul_test \
   --output_fn ${test_name} --input_fn ${template_name} \
   --lhs_rhs_type "i32" --acc_type "i32" \
   --m "128"  --n "128" --k "256"
run_test --test_file ${test_name} --pipeline "pack-peel" --rtol 0 --atol 0


template_name=${matmul_template_dir}/matmul_bias_MxK_KxN_N.mlir
generate_matmul_test \
   --output_fn ${test_name} --input_fn ${template_name} \
   --lhs_rhs_type "bf16" --acc_type "f32" \
   --m "1024"  --n "1024" --k "512"
run_test --test_file ${test_name} --pipeline "pack-peel"
run_test --test_file ${test_name} --pipeline "pack-peel" --use_ukernel 1

# Conv2d tests.
run_test --test_file ${THIS_DIR}/test_files/conv2d_nhwc_int32.mlir --pipeline "conv-decompose"
run_test --test_file ${THIS_DIR}/test_files/conv2d_nhwc_bf16.mlir --pipeline "conv-decompose"
run_test --test_file ${THIS_DIR}/test_files/conv2d_nhwc_int8.mlir --pipeline "conv-decompose"
run_test --test_file ${THIS_DIR}/test_files/conv2d_nhwc_q.mlir --pipeline "conv-decompose"
