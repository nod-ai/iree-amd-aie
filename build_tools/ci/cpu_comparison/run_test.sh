#!/bin/bash
#
# Copyright 2024 The IREE Authors

set -euox pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 6 ]; then

    echo -e "Illegal number of parameters: $#, expected 2-6 parameters." \
            "\n The parameters are as follows:" \
            "\n     1) <output-dir>               (required)" \
            "\n     2) <iree-install-dir>         (required)" \
            "\n     3) <mlir-aie-install-dir>     (optional)" \
            "\n     4) <peano-install-dir>        (optional)" \
            "\n     5) <xrt-dir>                  (optional)" \
            "\n     6) <vitis-install-dir>        (optional)" \
            "\n Example, dependent on environment variables:" \
            "\n     ./run_test.sh  " \
            "results_dir_tmp  \$IREE_INSTALL_DIR  \$MLIR_AIE_INSTALL_DIR  " \
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

# The name of the test file (will pass as command line argument in the future).
INPUT_GENERATOR="${THIS_DIR}/input_generator.py"
OUTPUT_COMPARER="${THIS_DIR}/output_comparer.py"

# Verify that the input generator and output comparer scripts exist
if [ ! -f "${INPUT_GENERATOR}" ]; then
  echo "Input generator script not found at ${INPUT_GENERATOR}"
  exit 1
fi
if [ ! -f "${OUTPUT_COMPARER}" ]; then
  echo "Output comparer script not found at ${OUTPUT_COMPARER}"
  exit 1
fi


source $XRT_DIR/setup.sh
# Circumvent xclbin security (no longer needed as of April 2024 XDNA driver)
export XRT_HACK_UNSECURE_LOADING_XCLBIN=1

cd ${OUTPUT_DIR}


function run_test() {

  # Options without defaults
  # ========================
  local lhs_rhs_type=""
  local acc_type=""
  local m=""
  local n=""
  local k=""

  # Options with defaults
  # =====================
  local peano_install_path="${PEANO}"
  local mlir_aie_install_path="${MLIR_AIE_INSTALL}"
  local vitis_path="${VITIS}"
  local pipeline="pad-pack"
  local rtol="1e-6"
  local atol="1e-6"


  while [ "$#" -gt 0 ]; do
    case "$1" in
      --rtol)
        rtol="$2"
        shift 2
        ;;
      --atol)
        atol="$2"
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

  # Generate a name for the test based on name_prefix and the matmul dimensions 
  # and data types, assuming m, n, k are just single integers:
  name="${name_prefix}_${m}x${n}x${k}_${lhs_rhs_type}_${acc_type}"

  # Disable exit on failure:
  set +e

  # Running 'python3 ${INPUT_GENERATOR} ${test_file} ${OUTPUT_DIR}' does 3 things.
  # 1) it creates a mlir file using the test template;
  # 2) it creates binary files with the data that will be consumed as inputs
  #    by iree-run-module;
  # 3) it prints a line with the names of the binary files, which
  #    iree-run-module will have appended (input and output flags).
  #
  python3 ${INPUT_GENERATOR} ${OUTPUT_DIR} ${name_prefix} ${m} ${n} ${k} ${lhs_rhs_type} ${acc_type}

  # Load the contents of OUTPUT_DIR/{name}_command_args.txt into a variable:
  input_output_line=$(cat ${OUTPUT_DIR}/${name}_input_args.txt)

  # Path of the test mlir and vmfb files
  test_file="${OUTPUT_DIR}/${name}.mlir"
  aie_vmfb="${OUTPUT_DIR}/${name}_aie.vmfb"
  cpu_vmfb="${OUTPUT_DIR}/${name}_cpu.vmfb"

  # Assert that the test file is exist
  if [ -z "${test_file}" ]; then
    echo "The test file must be provided."
    exit 1
  fi

  echo "**** Generating AIE .vmfb file for ${name} ****"
  ${IREE_COMPILE_EXE} "${test_file}"  \
      --iree-hal-target-backends=amd-aie \
      --iree-amdaie-use-pipeline=${pipeline} \
      --iree-amdaie-matmul-elementwise-fusion \
      --iree-amd-aie-peano-install-dir=${peano_install_path} \
      --iree-amd-aie-mlir-aie-install-dir=${mlir_aie_install_path} \
      --iree-amd-aie-vitis-install-dir=${vitis_path} \
      --iree-hal-dump-executable-files-to=$PWD \
      --mlir-disable-threading \
      -o "${aie_vmfb}"

   echo "**** Generating CPU .vmfb file ****"
   ${IREE_COMPILE_EXE} "${test_file}"  \
       --iree-hal-target-backends=llvm-cpu \
       -o "${cpu_vmfb}"

  # Extract function names from the mlir file
  function_names=$(grep -oP '@\K\S+(?=\()' ${test_file})

  # Running 'python3 ${INPUT_GENERATOR} ${test_file} ${OUTPUT_DIR}' does 2 things.
  # 1) it creates binary files with the data that will be consumed as inputs
  #    by iree-run-module
  # 2) it prints a line with the names of the binary files, which
  #    iree-run-module will have appended (input and output flags).
  #
  python3 ${INPUT_GENERATOR} ${test_file} ${OUTPUT_DIR}

  # Load the contents of OUTPUT_DIR/{name}_command_args.txt into a variable:
  input_output_line=$(cat ${OUTPUT_DIR}/${name}_input_args.txt)

  echo "Running the module through the CPU backend"
  eval "${IREE_RUN_EXE} --module=${cpu_vmfb} ${input_output_line} --output=@${name}_cpu.npy"

  echo "Running the module through the AIE backend"
  eval "${IREE_RUN_EXE} --module=${aie_vmfb} ${input_output_line} --device=xrt --output=@${name}_aie.npy"

  # Check if the output the files cpu.npy and aie.npy are the same. Pass in rtol and atol
  eval "python3 ${OUTPUT_COMPARER} ${name}_cpu.npy ${name}_aie.npy ${rtol} ${atol}"

  set +x
}

run_test \
   --name_prefix "matmul" \
   --lhs_rhs_type "bf16" \
   --acc_type "f32" \
   --m "8"  --n "32" --k "64"

run_test \
   --name_prefix "matmul_elementwise" \
   --pipeline "pack-peel" \
   --lhs_rhs_type "i32" \
   --acc_type "i32" \
   --m "128"  --n "128" --k "256"

run_test \
   --name_prefix "matmul_elementwise" \
   --pipeline "pack-peel" \
   --lhs_rhs_type "bf16" \
   --acc_type "f32" \
   --m "1024" --n "1024" --k "512"
