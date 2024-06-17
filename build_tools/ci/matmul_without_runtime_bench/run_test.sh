#!/bin/bash
#
# Copyright 2024 The LLVM Project
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
set -euox pipefail

# The usual boilerplate for getting the AIE directories setup.
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
            "\n     ./run_test.sh  " \
            "results_dir_tmp  \$IREE_BUILD_OR_INSTALL_DIR  \$MLIR_AIE_INSTALL_DIR  " \
            "\$PEANO_INSTALL_DIR  /opt/xilinx/xrt  \$VITIS_INSTALL_PATH"
    exit 1
fi

# All generated files should end up in the output directory.
OUTPUT_DIR=`realpath "$1"`
DIRECT_OUTPUT_DIR="${OUTPUT_DIR}/direct"
UKERNEL_OUTPUT_DIR="${OUTPUT_DIR}/ukernel"
for dir in ${OUTPUT_DIR} ${DIRECT_OUTPUT_DIR} ${UKERNEL_OUTPUT_DIR}; do
  mkdir -p ${dir}
  if [ ! -d "${dir}" ]; then
    echo "Failed to locate or construct directory '${dir}'."
    exit 1
  fi
done

IREE_INSTALL_DIR=`realpath "$2"`
if [ ! -d "${IREE_INSTALL_DIR}" ]; then
  echo "IREE_INSTALL_DIR must be a directory, '${IREE_INSTALL_DIR}' is not."
  exit 1
fi

# Search for iree-compile in the user provide directory.
IREE_COMPILE_EXE=""
for dir in "${IREE_INSTALL_DIR}" "${IREE_INSTALL_DIR}/bin" "${IREE_INSTALL_DIR}/tools"; do
  if [ -f "${dir}/iree-compile" ]; then
    IREE_COMPILE_EXE="${dir}/iree-compile"
  fi
done

if [ -z "${IREE_COMPILE_EXE}" ]; then
  echo "No 'iree-compile' found in any of the following directories: " \
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

source $XRT_DIR/setup.sh
export XRT_HACK_UNSECURE_LOADING_XCLBIN=1

M=2048
K=4096
N=2048

# Create a file ${OUTPUT_DIR}/generated_linalg_matmul.mlir
# by running 'generate_linalg_matmul.py with arguments M, K, and N.
GENERATE_SCRIPT="${THIS_DIR}/generate_linalg_matmul.py"
SOURCE_MLIR_FILE="${OUTPUT_DIR}/generated_linalg_matmul.mlir"
if [ ! -f "${GENERATE_SCRIPT}" ]; then
  echo "generate_linalg_matmul.py not found: ${GENERATE_SCRIPT}"
  exit 1
fi
GENERATE_COMMAND="python ${GENERATE_SCRIPT} M=${M} K=${K} N=${N} > ${SOURCE_MLIR_FILE}"
echo "Executing command: $GENERATE_COMMAND"
eval $GENERATE_COMMAND

THIS="$(cd $(dirname $0) && pwd)"

BASE_COMPILATION_FLAGS="-iree-hal-target-backends=amd-aie \
-iree-amd-aie-peano-install-dir=${PEANO} \
-iree-amd-aie-mlir-aie-install-dir=${MLIR_AIE_INSTALL} \
-iree-amd-aie-vitis-install-dir=${VITIS} \
-iree-amd-aie-show-invoked-commands"

DIRECT_COMPILATION_FLAGS="${BASE_COMPILATION_FLAGS} \
  --iree-hal-dump-executable-files-to=${DIRECT_OUTPUT_DIR} \
  -o ${DIRECT_OUTPUT_DIR}/test_artefact.vmfb"

UKERNEL_COMPILATION_FLAGS="${BASE_COMPILATION_FLAGS} \
  --iree-amdaie-enable-ukernels=all \
  --iree-hal-dump-executable-files-to=${UKERNEL_OUTPUT_DIR} \
  -o ${UKERNEL_OUTPUT_DIR}/test_artefact.vmfb"

if [ -f "${UKERNEL_OUTPUT_DIR}/mm.o" ]; then
  echo "File 'mm.o' already exists in ${UKERNEL_OUTPUT_DIR}."
else
  SRC_DIR="${MLIR_AIE_INSTALL}/aie_kernels/mm.o"
  ln -s ${SRC_DIR} ${OUTPUT_DIR}/mm.o
fi

cd ${OUTPUT_DIR}


IREE_DIRECT_COMPILE_COMMAND="${IREE_COMPILE_EXE} ${SOURCE_MLIR_FILE} ${DIRECT_COMPILATION_FLAGS}"
IREE_UKERNEL_COMPILE_COMMAND="${IREE_COMPILE_EXE} ${SOURCE_MLIR_FILE} ${UKERNEL_COMPILATION_FLAGS}"

# Execute the command to generate the .vmfb, .xclbin, .npu.txt files, etc.
echo "Executing command: $IREE_DIRECT_COMPILE_COMMAND"
eval $IREE_DIRECT_COMPILE_COMMAND


# Do the same as above, but for the UKERNEL compilation:
echo "Executing command: $IREE_UKERNEL_COMPILE_COMMAND"
eval $IREE_UKERNEL_COMPILE_COMMAND

for dir in ${DIRECT_OUTPUT_DIR} ${UKERNEL_OUTPUT_DIR}; do
  if [ ! -f "${dir}/test_artefact.vmfb" ]; then
    echo "test_artefact.vmfb was not created: ${dir}/test_artefact.vmfb"
    exit 1
  fi
done

# Get the full path of test.cpp (which is in the same directory as this script).
TEST_CPP="${THIS_DIR}/test.cpp"

FLAGS="-Wall -I${XRT_DIR}/include -L${XRT_DIR}/lib -luuid -lxrt_coreutil -lrt -lstdc++"
g++ ${TEST_CPP} -o ${OUTPUT_DIR}/test.exe ${FLAGS}

# Wrapped in "false" as most users don't use YCM:
false && {
  # =============================================
  # ==== YCM VIM SETUP =======================
  # =======================================
  YCM_EXTRA_CONF="${THIS_DIR}/.ycm_extra_conf.py"
  echo "def Settings( **kwargs ):" > ${YCM_EXTRA_CONF}
  echo "  return {" >> ${YCM_EXTRA_CONF}
  echo "    'flags': [ '-x', 'c++'," >> ${YCM_EXTRA_CONF}
  for flag in ${FLAGS}; do
    echo "              '${flag}'," >> ${YCM_EXTRA_CONF}
  done
  echo "            ]" >> ${YCM_EXTRA_CONF}
  echo "  }" >> ${YCM_EXTRA_CONF}
  # ====================
  # =================
  # ==============
}

# Verify that ${OUTPUT_DIR}/test.exe exists:
if [ ! -f "${OUTPUT_DIR}/test.exe" ]; then
  echo "test.exe was not created: ${OUTPUT_DIR}/test.exe"
  exit 1
fi

for dir in direct ukernel; do
  echo "Benchmarking ${dir}."
 
  XCLBIN_FILE=""
  for file in `find ${OUTPUT_DIR}/${dir} -name "*.xclbin"`; do
    XCLBIN_FILE="${file}"
  done
  if [ -z "${XCLBIN_FILE}" ]; then
    echo "No .xclbin file found in the output directory or any of its subdirectories."
    exit 1
  else
    echo "Found .xclbin file: ${XCLBIN_FILE}"
  fi

  IPU_TXT_FILE=""
  for file in `find ${OUTPUT_DIR}/${dir} -name "*.npu.txt"`; do
    IPU_TXT_FILE="${file}"
  done
  if [ -z "${IPU_TXT_FILE}" ]; then
    echo "No .npu.txt file found in the output directory or any of its subdirectories."
    exit 1
  else
    echo "Found .npu.txt file: ${IPU_TXT_FILE}"
  fi

  BENCHMARK_RESULTS_FN="${dir}/benchmark_results.txt"

  ${OUTPUT_DIR}/test.exe ${XCLBIN_FILE} ${IPU_TXT_FILE} ${M} ${K} ${N} ${BENCHMARK_RESULTS_FN}

  # TODO(newling) Lit test on results file confirming that the results are reasonable.
done


