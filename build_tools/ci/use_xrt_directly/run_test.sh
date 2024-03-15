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
            "\n     ./run_matmul_test.sh  " \
            "results_dir_tmp  \$IREE_BUILD_OR_INSTALL_DIR  \$MLIR_AIE_INSTALL_DIR  " \
            "\$PEANO_INSTALL_DIR  /opt/xilinx/xrt  \$VITIS_INSTALL_PATH"
    exit 1
fi


# All temporary files should end up in the output directory. 
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

source $XRT_DIR/setup.sh

THIS="$(cd $(dirname $0) && pwd)"
SOURCE_MLIR_FILE="${THIS_DIR}/linalg_matmul.mlir"

# Construct the iree-compile command. Use all the aie2xclbin printing options,
# and then test that the output is printed to stdout and stderr as expected.
IREE_COMPILE_COMMAND="${IREE_COMPILE_EXE} \
${SOURCE_MLIR_FILE} \
--iree-hal-target-backends=amd-aie \
--iree-amd-aie-peano-install-dir=${PEANO} \
--iree-amd-aie-mlir-aie-install-dir=${MLIR_AIE_INSTALL} \
--iree-amd-aie-vitis-install-dir=${VITIS} \
--iree-hal-dump-executable-files-to=${OUTPUT_DIR} \
--iree-amdaie-use-pipeline=simple-pack \
-o ${OUTPUT_DIR}/test_artefact.vmfb \
--iree-amd-aie-show-invoked-commands"

# Execute the command to generate the .vmfb, .xclbin, .ipu.txt files, etc.
echo "Executing command: $IREE_COMPILE_COMMAND"
eval $IREE_COMPILE_COMMAND 

if [ ! -f "${OUTPUT_DIR}/test_artefact.vmfb" ]; then
  echo "test_artefact.vmfb was not created: ${OUTPUT_DIR}/test_artefact.vmfb"
  exit 1
fi


g++ test.cpp -o ${OUTPUT_DIR}/test.exe -Wall -I${XRT_DIR}/include -L${XRT_DIR}/lib -luuid -lxrt_coreutil -lrt -lstdc++

# Verify that ${OUTPUT_DIR}/test.exe exists:
if [ ! -f "${OUTPUT_DIR}/test.exe" ]; then
  echo "test.exe was not created: ${OUTPUT_DIR}/test.exe"
  exit 1
fi

# We expect to find a file with the .xclbin extension in the output directory, 
# or in a subdirectory of the output directory. Locate it if possible. 
XCLBIN_FILE=""
for file in `find ${OUTPUT_DIR} -name "*.xclbin"`; do
  XCLBIN_FILE="${file}"
done
if [ -z "${XCLBIN_FILE}" ]; then
  echo "No .xclbin file found in the output directory or any of its subdirectories."
  exit 1
else
  echo "Found .xclbin file: ${XCLBIN_FILE}"
fi

# Maybe sign the XCLBIN file (CI moonshot). 
SIGNER=${XRT_DIR}/amdxdna/setup_xclbin_firmware.sh
if [ ! -f "$SIGNER" ]; then
  echo "**** Skipping XCLBIN signing: $SIGNER not found ****"
else
  sudo $SIGNER -dev Phoenix -xclbin "${XCLBIN_FILE}"
fi

# As for the XCLBIN file, we expect to find a .ipu.txt file in the output directory.
IPU_TXT_FILE=""
for file in `find ${OUTPUT_DIR} -name "*.ipu.txt"`; do
  IPU_TXT_FILE="${file}"
done
if [ -z "${IPU_TXT_FILE}" ]; then
  echo "No .ipu.txt file found in the output directory or any of its subdirectories."
  exit 1
else
  echo "Found .ipu.txt file: ${IPU_TXT_FILE}"
fi

# Run the test! 'true false' is the failure mode. 
${OUTPUT_DIR}/test.exe ${XCLBIN_FILE} ${IPU_TXT_FILE} true true


