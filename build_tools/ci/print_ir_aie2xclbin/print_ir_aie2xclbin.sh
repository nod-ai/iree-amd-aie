#!/bin/bash
#
# Copyright 2024 The LLVM Project
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

# Check for the number of provided arguments
if [ "$#" -ne 2 ] && [ "$#" -ne 5 ]; then
    echo -e "Illegal number of parameters: $#." \
            "\n For 2 parameters:" \
            "\n     1) <iree-compile-dir>" \
            "\n     2) <output-dir>" \
            "\n For 5 parameters:" \
            "\n     1) <iree-compile-dir>" \
            "\n     2) <output-dir>" \
            "\n     3) <peano-install-dir>" \
            "\n     4) <xrt-dir>" \
            "\n     5) <vitis-install-dir>" \
            "\n Example (dependent on environment variables):" \
            "\n     ./print_ir_aie2xclbin.sh " \
            "\$IREE_BUILD_DIR/tools " \
            "results_dir_tmp "\
            "\$PEANO_INSTALL_DIR "\
            "/opt/xilinx/xrt "\
            "\$VITIS_INSTALL_PATH"
    exit 1
fi


OUTPUT=`realpath "${2}"`
mkdir -p ${OUTPUT}

# The CI case:
if [ "$#" -eq 2 ]; then
  echo "Assuming that this is the 'CI case' as 2 parameters were provided."
  PEANO=/opt/llvm-aie
  XRT=/opt/xilinx/xrt
  VITIS=/opt/Xilinx/Vitis/2024.2
fi

echo "chess-clang: $(find $VITIS -name chess-clang)"
echo "xchesscc: $(find $VITIS -name xchesscc)"

# The local set-paths-manually case:
if [ "$#" -eq 5 ]; then
  PEANO="$3"
  XRT="$4"
  VITIS="$5"
fi

IREE_INSTALL_DIR="$1"
if [ ! -d "${IREE_INSTALL_DIR}/bin" ]; then
  echo "IREE_INSTALL_DIR/bin does not exist: ${IREE_INSTALL_DIR}/bin."
  exit 1
else
  IREE_INSTALL_DIR=`realpath "${IREE_INSTALL_DIR}"`
fi

for dir in "${IREE_INSTALL_DIR}" "${IREE_INSTALL_DIR}/bin" "${IREE_INSTALL_DIR}/tools"; do
  if [ -f "${dir}/iree-compile" ]; then
    IREE_COMPILE_EXE="${dir}/iree-compile"
  fi
done

if [ ! -x "${IREE_COMPILE_EXE}" ]; then
  echo "IREE_COMPILE_EXE does not exist or isn't executable: ${IREE_COMPILE_EXE}."
  exit 1
fi

if [ ! -d "${OUTPUT}" ]; then
  echo "OUTPUT does not exist: ${OUTPUT}"
  exit 1
fi

if [ -d "${PEANO}" ]; then
  PEANO=`realpath "${PEANO}"`
else
  echo "PEANO does not exist: ${PEANO}"
  exit 1
fi

if [ -d "${XRT}" ]; then
  XRT=`realpath "${XRT}"`
else
  echo "XRT does not exist: ${XRT}"
  exit 1
fi

if [ -d "${VITIS}" ]; then
  VITIS=${VITIS}
else
  echo "VITIS does not exist: ${VITIS}"
  exit 1
fi

# There might be a FileCheck program in the IREE_INSTALL_DIR. Check.
# Do not fail if it is not there, we can also check if it already on PATH.
if [ -x "${IREE_INSTALL_DIR}/bin/FileCheck" ]; then
  FILECHECK_EXE="${IREE_INSTALL_DIR}/bin/FileCheck"
elif [ -x "$(command -v FileCheck)" ]; then
  FILECHECK_EXE="$(command -v FileCheck)"
else
  echo "FileCheck does not exist or isn't executable in ${IREE_INSTALL_DIR}/bin or on PATH."
  exit 1
fi

source $XRT/setup.sh

THIS="$(cd $(dirname $0) && pwd)"
SOURCE_MLIR_FILE="${THIS}/linalg_matmul_f32.mlir"

# Construct the iree-compile command. Use all the aie2xclbin printing options,
# and then test that the output is printed to stdout and stderr as expected.
IREE_COMPILE_COMMAND="${IREE_COMPILE_EXE} \
${SOURCE_MLIR_FILE} \
--iree-hal-target-backends=amd-aie \
--iree-amd-aie-peano-install-dir=${PEANO} \
--iree-amd-aie-install-dir=${IREE_INSTALL_DIR} \
--iree-amd-aie-vitis-install-dir=${VITIS} \
--iree-hal-dump-executable-files-to=${OUTPUT} \
--aie2xclbin-print-ir-after-all \
--aie2xclbin-print-ir-before-all \
--aie2xclbin-print-ir-module-scope \
--aie2xclbin-timing \
--mlir-print-ir-after-all \
--mlir-print-ir-module-scope \
--mlir-disable-threading \
--iree-amdaie-tile-pipeline=pad-pack \
-o ${OUTPUT}/test_artefact.vmfb \
--iree-amd-aie-show-invoked-commands"


# set the files for stdout and stdin. These are what FileCheck will scrutinize.
STDOUT_FULLPATH="${OUTPUT}/stdout.txt"
STDERR_FULLPATH="${OUTPUT}/stderr.txt"

# Execute the command, piping all stdout and stderr to different files.
echo "Executing command: $IREE_COMPILE_COMMAND"
eval $IREE_COMPILE_COMMAND 1> ${STDOUT_FULLPATH} 2> ${STDERR_FULLPATH}

if [ ! -f "${STDOUT_FULLPATH}" ]; then
  echo "stdout file was not created: ${STDOUT_FULLPATH}"
  exit 1
fi

if [ ! -f "${STDERR_FULLPATH}" ]; then
  echo "stderr file was not created: ${STDERR_FULLPATH}"
  exit 1
fi

if [ ! -f "${OUTPUT}/test_artefact.vmfb" ]; then
  echo "test_artefact.vmfb was not created: ${OUTPUT}/test_artefact.vmfb"
  exit 1
fi


# Checks for some stdout from before aie2xclbin:
# CHECK-STDERR: linalg.matmul
#
# Checks for some stderr from during aie2xclbin:
# CHECK-STDERR-DAG: llvm.load
${FILECHECK_EXE} --input-file ${STDERR_FULLPATH} ${0} --check-prefix=CHECK-STDERR

# Checks that timing information is printed for aie2xclbin:
# CHECK-STDERRTIME: Execution time report
# CHECK-STDERRTIME-DAG: Total Execution Time
#   Check for a line of the form '0.0013 (  0.1%)  AIECoreToStandard'
# CHECK-STDERRTIME-DAG: AIECoreToStandard
#   Check for a line of the form: '1.1778 (100.0%)  Total'
# CHECK-STDERRTIME-DAG: Total
${FILECHECK_EXE} --input-file ${STDERR_FULLPATH} ${0} --check-prefix=CHECK-STDERRTIME

# CHECK-STDOUT-DAG: MEM_TOPOLOGY
${FILECHECK_EXE} --input-file ${STDOUT_FULLPATH} ${0} --check-prefix=CHECK-STDOUT

SOURCE_MLIR_FILE="${THIS}/buffers_xclbin.mlir"

IREE_COMPILE_COMMAND="${IREE_COMPILE_EXE} \
${SOURCE_MLIR_FILE} \
--compile-mode=hal-executable \
--iree-hal-target-backends=amd-aie \
--iree-amd-aie-peano-install-dir=${PEANO} \
--iree-amd-aie-install-dir=${IREE_INSTALL_DIR} \
--iree-amd-aie-vitis-install-dir=${VITIS} \
--iree-hal-dump-executable-intermediates-to=${OUTPUT} \
--iree-hal-dump-executable-files-to=${OUTPUT} \
--mlir-disable-threading \
--iree-amd-aie-show-invoked-commands"

echo "Executing command: $IREE_COMPILE_COMMAND"
eval $IREE_COMPILE_COMMAND 1> ${STDOUT_FULLPATH}
if [ ! -f "${STDOUT_FULLPATH}" ]; then
  echo "stdout file was not created: ${STDOUT_FULLPATH}"
  exit 1
fi

${FILECHECK_EXE} --input-file ${OUTPUT}/module_dummy1_amdaie_xclbin_fb/kernels.json $SOURCE_MLIR_FILE

SOURCE_MLIR_FILE="${THIS}/npu_instgen.mlir"

IREE_COMPILE_COMMAND="${IREE_COMPILE_EXE} \
${SOURCE_MLIR_FILE} \
--compile-mode=hal-executable \
--iree-hal-target-backends=amd-aie \
--iree-amd-aie-peano-install-dir=${PEANO} \
--iree-amd-aie-install-dir=${IREE_INSTALL_DIR} \
--iree-amd-aie-vitis-install-dir=${VITIS} \
--iree-hal-dump-executable-intermediates-to=${OUTPUT} \
--iree-hal-dump-executable-files-to=${OUTPUT} \
--mlir-disable-threading \
--iree-amd-aie-show-invoked-commands"

echo "Executing command: $IREE_COMPILE_COMMAND"
eval $IREE_COMPILE_COMMAND 1> ${STDOUT_FULLPATH}
if [ ! -f "${STDOUT_FULLPATH}" ]; then
  echo "stdout file was not created: ${STDOUT_FULLPATH}"
  exit 1
fi

${FILECHECK_EXE} --input-file ${OUTPUT}/module_dummy1_amdaie_xclbin_fb/dummy2_0.npu.txt $SOURCE_MLIR_FILE

echo "Test of printing in aie2xclbin passed."
