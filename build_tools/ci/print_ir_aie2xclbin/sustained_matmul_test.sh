#!/bin/bash
#
# Copyright 2024 The LLVM Project
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euox pipefail

# Check for the number of provided arguments
if [ "$#" -ne 2 ] && [ "$#" -ne 6 ]; then
    echo -e "Illegal number of parameters: $#." \
            "\n For 2 parameters:" \
            "\n     1) <iree-compile-dir>" \
            "\n     2) <output-dir>" \
            "\n For 6 parameters:" \
            "\n     1) <iree-compile-dir>" \
            "\n     2) <output-dir>" \
            "\n     3) <peano-install-dir>" \
            "\n     4) <xrt-dir>" \
            "\n     5) <vitis-install-dir>" \
            "\n     6) <mlir-aie-install-dir>"\
            "\n Example (dependent on environment variables):" \
            "\n     ./print_ir_aie2xclbin.sh " \
            "\$IREE_BUILD_DIR/tools " \
            "results_dir_tmp "\
            "\$PEANO_INSTALL_DIR "\
            "/opt/xilinx/xrt "\
            "\$VITIS_INSTALL_PATH "\
            "\$MLIR_AIE_INSTALL_DIR"
    exit 1
fi


OUTPUT=`realpath "${2}"`
mkdir -p ${OUTPUT}

# The CI case:
if [ "$#" -eq 2 ]; then
  PEANO=/opt/llvm-aie
  XRT=/opt/xilinx/xrt
  VITIS=/opt/Xilinx/Vitis/2023.2
  MLIR_AIE=`realpath .venv/lib/python3.10/site-packages/mlir_aie`
fi

# The local set-paths-manually case:
if [ "$#" -eq 6 ]; then
  PEANO="$3"
  XRT="$4"
  VITIS="$5"
  MLIR_AIE="$6"
fi

IREE_COMPILE="$1"
if [ -d "${IREE_COMPILE}" ]; then
   IREE_COMPILE=`realpath "${IREE_COMPILE}"`
else
  echo "IREE_COMPILE does not exist: ${IREE_COMPILE}."
  exit 1
fi

IREE_COMPILE_EXE="${IREE_COMPILE}/iree-compile"
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

if [ -d "${MLIR_AIE}" ]; then
  MLIR_AIE=`realpath "${MLIR_AIE}"`
else
  echo "MLIR_AIE does not exist: ${MLIR_AIE}"
  exit 1
fi

# There might be a FileCheck program in the IREE_COMPILE. Check.
# Do not fail if it is not there, we can also check if it already on PATH.
if [ -x "${IREE_COMPILE}/FileCheck" ]; then
  FILECHECK_EXE="${IREE_COMPILE}/FileCheck"
elif [ -x "$(command -v FileCheck)" ]; then
  FILECHECK_EXE="$(command -v FileCheck)"
else
  echo "FileCheck does not exist or isn't executable in ${IREE_COMPILE} or on PATH."
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
--iree-amd-aie-mlir-aie-install-dir=${MLIR_AIE} \
--iree-amd-aie-vitis-install-dir=${VITIS} \
--iree-hal-dump-executable-files-to=${OUTPUT} \
--iree-amdaie-use-pipeline=simple-pack \
-o ${OUTPUT}/module.vmfb "


# Execute the command, piping all stdout and stderr to different files.
echo "Executing command: $IREE_COMPILE_COMMAND"
eval $IREE_COMPILE_COMMAND 1> ${STDOUT_FULLPATH} 2> ${STDERR_FULLPATH}

if [ ! -f "${OUTPUT}/module.vmfb" ]; then
  echo "module.vmfb was not created: ${OUTPUT}/module.vmfb"
  exit 1
fi


# Checks for some stdout from before aie2xclbin:
# CHECK-STDERR: linalg.matmul
#
# Checks for some stderr from during aie2xclbin:
# CHECK-STDERR-DAG: aie.wire
# CHECK-STDERR-DAG: llvm.load
${FILECHECK_EXE} --input-file ${STDERR_FULLPATH} ${0} --check-prefix=CHECK-STDERR

# CHECK-STDOUT-DAG: Bootgen
# CHECK-STDOUT-DAG: MEM_TOPOLOGY
${FILECHECK_EXE} --input-file ${STDOUT_FULLPATH} ${0} --check-prefix=CHECK-STDOUT

