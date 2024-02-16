#!/usr/bin/env bash

set -xe
MLIR_AIE_INSTALL= `realpath .venv/lib/python3.10/site-packages/mlir_aie`
TESTDIR="$1"

BASE_DIR=`realpath "$(dirname $0)/../.."`
IREE_DIR="$2"
if [ -e "${IREE_DIR}/tools/iree-compile" ]; then
    IREE_BIN=`realpath "${IREE_DIR}/tools"`
else
    IREE_BIN=`realpath "${IREE_DIR}/bin"`
fi

MLIRFILE="${BASE_DIR}/tests/samples/pad_pipeline_e2e.mlir"

mkdir -p "$TESTDIR"
cd "$TESTDIR"

OUTPUT=output.vmfb
XRT_DIR=/opt/xilinx/xrt
PEANO=/opt/llvm-aie
VITIS=/opt/Xilinx/Vitis/2023.2
# An alternate to a full vitis install, will work here but not for a full build of mlir-aie
# https://riallto.ai/install-riallto.html#install-riallto
# VITIS=/opt/Riallto/Vitis/2023.1

source $XRT_DIR/setup.sh

"${IREE_BIN}/iree-compile" \
    --iree-hal-target-backends=amd-aie "${MLIRFILE}" \
    --iree-amd-aie-peano-install-dir "${PEANO}" \
    --iree-amd-aie-mlir-aie-install-dir "${MLIR_AIE_INSTALL}" \
    --iree-amd-aie-vitis-install-dir "${VITIS}" \
    --iree-hal-dump-executable-files-to=$PWD \
    --iree-amd-aie-show-invoked-commands \
    --iree-amdaie-use-pipeline=pad -o "${OUTPUT}"

XCLBIN="module_matmul_static_dispatch_0_amdaie_xclbin_fb.xclbin"
# ensure that we deploy with a unique file name to avoid conflicts with other jobs
XCLBIN_UNIQ="github.${GITHUB_RUN_ID}.${GITHUB_RUN_ATTEMPT}.${XCLBIN}"
cp "${XCLBIN}" "${XCLBIN_UNIQ}"
sudo $XRT_DIR/amdxdna/setup_xclbin_firmware.sh -dev Phoenix -xclbin "${XCLBIN_UNIQ}"
flock /tmp/ipu.lock "${IREE_BIN}/iree-run-module" --device=xrt --module="${OUTPUT}"  --input=8x16xi32=2 --input=16x8xi32=3
