#!/usr/bin/env bash

set -xe

TESTDIR="$1"

BASE_DIR=`realpath "$(dirname $0)/../.."`
IREE_DIR="${2:-${BASE_DIR}/../iree}"
if [ -d "${IREE_DIR}/tools" ]; then
    IREE_BIN=`realpath "${IREE_DIR}/tools"`
else
    IREE_BIN=`realpath "${IREE_DIR}/bin"`
fi

MLIRFILE="${BASE_DIR}/tests/samples/pad_pipeline_e2e.mlir"

mkdir -p "$TESTDIR"
cd "$TESTDIR"

python3 -m venv sandbox
source sandbox/bin/activate
MLIR_AIE=mlir_aie-0.0.1.2024021121+e6874fb
pip install https://github.com/Xilinx/mlir-aie/releases/download/dev-wheels/${MLIR_AIE}-py3-none-manylinux_2_35_x86_64.whl
MLIR_AIE_INSTALL=sandbox/lib/python3.10/site-packages/mlir_aie

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
    --iree-hal-dump-executable-intermediates-to=$PWD \
    --iree-amd-aie-show-invoked-commands \
    --iree-amdaie-use-pipeline=pad -o "${OUTPUT}"

XCLBIN="module_matmul_static_dispatch_0_amdaie_xclbin_fb.xclbin"
sudo $XRT_DIR/amdxdna/setup_xclbin_firmware.sh -dev Phoenix -xclbin $XCLBIN
"${IREE_BIN}/iree-run-module" --device=xrt --module="${OUTPUT}"  --input=8x16xi32=2 --input=16x8xi32=3
sudo $XRT_DIR/amdxdna/rm_xclbin.sh $XCLBIN
