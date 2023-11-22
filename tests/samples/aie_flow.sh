#!/bin/bash
set -e
set -x

IREE_DIR=${IREE_DIR:-${HOME}/iree}
IREE_BUILD_DIR=${IREE_BUILD_DIR:-${IREE_DIR}/build/Debug_AIE}
IREE_AMD_AIE_DIR=${IREE_AMD_AIE_DIR:-${IREE_DIR}/iree-amd-aie}
IREE_OPT=${IREE_OPT:-${IREE_BUILD_DIR}/tools/iree-opt}
SAMPLES_DIR=${SAMPLES_DIR:-${PWD}}
IREE_COMPILE=${IREE_COMPILE:-${IREE_BUILD_DIR}/tools/iree-compile}

DEBUG_FLAGS= # "--mlir-print-ir-after-all --mlir-print-ir-before-all --mlir-disable-threading"

${IREE_COMPILE} --iree-hal-target-backends=amd-aie --iree-codegen-transform-dialect-library=${SAMPLES_DIR}/matmul_fill_spec_pad.mlir ${SAMPLES_DIR}/matmul_fill_static_i32.mlir

