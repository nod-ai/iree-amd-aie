#!/bin/bash
set -e
set -x

IREE_DIR=${IREE_DIR:-${HOME}/iree}
IREE_BUILD_DIR=${IREE_BUILD_DIR:-${IREE_DIR}/build/Debug_AIE}
IREE_AMD_AIE_DIR=${IREE_AMD_AIE_DIR:-${IREE_DIR}/iree-amd-aie}
IREE_OPT=${IREE_OPT:-${IREE_BUILD_DIR}/tools/iree-opt}
SAMPLES_DIR=${SAMPLES_DIR:-${PWD}}

${IREE_OPT} ${SAMPLES_DIR}/matmul_fill_static.mlir \
  --iree-hal-target-backends=amd-aie \
  --iree-abi-transformation-pipeline \
  --iree-flow-transformation-pipeline \
  --iree-stream-transformation-pipeline \
  --iree-hal-configuration-pipeline | \
${IREE_OPT} \
  --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-codegen-materialize-user-configs, iree-amdaie-lower-executable-target, iree-codegen-erase-hal-descriptor-type-from-memref, iree-amdaie-bridge-to-air)))' \
  --iree-codegen-transform-dialect-library=${SAMPLES_DIR}/matmul_fill_spec_pad.mlir | \
${IREE_OPT} \
  --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(air-par-to-herd{depth=1}, air-par-to-launch{has-air-segment=true}, air-copy-to-dma, canonicalize, cse))))' | \
${IREE_OPT} \
  --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(air-dependency, air-dependency-schedule-opt, air-dma-to-channel, canonicalize, cse, air-dependency-canonicalize, canonicalize, cse, air-label-scf-for-to-ping-pong))))' | \
${IREE_OPT} \
  --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(air-ping-pong-transform{keep-memref-dealloc=true}, air-dealias-memref, canonicalize, cse, air-label-scf-for-in-segment, air-unroll-loop-for-pipelining-pattern))))' | \
${IREE_OPT} \
  --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(air-place-herds{num-rows=2 num-cols=2 row-anchor=2 col-anchor=0}, canonicalize, cse, func.func(air-renumber-dma, convert-linalg-to-loops)))))'
