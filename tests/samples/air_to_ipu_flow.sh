#!/bin/bash
set -e
set -x

IREE_DIR=${IREE_DIR:-${HOME}/iree}
IREE_BUILD_DIR=${IREE_BUILD_DIR:-${IREE_DIR}/build/Debug_AIE}
IREE_AMD_AIE_DIR=${IREE_AMD_AIE_DIR:-${IREE_DIR}/iree-amd-aie}
IREE_OPT=${IREE_OPT:-${IREE_BUILD_DIR}/tools/iree-opt}
SAMPLES_DIR=${SAMPLES_DIR:-${PWD}}

DEBUG_FLAGS= # "--mlir-print-ir-after-all --mlir-print-ir-before-all --mlir-disable-threading"

${IREE_OPT} ${SAMPLES_DIR}/air_to_ipu_expected_input.mlir \
  --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(air-par-to-herd{depth=1}, air-par-to-launch{has-air-segment=true}, air-copy-to-dma, canonicalize, cse))))' ${DEBUG_FLAGS} | \
${IREE_OPT} \
  --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(air-dependency, air-dependency-schedule-opt, air-specialize-dma-broadcast, air-dma-to-channel, canonicalize, cse, air-dependency-canonicalize, canonicalize, cse, air-label-scf-for-to-ping-pong))))' ${DEBUG_FLAGS} | \
${IREE_OPT} \
  --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(air-ping-pong-transform{keep-memref-dealloc=true}, air-dealias-memref, canonicalize, cse, air-fuse-channels, canonicalize, cse, air-label-scf-for-in-segment, air-unroll-loop-for-pipelining-pattern))))' ${DEBUG_FLAGS} | \
${IREE_OPT} \
    --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(air-collapse-herd), canonicalize, cse, air-place-herds{num-rows=4 num-cols=1 row-anchor=2 col-anchor=0}, canonicalize, cse, func.func(air-renumber-dma, convert-linalg-to-loops)))))' ${DEBUG_FLAGS} | \
${IREE_OPT} \
    --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(air-to-aie{row-offset=2 col-offset=0 device=ipu}, air-to-std, airrt-to-ipu, canonicalize))))' ${DEBUG_FLAGS}
