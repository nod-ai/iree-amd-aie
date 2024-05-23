export IREE_BUILD_DIR=${IREE_BUILD_DIR:-${WORK}/versal/iree-build5}

# ${IREE_BUILD_DIR}/tools/iree-opt --iree-transform-dialect-interpreter matmul_fill_spec_pad_pack_peel.mlir

# ${IREE_BUILD_DIR}/tools/iree-compile \
#     test.mlir \
#     --iree-hal-target-backends=amd-aie \
#     --compile-to=executable-sources |
# ${IREE_BUILD_DIR}/tools/iree-opt \
#     --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-codegen-materialize-user-configs, iree-amdaie-lower-executable-target, fold-memref-alias-ops)))' \
#     --iree-codegen-transform-dialect-library=matmul_fill_spec_pad_pack_peel.mlir
#     # --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" \
#     # --iree-codegen-transform-dialect-library=matmul_fill_spec_pad_pack_peel.mlir \

# pack_peel_pipeline_matmul.mlir \
# -debug-only=iree-amdaie-distribute-cores-and-objectfifos \
# ${IREE_BUILD_DIR}/tools/iree-compile \
#     pack_peel_pipeline_matmul.mlir  \
#     --iree-hal-target-backends=amd-aie \
#     --compile-to=executable-sources |
# ${IREE_BUILD_DIR}/tools/iree-opt \
#     --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" \
#     --iree-amdaie-use-pipeline=pack-peel \
#     --mlir-print-ir-before-all \
#     --iree-amdaie-enable-vectorization-passes=false \
#     -debug-only=iree-amdaie-distribute-cores-and-objectfifos \
#     > Log.cc

# matmul_peeled_objectfifo.mlir
# matmul_peeled_2x2.mlir
# ${IREE_BUILD_DIR}/tools/iree-opt matmul_peeled_2x2_small.mlir --mlir-print-ir-before-all --pass-pipeline="builtin.module(fold-memref-alias-ops,iree-amdaie-pack-to-dma,air-copy-to-dma,iree-amdaie-air-dma-to-amdaie-dma,iree-amdaie-normalize-loop-bounds,iree-amdaie-insert-cores,iree-amdaie-localize-logicalobjectfifo,cse,canonicalize,iree-amdaie-distribute-cores-and-objectfifos,cse,iree-amdaie-dma-to-circular-dma,func.func(iree-amdaie-create-aie-workgroup),canonicalize,cse,iree-amdaie-canonicalize-doubly-strided-op,iree-amdaie-access-to-acquire-release,cse,canonicalize,iree-amdaie-controlcode-loop-unroll,cse,canonicalize,iree-amdaie-create-logical-objectfifo-link,iree-amdaie-canonicalize-doubly-strided-op,iree-amdaie-lower-to-aie,iree-amdaie-convert-core-forall-to-for,canonicalize,convert-linalg-to-loops)"
# ${IREE_BUILD_DIR}/tools/iree-opt matmul_peeled_2x2_small.mlir --mlir-print-ir-before-all --debug-only=iree-amdaie-distribute-cores-and-objectfifos --pass-pipeline="builtin.module(fold-memref-alias-ops,iree-amdaie-pack-to-dma,air-copy-to-dma,iree-amdaie-air-dma-to-amdaie-dma,iree-amdaie-normalize-loop-bounds,iree-amdaie-insert-cores,iree-amdaie-localize-logicalobjectfifo,cse,canonicalize,iree-amdaie-distribute-cores-and-objectfifos,cse,iree-amdaie-dma-to-circular-dma,func.func(iree-amdaie-create-aie-workgroup),canonicalize,cse,iree-amdaie-canonicalize-doubly-strided-op,iree-amdaie-access-to-acquire-release,cse,canonicalize,iree-amdaie-controlcode-loop-unroll,cse,canonicalize,iree-amdaie-create-logical-objectfifo-link)"
# ${IREE_BUILD_DIR}/tools/iree-opt matmul_peeled_2x2_small.mlir --mlir-print-ir-before-all --pass-pipeline="builtin.module(fold-memref-alias-ops,iree-amdaie-pack-to-dma,air-copy-to-dma,iree-amdaie-air-dma-to-amdaie-dma,iree-amdaie-insert-cores,iree-amdaie-localize-logicalobjectfifo,cse,iree-amdaie-distribute-cores-and-objectfifos,cse,iree-amdaie-dma-to-circular-dma,func.func(iree-amdaie-create-aie-workgroup),canonicalize,cse,iree-amdaie-canonicalize-doubly-strided-op,iree-amdaie-access-to-acquire-release,cse,canonicalize,iree-amdaie-dma-loop-subsumption,iree-amdaie-combine-strided-ops,canonicalize)"
# ${IREE_BUILD_DIR}/tools/iree-opt matmul_peeled_2x2_small.mlir --mlir-print-ir-before-all --pass-pipeline="builtin.module(fold-memref-alias-ops,iree-amdaie-pack-to-dma,air-copy-to-dma,iree-amdaie-air-dma-to-amdaie-dma,iree-amdaie-insert-cores,iree-amdaie-localize-logicalobjectfifo,cse,iree-amdaie-distribute-cores-and-objectfifos,cse,iree-amdaie-dma-to-circular-dma,func.func(iree-amdaie-create-aie-workgroup),canonicalize,cse,iree-amdaie-canonicalize-doubly-strided-op,iree-amdaie-access-to-acquire-release,cse,canonicalize,iree-amdaie-dma-loop-subsumption,cse,canonicalize,iree-amdaie-assign-npu-dma-bd-ids,iree-amdaie-controlcode-loop-unroll,cse,canonicalize,iree-amdaie-create-logical-objectfifo-link,iree-amdaie-canonicalize-doubly-strided-op,iree-amdaie-lower-to-aie,iree-amdaie-convert-core-forall-to-for,canonicalize,convert-linalg-to-loops)"
# ${IREE_BUILD_DIR}/tools/iree-opt matmul_peeled_2x2_small.mlir --mlir-print-ir-before-all --pass-pipeline="builtin.module(fold-memref-alias-ops,iree-amdaie-pack-to-dma,air-copy-to-dma,iree-amdaie-air-dma-to-amdaie-dma,iree-amdaie-insert-cores,iree-amdaie-localize-logicalobjectfifo,cse,iree-amdaie-distribute-cores-and-objectfifos,cse,iree-amdaie-dma-to-circular-dma,func.func(iree-amdaie-create-aie-workgroup),canonicalize,cse,iree-amdaie-canonicalize-doubly-strided-op,iree-amdaie-access-to-acquire-release)"


# ${IREE_BUILD_DIR}/tools/iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-sources matmul_sample.mlir | 
# ${IREE_BUILD_DIR}/tools/iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" --iree-amdaie-enable-vectorization-passes=false --iree-amdaie-use-pipeline=pack-peel --mlir-print-ir-after-all &> debug_matmul_new.mlir

# --aie2xclbin-print-ir-before-all \
# --mlir-print-ir-before-all \

# ${IREE_BUILD_DIR}/tools/iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-sources matmul_peeled_objectfifo_e2e.mlir | ${IREE_BUILD_DIR}/tools/iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" --iree-amdaie-lower-to-aie-pipeline=objectFifo --iree-amdaie-tile-pipeline=pack-peel --mlir-print-ir-after-all &> Log.cc

# # --iree-amd-aie-enable-chess \
${IREE_BUILD_DIR}/tools/iree-compile --mlir-elide-elementsattrs-if-larger=2 \
    --iree-hal-target-backends=amd-aie \
    --iree-amdaie-tile-pipeline=pack-peel \
    --iree-amdaie-lower-to-aie-pipeline=objectFifo \
    matmul_elemwise_sample.mlir \
    --iree-amd-aie-peano-install-dir=/opt/llvm-aie \
    --iree-amd-aie-vitis-install-dir=/proj/xbuilds/2024.1_released/installs/lin64/Vitis/2024.1 \
    --iree-amd-aie-install-dir=${WORK}/versal/iree-build5 \
    --iree-hal-dump-executable-files-to=$PWD \
    --iree-amd-aie-show-invoked-commands -o pack_peel.vmfb \
    --mlir-disable-threading \
    --iree-amd-aie-show-invoked-commands \
    --mlir-print-ir-before-all \
    &> Log.cc

# ${IREE_BUILD_DIR}/tools/iree-compile failed_ir.mlir \
#     --iree-hal-target-backends=amd-aie \
#     --iree-amdaie-use-pipeline=pack-peel \
#     --iree-amdaie-use-aie-backend=objectfifo \
#     --iree-amd-aie-peano-install-dir=../../../llvm-aie/install \
#      --iree-amd-aie-mlir-aie-install-dir=../../../mlir-aie/install \
#      --iree-amd-aie-vitis-install-dir=/opt/Xilinx/Vitis/2023.2 \
#      --iree-hal-dump-executable-files-to=test_matmuls \
#      --iree-amd-aie-show-invoked-commands \
#      -o objectFifo.vmfb
 