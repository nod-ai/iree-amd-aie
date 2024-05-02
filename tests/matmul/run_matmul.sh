export IREE_BUILD_DIR=${IREE_BUILD_DIR:-${WORK}/versal/iree-build5}
export IREE_AMD_AIE_DIR=${IREE_AMD_AIE_DIR:-${WORK}/versal/iree-amd-aie} 

export PASSES="fold-memref-alias-ops,iree-amdaie-pack-to-dma,air-copy-to-dma,iree-amdaie-air-dma-to-amdaie-dma"
# export PASSES="${PASSES},func.func(iree-amdaie-map-forall-to-cores)"
# export PASSES="${PASSES},scf-for-loop-specialization"
# export PASSES="${PASSES},func.func(iree-amdaie-insert-aie-workgroup2)"
# export PASSES="${PASSES},iree-amdaie-unroll-and-distribute-workgroup2"
# export PASSES="${PASSES},iree-amdaie-insert-aie-workgroup,iree-amdaie-fuse-logicalobjectfifo-into-workgroup,cse"
# export PASSES="${PASSES},iree-amdaie-unroll-and-distribute-workgroup" # ,canonicalize,cse

export PASSES="${PASSES},iree-amdaie-insert-aie-workgroup,iree-amdaie-fuse-logicalobjectfifo-into-workgroup,cse"
export PASSES="${PASSES},iree-amdaie-unroll-and-distribute-workgroup" # ,cse
export PASSES="${PASSES},cse,iree-amdaie-dma-to-circular-dma"
export PASSES="${PASSES},func.func(iree-amdaie-create-aie-workgroup),cse"
export PASSES="${PASSES},iree-amdaie-canonicalize-doubly-strided-op"
export PASSES="${PASSES},iree-amdaie-consume-produce-to-acquire-release,cse,canonicalize"
export PASSES="${PASSES},iree-amdaie-controlcode-loop-unroll,cse,canonicalize"
export PASSES="${PASSES},canonicalize,iree-amdaie-lower-to-aie,canonicalize"
export PASSES="${PASSES},convert-linalg-to-loops"
${IREE_BUILD_DIR}/tools/iree-opt \
    --pass-pipeline="builtin.module(${PASSES})" \
    --mlir-print-ir-after-all matmul_7.mlir \
    -debug-only=iree-amdaie-lower-to-aie
    # --pass-pipeline="builtin.module(${passes})" \
    #--pass-pipeline="builtin.module(fold-memref-alias-ops,iree-amdaie-pack-to-dma,air-copy-to-dma,iree-amdaie-to-objectfifo,iree-amdaie-logical-objectfifo-from-memref-cleanup,iree-amdaie-fuse-dma-copy-into-aie-region,iree-amdaie-fuse-logicalobjectfifo-from-memref-into-workgroup,iree-amdaie-fuse-scf-for-into-aie-region,iree-amdaie-fuse-scf-for-into-aie-core,iree-amdaie-fuse-aie-regions,iree-amdaie-fuse-logicalobjectfifo-from-memref-into-workgroup,iree-amdaie-logical-objectfifo-from-memref-cleanup)" \
    

    
# ${IREE_BUILD_DIR}/tools/iree-opt \
#     matmul_1.mlir \
#     --pass-pipeline="builtin.module(iree-amdaie-aie-lowering-pipeline)" \
#     --mlir-print-ir-after-all \
#     > module_dump.mlir 2>&1