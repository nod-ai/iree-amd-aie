// This script shows an example lowering matmul through IREE for a special accelerator.
//
// ```
//   export IREE_DIR=${HOME}/iree/iree; \
//   export IREE_AMD_AIE_DIR=${IREE_AMD_AIE_DIR:-${HOME}/iree/iree-amd-aie}
//   ${IREE_DIR}/build/tools/iree-opt \
//     ${IREE_AMD_AIE_DIR}/tests/samples/matmul_fill_static.mlir \
//     --iree-hal-target-backends=amd-aie \
//     --iree-abi-transformation-pipeline \
//     --iree-flow-transformation-pipeline \
//     --iree-stream-transformation-pipeline \
//     --iree-hal-configuration-pipeline | \
//   ${IREE_DIR}/build/tools/iree-opt \
//      --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))' \
//      --iree-codegen-transform-dialect-library=${IREE_AMD_AIE_DIR}/tests/samples/matmul_fill_spec_pad_new.mlir \
//      --iree-amd-aie-cpp-passes
// ```


// The first level tiling + fusion + padding + DSP + bufferization + cleanup is done in a C++ pass
// compared to base script.
module attributes { transform.with_named_sequence } {
  transform.named_sequence @cleanup(%variant_op: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
      transform.apply_patterns to %func {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %func : !transform.any_op
    transform.iree.apply_cse %func : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.read_only}) {
    // Bufferize and drop HAL decriptor from memref ops.
    transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
    %variant_op_3 = transform.iree.bufferize %variant_op : (!transform.any_op) -> !transform.any_op

    transform.include @cleanup failures(propagate) (%variant_op_3) : (!transform.any_op) -> ()
    transform.yield
  }
}
