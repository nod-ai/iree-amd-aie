// This script shows an example lowering matmul through IREE for a special accelerator.
//
// ```
//   export IREE_DIR=${HOME}/iree/iree; \
//   export IREE_AMD_AIE_DIR=${IREE_AMD_AIE_DIR:-${HOME}/iree/iree-amd-aie}
//   ${IREE_DIR}/build/tools/iree-opt \
//     ${IREE_AMD_AIE_DIR}/tests/samples/matmul_static.mlir \
//     --iree-hal-target-backends=llvm-cpu \
//     --iree-abi-transformation-pipeline \
//     --iree-flow-transformation-pipeline \
//     --iree-stream-transformation-pipeline \
//     --iree-hal-configuration-pipeline | \
//   ${IREE_DIR}/build/tools/iree-opt \
//      --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target, builtin.module(func.func(iree-hoist-statically-bound-allocations)))))' \
//      --iree-codegen-llvmcpu-use-transform-dialect=${IREE_AMD_AIE_DIR}/tests/samples/matmul_spec_pad.mlir
// ```

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
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!transform.any_op) -> !transform.any_op

    // First level tile to forall with tile_sizes [16, 32].
    %tiled_matmul, %forall =
      transform.structured.tile_using_forall %matmul tile_sizes [16, 32]
        ( mapping = [#gpu.block<y>, #gpu.block<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall
      : (!transform.any_op) -> ()

    // Pad operation.
    %padded, %pad, %__ = transform.structured.pad %tiled_matmul {
      padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32],
      padding_dimensions=[0, 1, 2],
      pack_paddings=[1, 1, 1],
      copy_back_op="none"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %pad_dps = transform.structured.rewrite_in_destination_passing_style %pad : (!transform.any_op) -> !transform.any_op

    // Second level tile to forall with tile_sizes [8, 8].
    %tiled_matmul_1, %forall_1 =
      transform.structured.tile_using_forall %padded tile_sizes [8, 8]
        ( mapping = [#gpu.thread<y>, #gpu.thread<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Pad operation.
    %padded_1, %pad_1, %_ = transform.structured.pad %tiled_matmul_1 {
      padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32],
      padding_dimensions=[0, 1, 2],
      pack_paddings=[0, 0, 1],
      copy_back_op="none"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %pad_1_dps = transform.structured.rewrite_in_destination_passing_style %pad_1 : (!transform.any_op) -> !transform.any_op

    // Tile reduction dimension.
    %tiled_reduction, %loop =
      transform.structured.tile_using_for %padded_1 [0, 0, 8]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Clean up.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
    transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()

    // Bufferize and drop HAL decriptor from memref ops.
    %variant_op_3 = transform.iree.bufferize %variant_op : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
