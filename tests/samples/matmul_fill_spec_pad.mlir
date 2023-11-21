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
//      --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-codegen-materialize-user-configs, iree-amdaie-lower-executable-target, iree-codegen-erase-hal-descriptor-type-from-memref, iree-amdaie-bridge-to-air)))' \
//      --iree-codegen-transform-dialect-library=${IREE_AMD_AIE_DIR}/tests/samples/matmul_fill_spec_pad.mlir
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
    %ops = transform.structured.match ops{["linalg.fill", "linalg.matmul"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %fill, %matmul = transform.split_handle %ops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // First level tile to forall with tile_sizes [8, 8].
    %tiled_matmul, %forall =
      transform.structured.tile_using_forall %matmul tile_sizes [32, 32, 0]
        ( mapping = [#gpu.block<y>, #gpu.block<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall
      : (!transform.any_op) -> ()

    // Fuse fill operation into the loop
    %fused_fill, %fused_for_all = transform.structured.fuse_into_containing_op %fill into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Pad operation.
    %padded, %pad, %__ = transform.structured.pad %tiled_matmul {
      padding_values=[0 : i32, 0 : i32, 0 : i32],
      padding_dimensions=[0, 1],
      pack_paddings=[1, 1, 1],
      copy_back_op="linalg.copy"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %pad_dps = transform.structured.rewrite_in_destination_passing_style %pad : (!transform.any_op) -> !transform.any_op

    // Promote the operands to shared memory.
    %padded_lhs = transform.get_producer_of_operand %padded[0] : (!transform.any_op) -> (!transform.any_op)
    %padded_lhs_buffer, %padded_lhs_new = transform.structured.bufferize_to_allocation %padded_lhs
        {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    %padded_rhs = transform.get_producer_of_operand %padded[1] : (!transform.any_op) -> (!transform.any_op)
    %padded_rhs_buffer, %padded_rhs_new = transform.structured.bufferize_to_allocation %padded_rhs
        {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    %padded_out = transform.get_producer_of_operand %padded[2] : (!transform.any_op) -> (!transform.any_op)
    %padded_out_buffer, %padded_out_new = transform.structured.bufferize_to_allocation %padded_out
        {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Run canonicalizations.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Find the matmul and fill again
    %tiled_ops = transform.structured.match ops{["linalg.fill", "linalg.matmul"]} in %fused_for_all : (!transform.any_op) -> !transform.any_op
    %tiled_fill_op, %tiled_padded_matmul = transform.split_handle %tiled_ops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Second level tile to forall with tile_sizes [4, 4].
    %tiled_matmul_1, %forall_1 =
      transform.structured.tile_using_forall %tiled_padded_matmul tile_sizes [16, 16, 0]
        ( mapping = [#gpu.thread<y>, #gpu.thread<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_fill_2, %fused_for_all_2 = transform.structured.fuse_into_containing_op %tiled_fill_op into %forall_1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Pad operation.
    %padded_1, %pad_1, %_ = transform.structured.pad %tiled_matmul_1 {
      padding_values=[0 : i32, 0 : i32, 0 : i32],
      padding_dimensions=[0, 1],
      pack_paddings=[0, 0, 1],
      copy_back_op="linalg.copy"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %pad_1_dps = transform.structured.rewrite_in_destination_passing_style %pad_1 : (!transform.any_op) -> !transform.any_op

    // Promote the result to local memory.
    %padded_result = transform.get_producer_of_operand %padded_1[2] : (!transform.any_op) -> (!transform.any_op)
    %padded_result_buffer, %padded_result_new = transform.structured.bufferize_to_allocation %padded_result
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Run canonicalizations.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Tile reduction dimension.
    %tiled_reduction, %loop =
      transform.structured.tile_using_for %padded_1 [0, 0, 8]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Clean up.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Pad operation.
    %padded_reduction, %pad_reduction, %___ = transform.structured.pad %tiled_reduction {
      padding_values=[0 : i32, 0 : i32, 0 : i32],
      padding_dimensions=[0, 1, 2],
      pack_paddings=[1, 1, 0],
      copy_back_op="none"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %pad_2_dps = transform.structured.rewrite_in_destination_passing_style %pad_reduction : (!transform.any_op) -> !transform.any_op

    // Promote to local memory
    %padded_reduction_lhs = transform.get_producer_of_operand %padded_reduction[0] : (!transform.any_op) -> (!transform.any_op)
    %padded_reduction_lhs_buffer, %padded_reduction_lhs_new = transform.structured.bufferize_to_allocation %padded_reduction_lhs
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    %padded_reduction_rhs = transform.get_producer_of_operand %padded_reduction[1] : (!transform.any_op) -> (!transform.any_op)
    %padded_reduction_rhs_buffer, %padded_reduction_rhs_new = transform.structured.bufferize_to_allocation %padded_reduction_rhs
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Clean up.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Bufferize and drop HAL decriptor from memref ops.
    transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
    %variant_op_3 = transform.iree.bufferize %variant_op : (!transform.any_op) -> !transform.any_op

    // Hoist static allocations.
    %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3
      : (!transform.any_op) -> !transform.any_op
    // transform.iree.hoist_static_alloc %memref_func : (!transform.any_op) -> ()
    transform.yield
  }
}
