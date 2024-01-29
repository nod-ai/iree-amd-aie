// This script shows an example lowering matmul through IREE for a special accelerator.
//
// ```
//   export IREE_BUILD_DIR=${IREE_BUILD_DIR:-${HOME}/iree/build/Debug}
//   export IREE_AMD_AIE_DIR=${IREE_AMD_AIE_DIR:-${HOME}/iree/iree-amd-aie}
//   ${IREE_BUILD_DIR}/tools/iree-opt \
//     ${IREE_AMD_AIE_DIR}/tests/samples/matmul_fill_static_i8_i32.mlir \
//     --iree-hal-target-backends=amd-aie \
//     --iree-abi-transformation-pipeline \
//     --iree-flow-transformation-pipeline \
//     --iree-stream-transformation-pipeline \
//     --iree-hal-configuration-pipeline | \
//   ${IREE_BUILD_DIR}/tools/iree-opt \
//      --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-codegen-materialize-user-configs, iree-amdaie-lower-executable-target, fold-memref-alias-ops)))' \
//      --iree-codegen-transform-dialect-library=${IREE_AMD_AIE_DIR}/tests/samples/matmul_fill_spec_pack_peel.mlir | \
//   ${IREE_BUILD_DIR}/tools/iree-aie-translate \
//      --serialize-accel \
//      --allow-unregistered-dialect
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
    transform.apply_cse to %func : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.read_only}) {
    %ops = transform.structured.match ops{["linalg.fill", "linalg.matmul"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %fill, %matmul = transform.split_handle %ops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // First level tile to forall with tile_sizes [16, 256] to target 4 cores. Adjust to [16, 64] for 1 core.
    %tiled_matmul, %forall =
      transform.structured.tile_using_forall %matmul tile_sizes [16, 256]
        ( mapping = [#gpu.block<y>, #gpu.block<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall
      : (!transform.any_op) -> ()

    // Fuse fill operation into the forall loop.
    %fused_fill, %_ = transform.structured.fuse_into_containing_op %fill into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Tile reduction dimension.
    %tiled_reduction, %loop =
      transform.structured.tile_using_for %tiled_matmul [0, 0, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)

    // Pack by applying data tiling, and the linalg.matmul becomes linalg.generic.
    %packed = transform.structured.pack %tiled_reduction packed_sizes = [16, 64, 64]
      : (!transform.any_op) -> (!transform.any_op)

    // Transpose B matrix from [K N n k] to [K N k n]
    %pack_producer_b0 = transform.get_producer_of_operand %packed[1]
      : (!transform.any_op) -> (!transform.any_op)
    %packed_b0, %pack_b0, %empty_unpack_b0 =
      transform.structured.pack_transpose %pack_producer_b0 with_compute_op(%packed)
      inner_perm = [1, 0] : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Bufferize to shared memory allocation
    %pack_producer_a0 = transform.get_producer_of_operand %packed_b0[0]
      : (!transform.any_op) -> (!transform.any_op)
    %pack_producer_c0 = transform.get_producer_of_operand %packed_b0[2]
      : (!transform.any_op) -> (!transform.any_op)
    %buffer_a0, %new_a0 = transform.structured.bufferize_to_allocation %pack_b0
      {memory_space = "shared", bufferize_destination_only, emit_dealloc} : !transform.any_op
    %buffer_b0, %new_b0 = transform.structured.bufferize_to_allocation %pack_producer_a0
      {memory_space = "shared", bufferize_destination_only, emit_dealloc} : !transform.any_op
    %buffer_c0, %new_c0 = transform.structured.bufferize_to_allocation %pack_producer_c0
      {memory_space = "shared", bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Second level tile to forall with tile_sizes [1, 1].
    %tiled_matmul_1, %forall_1 =
      transform.structured.tile_using_forall %packed_b0 tile_sizes [1, 1]
        ( mapping = [#gpu.thread<y>, #gpu.thread<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Pack by applying data tiling, and the linalg.matmul becomes linalg.generic.
    %packed_2 = transform.structured.pack %tiled_matmul_1 packed_sizes = [0, 0, 0, 4, 8, 8]
      : (!transform.any_op) -> (!transform.any_op)

    // Transpose A matrix from [M K m k m0 k0] to [M K k m m0 k0]
    %pack_producer_a = transform.get_producer_of_operand %packed_2[0]
      : (!transform.any_op) -> (!transform.any_op)
    %packed_a, %pack_a, %empty_unpack_a =
      transform.structured.pack_transpose %pack_producer_a with_compute_op(%packed_2)
      outer_perm = [0, 1, 3, 2] : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Transpose B matrix from [K N k n n0 k0] to [K N n k k0 n0]
    %pack_producer_b = transform.get_producer_of_operand %packed_a[1]
      : (!transform.any_op) -> (!transform.any_op)
    %packed_b, %pack_b, %empty_unpack_b =
      transform.structured.pack_transpose %pack_producer_b with_compute_op(%packed_a)
      outer_perm = [0, 1, 3, 2] inner_perm = [1, 0] : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Transpose C matrix from [M N m n m0 n0] to [M N n m m0 n0]
    %unpack = transform.get_consumers_of_result %packed_b[0]
      : (!transform.any_op) -> (!transform.any_op)
    %packed_c, %pack_c, %unpack_c =
      transform.structured.pack_transpose %unpack with_compute_op(%packed_b)
      outer_perm = [0, 1, 3, 2] : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Bufferize to local memory allocation
    %buffer_a, %new_a = transform.structured.bufferize_to_allocation %pack_a
      {memory_space = "local", bufferize_destination_only} : !transform.any_op
    %buffer_b, %new_b = transform.structured.bufferize_to_allocation %pack_b
      {memory_space = "local", bufferize_destination_only} : !transform.any_op
    %buffer_c, %new_c = transform.structured.bufferize_to_allocation %pack_c
      {memory_space = "local", bufferize_destination_only} : !transform.any_op

    // Hoist static alloc out of the loops
    %memref_func = transform.structured.match ops{["func.func"]} in %variant_op
      : (!transform.any_op) -> !transform.any_op
    transform.iree.hoist_static_alloc %memref_func : (!transform.any_op) -> ()

    // Peel the first iteration out of the for loop.
    // This only works when the for loop has more than one iteration.
    %1 = transform.get_parent_op %packed_c {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    %main_loop, %remainder = transform.loop.peel %1 {peel_front = true}
      : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Find the fill and the second forall operations.
    %fused_fill_1 = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %fill_consumer = transform.get_consumers_of_result %fused_fill_1[0] : (!transform.any_op) -> (!transform.any_op)

    // Fuse fill operation into the forall loop
    %fused_fill_2, %__ = transform.structured.fuse_into_containing_op %fused_fill_1 into %fill_consumer
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Clean up.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
    transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()

    // Bufferize and drop HAL decriptor from memref ops.
    %variant_op_3 = transform.iree.bufferize %variant_op : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
