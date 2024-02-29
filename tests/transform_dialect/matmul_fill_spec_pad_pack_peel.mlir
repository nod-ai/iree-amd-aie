// RUN: iree-opt --iree-transform-dialect-interpreter %s

func.func @matmul_i32(%arg0 : tensor<1024x2048xi32>, %arg1 : tensor<2048x512xi32>) -> tensor<1024x512xi32> {
  %c0 = arith.constant 0: i32
  %empty = tensor.empty() : tensor<1024x512xi32>
  %0 = linalg.fill ins(%c0 : i32) outs(%empty : tensor<1024x512xi32>) -> tensor<1024x512xi32>
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<1024x2048xi32>, tensor<2048x512xi32>)
      outs(%0 : tensor<1024x512xi32>) -> tensor<1024x512xi32>
  return %1 : tensor<1024x512xi32>
}

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

    // First level tile to forall.
    %tiled_matmul, %forall =
      transform.structured.tile_using_forall %matmul tile_sizes [64, 64]
        ( mapping = [#gpu.block<y>, #gpu.block<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse fill operation into the forall loop.
    %fused_fill, %fused_loop = transform.structured.fuse_into_containing_op %fill into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // First level for loop.
    %tiled_reduction, %for_loop =
      transform.structured.tile_using_for %tiled_matmul [0, 0, 256]
      : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)

    // Pad operation.
    %padded, %pad, %__ = transform.structured.pad %tiled_reduction {
      padding_values=[0 : i32, 0 : i32, 0 : i32],
      padding_dimensions=[0, 1, 2],
      pad_to_multiple_of=[64, 64, 64],
      pack_paddings=[1, 1, 1],
      copy_back_op="linalg.copy"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %pad_dps = transform.structured.rewrite_in_destination_passing_style %pad : (!transform.any_op) -> !transform.any_op

    // Promote the input operands to shared memory.
    %padded_lhs = transform.get_producer_of_operand %padded[0] : (!transform.any_op) -> (!transform.any_op)
    %padded_lhs_buffer, %padded_lhs_new = transform.structured.bufferize_to_allocation %padded_lhs
        {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    %padded_rhs = transform.get_producer_of_operand %padded[1] : (!transform.any_op) -> (!transform.any_op)
    %padded_rhs_buffer, %padded_rhs_new = transform.structured.bufferize_to_allocation %padded_rhs
        {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Promote the result to shared memrory
    %padded_result = transform.get_producer_of_operand %padded[2] : (!transform.any_op) -> (!transform.any_op)
    %padded_result_buffer, %padded_result_new = transform.structured.bufferize_to_allocation %padded_result
        {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Clean up.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Second level tile to forall with tile_sizes.
    %tiled_matmul_1, %forall_1 =
      transform.structured.tile_using_forall %padded tile_sizes [32, 32]
        ( mapping = [#gpu.thread<y>, #gpu.thread<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Pack by applying data tiling, and the linalg.matmul becomes linalg.generic.
    %packed = transform.structured.pack %tiled_matmul_1 packed_sizes = [4, 8, 8]
      : (!transform.any_op) -> (!transform.any_op)

    // Transpose A matrix.
    %pack_producer_a = transform.get_producer_of_operand %packed[0]
      : (!transform.any_op) -> (!transform.any_op)
    %packed_a, %pack_a, %empty_unpack_a =
      transform.structured.pack_transpose %pack_producer_a with_compute_op(%packed)
      outer_perm = [1, 0] : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Transpose B matrix.
    %pack_producer_b = transform.get_producer_of_operand %packed_a[1]
      : (!transform.any_op) -> (!transform.any_op)
    %packed_b, %pack_b, %empty_unpack_b =
      transform.structured.pack_transpose %pack_producer_b with_compute_op(%packed_a)
      outer_perm = [1, 0] inner_perm = [1, 0] : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Transpose C matrix.
    %unpack = transform.get_consumers_of_result %packed_b[0]
      : (!transform.any_op) -> (!transform.any_op)
    %packed_c, %pack_c, %unpack_c =
      transform.structured.pack_transpose %unpack with_compute_op(%packed_b)
      outer_perm = [1, 0] : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Promote the result to local memory.
    %buffer_c, %new_c = transform.structured.bufferize_to_allocation %pack_c
      {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Second level for loop.
    %tiled_reduction_1, %for_loop_1 =
      transform.structured.tile_using_for %packed_c [0, 0, 4]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse pack ops into the for loop.
    %fused_pack_a, %fused_loop_a = transform.structured.fuse_into_containing_op %pack_a into %for_loop_1
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_pack_b, %fused_loop_b = transform.structured.fuse_into_containing_op %pack_b into %for_loop_1
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Promote the inputs to local memory.
    %buffer_a, %new_a = transform.structured.bufferize_to_allocation %fused_pack_a
      {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op
    %buffer_b, %new_b = transform.structured.bufferize_to_allocation %fused_pack_b
      {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Clean up.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Hoist static alloc out of the loops
    %memref_func = transform.structured.match ops{["func.func"]} in %variant_op
      : (!transform.any_op) -> !transform.any_op
    transform.iree.hoist_static_alloc %memref_func : (!transform.any_op) -> ()

    // Peel the first iteration out of the for loop.
    %1 = transform.get_consumers_of_result %fused_fill[0] : (!transform.any_op) -> !transform.op<"scf.for">
    %main_loop, %remainder = transform.loop.peel %1 {peel_front = true}
      : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Find the fill and the second forall operations.
    %fused_fill_1 = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %fill_consumer = transform.get_consumers_of_result %fused_fill_1[0] : (!transform.any_op) -> (!transform.any_op)

    // Fuse fill operation into the forall loop
    %fused_fill_2, %fused_loop_2 = transform.structured.fuse_into_containing_op %fused_fill_1 into %fill_consumer
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Clean up.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
    transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()

    // Bufferize and drop HAL decriptor from memref ops.
    %variant_op_3 = transform.iree.bufferize %variant_op : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
