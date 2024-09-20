// RUN: iree-opt --iree-transform-dialect-interpreter %s | FileCheck %s
// This script shows an example lowering matmul for AIE device.
// In this strategy, we use pack operations for data movement from L3 to L2, and L2 to L1.
// In order to keep initialization in L1, the first iteration of scf.for loop is peeled.

#pipeline_layout = #hal.pipeline.layout<bindings= [
    #hal.pipeline.binding<storage_buffer, ReadOnly>,
    #hal.pipeline.binding<storage_buffer, ReadOnly>,
    #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul_i32() {
  %c0_i32 = arith.constant 0: i32
  %c0 = arith.constant 0 : index
  %arg0_binding = hal.interface.binding.subspan layout(#pipeline_layout)  binding(0) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x2048xi32>>
  %arg0 = flow.dispatch.tensor.load %arg0_binding, offsets = [0, 0], sizes = [1024, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x2048xi32>> -> tensor<1024x2048xi32>
  %arg1_binding = hal.interface.binding.subspan layout(#pipeline_layout)  binding(1) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x512xi32>>
  %arg1 = flow.dispatch.tensor.load %arg1_binding, offsets = [0, 0], sizes = [2048, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x512xi32>> -> tensor<2048x512xi32>
  %arg2_binding = hal.interface.binding.subspan layout(#pipeline_layout)  binding(2) offset(%c0) flags(None) : !flow.dispatch.tensor<writeonly:tensor<1024x512xi32>>
  %empty = tensor.empty() : tensor<1024x512xi32>
  %0 = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<1024x512xi32>) -> tensor<1024x512xi32>
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<1024x2048xi32>, tensor<2048x512xi32>)
      outs(%0 : tensor<1024x512xi32>) -> tensor<1024x512xi32>
  flow.dispatch.tensor.store %1, %arg2_binding, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : tensor<1024x512xi32> -> !flow.dispatch.tensor<writeonly:tensor<1024x512xi32>>
  return
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
    %first_level_tiled_matmul, %outer_forall =
      transform.structured.tile_using_forall %matmul tile_sizes [64, 64]
        ( mapping = [#gpu.block<y>, #gpu.block<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse fill operation into the forall loop.
    %fused_fill, %1 = transform.structured.fuse_into_containing_op %fill into %outer_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // First level pack the matmul.
    %first_level_tiled_transposed_l2_packed_matmul = transform.structured.pack %first_level_tiled_matmul packed_sizes = [64, 64, 256]
      : (!transform.any_op) -> (!transform.any_op)

    %lhs_l2_pack = transform.get_producer_of_operand %first_level_tiled_transposed_l2_packed_matmul[0] : (!transform.any_op) -> (!transform.any_op)

    %rhs_transposed_l2_pack_op = transform.get_producer_of_operand %first_level_tiled_transposed_l2_packed_matmul[1] : (!transform.any_op) -> (!transform.any_op)
    %first_level_tiled_l2_packed_matmul, %rhs_l2_pack, %rhs_unpack =
      transform.structured.pack_transpose %rhs_transposed_l2_pack_op with_compute_op(%first_level_tiled_transposed_l2_packed_matmul)
      outer_perm = [0, 1] inner_perm = [1, 0] : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Clean up.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Promote the fused fill to shared memory
    %result_l2 = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %result_l2_buffer, %result_t2_new = transform.structured.bufferize_to_allocation %result_l2
        {memory_space = 1, bufferize_destination_only, mempcy = "linalg.copy", emit_dealloc} : !transform.any_op

    // Second level pack the matmul.
    %generic_op = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %l1_packed = transform.structured.pack %generic_op packed_sizes = [0, 0, 0, 4, 8, 8]
      : (!transform.any_op) -> (!transform.any_op)

    // Transpose A matrix from [M K m k m0 k0] to [M K k m m0 k0]
    %l1_packed_lhs = transform.get_producer_of_operand %l1_packed[0]
      : (!transform.any_op) -> (!transform.any_op)
    %lhs_l1_packed_matmul, %lhs_l1_pack_op, %lhs_l1_unpack_op =
      transform.structured.pack_transpose %l1_packed_lhs with_compute_op(%l1_packed)
      outer_perm = [0, 1, 3, 2] : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Transpose B matrix from [K N k n n0 k0] to [K N n k k0 n0]
    %l1_packed_rhs = transform.get_producer_of_operand %lhs_l1_packed_matmul[1]
      : (!transform.any_op) -> (!transform.any_op)
    %operands_l1_packed_matmul, %rhs_l1_pack_op, %rhs_l1_unpack_op =
      transform.structured.pack_transpose %l1_packed_rhs with_compute_op(%lhs_l1_packed_matmul)
      outer_perm = [0, 1, 3, 2] inner_perm = [1, 0] : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Transpose C matrix from [M N m n m0 n0] to [M N n m m0 n0]
    %l1_packed_output = transform.get_consumers_of_result %operands_l1_packed_matmul[0]
      : (!transform.any_op) -> (!transform.any_op)
    %l1_packed_matmul, %output_l1_pack_op, %output_l1_unpack_op =
      transform.structured.pack_transpose %l1_packed_output with_compute_op(%operands_l1_packed_matmul)
      outer_perm = [0, 1, 3, 2] : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Promote the result to local memory
    %output_l1_pack_op_source_buffer, %output_l1_pack_op_new = transform.structured.bufferize_to_allocation %output_l1_pack_op
        {memory_space = 2, bufferize_destination_only, memcpy_op = "linalg.copy", emit_dealloc} : !transform.any_op

    // First level for loop.
    %first_level_tiled_reduction_matmul, %outer_for_loop =
      transform.structured.tile_using_for %l1_packed_matmul tile_sizes [0, 0, 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse the pack operations in the outer for loop.
    %fused_lhs_l1_pack, %2 = transform.structured.fuse_into_containing_op %lhs_l1_pack_op into %outer_for_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_rhs_l1_pack, %3 = transform.structured.fuse_into_containing_op %rhs_l1_pack_op into %outer_for_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_lhs_l2_pack, %4 = transform.structured.fuse_into_containing_op %lhs_l2_pack into %outer_for_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_rhs_l2_pack, %5 = transform.structured.fuse_into_containing_op %rhs_l2_pack into %outer_for_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Promote the lhs to shared memory
    %lhs_l2_pack_buffer, %lhs_l2_pack_new = transform.structured.bufferize_to_allocation %fused_lhs_l2_pack
       {memory_space = 1, bufferize_destination_only, memcpy_op = "linalg.copy", emit_dealloc} : !transform.any_op

    // Promote the rhs to shared memory
    %rhs_l2_pack_buffer, %rhs_l2_pack_new = transform.structured.bufferize_to_allocation %fused_rhs_l2_pack
       {memory_space = 1, bufferize_destination_only, memcpy_op = "linalg.copy", emit_dealloc} : !transform.any_op

    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Second level tile to forall with tile_sizes.
    %second_level_tiled_matmul, %inner_forall =
      transform.structured.tile_using_forall %first_level_tiled_reduction_matmul tile_sizes [0, 0, 0, 8, 4, 0]
        ( mapping = [#gpu.thread<y>, #gpu.thread<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse the pack operations in inner forall loop.
    %fused_lhs_l1_pack2, %6 = transform.structured.fuse_into_containing_op %fused_lhs_l1_pack into %inner_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_rhs_l1_pack2, %7 = transform.structured.fuse_into_containing_op %fused_rhs_l1_pack into %inner_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Clean up.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Second level for loop.
    %generic_op1 = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %second_level_tiled_reduction_matmul, %inner_for_loop =
      transform.structured.tile_using_for %generic_op1 tile_sizes [0, 0, 0, 0, 0, 4]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse the pack operations in inner for loop.
    %fused_lhs_l1_pack3, %8 = transform.structured.fuse_into_containing_op %fused_lhs_l1_pack2 into %inner_for_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_rhs_l1_pack3, %9 = transform.structured.fuse_into_containing_op %fused_rhs_l1_pack2 into %inner_for_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Promote the LHS to local memory.
    %lhs_l1_pack_buffer, %lhs_l1_pack_new = transform.structured.bufferize_to_allocation %fused_lhs_l1_pack3
      {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Promote the RHS to local memory.
    %rhs_l1_pack_buffer, %rhs_l1_pack_new = transform.structured.bufferize_to_allocation %fused_rhs_l1_pack3
      {memory_space = 2, bufferize_destination_only, memcpy_op = "linalg.copy", emit_dealloc} : !transform.any_op

    // Clean up.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Hoist static alloc out of the loops
    %func = transform.structured.match ops{["func.func"]} in %variant_op
      : (!transform.any_op) -> !transform.any_op
    transform.iree.hoist_static_alloc %func : (!transform.any_op) -> ()

    // Peel the for loop
    %for_ops = transform.structured.match ops{["scf.for"]} in %variant_op : (!transform.any_op) -> !transform.op<"scf.for">
    %inner_for, %outer_for = transform.split_handle %for_ops : (!transform.op<"scf.for">) -> (!transform.op<"scf.for">, !transform.op<"scf.for">)
    %peeled_outer_loop, %remainder = transform.loop.peel %outer_for {peel_front = true}
      : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)

    // Clean up.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Find the fill operation to fuse
    %fill_op = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    // Get the consumers of the fill. It has to be a scf.forall op.
    %peeled_loop = transform.get_consumers_of_result %fill_op[0] : (!transform.any_op) -> (!transform.op<"scf.forall">)

    // Fuse the fill within the loop.
    %peel_fused, %13 = transform.structured.fuse_into_containing_op %fill_op into %peeled_loop : (!transform.any_op, !transform.op<"scf.forall">) -> (!transform.any_op, !transform.any_op)

    // Clean up.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Bufferize and drop HAL decriptor from memref ops.
    %func_op = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.eliminate_empty_tensors %func_op : (!transform.any_op) -> ()
    %memref_func = transform.iree.bufferize %func_op : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}

// CHECK-LABEL: @matmul_i32
//       CHECK: memref.alloc() : memref<1x1x4x4x8x8xi32, 2>
//       CHECK: memref.alloc() : memref<1x1x4x8x4x8xi32, 2>
//       CHECK: memref.alloc() : memref<1x1x256x64xi32, 1>
//       CHECK: memref.alloc() : memref<1x1x64x256xi32, 1>
//       CHECK: memref.alloc() : memref<1x1x8x16x4x8xi32, 2>
//       CHECK: memref.alloc() : memref<1x1x64x64xi32, 1>
//       CHECK: scf.forall
//       CHECK: {
//       CHECK:   iree_linalg_ext.pack %{{.*}} : (memref<64x256xi32, strided<[2048, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x1x64x256xi32, 1>)
//       CHECK:   iree_linalg_ext.pack %{{.*}} : (memref<256x64xi32, strided<[512, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x1x256x64xi32, 1>)
//       CHECK:   scf.forall
//       CHECK:   {
//       CHECK:     memref.subview %{{.*}} : memref<1x1x8x16x4x8xi32, 2> to memref<1x1x4x8x4x8xi32, strided<[4096, 4096, 512, 32, 8, 1], offset: ?>, 2>
//       CHECK:     linalg.fill ins(%{{.*}}) outs(%{{.*}} : memref<1x1x4x8x4x8xi32, strided<[4096, 4096, 512, 32, 8, 1], offset: ?>, 2>)
//       CHECK:     scf.for
//       CHECK:     {
//       CHECK:       iree_linalg_ext.pack %{{.*}} : (memref<1x1x32x32xi32, strided<[16384, 16384, 256, 1], offset: ?>, 1> memref<1x1x4x8x4x8xi32, 2>)
//       CHECK:       iree_linalg_ext.pack %{{.*}} : (memref<1x1x32x32xi32, strided<[16384, 16384, 64, 1], offset: ?>, 1> memref<1x1x4x4x8x8xi32, 2>)
//       CHECK:       linalg.generic
//       CHECK:     }
//       CHECK:   }
//       CHECK:   scf.for
//       CHECK:   {
//       CHECK:     iree_linalg_ext.pack %{{.*}} : (memref<64x256xi32, strided<[2048, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x1x64x256xi32, 1>)
//       CHECK:     iree_linalg_ext.pack %{{.*}} : (memref<256x64xi32, strided<[512, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x1x256x64xi32, 1>)
//       CHECK:     scf.forall
//       CHECK:     {
//       CHECK:       scf.for
//       CHECK:       {
//       CHECK:         iree_linalg_ext.pack %{{.*}} : (memref<1x1x32x32xi32, strided<[16384, 16384, 256, 1], offset: ?>, 1> memref<1x1x4x8x4x8xi32, 2>)
//       CHECK:         iree_linalg_ext.pack %{{.*}} : (memref<1x1x32x32xi32, strided<[16384, 16384, 64, 1], offset: ?>, 1> memref<1x1x4x4x8x8xi32, 2>)
//       CHECK:         linalg.generic
//       CHECK:       }
//       CHECK:     }
//       CHECK:   }
//       CHECK:   iree_linalg_ext.unpack %{{.*}} : (memref<1x1x8x16x4x8xi32, 2> memref<1x1x64x64xi32, 1>)
//       CHECK:   iree_linalg_ext.unpack %{{.*}} : (memref<1x1x64x64xi32, 1> memref<64x64xi32, strided<[512, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>)
//       CHECK: }
//       CHECK: memref.dealloc %{{.*}} : memref<1x1x64x64xi32, 1>
//       CHECK: memref.dealloc %{{.*}} : memref<1x1x8x16x4x8xi32, 2>
//       CHECK: memref.dealloc %{{.*}} : memref<1x1x64x256xi32, 1>
//       CHECK: memref.dealloc %{{.*}} : memref<1x1x256x64xi32, 1>
//       CHECK: memref.dealloc %{{.*}} : memref<1x1x4x8x4x8xi32, 2>
//       CHECK: memref.dealloc %{{.*}} : memref<1x1x4x4x8x8xi32, 2>
