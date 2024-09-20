// RUN: iree-opt --iree-transform-dialect-interpreter %s | FileCheck %s
// This script shows an example lowering matmul through pad based pipeline for AIE device.
// In this strategy, we use pad operations for data movement from L3 to L2, and L2 to L1.

#pipeline_layout = #hal.pipeline.layout<bindings= [
    #hal.pipeline.binding<storage_buffer, ReadOnly>,
    #hal.pipeline.binding<storage_buffer, ReadOnly>,
    #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul_i32() {
  %c0_i32 = arith.constant 0: i32
  %c0 = arith.constant 0 : index
  %arg0_binding = hal.interface.binding.subspan layout(#pipeline_layout)  binding(0) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8x16xi32>>
  %arg0 = flow.dispatch.tensor.load %arg0_binding, offsets = [0, 0], sizes = [8, 16], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x16xi32>> -> tensor<8x16xi32>
  %arg1_binding = hal.interface.binding.subspan layout(#pipeline_layout)  binding(1) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x8xi32>>
  %arg1 = flow.dispatch.tensor.load %arg1_binding, offsets = [0, 0], sizes = [16, 8], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x8xi32>> -> tensor<16x8xi32>
  %arg2_binding = hal.interface.binding.subspan layout(#pipeline_layout)  binding(2) offset(%c0) flags(None) : !flow.dispatch.tensor<writeonly:tensor<8x8xi32>>
  %empty = tensor.empty() : tensor<8x8xi32>
  %0 = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<8x8xi32>) -> tensor<8x8xi32>
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<8x16xi32>, tensor<16x8xi32>)
      outs(%0 : tensor<8x8xi32>) -> tensor<8x8xi32>
  flow.dispatch.tensor.store %1, %arg2_binding, offsets = [0, 0], sizes = [8, 8], strides = [1, 1] : tensor<8x8xi32> -> !flow.dispatch.tensor<writeonly:tensor<8x8xi32>>
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

    // First level tile to forall with tile_sizes [8, 8].
    %tiled_matmul, %forall =
      transform.structured.tile_using_forall %matmul tile_sizes [8, 8]
        ( mapping = [#gpu.block<y>, #gpu.block<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse fill operation into the loop
    %fused_fill, %fused_for_all = transform.structured.fuse_into_containing_op %fill into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Pad operation.
    %padded, %pad, %__ = transform.structured.pad %tiled_matmul {
      padding_values=[0 : i32, 0 : i32, 0 : i32],
      padding_dimensions=[0, 1, 2],
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

    // Promote the result to shared memrory
    %padded_result = transform.get_producer_of_operand %padded[2] : (!transform.any_op) -> (!transform.any_op)
    %padded_result_buffer, %padded_result_new = transform.structured.bufferize_to_allocation %padded_result
        {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Run canonicalizations.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Find the matmul and fill again
    %tiled_ops = transform.structured.match ops{["linalg.fill", "linalg.matmul"]} in %fused_for_all : (!transform.any_op) -> !transform.any_op
    %tiled_fill_op, %tiled_padded_matmul = transform.split_handle %tiled_ops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Second level tile to forall with tile_sizes [4, 4].
    %tiled_matmul_1, %forall_1 =
      transform.structured.tile_using_forall %tiled_padded_matmul tile_sizes [4, 4]
        ( mapping = [#gpu.thread<y>, #gpu.thread<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_fill_2, %fused_for_all_2 = transform.structured.fuse_into_containing_op %tiled_fill_op into %forall_1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Pad operation.
    %padded_1, %pad_1, %_ = transform.structured.pad %tiled_matmul_1 {
      padding_values=[0 : i32, 0 : i32, 0 : i32],
      padding_dimensions=[0, 1, 2],
      pack_paddings=[0, 0, 1],
      copy_back_op="linalg.copy"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %pad_1_dps = transform.structured.rewrite_in_destination_passing_style %pad_1 : (!transform.any_op) -> !transform.any_op

    // Promote the result to local memory.
    %padded_result_local = transform.get_producer_of_operand %padded_1[2] : (!transform.any_op) -> (!transform.any_op)
    %padded_result_local_buffer, %padded_result_local_new = transform.structured.bufferize_to_allocation %padded_result_local
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Run canonicalizations.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Tile reduction dimension.
    %tiled_reduction, %loop =
      transform.structured.tile_using_for %padded_1 tile_sizes [0, 0, 4]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Clean up.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Pad operation.
    %padded_reduction, %pad_reduction, %___ = transform.structured.pad %tiled_reduction {
      padding_values=[0 : i32, 0 : i32, 0 : i32],
      padding_dimensions=[0, 1, 2],
      pack_paddings=[1, 1, 0],
      copy_back_op="linalg.copy"
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
    %func_op = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.eliminate_empty_tensors %func_op : (!transform.any_op) -> ()
    %memref_func = transform.iree.bufferize %func_op : (!transform.any_op) -> !transform.any_op

    transform.include @cleanup failures(propagate) (%memref_func) : (!transform.any_op) -> ()
    transform.yield
  }
}

// CHECK-LABEL: @matmul_i32
//       CHECK: scf.forall
//       CHECK: {
//       CHECK:   memref.alloc() : memref<8x16xi32, 1>
//       CHECK:   linalg.copy ins(%{{.*}} : memref<8x16xi32, strided<[16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>) outs(%{{.*}} : memref<8x16xi32, 1>)
//       CHECK:   memref.alloc() : memref<16x8xi32, 1>
//       CHECK:   linalg.copy ins(%{{.*}} : memref<16x8xi32, strided<[8, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>) outs(%{{.*}} : memref<16x8xi32, 1>)
//       CHECK:   memref.alloc() : memref<8x8xi32, 1>
//       CHECK:   scf.forall
//       CHECK:   {
//       CHECK:     memref.alloc() : memref<4x4xi32, 2>
//       CHECK:     linalg.fill ins(%{{.*}}) outs(%{{.*}} : memref<4x4xi32, 2>)
//       CHECK:     scf.for
//       CHECK:     {
//       CHECK:        memref.alloc() : memref<4x4xi32, 2>
//       CHECK:        linalg.copy ins(%{{.*}} : memref<4x4xi32, strided<[16, 1], offset: ?>, 1>) outs(%{{.*}} : memref<4x4xi32, 2>)
//       CHECK:        memref.alloc() : memref<4x4xi32, 2>
//       CHECK:        linalg.copy ins(%{{.*}} : memref<4x4xi32, strided<[8, 1], offset: ?>, 1>) outs(%{{.*}} : memref<4x4xi32, 2>)
//       CHECK:        linalg.matmul
//       CHECK:        memref.dealloc %{{.*}} : memref<4x4xi32, 2>
//       CHECK:        memref.dealloc %{{.*}} : memref<4x4xi32, 2>
//       CHECK:     }
//       CHECK:     linalg.copy ins(%{{.*}} : memref<4x4xi32, 2>) outs(%{{.*}} : memref<4x4xi32, strided<[8, 1], offset: ?>, 1>)
//       CHECK:     memref.dealloc %{{.*}} : memref<4x4xi32, 2>
//       CHECK:   }
//       CHECK:   linalg.copy ins(%{{.*}} : memref<8x8xi32, 1>) outs(%{{.*}} : memref<8x8xi32, strided<[8, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>)
//       CHECK:   memref.dealloc %{{.*}} : memref<8x16xi32, 1>
//       CHECK:   memref.dealloc %{{.*}} : memref<16x8xi32, 1>
//       CHECK:   memref.dealloc %{{.*}} : memref<8x8xi32, 1>
//       CHECK: }
