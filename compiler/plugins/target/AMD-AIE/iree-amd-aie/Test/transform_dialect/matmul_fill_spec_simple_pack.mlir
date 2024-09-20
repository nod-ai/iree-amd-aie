// RUN: iree-opt --iree-transform-dialect-interpreter %s | FileCheck %s
// This script shows an example lowering matmul through simple-pack pipeline for AIE device.
// In this strategy, we use pack operations for data movement from L3 to L2, and L2 to L1.

#pipeline_layout = #hal.pipeline.layout<bindings= [
    #hal.pipeline.binding<storage_buffer, ReadOnly>,
    #hal.pipeline.binding<storage_buffer, ReadOnly>,
    #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul_i32() {
  %c0_i32 = arith.constant 0: i32
  %c0 = arith.constant 0 : index
  %arg0_binding = hal.interface.binding.subspan layout(#pipeline_layout)  binding(0) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x512xi32>>
  %arg0 = flow.dispatch.tensor.load %arg0_binding, offsets = [0, 0], sizes = [2048, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x512xi32>> -> tensor<2048x512xi32>
  %arg1_binding = hal.interface.binding.subspan layout(#pipeline_layout)  binding(1) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<512x2048xi32>>
  %arg1 = flow.dispatch.tensor.load %arg1_binding, offsets = [0, 0], sizes = [512, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x2048xi32>> -> tensor<512x2048xi32>
  %arg2_binding = hal.interface.binding.subspan layout(#pipeline_layout)  binding(2) offset(%c0) flags(None) : !flow.dispatch.tensor<writeonly:tensor<2048x2048xi32>>
  %empty = tensor.empty() : tensor<2048x2048xi32>
  %0 = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<2048x2048xi32>) -> tensor<2048x2048xi32>
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<2048x512xi32>, tensor<512x2048xi32>)
      outs(%0 : tensor<2048x2048xi32>) -> tensor<2048x2048xi32>
  flow.dispatch.tensor.store %1, %arg2_binding, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : tensor<2048x2048xi32> -> !flow.dispatch.tensor<writeonly:tensor<2048x2048xi32>>
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
    %tiled_matmul, %forall =
      transform.structured.tile_using_forall %matmul tile_sizes [64, 64]
        ( mapping = [#gpu.block<y>, #gpu.block<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse fill operation into the loop
    %fused_fill, %_ = transform.structured.fuse_into_containing_op %fill into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // First level of packing, move data from L3 to L2.
    %packed = transform.structured.pack %tiled_matmul packed_sizes = [64, 64, 512]
      : (!transform.any_op) -> (!transform.any_op)

    // Transpose B matrix from [K N n k] to [K N k n]
    %pack_producer_b0 = transform.get_producer_of_operand %packed[1]
      : (!transform.any_op) -> (!transform.any_op)
    %packed_b0, %pack_b0, %empty_unpack_b0 =
      transform.structured.pack_transpose %pack_producer_b0 with_compute_op(%packed)
      inner_perm = [1, 0] : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Run canonicalization to fold fill with pack and unpack operations.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Bufferize to shared memory allocation
    %pack_producer_a0 = transform.get_producer_of_operand %packed_b0[0]
      : (!transform.any_op) -> (!transform.any_op)
    %pack_producer_c0 = transform.get_producer_of_operand %packed_b0[2]
      : (!transform.any_op) -> (!transform.any_op)
    %buffer_a0, %new_a0 = transform.structured.bufferize_to_allocation %pack_b0
      {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op
    %buffer_b0, %new_b0 = transform.structured.bufferize_to_allocation %pack_producer_a0
      {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op
    %buffer_c0, %new_c0 = transform.structured.bufferize_to_allocation %pack_producer_c0
      {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Second level tile to forall.
    %tiled_matmul_1, %forall_1 =
      transform.structured.tile_using_forall %packed_b0 tile_sizes [0, 0, 0, 32, 32]
        ( mapping = [#gpu.thread<y>, #gpu.thread<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Find the fill operation to fuse.
    %fused_fill_1 = transform.get_producer_of_operand %forall_1[0] : (!transform.any_op) -> (!transform.any_op)
    %fused_fill_2, %__ = transform.structured.fuse_into_containing_op %fused_fill_1 into %forall_1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Second level of packing, move data from L2 to L1.
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

    // Promote the result to local memory.
    %buffer_c, %new_c = transform.structured.bufferize_to_allocation %pack_c
      {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Tile reduction dimension.
    %tiled_reduction, %loop =
      transform.structured.tile_using_for %packed_c tile_sizes [0, 0, 0, 0, 0, 4]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Find the for op and fuse the pack ops into the loop.
    %for_op = transform.structured.match ops{["scf.for"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %fused_pack_a, %e1 = transform.structured.fuse_into_containing_op %pack_a into %for_op
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_pack_b, %e2 = transform.structured.fuse_into_containing_op %pack_b into %for_op
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Promote the inputs to local memory.
    %buffer_a, %new_a = transform.structured.bufferize_to_allocation %fused_pack_a
      {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op
    %buffer_b, %new_b = transform.structured.bufferize_to_allocation %fused_pack_b
      {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

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
//       CHECK: scf.forall
//       CHECK: {
//       CHECK:   memref.alloc() : memref<1x1x64x512xi32, 1>
//       CHECK:   iree_linalg_ext.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [64, 512] into %{{.*}} : (memref<64x512xi32, strided<[512, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x1x64x512xi32, 1>)
//       CHECK:   memref.alloc() : memref<1x1x512x64xi32, 1>
//       CHECK:   iree_linalg_ext.pack %{{.*}} outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [512, 64] into %{{.*}} : (memref<512x64xi32, strided<[2048, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x1x512x64xi32, 1>)
//       CHECK:   memref.alloc() : memref<1x1x64x64xi32, 1>
//       CHECK:   scf.forall
//       CHECK:   {
//       CHECK:     memref.alloc() : memref<1x1x4x8x4x8xi32, 2>
//       CHECK:     linalg.fill ins(%{{.*}}) outs(%{{.*}} : memref<1x1x4x8x4x8xi32, 2>)
//       CHECK:     scf.for
//       CHECK:     {
//       CHECK:        memref.alloc() : memref<1x1x4x8x4x8xi32, 2>
//       CHECK:        iree_linalg_ext.pack %{{.*}} outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %{{.*}} : (memref<1x1x32x32xi32, strided<[32768, 32768, 512, 1], offset: ?>, 1> memref<1x1x4x8x4x8xi32, 2>)
//       CHECK:        memref.alloc() : memref<1x1x4x4x8x8xi32, 2>
//       CHECK:        iree_linalg_ext.pack %{{.*}} outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 8] into %{{.*}} : (memref<1x1x32x32xi32, strided<[32768, 32768, 64, 1], offset: ?>, 1> memref<1x1x4x4x8x8xi32, 2>)
//       CHECK:        linalg.generic
//       CHECK:        memref.dealloc %{{.*}} : memref<1x1x4x8x4x8xi32, 2>
//       CHECK:        memref.dealloc %{{.*}} : memref<1x1x4x4x8x8xi32, 2>
//       CHECK:     }
//       CHECK:     iree_linalg_ext.unpack %{{.*}} outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %{{.*}} : (memref<1x1x4x8x4x8xi32, 2> memref<1x1x32x32xi32, strided<[4096, 4096, 64, 1], offset: ?>, 1>)
//       CHECK:     memref.dealloc %alloc_7 : memref<1x1x4x8x4x8xi32, 2>
//       CHECK:   }
//       CHECK:   iree_linalg_ext.unpack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %{{.*}} : (memref<1x1x64x64xi32, 1> memref<64x64xi32, strided<[2048, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>)
//       CHECK:   memref.dealloc %{{.*}} : memref<1x1x512x64xi32, 1>
//       CHECK:   memref.dealloc %{{.*}} : memref<1x1x64x512xi32, 1>
//       CHECK:   memref.dealloc %{{.*}} : memref<1x1x64x64xi32, 1>
//       CHECK: }

