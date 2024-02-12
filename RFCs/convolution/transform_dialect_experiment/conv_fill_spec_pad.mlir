// This script shows an example of lowering conv through IREE to (eventually)
// AIE.  It is based on the lowering of conv2d in IREE for llvm-cpu.
//
// See conv_linalg.mlir for the problem size I'm currently applying this to:
//
// !input = tensor<2x32x14x14xf32>
// !weight = tensor<64x32x3x3xf32>
// !output = tensor<2x64x12x12xf32>

!any = !transform.any_op

module attributes { transform.with_named_sequence } {

  transform.named_sequence @cleanup(%variant_op: !any
                                     {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %variant_op
            : (!any) -> !any
    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !any
    transform.iree.apply_licm %func : !any
    transform.apply_cse to %func : !any
    transform.yield
  }

  transform.named_sequence @replace_conv2d_with_conv1d(%variant_op: !any
                                          {transform.readonly}) {
    %f0 = transform.structured.match ops{["func.func"]} in %variant_op
            : (!any) -> !any
    %f1 = transform.apply_registered_pass
         "iree-codegen-decompose-convolution-to-lower-dim-ops"
         to %f0 : (!any) -> !any
    transform.yield
  }

  transform.named_sequence @iree_generic_vectorization(%variant_op: !any
                                          {transform.readonly}) {
    %f0 = transform.structured.match ops{["func.func"]} in %variant_op
            : (!any) -> !any
    %f1 = transform.apply_registered_pass "iree-codegen-generic-vectorization"
         to %f0 : (!any) -> !any
    transform.yield
  }

  transform.named_sequence @__transform_main(%variant_op: !any
                                              {transform.read_only}) {
    %ops = transform.structured.match ops{["linalg.fill",
                                           "linalg.conv_2d_nchw_fchw"]}
          in %variant_op : (!any) -> !any

    %fill, %conv = transform.split_handle %ops : (!any) -> (!any, !any)

    // Each air launch & segment will process 4 output channels of 1 image, 
    // and a single patch of size 4x4. This call inserts the outermost forall 
    // loop, corresponding to the air launch. 
    %tiled_conv, %forall =
      transform.structured.tile_using_forall %conv tile_sizes [1, 4, 4, 4]
      : (!any) -> (!any, !any)

    // Fuse fill operation into the forall loop
    %fused_fill, %fused_for_all =
      transform.structured.fuse_into_containing_op %fill into %forall
      : (!any, !any) -> (!any, !any)

    // Pad and bufferize all 3 convolution tensors (input patch, output patch, 
    // kernel slice) to shared memory (memory level 1). These three steps (pad, 
    // DPS, bufferize) effectively extend IR like

    // ```
    // %extracted_slice_0 = tensor.extract_slice %4 ...  to tensor<4x32x3x3xf32>
    // ```
    //
    // with IR like 
    //
    // ```
    // %13 = bufferization.alloc_tensor() : tensor<4x32x3x3xf32>
    // %alloc_2 = memref.alloc() : memref<4x32x3x3xf32, 1>
    // %14 = bufferization.to_tensor %alloc_2 ... : memref<4x32x3x3xf32, 1>
    // %15 = linalg.copy ins(%extracted_slice_0 : tensor<4x32x3x3xf32>) 
    //             outs(%14 : tensor<4x32x3x3xf32>) -> tensor<4x32x3x3xf32>
    // ```

     %padded, %pad, %__ = transform.structured.pad %tiled_conv {
       padding_values=[0. : f32, 0. : f32, 0. : f32],
       padding_dimensions=[0, 1, 2],
       pack_paddings=[1, 1, 1],  
       copy_back_op="linalg.copy"
     } : (!any) -> (!any, !any, !any)

     %____ = transform.structured.rewrite_in_destination_passing_style
                %pad : (!any) -> !any

     %padded_lhs = transform.get_producer_of_operand %padded[0]
                   : (!any) -> (!any)
     %padded_lhs_buffer, %padded_lhs_new =
       transform.structured.bufferize_to_allocation %padded_lhs
       {memory_space = 1, bufferize_destination_only, emit_dealloc}
       : !any

     %padded_rhs = transform.get_producer_of_operand %padded[1]
                   : (!any) -> (!any)
     %padded_rhs_buffer, %padded_rhs_new =
       transform.structured.bufferize_to_allocation %padded_rhs
       {memory_space = 1, bufferize_destination_only, emit_dealloc}
       : !any

     %padded_result = transform.get_producer_of_operand %padded[2]
                      : (!any) -> (!any)
     %padded_result_buffer, %padded_result_new =
       transform.structured.bufferize_to_allocation %padded_result
       {memory_space = 1, bufferize_destination_only, emit_dealloc}
       : !any

    transform.include @cleanup failures(propagate) (%variant_op)
      : (!any) -> ()

     // Insert second level of tiling, which corresponds to the air herd. 
     // At this level we promote just the result to local memory 
     // (memory level 2): this is just a copy of the logic of the matmul 
     // example in iree-amd-aie. Each of the 4 cores/tiles 
     // (herd members) processes one row of the 4x4 output patch.
     %tiled_ops = transform.structured.match ops{["linalg.fill",
                                                  "linalg.conv_2d_nchw_fchw"]}
                  in %fused_for_all : (!any) -> !any

     %tiled_fill_op, %tiled_padded_conv = transform.split_handle %tiled_ops
                  : (!any) -> (!any, !any)

     %tiled_conv_1, %forall_1 =
       transform.structured.tile_using_forall %tiled_padded_conv
       tile_sizes [1, 4, 1, 4] : (!any) -> (!any, !any)

     %fused_fill_2, %fused_for_all_2 =
       transform.structured.fuse_into_containing_op %tiled_fill_op into %forall_1
       : (!any, !any) -> (!any, !any)
 
     %padded_1, %pad_1, %_ = transform.structured.pad %tiled_conv_1 {
       padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32],
       padding_dimensions=[0, 1, 2],
       pack_paddings=[0, 0, 1],
       copy_back_op="linalg.copy"
     } : (!any) -> (!any, !any, !any)

     %pad_1_dps = transform.structured.rewrite_in_destination_passing_style
                  %pad_1 : (!any) -> !any
 
     %padded_result_local = transform.get_producer_of_operand %padded_1[2]
                            : (!any) -> (!any)

     %padded_result_local_buffer, %padded_result_local_new =
       transform.structured.bufferize_to_allocation %padded_result_local
       {memory_space = 2, bufferize_destination_only, emit_dealloc} : !any



     // Now create the loop structure that each tile will execute. This third 
     // level consists of 3 for loops: process each of the 9 spatial positions 
     // of the kernel is sequence, and process the 32 input channels in chunks 
     // of size 8. The 7 values below therefore correspond to:
     // -- leading 0's do not tile N, H, W, or K further. 
     // -- the 8 : tile the input channel (C) with size 8. 
     // -- the two trailing 1's : tile the kernel spatial dimensions with size 1. 
 
     // Tile the work that each core does.
     %tiled_reduction, %loop0, %loop1, %loop2  =
     transform.structured.tile_using_for %padded_1 [0,0,0,0,8,1,1]
       : (!any) -> (!any, !any, !any, !any)

     transform.include @replace_conv2d_with_conv1d failures(propagate) 
           (%variant_op) : (!any) -> ()

     transform.include @cleanup failures(propagate) (%variant_op) : (!any) -> ()
 
     %inner_conv = transform.structured.match ops{["linalg.conv_1d_ncw_fcw"]}
                    in %fused_for_all : (!any) -> !any
 
     %padded_2, %pad_2, %___ = transform.structured.pad %inner_conv {
       padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32],
       padding_dimensions=[0, 1, 2],
       pack_paddings=[1, 1, 0],
       copy_back_op="linalg.copy"
     } : (!any) -> (!any, !any, !any)


     %pad_2_dps = transform.structured.rewrite_in_destination_passing_style
                  %pad_2 : (!any) -> !any

     %padded_2_lhs = transform.get_producer_of_operand %padded_2[0]
                             : (!any) -> (!any)


     %padded_2_lhs_buffer, %padded_2_lhs_new =
       transform.structured.bufferize_to_allocation %padded_2_lhs
       {memory_space = 2, bufferize_destination_only, emit_dealloc}
       : !any
 
     %padded_2_rhs = transform.get_producer_of_operand %padded_2[1]
                             : (!any) -> (!any)
     %padded_2_rhs_buffer, %padded_2_rhs_new =
       transform.structured.bufferize_to_allocation %padded_2_rhs
       {memory_space = 2, bufferize_destination_only, emit_dealloc}
       : !any
 
 
     // Clean up.
     transform.include @cleanup failures(propagate) (%variant_op)
       : (!any) -> ()
 
     // Bufferize and drop HAL descriptor from memref ops.
     transform.iree.eliminate_empty_tensors %variant_op: (!any) -> ()

    %variant_op_3 = transform.iree.bufferize %variant_op : (!any) -> !any
 
     transform.include @cleanup failures(propagate) (%variant_op_3)
       : (!any) -> ()

// TODO(jn) use vectorization to go from conv1d to vector contract. 
// running the lass creates a large change to the IR. 
// I could maybe write a pass which converts it to a matmul manually. 
//  transform.include @iree_generic_vectorization failures(propagate) (%variant_op_3)
//    : (!any) -> ()

    transform.yield
  }
}

