// RUN: iree-opt --iree-transform-dialect-interpreter %s | FileCheck %s
// This script demonstrates lowering conv through IREE to eventually target AIE.
// It's based on conv2d lowering in IREE for llvm-cpu.
//
// The trick is to tile the 2-d convolution into 1-d convolution, and then
// convert the 1-d convolution to a vector.contract.

#pipeline_layout = #hal.pipeline.layout<bindings= [
    #hal.pipeline.binding<storage_buffer, ReadOnly>,
    #hal.pipeline.binding<storage_buffer, ReadOnly>,
    #hal.pipeline.binding<storage_buffer>
]>

func.func @conv_static_dispatch_0_conv_2d_nchw_fchw_2x64x12x12x32x3x3_f32(){
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x32x14x14xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x32x3x3xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x64x12x12xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 32, 14, 14], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x32x14x14xf32>> -> tensor<2x32x14x14xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [64, 32, 3, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x32x3x3xf32>> -> tensor<64x32x3x3xf32>
  %5 = tensor.empty() : tensor<2x64x12x12xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x64x12x12xf32>) -> tensor<2x64x12x12xf32>
  %7 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%3, %4 : tensor<2x32x14x14xf32>, tensor<64x32x3x3xf32>) outs(%6 : tensor<2x64x12x12xf32>) -> tensor<2x64x12x12xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 64, 12, 12], strides = [1, 1, 1, 1] : tensor<2x64x12x12xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x64x12x12xf32>>
  return
}

!any = !transform.any_op

module attributes { transform.with_named_sequence } {
  transform.named_sequence @cleanup(%variant_op: !any {transform.readonly}) {
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

  transform.named_sequence @full_pipeline(%variant_op: !any {transform.readonly}) {
    %ops = transform.structured.match ops{["linalg.fill", "linalg.conv_2d_nchw_fchw"]}
          in %variant_op : (!any) -> !any
    %fill, %conv = transform.split_handle %ops : (!any) -> (!any, !any)

    // Air launch & segment processes 4 output channels of 1 image, patch size 4x4.
    // conv_2d_nchw_fchw tiling dims: [N, OC, OH, OW, IC, KH, KW].
    %tiled_conv, %forall =
      transform.structured.tile_using_forall %conv tile_sizes [1, 4, 4, 4]
      : (!any) -> (!any, !any)

    // Fuse fill operation into the forall loop.
    %fused_fill, %fused_for_all =
      transform.structured.fuse_into_containing_op %fill into %forall
      : (!any, !any) -> (!any, !any)

    // Pad and bufferize convolution tensors to shared memory (level 1).
    // Effectively extend IR like
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
    //
    %padded, %pad, %_ = transform.structured.pad %tiled_conv {
      padding_values=[0. : f32, 0. : f32, 0. : f32],
      padding_dimensions=[0, 1, 2],
      pack_paddings=[1, 1, 1],
      copy_back_op="linalg.copy"
    } : (!any) -> (!any, !any, !any)

    %__ = transform.structured.rewrite_in_destination_passing_style %pad
            : (!any) -> !any

    // Bufferize input, weight, and output tensors.
    %padded_lhs = transform.get_producer_of_operand %padded[0] : (!any) -> (!any)
    %padded_lhs_buffer, %padded_lhs_new =
      transform.structured.bufferize_to_allocation %padded_lhs
      {memory_space = 1, bufferize_destination_only, emit_dealloc} : !any

    %padded_rhs = transform.get_producer_of_operand %padded[1] : (!any) -> (!any)
    %padded_rhs_buffer, %padded_rhs_new =
      transform.structured.bufferize_to_allocation %padded_rhs
      {memory_space = 1, bufferize_destination_only, emit_dealloc} : !any

    %padded_result = transform.get_producer_of_operand %padded[2] : (!any) -> (!any)
    %padded_result_buffer, %padded_result_new =
      transform.structured.bufferize_to_allocation %padded_result
      {memory_space = 1, bufferize_destination_only, emit_dealloc} : !any

    transform.include @cleanup failures(propagate) (%variant_op) : (!any) -> ()

    // Second level of tiling for air herd, promoting result to local memory.
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

    %padded_1, %pad_1, %___ = transform.structured.pad %tiled_conv_1 {
      padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32],
      padding_dimensions=[0, 1, 2],
      pack_paddings=[0, 0, 1],
      copy_back_op="linalg.copy"
    } : (!any) -> (!any, !any, !any)

    %pad_1_dps = transform.structured.rewrite_in_destination_passing_style %pad_1
                 : (!any) -> !any

    %padded_result_local = transform.get_producer_of_operand %padded_1[2]
                           : (!any) -> (!any)
    %padded_result_local_buffer, %padded_result_local_new =
      transform.structured.bufferize_to_allocation %padded_result_local
      {memory_space = 2, bufferize_destination_only, emit_dealloc} : !any

    // Create loop structure for each tile's execution.
    // [0, 0, 0, 0, 8, 1, 1]
    //  ^  ^  ^  ^  ^  ^  ^
    //  |  |  |  |  |  |  |
    //  N  OC OH OW IC KH KW ===> 3 reduction loops using scf.for.

    %tiled_reduction, %loop0, %loop1, %loop2  =
      transform.structured.tile_using_for %padded_1 tile_sizes [0,0,0,0,8,1,1]
      : (!any) -> (!any, !any, !any, !any)

    transform.include @replace_conv2d_with_conv1d failures(propagate)
          (%variant_op) : (!any) -> ()

    transform.include @cleanup failures(propagate) (%variant_op) : (!any) -> ()

    %inner_conv = transform.structured.match ops{["linalg.conv_1d_ncw_fcw"]}
                   in %fused_for_all : (!any) -> !any

    %padded_2, %pad_2, %____ = transform.structured.pad %inner_conv {
      padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32],
      padding_dimensions=[0, 1, 2],
      pack_paddings=[1, 1, 0],
      copy_back_op="linalg.copy"
    } : (!any) -> (!any, !any, !any)

    %pad_2_dps = transform.structured.rewrite_in_destination_passing_style %pad_2
                 : (!any) -> !any

    %padded_2_lhs = transform.get_producer_of_operand %padded_2[0] : (!any) -> (!any)
    %padded_2_lhs_buffer, %padded_2_lhs_new =
      transform.structured.bufferize_to_allocation %padded_2_lhs
      {memory_space = 2, bufferize_destination_only, emit_dealloc} : !any

    %padded_2_rhs = transform.get_producer_of_operand %padded_2[1] : (!any) -> (!any)
    %padded_2_rhs_buffer, %padded_2_rhs_new =
      transform.structured.bufferize_to_allocation %padded_2_rhs
      {memory_space = 2, bufferize_destination_only, emit_dealloc} : !any

    transform.include @cleanup failures(propagate) (%variant_op) : (!any) -> ()

    %conv_pre_contract =
       transform.structured.match ops{["linalg.conv_1d_ncw_fcw"]}
               in %fused_for_all : (!any) -> !any

    transform.structured.vectorize %conv_pre_contract : !any

    %func_op = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.eliminate_empty_tensors %func_op : (!any) -> ()
    %memref_func = transform.iree.bufferize %func_op : (!any) -> !any

    transform.include @cleanup failures(propagate) (%memref_func) : (!any) -> ()

    transform.yield
  }

  transform.named_sequence @__transform_main(%variant_op: !any
                                              {transform.read_only}) {
    transform.include @full_pipeline failures(propagate) (%variant_op)
      : (!any) -> ()
    transform.yield
  }
}

// CHECK-LABEL:  func.func @conv_static
// CHECK:        scf.forall
// CHECK:        scf.forall
// CHECK-NOT:    scf.forall
// CHECK:        scf.for
// CHECK:        scf.for
// CHECK:        scf.for
// CHECK-NOT:    scf.for
// CHECK:        vector.contract
// CHECK:        return
