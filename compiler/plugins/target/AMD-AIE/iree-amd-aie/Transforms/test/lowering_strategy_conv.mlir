// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy{use-pass-pipeline=conv-decompose})' %s | FileCheck %s

// CHECK{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[0, 4, 4, 4, 0, 0, 0], [1, 4, 1, 4, 0, 0, 0], [0, 0, 0, 0, 8, 1, 1]]>
builtin.module {
  func.func @conv_2d_nchw_fchw_2x64x12x12x32x3x3_i32() {
    %cst = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x32x14x14xi32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x32x3x3xi32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x64x12x12xi32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 32, 14, 14], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x32x14x14xi32>> -> tensor<2x32x14x14xi32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [64, 32, 3, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x32x3x3xi32>> -> tensor<64x32x3x3xi32>
    %5 = tensor.empty() : tensor<2x64x12x12xi32>
    %6 = linalg.fill ins(%cst : i32) outs(%5 : tensor<2x64x12x12xi32>) -> tensor<2x64x12x12xi32>
    %7 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%3, %4 : tensor<2x32x14x14xi32>, tensor<64x32x3x3xi32>) outs(%6 : tensor<2x64x12x12xi32>) -> tensor<2x64x12x12xi32>
    // CHECK: linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, lowering_config = #config, strides = dense<1> : vector<2xi64>}
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 64, 12, 12], strides = [1, 1, 1, 1] : tensor<2x64x12x12xi32> -> !flow.dispatch.tensor<writeonly:tensor<2x64x12x12xi32>>
    return
  }
}

// -----

// CHECK{LITERAL}: #iree_codegen.lowering_config<tile_sizes = [[0, 4, 4, 4, 0, 0, 0], [1, 1, 4, 4, 0, 0, 0], [0, 0, 0, 0, 1, 1, 8]]>
func.func @conv_static_dispatch_0_conv_2d_nhwc_hwcf_2x12x12x64x3x3x32_bf16xbf16xf32() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x14x14x32xbf16>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x32x64xbf16>>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x12x12x64xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 14, 14, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x14x14x32xbf16>> -> tensor<2x14x14x32xbf16>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 32, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x32x64xbf16>> -> tensor<3x3x32x64xbf16>
  %5 = tensor.empty() : tensor<2x12x12x64xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x12x12x64xf32>) -> tensor<2x12x12x64xf32>
  %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%3, %4 : tensor<2x14x14x32xbf16>, tensor<3x3x32x64xbf16>) outs(%6 : tensor<2x12x12x64xf32>) -> tensor<2x12x12x64xf32>
  // CHECK: linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, lowering_config = #config, strides = dense<1> : vector<2xi64>}
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 12, 12, 64], strides = [1, 1, 1, 1] : tensor<2x12x12x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x12x12x64xf32>>
  return
}

// -----

// CHECK{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[1, 4, 4, 16, 0, 0], [1, 1, 4, 4, 0, 0], [1, 1, 4, 4, 1, 1]]>
func.func @conv_depthwise_channel_last_bf16(){
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x14x14x64xbf16>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x64xbf16>>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x12x12x64xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 14, 14, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x14x14x64xbf16>> -> tensor<2x14x14x64xbf16>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [3, 3, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x64xbf16>> -> tensor<3x3x64xbf16>
  %5 = tensor.empty() : tensor<2x12x12x64xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x12x12x64xf32>) -> tensor<2x12x12x64xf32>
  %7 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%3, %4 : tensor<2x14x14x64xbf16>, tensor<3x3x64xbf16>) outs(%6 : tensor<2x12x12x64xf32>) -> tensor<2x12x12x64xf32>
  // CHECK: linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, lowering_config = #config, strides = dense<1> : vector<2xi64>}
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 12, 12, 64], strides = [1, 1, 1, 1] : tensor<2x12x12x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x12x12x64xf32>>
  return
}

// -----
// Same test as above, but where the operand type is i8. In this case we expect OC tile size of 32 (not 16) at level 0, and 8 at levels 1 and 2. This is because of the instruction size of AIE. 

// CHECK{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[1, 4, 4, 32, 0, 0], [1, 1, 4, 8, 0, 0], [0, 0, 0, 0, 1, 1]]>
func.func @conv_depthwise_channel_last_i8(){
  %cst = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x14x14x64xi8>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x64xi8>>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x12x12x64xi32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 14, 14, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x14x14x64xi8>> -> tensor<2x14x14x64xi8>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [3, 3, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x64xi8>> -> tensor<3x3x64xi8>
  %5 = tensor.empty() : tensor<2x12x12x64xi32>
  %6 = linalg.fill ins(%cst : i32) outs(%5 : tensor<2x12x12x64xi32>) -> tensor<2x12x12x64xi32>
  %7 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%3, %4 : tensor<2x14x14x64xi8>, tensor<3x3x64xi8>) outs(%6 : tensor<2x12x12x64xi32>) -> tensor<2x12x12x64xi32>
  // CHECK: linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, lowering_config = #config, strides = dense<1> : vector<2xi64>}
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 12, 12, 64], strides = [1, 1, 1, 1] : tensor<2x12x12x64xi32> -> !flow.dispatch.tensor<writeonly:tensor<2x12x12x64xi32>>
  return
}
