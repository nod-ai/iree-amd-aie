// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy{use-pass-pipeline=conv-decompose})' %s | FileCheck %s



// CHECK{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[1, 4, 4, 4, 0, 0, 0], [1, 1, 4, 4, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 0, 0]]>
// CHECK{LITERAL}: #packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [0, 0, 0, 4, 0, 0, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[], [1, 0], []], outerPerm = [[0, 1, 3, 2], [], [0, 1, 2, 3]]}]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
func.func @conv_static_dispatch_0_conv_2d_nhwc_hwcf_2x12x12x64x3x3x32_bf16xbf16xf32() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x14x14x32xbf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x32x64xbf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x12x12x64xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 14, 14, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x14x14x32xbf16>> -> tensor<2x14x14x32xbf16>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 32, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x32x64xbf16>> -> tensor<3x3x32x64xbf16>
  %5 = tensor.empty() : tensor<2x12x12x64xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x12x12x64xf32>) -> tensor<2x12x12x64xf32>
  %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%3, %4 : tensor<2x14x14x32xbf16>, tensor<3x3x32x64xbf16>) outs(%6 : tensor<2x12x12x64xf32>) -> tensor<2x12x12x64xf32>
  // CHECK: linalg.conv_2d_nhwc_hwcf
  // CHECK-SAME: lowering_config = #config,
  // CHECK-SAME: packing_config = #packingConfig,
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 12, 12, 64], strides = [1, 1, 1, 1] : tensor<2x12x12x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x12x12x64xf32>>
  return
}

// -----

// CHECK{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[1, 4, 4, 4, 0, 0], [1, 1, 4, 4, 0, 0], [0, 0, 0, 0, 1, 1, 0]]>
// CHECK{LITERAL}: #packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [0, 0, 0, 4, 0, 0], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[], [], []], outerPerm = [[0, 1, 2, 3], [0, 1, 2], [0, 1, 2, 3]]}]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
func.func @conv_depthwise_channel_last_bf16(){
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x14x14x64xbf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x64xbf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x12x12x64xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 14, 14, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x14x14x64xbf16>> -> tensor<2x14x14x64xbf16>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [3, 3, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x64xbf16>> -> tensor<3x3x64xbf16>
  %5 = tensor.empty() : tensor<2x12x12x64xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x12x12x64xf32>) -> tensor<2x12x12x64xf32>
  %7 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%3, %4 : tensor<2x14x14x64xbf16>, tensor<3x3x64xbf16>) outs(%6 : tensor<2x12x12x64xf32>) -> tensor<2x12x12x64xf32>
  // CHECK: linalg.depthwise_conv_2d_nhwc_hwc
  // CHECK-SAME: lowering_config = #config,
  // CHECK-SAME: packing_config = #packingConfig,
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 12, 12, 64], strides = [1, 1, 1, 1] : tensor<2x12x12x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x12x12x64xf32>>
  return
}

// -----
// Same test as above, but where the operand type is i8. In this case we expect OC tile size 8  (not 4) at level 1. This is because of the instruction size of AIE.

// CHECK{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[1, 4, 4, 8, 0, 0], [1, 1, 4, 8, 0, 0], [0, 0, 0, 0, 1, 1, 0]]>
// CHECK{LITERAL}: #packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [0, 0, 0, 8, 0, 0], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[], [], []], outerPerm = [[0, 1, 2, 3], [0, 1, 2], [0, 1, 2, 3]]}]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
func.func @conv_depthwise_channel_last_i8(){
  %cst = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x14x14x64xi8>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x64xi8>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x12x12x64xi32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 14, 14, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x14x14x64xi8>> -> tensor<2x14x14x64xi8>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [3, 3, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x64xi8>> -> tensor<3x3x64xi8>
  %5 = tensor.empty() : tensor<2x12x12x64xi32>
  %6 = linalg.fill ins(%cst : i32) outs(%5 : tensor<2x12x12x64xi32>) -> tensor<2x12x12x64xi32>
  %7 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%3, %4 : tensor<2x14x14x64xi8>, tensor<3x3x64xi8>) outs(%6 : tensor<2x12x12x64xi32>) -> tensor<2x12x12x64xi32>
  // CHECK: linalg.depthwise_conv_2d_nhwc_hwc
  // CHECK-SAME: lowering_config = #config,
  // CHECK-SAME: packing_config = #packingConfig,
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 12, 12, 64], strides = [1, 1, 1, 1] : tensor<2x12x12x64xi32> -> !flow.dispatch.tensor<writeonly:tensor<2x12x12x64xi32>>
  return
}
