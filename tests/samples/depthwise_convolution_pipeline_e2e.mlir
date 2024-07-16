// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-sources %s | iree-opt --mlir-disable-threading  --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" --iree-amdaie-tile-pipeline=depthwise-convolution-decompose --split-input-file | FileCheck %s

func.func @conv_2d_nhwc_hwcf(%arg0: tensor<2x14x14x64xbf16>, %arg1: tensor<3x3x64xbf16>) -> tensor<2x12x12x64xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<2x12x12x64xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x12x12x64xf32>) -> tensor<2x12x12x64xf32>
  %2 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %arg1 : tensor<2x14x14x64xbf16>, tensor<3x3x64xbf16>) outs(%1 : tensor<2x12x12x64xf32>) -> tensor<2x12x12x64xf32>
  return %2 : tensor<2x12x12x64xf32>
}

// CHECK-LABEL: hal.executable.export public @conv_2d_nhwc_hwcf_dispatch_0_depthwise_conv_2d_nhwc_hwc_2x12x12x64x3x3_bf16xbf16xf32 
//       CHECK:    aie.device(npu1_4col)
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aie.shim_dma_allocation
// CHECK: func.func @conv_2d_nhwc_hwcf_dispatch_0_depthwise_conv_2d_nhwc_hwc_2x12x12x64x3x3_bf16xbf16xf32(%arg0: memref<12544xi32>, %arg1: memref<288xi32>, %arg2: memref<2x12x12x64xf32>) {
//       CHECK:      aiex.npu.dma_memcpy_nd
//       CHECK:      aiex.npu.dma_memcpy_nd
//       CHECK:      aiex.npu.dma_memcpy_nd
//       CHECK:      aiex.npu.sync

