// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-sources %s | iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" --iree-amdaie-tile-pipeline=conv-decompose --split-input-file | FileCheck %s

func.func @conv_2d_nhwc_hwcf(%arg0: tensor<2x14x14x32xi32>, %arg1: tensor<3x3x32x64xi32>) -> tensor<2x12x12x64xi32> {
  %cst = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<2x12x12x64xi32>
  %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<2x12x12x64xi32>) -> tensor<2x12x12x64xi32>
  %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %arg1 : tensor<2x14x14x32xi32>, tensor<3x3x32x64xi32>) outs(%1 : tensor<2x12x12x64xi32>) -> tensor<2x12x12x64xi32>
  return %2 : tensor<2x12x12x64xi32>
}

// CHECK-LABEL: hal.executable.export public @conv_2d_nhwc_hwcf_dispatch_0_conv_2d_nhwc_hwcf_2x12x12x64x3x3x32_i32
//       CHECK:    aie.device(npu1_4col)
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aiex.runtime_sequence @conv_2d_nhwc_hwcf_dispatch_0_conv_2d_nhwc_hwcf_2x12x12x64x3x3x32_i32(%arg0: memref<2x14x14x32xi32>, %arg1: memref<3x3x32x64xi32>, %arg2: memref<2x12x12x64xi32>)
//       CHECK:      aiex.npu.dma_memcpy_nd
//       CHECK:      aiex.npu.dma_memcpy_nd
//       CHECK:      aiex.npu.dma_memcpy_nd
//       CHECK:      aiex.npu.sync

// -----

func.func @conv_2d_nhwc_hwcf_q(%arg0: tensor<2x14x14x32xi8>, %arg1: tensor<3x3x32x64xi8>) -> tensor<2x12x12x64xi32> {
  %cst = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<2x12x12x64xi32>
  %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<2x12x12x64xi32>) -> tensor<2x12x12x64xi32>
  %2 = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %arg1, %cst, %cst : tensor<2x14x14x32xi8>, tensor<3x3x32x64xi8>, i32, i32) outs(%1 : tensor<2x12x12x64xi32>) -> tensor<2x12x12x64xi32>
  return %2 : tensor<2x12x12x64xi32>
}

// CHECK-LABEL: hal.executable.export public @conv_2d_nhwc_hwcf_q_dispatch_0_conv_2d_nhwc_hwcf_q_2x12x12x64x3x3x32_i8xi8xi32xi32xi32
//       CHECK:    aie.device(npu1_4col)
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aiex.runtime_sequence @conv_2d_nhwc_hwcf_q_dispatch_0_conv_2d_nhwc_hwcf_q_2x12x12x64x3x3x32_i8xi8xi32xi32xi32(%arg0: memref<3136xi32>, %arg1: memref<4608xi32>, %arg2: memref<2x12x12x64xi32>)
//       CHECK:      aiex.npu.dma_memcpy_nd
//       CHECK:      aiex.npu.dma_memcpy_nd
//       CHECK:      aiex.npu.dma_memcpy_nd
//       CHECK:      aiex.npu.sync
