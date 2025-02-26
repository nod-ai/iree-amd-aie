// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-targets --iree-amdaie-lower-to-aie-pipeline=air --iree-amdaie-tile-pipeline=pack-peel-4-level-tiling --split-input-file %s | FileCheck %s

func.func @matmul_bf16(%lhs: tensor<512x512xbf16>, %rhs: tensor<512x512xbf16>) -> tensor<512x512xf32>
{
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<512x512xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<512x512xf32>) -> tensor<512x512xf32>
  %res = linalg.matmul ins(%lhs, %rhs: tensor<512x512xbf16>, tensor<512x512xbf16>)
                    outs(%1: tensor<512x512xf32>) -> tensor<512x512xf32>
  return %res : tensor<512x512xf32>
}

//   CHECK-LABEL: hal.executable.export public @matmul_bf16_dispatch_0_matmul_512x512x512_bf16xbf16xf32
//         CHECK:    aie.device(npu1_4col)
// CHECK-COUNT-3:    aie.shim_dma_allocation
