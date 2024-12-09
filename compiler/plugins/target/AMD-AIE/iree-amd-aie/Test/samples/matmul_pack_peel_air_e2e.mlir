// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-targets --iree-amdaie-lower-to-aie-pipeline=air --iree-amdaie-tile-pipeline=pack-peel --split-input-file %s | FileCheck %s

func.func @matmul_i8_i32(%lhs: tensor<32x16xi8>, %rhs: tensor<16x32xi8>) -> tensor<32x32xi32>
{
  %cst = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<32x32xi32>
  %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<32x32xi32>) -> tensor<32x32xi32>
  %res = linalg.matmul ins(%lhs, %rhs: tensor<32x16xi8>, tensor<16x32xi8>)
                    outs(%1: tensor<32x32xi32>) -> tensor<32x32xi32>
  return %res : tensor<32x32xi32>
}

//   CHECK-LABEL: hal.executable.export public @matmul_i8_i32_dispatch_0_matmul_32x32x16_i8xi8xi32
//         CHECK:    aie.device(npu1_4col)
// CHECK-COUNT-3:    aie.shim_dma_allocation

// -----

func.func @matmul_bf16(%lhs: tensor<16x32xbf16>, %rhs: tensor<32x16xbf16>) -> tensor<16x16xf32>
{
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<16x16xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %res = linalg.matmul ins(%lhs, %rhs: tensor<16x32xbf16>, tensor<32x16xbf16>)
                    outs(%1: tensor<16x16xf32>) -> tensor<16x16xf32>
  return %res : tensor<16x16xf32>
}

//   CHECK-LABEL: hal.executable.export public @matmul_bf16_dispatch_0_matmul_16x16x32_bf16
//         CHECK:    aie.device(npu1_4col)
// CHECK-COUNT-3:    aie.shim_dma_allocation
