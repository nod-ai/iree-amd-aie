// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-sources %s | iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" --iree-amdaie-use-pipeline=pack-peel --split-input-file | FileCheck %s

func.func @matmul_i8_i32(%lhs: tensor<32x16xi8>, %rhs: tensor<16x32xi8>) -> tensor<32x32xi32>
{
  %cst = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<32x32xi32>
  %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<32x32xi32>) -> tensor<32x32xi32>
  %res = linalg.matmul ins(%lhs, %rhs: tensor<32x16xi8>, tensor<16x32xi8>)
                    outs(%1: tensor<32x32xi32>) -> tensor<32x32xi32>
  return %res : tensor<32x32xi32>
}

// CHECK-LABEL: hal.executable.export public @matmul_i8_i32_dispatch_0_matmul_32x32x16_i8xi8xi32
//       CHECK:    aie.device(npu1_4col)
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    func.func @matmul_i8_i32_dispatch_0_matmul_32x32x16_i8xi8xi32(%arg0: memref<128xi32>, %arg1: memref<128xi32>, %arg2: memref<32x32xi32>)
//       CHECK:      aiex.npu.dma_memcpy_nd
//       CHECK:      aiex.npu.dma_memcpy_nd
//       CHECK:      aiex.npu.dma_memcpy_nd
//       CHECK:      aiex.npu.sync

// -----

func.func @matmul_bf16(%lhs: tensor<16x32xbf16>, %rhs: tensor<32x16xbf16>) -> tensor<16x16xbf16>
{
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = tensor.empty() : tensor<16x16xbf16>
  %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<16x16xbf16>) -> tensor<16x16xbf16>
  %res = linalg.matmul ins(%lhs, %rhs: tensor<16x32xbf16>, tensor<32x16xbf16>)
                    outs(%1: tensor<16x16xbf16>) -> tensor<16x16xbf16>
  return %res : tensor<16x16xbf16>
}

// CHECK-LABEL: hal.executable.export public @matmul_bf16_dispatch_0_matmul_16x16x32_bf16
//       CHECK:    aie.device(npu1_4col)
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    func.func @matmul_bf16_dispatch_0_matmul_16x16x32_bf16(%arg0: memref<256xi32>, %arg1: memref<256xi32>, %arg2: memref<128xi32>)
//       CHECK:      aiex.npu.dma_memcpy_nd
//       CHECK:      aiex.npu.dma_memcpy_nd
//       CHECK:      aiex.npu.dma_memcpy_nd
//       CHECK:      aiex.npu.sync
