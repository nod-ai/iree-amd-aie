// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-sources --mlir-print-ir-after=fold-memref-alias-ops %s | iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" --iree-amdaie-use-pipeline=pack-peel | FileCheck %s

func.func @matmul_example(%lhs: tensor<1024x512xi8>, %rhs: tensor<512x1024xi8>) -> tensor<1024x1024xi32>
{
  %cst = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<1024x1024xi32>
  %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<1024x1024xi32>) -> tensor<1024x1024xi32>
  %res = linalg.matmul ins(%lhs, %rhs: tensor<1024x512xi8>, tensor<512x1024xi8>)
                    outs(%1: tensor<1024x1024xi32>) -> tensor<1024x1024xi32>
  return %res : tensor<1024x1024xi32>
}

// CHECK-LABEL: hal.executable.export public @matmul_example_dispatch_0_matmul_1024x1024x512_i8xi8xi32
//       CHECK:    aie.device(ipu)
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    func.func @matmul_example_dispatch_0_matmul_1024x1024x512_i8xi8xi32(%arg0: memref<131072xi32>, %arg1: memref<131072xi32>, %arg2: memref<1024x1024xi32>)
//       CHECK:      aiex.ipu.dma_memcpy_nd
//       CHECK:      aiex.ipu.dma_memcpy_nd
//       CHECK:      aiex.ipu.dma_memcpy_nd
//       CHECK:      aiex.ipu.sync

func.func @matmul_bf16(%lhs: tensor<512x1024xbf16>, %rhs: tensor<1024x512xbf16>) -> tensor<512x512xbf16>
{
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = tensor.empty() : tensor<512x512xbf16>
  %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<512x512xbf16>) -> tensor<512x512xbf16>
  %res = linalg.matmul ins(%lhs, %rhs: tensor<512x1024xbf16>, tensor<1024x512xbf16>)
                    outs(%1: tensor<512x512xbf16>) -> tensor<512x512xbf16>
  return %res : tensor<512x512xbf16>
}

// CHECK-LABEL: hal.executable.export public @matmul_large_dispatch_0_matmul_512x512x1024_bf16
//       CHECK:    aie.device(ipu)
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    func.func @matmul_large_dispatch_0_matmul_512x512x1024_bf16(%arg0: memref<262144xi32>, %arg1: memref<262144xi32>, %arg2: memref<131072xi32>)
//       CHECK:      aiex.ipu.dma_memcpy_nd
//       CHECK:      aiex.ipu.dma_memcpy_nd
//       CHECK:      aiex.ipu.dma_memcpy_nd
//       CHECK:      aiex.ipu.sync
