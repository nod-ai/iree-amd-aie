// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-sources %s | iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" --iree-amdaie-use-pipeline=pad | FileCheck %s --check-prefix=CPP

// This test demonstrates Pad pipeline based e2e lowering.

// To check the cpp path equivalent to the transform dialect script.
// CPP-LABEL: hal.executable.export public @matmul_static_dispatch_0_matmul_8x8x16_i32
//       CPP:    aie.device(ipu)
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    func.func @matmul_static_dispatch_0_matmul_8x8x16_i32(%arg0: memref<8x16xi32>, %arg1: memref<16x8xi32>, %arg2: memref<8x8xi32>)
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.sync
func.func @matmul_static(%lhs : tensor<8x16xi32>,
    %rhs : tensor<16x8xi32>) -> tensor<8x8xi32> {
  %empty = tensor.empty() : tensor<8x8xi32>
  %cst = arith.constant 0 : i32
  %fill = linalg.fill ins(%cst : i32) outs(%empty : tensor<8x8xi32>) -> tensor<8x8xi32>
  %2 = linalg.matmul ins(%lhs, %rhs : tensor<8x16xi32>, tensor<16x8xi32>)
      outs(%fill : tensor<8x8xi32>) -> tensor<8x8xi32>
  return %2 : tensor<8x8xi32>
}
