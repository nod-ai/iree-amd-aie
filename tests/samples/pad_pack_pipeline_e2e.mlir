// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-sources %s | iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" --iree-amdaie-use-pipeline=pad-pack --split-input-file | FileCheck %s --check-prefix=CPP

// This test demonstrates Pad-Pack pipeline based e2e lowering.

// CPP-LABEL: hal.executable.export public @matmul_small_dispatch_0_matmul_8x32x16_i32
//       CPP:    aie.device(ipu)
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    func.func @matmul_small_dispatch_0_matmul_8x32x16_i32(%arg0: memref<8x16xi32>, %arg1: memref<16x32xi32>, %arg2: memref<8x32xi32>)
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.sync
func.func @matmul_small(%lhs : tensor<8x16xi32>,
    %rhs : tensor<16x32xi32>) -> tensor<8x32xi32> {
  %empty = tensor.empty() : tensor<8x32xi32>
  %cst = arith.constant 0 : i32
  %fill = linalg.fill ins(%cst : i32) outs(%empty : tensor<8x32xi32>) -> tensor<8x32xi32>
  %2 = linalg.matmul ins(%lhs, %rhs : tensor<8x16xi32>, tensor<16x32xi32>)
      outs(%fill : tensor<8x32xi32>) -> tensor<8x32xi32>
  return %2 : tensor<8x32xi32>
}

// -----

// CPP-LABEL: hal.executable.export public @matmul_large_dispatch_0_matmul_2048x2048x2048_i32
//       CPP:    aie.device(ipu)
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    func.func @matmul_large_dispatch_0_matmul_2048x2048x2048_i32(%arg0: memref<2048x2048xi32>, %arg1: memref<2048x2048xi32>, %arg2: memref<2048x2048xi32>)
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.sync

func.func @matmul_large(%lhs: tensor<2048x2048xi32>, %rhs: tensor<2048x2048xi32>) -> tensor<2048x2048xi32> {
  %empty = tensor.empty() : tensor<2048x2048xi32>
  %cst = arith.constant 0 : i32
  %fill = linalg.fill ins(%cst : i32) outs(%empty : tensor<2048x2048xi32>) -> tensor<2048x2048xi32>
  %res = linalg.matmul ins(%lhs, %rhs: tensor<2048x2048xi32>, tensor<2048x2048xi32>)
                    outs(%fill: tensor<2048x2048xi32>) -> tensor<2048x2048xi32>
  return %res : tensor<2048x2048xi32>
}
