// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-sources %s | iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" --iree-amdaie-enable-large-gemm | FileCheck %s --check-prefix=CPP

// This test demonstrates Pack pipeline based e2e lowering.

// To check the cpp path equivalent to the transform dialect script.
// CPP-LABEL: hal.executable.export public @matmul_static_dispatch_0_matmul_2048x2048x512_i32
//       CPP:    aie.device(ipu)
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    func.func @matmul_static_dispatch_0_matmul_2048x2048x512_i32(%arg0: memref<2048x512xi32>, %arg1: memref<512x2048xi32>, %arg2: memref<2048x2048xi32>)
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.sync

func.func @matmul_static(%lhs: tensor<2048x512xi32>, %rhs: tensor<512x2048xi32>) -> tensor<2048x2048xi32> {
  %empty = tensor.empty() : tensor<2048x2048xi32>
  %cst = arith.constant 0 : i32
  %fill = linalg.fill ins(%cst : i32) outs(%empty : tensor<2048x2048xi32>) -> tensor<2048x2048xi32>
  %res = linalg.matmul ins(%lhs, %rhs: tensor<2048x512xi32>, tensor<512x2048xi32>)
                    outs(%fill: tensor<2048x2048xi32>) -> tensor<2048x2048xi32>
  return %res : tensor<2048x2048xi32>
}
