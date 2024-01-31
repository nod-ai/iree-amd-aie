// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-sources %s | iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" | FileCheck %s --check-prefix=CPP
// XFAIL: *
// This test demonstrates Pack pipeline based e2e lowering.

// To check the cpp path equivalent to the transform dialect script.
// CPP-LABEL: hal.executable.export public @matmul_static_dispatch_0_matmul_8x32x16_i32
//       CPP:    aie.device(ipu)
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    func.func @matmul_static_dispatch_0_matmul_8x32x16_i32(%arg0: memref<8x16xi32>, %arg1: memref<16x32xi32>, %arg2: memref<8x32xi32>)
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.sync
func.func @matmul_accumulate_i32xi32(%lhs: tensor<8x16xi32>, %rhs: tensor<16x32xi32>, %acc: tensor<8x32xi32>) -> tensor<8x32xi32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<8x16xi32>, tensor<16x32xi32>) outs(%acc: tensor<8x32xi32>) -> tensor<8x32xi32>
  return %result: tensor<8x32xi32>
}
