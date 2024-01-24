// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-sources %s | iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" | FileCheck %s

// CHECK-LABEL: hal.executable.export public @matmul_static_dispatch_0_matmul_8x32x16_i32
//       CHECK:    aie.device(ipu)
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    func.func @matmul_static_dispatch_0_matmul_8x32x16_i32(%arg0: memref<8x16xi32>, %arg1: memref<16x32xi32>, %arg2: memref<8x32xi32>)
//       CHECK:      aiex.ipu.dma_memcpy_nd
//       CHECK:      aiex.ipu.dma_memcpy_nd
//       CHECK:      aiex.ipu.dma_memcpy_nd
//       CHECK:      aiex.ipu.sync

#compilation = #iree_codegen.compilation_info<lowering_config = <tile_sizes = [[8, 16], [1, 1], [0, 0, 1]]>, translation_info = <CPUDefault>>
func.func @matmul_static(%lhs : tensor<8x16xi32>,
    %rhs : tensor<16x32xi32>) -> tensor<8x32xi32> {
  %empty = tensor.empty() : tensor<8x32xi32>
  %cst = arith.constant 0 : i32
  %fill = linalg.fill ins(%cst : i32) outs(%empty : tensor<8x32xi32>) -> tensor<8x32xi32>
  %2 = linalg.matmul {compilation_info = #compilation} ins(%lhs, %rhs : tensor<8x16xi32>, tensor<16x32xi32>)
      outs(%fill : tensor<8x32xi32>) -> tensor<8x32xi32>
  return %2 : tensor<8x32xi32>
}
