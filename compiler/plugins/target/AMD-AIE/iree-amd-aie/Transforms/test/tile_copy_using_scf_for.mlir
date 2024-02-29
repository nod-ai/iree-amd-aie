// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-tile{tiling-level=1}))' --split-input-file %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [0, 0, 256]]>
func.func @matmul_example(%arg0 : tensor<64x2048xi32>, %arg1 : tensor<2048x64xi32>) -> tensor<64x64xi32> {
  %c0_i32 = arith.constant 0 : i32
  %9 = bufferization.alloc_tensor() : tensor<64x2048xi32>
  %10 = linalg.copy {lowering_config = #config} ins(%arg0: tensor<64x2048xi32>) outs(%9 : tensor<64x2048xi32>) -> tensor<64x2048xi32>
  %11 = bufferization.alloc_tensor() : tensor<2048x64xi32>
  %12 = linalg.copy {lowering_config = #config} ins(%arg1 : tensor<2048x64xi32>) outs(%11 : tensor<2048x64xi32>) -> tensor<2048x64xi32>
  %13 = bufferization.alloc_tensor() : tensor<64x64xi32>
  %14 = linalg.fill ins(%c0_i32 : i32) outs(%13 : tensor<64x64xi32>) -> tensor<64x64xi32>
  %15 = linalg.matmul ins(%10, %12 : tensor<64x2048xi32>, tensor<2048x64xi32>) outs(%14 : tensor<64x64xi32>) -> tensor<64x64xi32>
  %16 = linalg.copy ins(%15 : tensor<64x64xi32>) outs(%13 : tensor<64x64xi32>) -> tensor<64x64xi32>
  return %16 : tensor<64x64xi32>
}

// CHECK:   @matmul_example
// CHECK:   {
// CHECK:       scf.for
// CHECK:       {
// CHECK:           linalg.copy
// CHECK:       }
// CHECK:       scf.for
// CHECK:       {
// CHECK:           linalg.copy
// CHECK:       }
// CHECK:       linalg.fill
// CHECK:       linalg.matmul
// CHECK:       linalg.copy
// CHECK:   }

