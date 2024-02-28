// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-tile-and-fuse{use-scf-for tiling-level=2 tiling-op=rhsCopyOp}))' --split-input-file %s | FileCheck %s --check-prefix=CHECK-RHS-COPY
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-tile-and-fuse{use-scf-for tiling-level=1 tiling-op=lhsCopyOp}))' --split-input-file %s | FileCheck %s --check-prefix=CHECK-LHS-COPY

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [0, 256], [256, 0]]>
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

// CHECK-RHS-COPY: @matmul_example
// CHECK-RHS-COPY:   {
// CHECK-RHS-COPY:       linalg.copy
// CHECK-RHS-COPY:       scf.for
// CHECK-RHS-COPY:       {
// CHECK-RHS-COPY:           linalg.copy
// CHECK-RHS-COPY:       }
// CHECK-RHS-COPY:       linalg.fill
// CHECK-RHS-COPY:       linalg.matmul
// CHECK-RHS-COPY:       linalg.copy
// CHECK-RHS-COPY:   }

// CHECK-LHS-COPY: @matmul_example
// CHECK-LHS-COPY:   {
// CHECK-LHS-COPY:       scf.for
// CHECK-LHS-COPY:       {
// CHECK-LHS-COPY:           linalg.copy
// CHECK-LHS-COPY:       }
// CHECK-LHS-COPY:       linalg.copy
// CHECK-LHS-COPY:       linalg.fill
// CHECK-LHS-COPY:       linalg.matmul
// CHECK-LHS-COPY:       linalg.copy
// CHECK-LHS-COPY:   }
