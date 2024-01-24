// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-pack-and-transpose{pack-level=0}))' --split-input-file %s | FileCheck %s

func.func @matmul_example_dispatch_0_matmul_16x256x256_i8xi8xi32(%arg0 : tensor<16x256xi8>, %arg1 : tensor<256x256xi8>) -> tensor<16x256xi32> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %5 = tensor.empty() : tensor<16x256xi32>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<16x256xi32>) -> tensor<16x256xi32>
  // CHECK: tensor.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [8, 16] into %{{.*}} : tensor<16x256xi8> -> tensor<2x16x8x16xi8>
  // CHECK: tensor.pack %{{.*}} outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %{{.*}} : tensor<256x256xi8> -> tensor<16x16x16x16xi8>
  // CHECK: tensor.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [8, 16] into %{{.*}} : tensor<16x256xi32> -> tensor<2x16x8x16xi32>
  %7 = linalg.matmul ins(%arg0, %arg1 : tensor<16x256xi8>, tensor<256x256xi8>) outs(%6 : tensor<16x256xi32>) -> tensor<16x256xi32>
  return %7 : tensor<16x256xi32>
}
