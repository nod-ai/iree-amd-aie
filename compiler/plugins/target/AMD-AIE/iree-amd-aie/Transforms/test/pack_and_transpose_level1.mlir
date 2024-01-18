// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-pack-and-transpose{pack-level=1}))' --split-input-file %s | FileCheck --check-prefix=CHECK-1 %s

func.func @matmul_example_dispatch_0_matmul_16x256x256_i8xi8xi32(%arg0 : tensor<16x256xi8>, %arg1 : tensor<256x256xi8>) -> tensor<16x256xi32> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %5 = tensor.empty() : tensor<16x256xi32>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<16x256xi32>) -> tensor<16x256xi32>
  // CHECK-1: tensor.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %{{.*}} : tensor<16x256xi8> -> tensor<1x4x16x64xi8>
  // CHECK-1: tensor.pack %{{.*}} outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %{{.*}} : tensor<256x256xi8> -> tensor<4x4x64x64xi8>
  // CHECK-1: tensor.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %{{.*}} : tensor<16x256xi32> -> tensor<1x4x16x64xi32>
  %7 = linalg.matmul ins(%arg0, %arg1 : tensor<16x256xi8>, tensor<256x256xi8>) outs(%6 : tensor<16x256xi32>) -> tensor<16x256xi32>
  return %7 : tensor<16x256xi32>
}
