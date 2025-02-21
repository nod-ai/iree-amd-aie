// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-pack-and-transpose{pack-level=0}))' --split-input-file %s | FileCheck %s

// CHECK: #config
#config = #iree_codegen.lowering_config<tile_sizes = [[16, 256], [0, 0, 1], [0, 0, 0, 4, 16, 0]]>
// CHECK: #packingConfig
#packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [16, 256, 128], transposePackIndices = [1], unpackEmpty = [false], innerPerm = [[1, 0]], outerPerm = [[0, 1]]}, {packedSizes = [0, 0, 0, 4, 8, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
// CHECK: @matmul_example_dispatch_0_matmul_16x256x256_i8xi8xi32
func.func @matmul_example_dispatch_0_matmul_16x256x256_i8xi8xi32(%arg0 : tensor<16x256xi8>, %arg1 : tensor<256x256xi8>) -> tensor<16x256xi32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<16x256xi32>
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<16x256xi32>) -> tensor<16x256xi32>
  // CHECK:       linalg.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [16, 128] into %{{.*}} : tensor<16x256xi8> -> tensor<1x2x16x128xi8>
  // CHECK:       linalg.pack %{{.*}} outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [128, 256] into %{{.*}} : tensor<256x256xi8> -> tensor<2x1x128x256xi8>
  // CHECK:       linalg.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [16, 256] into %{{.*}} : tensor<16x256xi32> -> tensor<1x1x16x256xi32>
  // CHECK:       linalg.generic
  // CHECK-SAME:  attrs =  {lowering_config = #config, packing_config = #packingConfig}
  %2 = linalg.matmul {lowering_config = #config, packing_config = #packingConfig} ins(%arg0, %arg1 : tensor<16x256xi8>, tensor<256x256xi8>) outs(%1 : tensor<16x256xi32>) -> tensor<16x256xi32>
  return %2 : tensor<16x256xi32>
}

// -----

// CHECK: #config
#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [0, 0, 1], [0, 0, 0, 8, 8, 0]]>
// CHECK: #packingConfig
#packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [64, 64, 32], transposePackIndices = [1], unpackEmpty = [false], innerPerm = [[0, 1]], outerPerm = [[0, 1]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [0, 1], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
// CHECK: @matmul_transpose_b_dispatch_0_matmul_transpose_b_256x1024x512_i32
func.func @matmul_transpose_b_dispatch_0_matmul_transpose_b_256x1024x512_i32(%arg0: tensor<256x512xi32>, %arg1: tensor<1024x512xi32>) -> tensor<256x1024xi32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<256x1024xi32>
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<256x1024xi32>) -> tensor<256x1024xi32>
  // CHECK:       linalg.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [64, 32] into %2 : tensor<256x512xi32> -> tensor<4x16x64x32xi32>
  // CHECK:       linalg.pack %{{.*}} outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [64, 32] into %{{.*}} : tensor<1024x512xi32> -> tensor<16x16x64x32xi32>
  // CHECK:       linalg.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %{{.*}} : tensor<256x1024xi32> -> tensor<4x16x64x64xi32>
  // CHECK:       linalg.generic
  // CHECK-SAME:  attrs =  {lowering_config = #config, packing_config = #packingConfig}
  %2 = linalg.matmul_transpose_b {lowering_config = #config, packing_config = #packingConfig} ins(%arg0, %arg1 : tensor<256x512xi32>, tensor<1024x512xi32>) outs(%1 : tensor<256x1024xi32>) -> tensor<256x1024xi32>
  return %2 : tensor<256x1024xi32>
}
