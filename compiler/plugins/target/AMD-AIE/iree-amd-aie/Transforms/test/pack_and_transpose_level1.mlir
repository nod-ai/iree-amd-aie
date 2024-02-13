// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-pack-and-transpose{pack-level=0}))' --split-input-file %s | FileCheck %s

// CHECK: #config
#config = #iree_codegen.lowering_config<tile_sizes = [[8, 8], [4, 4], [0, 0, 4]]>
// CHECK: #packingConfig
#packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [8, 8, 256], transposePackIndices = [1], unpackEmpty = [0], innerPerm = [[1, 0]], outerPerm = [[0, 1]]}, {packedSizes = [0, 0, 0, 4, 8, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [0, 0, 1], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
// CHECK: @matmul_example_dispatch_0_matmul_16x256x256_i8xi8xi32
func.func @matmul_example_dispatch_0_matmul_16x256x256_i8xi8xi32(%arg0 : tensor<16x256xi8>, %arg1 : tensor<256x256xi8>) -> tensor<16x256xi32> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %5 = tensor.empty() : tensor<16x256xi32>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<16x256xi32>) -> tensor<16x256xi32>
  // CHECK:       tensor.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [8, 256] into %{{.*}} : tensor<16x256xi8> -> tensor<2x1x8x256xi8>
  // CHECK:       tensor.pack %{{.*}} outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [256, 8] into %{{.*}} : tensor<256x256xi8> -> tensor<1x32x256x8xi8>
  // CHECK:       tensor.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %{{.*}} : tensor<16x256xi32> -> tensor<2x32x8x8xi32>
  // CHECK:       linalg.generic
  // CHECK-SAME:  attrs =  {lowering_config = #config, packing_config = #packingConfig}
  %7 = linalg.matmul {lowering_config = #config, packing_config = #packingConfig} ins(%arg0, %arg1 : tensor<16x256xi8>, tensor<256x256xi8>) outs(%6 : tensor<16x256xi32>) -> tensor<16x256xi32>
  return %7 : tensor<16x256xi32>
}
