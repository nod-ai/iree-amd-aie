// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-pack-and-transpose{pack-level=1}))' --split-input-file %s | FileCheck %s

// CHECK: #config
#config = #iree_codegen.lowering_config<tile_sizes = [[16, 256], [0, 0, 1], [0, 0, 0, 4, 16, 0]]>
#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// CHECK: #packingConfig
#packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [16, 256, 128], transposePackIndices = [1], unpackEmpty = [false], innerPerm = [[1, 0]], outerPerm = [[0, 1]]}, {packedSizes = [0, 0, 0, 4, 8, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
func.func @matmul_example_dispatch_0_matmul_16x256x256_i8xi8xi32(%arg0: tensor<16x256xi8>, %arg1: tensor<256x256xi8>) -> tensor<16x256xi32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<16x256xi32>
  %1 = scf.forall (%arg2, %arg3) = (0, 0) to (16, 256) step (16, 256) shared_outs(%arg4 = %0) -> (tensor<16x256xi32>) {
    %extracted_slice = tensor.extract_slice %arg0[%arg2, 0] [16, 256] [1, 1] : tensor<16x256xi8> to tensor<16x256xi8>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg3] [256, 256] [1, 1] : tensor<256x256xi8> to tensor<256x256xi8>
    %extracted_slice_1 = tensor.extract_slice %arg4[%arg2, %arg3] [16, 256] [1, 1] : tensor<16x256xi32> to tensor<16x256xi32>
    %2 = tensor.empty() : tensor<1x2x16x128xi8>
    %pack = linalg.pack %extracted_slice inner_dims_pos = [0, 1] inner_tiles = [16, 128] into %2 : tensor<16x256xi8> -> tensor<1x2x16x128xi8>
    %3 = tensor.empty() : tensor<2x1x128x256xi8>
    %pack_2 = linalg.pack %extracted_slice_0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [128, 256] into %3 : tensor<256x256xi8> -> tensor<2x1x128x256xi8>
    %4 = tensor.empty() : tensor<1x1x16x256xi32>
    %5 = linalg.fill ins(%c0_i32 : i32) outs(%4 : tensor<1x1x16x256xi32>) -> tensor<1x1x16x256xi32>
    // CHECK: linalg.pack %{{.*}} outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %{{.*}} : tensor<1x2x16x128xi8> -> tensor<1x2x16x4x4x8xi8>
    // CHECK: linalg.pack %{{.*}} outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 8] into %{{.*}} : tensor<2x1x128x256xi8> -> tensor<2x1x32x16x8x8xi8>
    // CHECK: linalg.pack %{{.*}} outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %{{.*}} : tensor<1x1x16x256xi32> -> tensor<1x1x32x4x4x8xi32>
    // CHECK:       linalg.generic
    // CHECK-SAME:  attrs =  {lowering_config = #config, packing_config = #packingConfig}
    %6 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_2 : tensor<1x2x16x128xi8>, tensor<2x1x128x256xi8>) outs(%5 : tensor<1x1x16x256xi32>) attrs =  {lowering_config = #config, packing_config = #packingConfig} {
    ^bb0(%in: i8, %in_3: i8, %out: i32):
      %7 = arith.extsi %in : i8 to i32
      %8 = arith.extsi %in_3 : i8 to i32
      %9 = arith.muli %7, %8 : i32
      %10 = arith.addi %out, %9 : i32
      linalg.yield %10 : i32
    } -> tensor<1x1x16x256xi32>
    %unpack = linalg.unpack %6 inner_dims_pos = [0, 1] inner_tiles = [16, 256] into %extracted_slice_1 : tensor<1x1x16x256xi32> -> tensor<16x256xi32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %unpack into %arg4[%arg2, %arg3] [16, 256] [1, 1] : tensor<16x256xi32> into tensor<16x256xi32>
    }
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  return %1 : tensor<16x256xi32>
}

// -----

// CHECK: #config
#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [0, 0, 1], [0, 0, 0, 8, 8, 0]]>
#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// CHECK: #packingConfig
#packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [64, 64, 32], transposePackIndices = [1], unpackEmpty = [false], innerPerm = [[0, 1]], outerPerm = [[0, 1]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [0, 1], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
// CHECK: @matmul_transpose_b_dispatch_0_matmul_transpose_b_256x1024x512_i32
func.func @matmul_transpose_b_dispatch_0_matmul_transpose_b_256x1024x512_i32(%arg0: tensor<256x512xi32>, %arg1: tensor<1024x512xi32>) -> tensor<256x1024xi32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<256x1024xi32>
  %1 = scf.forall (%arg2, %arg3) = (0, 0) to (256, 1024) step (64, 64) shared_outs(%arg4 = %0) -> (tensor<256x1024xi32>) {
    %extracted_slice = tensor.extract_slice %arg0[%arg2, 0] [64, 512] [1, 1] : tensor<256x512xi32> to tensor<64x512xi32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg3, 0] [64, 512] [1, 1] : tensor<1024x512xi32> to tensor<64x512xi32>
    %extracted_slice_1 = tensor.extract_slice %arg4[%arg2, %arg3] [64, 64] [1, 1] : tensor<256x1024xi32> to tensor<64x64xi32>
    %2 = tensor.empty() : tensor<1x16x64x32xi32>
    %pack = linalg.pack %extracted_slice inner_dims_pos = [0, 1] inner_tiles = [64, 32] into %2 : tensor<64x512xi32> -> tensor<1x16x64x32xi32>
    %pack_2 = linalg.pack %extracted_slice_0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [64, 32] into %2 : tensor<64x512xi32> -> tensor<1x16x64x32xi32>
    %3 = tensor.empty() : tensor<1x1x64x64xi32>
    %4 = linalg.fill ins(%c0_i32 : i32) outs(%3 : tensor<1x1x64x64xi32>) -> tensor<1x1x64x64xi32>
    // CHECK: linalg.pack %{{.*}} outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %{{.*}} : tensor<1x16x64x32xi32> -> tensor<1x16x4x16x4x8xi32>
    // CHECK: linalg.pack %{{.*}} outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %{{.*}} : tensor<1x16x64x32xi32> -> tensor<1x16x4x16x4x8xi32>
    // CHECK: linalg.pack %{{.*}} outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 4] into %{{.*}} : tensor<1x1x64x64xi32> -> tensor<1x1x16x16x4x4xi32>
    // CHECK:       linalg.generic
    // CHECK-SAME:  attrs =  {lowering_config = #config, packing_config = #packingConfig}
    %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_2 : tensor<1x16x64x32xi32>, tensor<1x16x64x32xi32>) outs(%4 : tensor<1x1x64x64xi32>) attrs =  {lowering_config = #config, packing_config = #packingConfig} {
    ^bb0(%in: i32, %in_3: i32, %out: i32):
      %6 = arith.muli %in, %in_3 : i32
      %7 = arith.addi %out, %6 : i32
      linalg.yield %7 : i32
    } -> tensor<1x1x64x64xi32>
    %unpack = linalg.unpack %5 inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %extracted_slice_1 : tensor<1x1x64x64xi32> -> tensor<64x64xi32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %unpack into %arg4[%arg2, %arg3] [64, 64] [1, 1] : tensor<64x64xi32> into tensor<256x1024xi32>
    }
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  return %1 : tensor<256x1024xi32>
}
