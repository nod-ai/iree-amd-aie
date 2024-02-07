// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-fuse-pack-into-for))' %s | FileCheck %s

#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
func.func @fuse_pack_into_for(%arg0: tensor<1x1x32x512xi32>, %arg1: tensor<1x1x512x32xi32>) -> tensor<1x1x4x8x4x8xi32> {
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %15 = tensor.empty() : tensor<1x1x64x8x4x8xi32>
  %pack_8 = tensor.pack %arg0 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %15 : tensor<1x1x32x512xi32> -> tensor<1x1x64x8x4x8xi32>
  %16 = tensor.empty() : tensor<1x1x4x64x8x8xi32>
  %pack_9 = tensor.pack %arg1 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 8] into %16 : tensor<1x1x512x32xi32> -> tensor<1x1x4x64x8x8xi32>
  %17 = tensor.empty() : tensor<1x1x4x8x4x8xi32>
  %18 = linalg.fill ins(%c0_i32 : i32) outs(%17 : tensor<1x1x4x8x4x8xi32>) -> tensor<1x1x4x8x4x8xi32>
  %19 = scf.for %arg6 = %c0 to %c64 step %c4 iter_args(%arg7 = %18) -> (tensor<1x1x4x8x4x8xi32>) {
    %extracted_slice_12 = tensor.extract_slice %pack_8[0, 0, %arg6, 0, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : tensor<1x1x64x8x4x8xi32> to tensor<1x1x4x8x4x8xi32>
    %extracted_slice_13 = tensor.extract_slice %pack_9[0, 0, 0, %arg6, 0, 0] [1, 1, 4, 4, 8, 8] [1, 1, 1, 1, 1, 1] : tensor<1x1x4x64x8x8xi32> to tensor<1x1x4x4x8x8xi32>
    %20 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%extracted_slice_12, %extracted_slice_13 : tensor<1x1x4x8x4x8xi32>, tensor<1x1x4x4x8x8xi32>) outs(%arg7 : tensor<1x1x4x8x4x8xi32>) {
    ^bb0(%in: i32, %in_14: i32, %out: i32):
      %21 = arith.muli %in, %in_14 : i32
      %22 = arith.addi %out, %21 : i32
      linalg.yield %22 : i32
    } -> tensor<1x1x4x8x4x8xi32>
    scf.yield %20 : tensor<1x1x4x8x4x8xi32>
  }
  return %19 : tensor<1x1x4x8x4x8xi32>
}

// CHECK:  @fuse_pack_into_for
// CHECK:  scf.for
// CHECK:  {
// CHECK:     tensor.extract_slice %{{.*}} : tensor<1x1x32x512xi32> to tensor<1x1x32x32xi32>
// CHECK:     tensor.extract_slice %{{.*}} : tensor<1x1x64x8x4x8xi32> to tensor<1x1x4x8x4x8xi32>
// CHECK:     tensor.pack %{{.*}} outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %{{.*}} : tensor<1x1x32x32xi32> -> tensor<1x1x4x8x4x8xi32>
// CHECK:     tensor.extract_slice %{{.*}} : tensor<1x1x512x32xi32> to tensor<1x1x32x32xi32>
// CHECK:     tensor.extract_slice %{{.*}} : tensor<1x1x4x64x8x8xi32> to tensor<1x1x4x4x8x8xi32>
// CHECK:     tensor.pack %{{.*}} outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 8] into %{{.*}} : tensor<1x1x32x32xi32> -> tensor<1x1x4x4x8x8xi32>
// CHECK:     linalg.generic
// CHECK:  }
