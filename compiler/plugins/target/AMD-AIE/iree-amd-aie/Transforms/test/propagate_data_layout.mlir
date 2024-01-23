// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-propagate-data-layout, canonicalize, cse))' --split-input-file %s | FileCheck %s


#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  func.func @matmul_static(%arg0: tensor<1x4x16x64xi32>, %arg1: tensor<4x1x64x64xi32>) -> tensor<1x1x16x64xi32> {
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<1x4x8x4x4x8xi32>
    %pack = tensor.pack %arg0 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %0 : tensor<1x4x16x64xi32> -> tensor<1x4x8x4x4x8xi32>
    %1 = tensor.empty() : tensor<4x1x8x8x8x8xi32>
    %pack_0 = tensor.pack %arg1 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 8] into %1 : tensor<4x1x64x64xi32> -> tensor<4x1x8x8x8x8xi32>
    %2 = tensor.empty() : tensor<1x1x8x4x4x8xi32>
    %3 = linalg.fill ins(%c0_i32 : i32) outs(%2 : tensor<1x1x8x4x4x8xi32>) -> tensor<1x1x8x4x4x8xi32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_0 : tensor<1x4x8x4x4x8xi32>, tensor<4x1x8x8x8x8xi32>) outs(%3 : tensor<1x1x8x4x4x8xi32>) {
    ^bb0(%in: i32, %in_1: i32, %out: i32):
      %6 = arith.muli %in, %in_1 : i32
      %7 = arith.addi %out, %6 : i32
      linalg.yield %7 : i32
    } -> tensor<1x1x8x4x4x8xi32>
    %empty = tensor.empty() : tensor<1x1x16x64xi32>
    %unpack = tensor.unpack %4 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %empty : tensor<1x1x8x4x4x8xi32> -> tensor<1x1x16x64xi32>
    %empty2 = tensor.empty() : tensor<1x1x16x64xi32>
    %fill = linalg.fill ins(%c0_i32 : i32) outs(%empty2 : tensor<1x1x16x64xi32>) -> tensor<1x1x16x64xi32>
    %5 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%unpack: tensor<1x1x16x64xi32>) outs(%fill : tensor<1x1x16x64xi32>) {
    ^bb0(%in: i32, %out: i32):
      %6 = arith.muli %in, %out : i32
      %7 = arith.addi %out, %6 : i32
      linalg.yield %7 : i32
    } -> tensor<1x1x16x64xi32>
    return %5 : tensor<1x1x16x64xi32>
  }
// CHECK-LABEL: matmul_static
// CHECK-COUNT-2: tensor.pack
// CHECK: linalg.generic
// CHECK-NOT: tensor.pack
// CHECK-NOT: tensor.unpack
// CHECK: linalg.generic
// CHECK: tensor.unpack
