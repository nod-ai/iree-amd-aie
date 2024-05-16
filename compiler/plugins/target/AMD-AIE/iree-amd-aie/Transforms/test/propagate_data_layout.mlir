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
//       CHECK: %[[PACK_0:.*]] = tensor.pack {{.*}} : tensor<1x4x16x64xi32> -> tensor<1x4x8x4x4x8xi32>
//       CHECK: %[[PACK_1:.*]] = tensor.pack {{.*}} : tensor<4x1x64x64xi32> -> tensor<4x1x8x8x8x8xi32>
//       CHECK: %[[FILL:.*]] = linalg.fill {{.*}} -> tensor<1x1x8x4x4x8xi32>
//       CHECK: %[[MATMUL_0:.*]] = linalg.generic {{.*}} ins(%[[PACK_0]], %[[PACK_1]] : tensor<1x4x8x4x4x8xi32>, tensor<4x1x8x8x8x8xi32>) outs(%[[FILL]] : tensor<1x1x8x4x4x8xi32>)
//   CHECK-NOT: tensor.unpack
//   CHECK-NOT: tensor.pack
//       CHECK: %[[MATMUL_1:.*]] = linalg.generic {{.*}} ins(%[[MATMUL_0]] : tensor<1x1x8x4x4x8xi32>) outs(%[[FILL]] : tensor<1x1x8x4x4x8xi32>)
//       CHECK: %[[UNPACK:.*]] = tensor.unpack %[[MATMUL_1:.*]] {{.*}} : tensor<1x1x8x4x4x8xi32> -> tensor<1x1x16x64xi32>

// -----

func.func @matmul_elementwise_1024x1024x512_i8xi8xi32(%arg0: tensor<1024x512xi8>, %arg1: tensor<512x1024xi8>, %arg2: tensor<1024x1024xi32>) -> tensor<1024x1024xi32> {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<1024x1024xi32>
  %1 = scf.forall (%arg3, %arg4) = (0, 0) to (1024, 1024) step (64, 64) shared_outs(%arg5 = %0) -> (tensor<1024x1024xi32>) {
    %extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [64, 512] [1, 1] : tensor<1024x512xi8> to tensor<64x512xi8>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg4] [512, 64] [1, 1] : tensor<512x1024xi8> to tensor<512x64xi8>
    %extracted_slice_1 = tensor.extract_slice %0[%arg3, %arg4] [64, 64] [1, 1] : tensor<1024x1024xi32> to tensor<64x64xi32>
    %2 = linalg.fill ins(%c0_i32 : i32) outs(%extracted_slice_1 : tensor<64x64xi32>) -> tensor<64x64xi32>
    %3 = tensor.empty() : tensor<1x16x64x32xi8>
    %pack = tensor.pack %extracted_slice inner_dims_pos = [0, 1] inner_tiles = [64, 32] into %3 : tensor<64x512xi8> -> tensor<1x16x64x32xi8>
    %4 = tensor.empty() : tensor<16x1x64x32xi8>
    %5 = tensor.empty() : tensor<16x1x32x64xi8>
    %pack_2 = tensor.pack %extracted_slice_0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 64] into %5 : tensor<512x64xi8> -> tensor<16x1x32x64xi8>
    %6 = tensor.empty() : tensor<1x1x64x64xi32>
    %pack_3 = tensor.pack %2 inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %6 : tensor<64x64xi32> -> tensor<1x1x64x64xi32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d5, d4)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_2 : tensor<1x16x64x32xi8>, tensor<16x1x32x64xi8>) outs(%pack_3 : tensor<1x1x64x64xi32>) {
    ^bb0(%in: i8, %in_6: i8, %out: i32):
      %9 = arith.extsi %in : i8 to i32
      %10 = arith.extsi %in_6 : i8 to i32
      %11 = arith.muli %9, %10 : i32
      %12 = arith.addi %out, %11 : i32
      linalg.yield %12 : i32
    } -> tensor<1x1x64x64xi32>
    %unpack = tensor.unpack %7 inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %2 : tensor<1x1x64x64xi32> -> tensor<64x64xi32>
    %extracted_slice_4 = tensor.extract_slice %arg2[%arg3, %arg4] [64, 64] [1, 1] : tensor<1024x1024xi32> to tensor<64x64xi32>
    %extracted_slice_5 = tensor.extract_slice %arg5[%arg3, %arg4] [64, 64] [1, 1] : tensor<1024x1024xi32> to tensor<64x64xi32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%unpack, %extracted_slice_4 : tensor<64x64xi32>, tensor<64x64xi32>) outs(%extracted_slice_5 : tensor<64x64xi32>) {
    ^bb0(%in: i32, %in_6: i32, %out: i32):
      %9 = arith.addi %in, %in_6 : i32
      linalg.yield %9 : i32
    } -> tensor<64x64xi32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %8 into %arg5[%arg3, %arg4] [64, 64] [1, 1] : tensor<64x64xi32> into tensor<1024x1024xi32>
    }
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  return %1 : tensor<1024x1024xi32>
}

// CHECK-LABEL: matmul_elementwise_1024x1024x512_i8xi8xi32
//       CHECK: %[[PACK_0:.*]] = tensor.pack {{.*}} : tensor<64x512xi8> -> tensor<1x16x64x32xi8>
//       CHECK: %[[PACK_1:.*]] = tensor.pack {{.*}} : tensor<512x64xi8> -> tensor<16x1x32x64xi8>
//       CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1x1x64x64xi32>
//       CHECK: %[[FILL:.*]] = linalg.fill {{.*}} -> tensor<1x1x64x64xi32>
//       CHECK: %[[MATMUL:.*]] = linalg.generic {{.*}} ins(%[[PACK_0]], %[[PACK_1]] : tensor<1x16x64x32xi8>, tensor<16x1x32x64xi8>) outs(%[[FILL]] : tensor<1x1x64x64xi32>)
//   CHECK-NOT: tensor.unpack
//       CHECK: %[[PACK_2:.*]] = tensor.pack {{.*}} : tensor<64x64xi32> -> tensor<1x1x64x64xi32>
//       CHECK: %[[ELEMENT:.*]] = linalg.generic {{.*}} ins(%[[MATMUL]], %[[PACK_2]] : tensor<1x1x64x64xi32>, tensor<1x1x64x64xi32>) outs(%[[EMPTY]] : tensor<1x1x64x64xi32>)
//       CHECK: %[[UNPACK:.*]] = tensor.unpack %[[ELEMENT:.*]] {{.*}} :  tensor<1x1x64x64xi32> -> tensor<64x64xi32>
