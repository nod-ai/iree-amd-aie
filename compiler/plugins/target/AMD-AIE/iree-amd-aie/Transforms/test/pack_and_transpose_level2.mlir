// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-pack-and-transpose{pack-level=1}))' --split-input-file %s | FileCheck %s

#map = affine_map<(d0) -> (d0 * 16)>
#map1 = affine_map<(d0) -> (d0 * 64)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d5, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
func.func @matmul_example_dispatch_0_matmul_16x256x256_i8xi8xi32(%arg0: tensor<16x256xi8>, %arg1 : tensor<256x256xi8>) -> tensor<16x256xi32> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %5 = tensor.empty() : tensor<16x256xi32>
  %6 = scf.forall (%iv0, %iv1) in (1, 4) shared_outs(%arg2 = %5) -> (tensor<16x256xi32>) {
    %7 = affine.apply #map(%iv0)
    %8 = affine.apply #map1(%iv1)
    %extracted_slice = tensor.extract_slice %arg0[%7, 0] [16, 256] [1, 1] : tensor<16x256xi8> to tensor<16x256xi8>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, %8] [256, 64] [1, 1] : tensor<256x256xi8> to tensor<256x64xi8>
    %extracted_slice_1 = tensor.extract_slice %arg2[%7, %8] [16, 64] [1, 1] : tensor<16x256xi32> to tensor<16x64xi32>
    %9 = tensor.empty() : tensor<1x4x16x64xi8>
    %pack = tensor.pack %extracted_slice inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %9 : tensor<16x256xi8> -> tensor<1x4x16x64xi8>
    %10 = tensor.empty() : tensor<4x1x64x64xi8>
    %pack_2 = tensor.pack %extracted_slice_0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %10 : tensor<256x64xi8> -> tensor<4x1x64x64xi8>
    %11 = tensor.empty() : tensor<1x1x16x64xi32>
    %12 = scf.forall (%arg3, %arg4) in (1, 1) shared_outs(%arg5 = %11) -> (tensor<1x1x16x64xi32>) {
      %extracted_slice_3 = tensor.extract_slice %pack[%arg3, 0, 0, 0] [1, 4, 16, 64] [1, 1, 1, 1] : tensor<1x4x16x64xi8> to tensor<1x4x16x64xi8>
      %extracted_slice_4 = tensor.extract_slice %pack_2[0, %arg4, 0, 0] [4, 1, 64, 64] [1, 1, 1, 1] : tensor<4x1x64x64xi8> to tensor<4x1x64x64xi8>
      %extracted_slice_5 = tensor.extract_slice %arg5[%arg3, %arg4, 0, 0] [1, 1, 16, 64] [1, 1, 1, 1] : tensor<1x1x16x64xi32> to tensor<1x1x16x64xi32>
      %13 = linalg.fill ins(%c0_i32 : i32) outs(%extracted_slice_5 : tensor<1x1x16x64xi32>) -> tensor<1x1x16x64xi32>
      // CHECK: tensor.pack %{{.*}} outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %{{.*}} : tensor<1x4x16x64xi8> -> tensor<1x4x8x4x4x8xi8>
      // CHECK: tensor.pack %{{.*}} outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 8] into %{{.*}} : tensor<4x1x64x64xi8> -> tensor<4x1x8x8x8x8xi8>
      // CHECK: tensor.pack %{{.*}} outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %{{.*}} : tensor<1x1x16x64xi32> -> tensor<1x1x8x4x4x8xi32>
      %14 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%extracted_slice_3, %extracted_slice_4 : tensor<1x4x16x64xi8>, tensor<4x1x64x64xi8>) outs(%13 : tensor<1x1x16x64xi32>) {
      ^bb0(%in: i8, %in_6: i8, %out: i32):
        %15 = arith.extsi %in : i8 to i32
        %16 = arith.extsi %in_6 : i8 to i32
        %17 = arith.muli %15, %16 : i32
        %18 = arith.addi %out, %17 : i32
        linalg.yield %18 : i32
      } -> tensor<1x1x16x64xi32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %14 into %arg5[%arg3, %arg4, 0, 0] [1, 1, 16, 64] [1, 1, 1, 1] : tensor<1x1x16x64xi32> into tensor<1x1x16x64xi32>
      }
    } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    %unpack = tensor.unpack %12 inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %extracted_slice_1 : tensor<1x1x16x64xi32> -> tensor<16x64xi32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %unpack into %arg2[%7, %8] [16, 64] [1, 1] : tensor<16x64xi32> into tensor<16x256xi32>
    }
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  return %6 : tensor<16x256xi32>
}
