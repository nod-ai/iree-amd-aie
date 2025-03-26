// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-amdaie-fuse-producer-into-loop))' %s | FileCheck %s --check-prefix=DEPTH-1
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-amdaie-fuse-producer-into-loop{use-scf-for=false}))' %s | FileCheck %s --check-prefix=FORALL-DEPTH-1
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-amdaie-fuse-producer-into-loop{fuse-depth=2}))' %s | FileCheck %s --check-prefix=DEPTH-2
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-amdaie-fuse-producer-into-loop{fuse-depth=2 use-scf-for=false}))' %s | FileCheck %s --check-prefix=FORALL-DEPTH-2

// -----

#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
func.func @fuse_pack_into_for(%arg0: tensor<1x1x32x512xi32>, %arg1: tensor<1x1x512x32xi32>) -> tensor<1x1x4x8x4x8xi32> {
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %15 = tensor.empty() : tensor<1x1x64x8x4x8xi32>
  %pack_8 = linalg.pack %arg0 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %15 : tensor<1x1x32x512xi32> -> tensor<1x1x64x8x4x8xi32>
  %16 = tensor.empty() : tensor<1x1x4x64x8x8xi32>
  %pack_9 = linalg.pack %arg1 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 8] into %16 : tensor<1x1x512x32xi32> -> tensor<1x1x4x64x8x8xi32>
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

// DEPTH-1:  @fuse_pack_into_for
// DEPTH-1:  scf.for
// DEPTH-1:  {
// DEPTH-1:     tensor.extract_slice %{{.*}} : tensor<1x1x32x512xi32> to tensor<1x1x32x32xi32>
// DEPTH-1:     tensor.extract_slice %{{.*}} : tensor<1x1x64x8x4x8xi32> to tensor<1x1x4x8x4x8xi32>
// DEPTH-1:     linalg.pack %{{.*}} outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %{{.*}} : tensor<1x1x32x32xi32> -> tensor<1x1x4x8x4x8xi32>
// DEPTH-1:     tensor.extract_slice %{{.*}} : tensor<1x1x512x32xi32> to tensor<1x1x32x32xi32>
// DEPTH-1:     tensor.extract_slice %{{.*}} : tensor<1x1x4x64x8x8xi32> to tensor<1x1x4x4x8x8xi32>
// DEPTH-1:     linalg.pack %{{.*}} outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 8] into %{{.*}} : tensor<1x1x32x32xi32> -> tensor<1x1x4x4x8x8xi32>
// DEPTH-1:     linalg.generic
// DEPTH-1:  }

// -----

// Test to fuse multilevel pack ops into for loop.

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
func.func @fuse_multilevel_pack_into_for(%arg0: tensor<2048x2048xi32>, %arg1: tensor<2048x2048xi32>) -> tensor<2048x2048xi32> {
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<2048x2048xi32>
  %1 = scf.forall (%arg2, %arg3) = (0, 0) to (2048, 2048) step (64, 64) shared_outs(%arg4 = %0) -> (tensor<2048x2048xi32>) {
    %extracted_slice = tensor.extract_slice %arg0[%arg2, 0] [64, 2048] [1, 1] : tensor<2048x2048xi32> to tensor<64x2048xi32>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg3] [2048, 64] [1, 1] : tensor<2048x2048xi32> to tensor<2048x64xi32>
    %extracted_slice_1 = tensor.extract_slice %arg4[%arg2, %arg3] [64, 64] [1, 1] : tensor<2048x2048xi32> to tensor<64x64xi32>
    %2 = tensor.empty() : tensor<1x64x64x32xi32>
    %pack = linalg.pack %extracted_slice inner_dims_pos = [0, 1] inner_tiles = [64, 32] into %2 : tensor<64x2048xi32> -> tensor<1x64x64x32xi32>
    %3 = tensor.empty() : tensor<64x1x32x64xi32>
    %pack_2 = linalg.pack %extracted_slice_0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 64] into %3 : tensor<2048x64xi32> -> tensor<64x1x32x64xi32>
    %alloc = memref.alloc() : memref<1x1x64x64xi32, 1 : i32>
    %4 = bufferization.to_tensor %alloc restrict writable : memref<1x1x64x64xi32, 1 : i32> to tensor<1x1x64x64xi32>
    %5 = tensor.empty() : tensor<1x64x4x16x4x8xi32>
    %pack_3 = linalg.pack %pack outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %5 : tensor<1x64x64x32xi32> -> tensor<1x64x4x16x4x8xi32>
    %6 = tensor.empty() : tensor<64x1x16x4x8x4xi32>
    %pack_4 = linalg.pack %pack_2 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 4] into %6 : tensor<64x1x32x64xi32> -> tensor<64x1x16x4x8x4xi32>
    %alloc_5 = memref.alloc() : memref<1x1x16x16x4x4xi32, 2 : i32>
    %7 = bufferization.to_tensor %alloc_5 restrict writable : memref<1x1x16x16x4x4xi32, 2 : i32> to tensor<1x1x16x16x4x4xi32>
    %8 = linalg.fill ins(%c0_i32 : i32) outs(%7 : tensor<1x1x16x16x4x4xi32>) -> tensor<1x1x16x16x4x4xi32>
    %9 = scf.for %arg5 = %c0 to %c64 step %c1 iter_args(%arg6 = %8) -> (tensor<1x1x16x16x4x4xi32>) {
      %extracted_slice_7 = tensor.extract_slice %pack_3[0, %arg5, 0, 0, 0, 0] [1, 1, 4, 16, 4, 8] [1, 1, 1, 1, 1, 1] : tensor<1x64x4x16x4x8xi32> to tensor<1x1x4x16x4x8xi32>
      %extracted_slice_8 = tensor.extract_slice %pack_4[%arg5, 0, 0, 0, 0, 0] [1, 1, 16, 4, 8, 4] [1, 1, 1, 1, 1, 1] : tensor<64x1x16x4x8x4xi32> to tensor<1x1x16x4x8x4xi32>
      %10 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%extracted_slice_7, %extracted_slice_8 : tensor<1x1x4x16x4x8xi32>, tensor<1x1x16x4x8x4xi32>) outs(%arg6 : tensor<1x1x16x16x4x4xi32>) {
      ^bb0(%in: i32, %in_9: i32, %out: i32):
        %11 = arith.muli %in, %in_9 : i32
        %12 = arith.addi %out, %11 : i32
        linalg.yield %12 : i32
      } -> tensor<1x1x16x16x4x4xi32>
      scf.yield %10 : tensor<1x1x16x16x4x4xi32>
    }
    %unpack = linalg.unpack %9 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 4] into %4 : tensor<1x1x16x16x4x4xi32> -> tensor<1x1x64x64xi32>
    %unpack_6 = linalg.unpack %unpack inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %extracted_slice_1 : tensor<1x1x64x64xi32> -> tensor<64x64xi32>
    memref.dealloc %alloc : memref<1x1x64x64xi32, 1 : i32>
    memref.dealloc %alloc_5 : memref<1x1x16x16x4x4xi32, 2 : i32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %unpack_6 into %arg4[%arg2, %arg3] [64, 64] [1, 1] : tensor<64x64xi32> into tensor<2048x2048xi32>
    }
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  return %1 : tensor<2048x2048xi32>
}

// DEPTH-1:  @fuse_multilevel_pack_into_for
// DEPTH-1:  scf.for
// DEPTH-1:  {
// DEPTH-1:        %[[PACK_1_SOURCE:.*]] = tensor.extract_slice %{{.*}} : tensor<1x64x64x32xi32> to tensor<1x1x64x32xi32>
// DEPTH-1:        %[[PACK_1_DEST:.*]] = tensor.extract_slice %{{.*}} : tensor<1x64x4x16x4x8xi32> to tensor<1x1x4x16x4x8xi32>
// DEPTH-1:        %[[PACK_1:.*]] = linalg.pack %[[PACK_1_SOURCE]] {{.*}} into %[[PACK_1_DEST]] : tensor<1x1x64x32xi32> -> tensor<1x1x4x16x4x8xi32>
// DEPTH-1:        %[[PACK_2_SOURCE:.*]] = tensor.extract_slice %{{.*}} : tensor<64x1x32x64xi32> to tensor<1x1x32x64xi32>
// DEPTH-1:        %[[PACK_2_DEST:.*]] = tensor.extract_slice %{{.*}} : tensor<64x1x16x4x8x4xi32> to tensor<1x1x16x4x8x4xi32>
// DEPTH-1:        %[[PACK_2:.*]] = linalg.pack %[[PACK_2_SOURCE]] {{.*}} into %[[PACK_2_DEST]] : tensor<1x1x32x64xi32> -> tensor<1x1x16x4x8x4xi32>
// DEPTH-1:        linalg.generic {{.*}} ins(%[[PACK_1]], %[[PACK_2]] :
// DEPTH-1:  }

// DEPTH-2:  @fuse_multilevel_pack_into_for
// DEPTH-2:  scf.for
// DEPTH-2:  {
// DEPTH-2:        %[[PACK_1_SOURCE:.*]] = tensor.extract_slice %{{.*}} : tensor<64x2048xi32> to tensor<64x32xi32>
// DEPTH-2:        %[[PACK_1_DEST:.*]] = tensor.extract_slice %{{.*}} : tensor<1x64x64x32xi32> to tensor<1x1x64x32xi32>
// DEPTH-2:        %[[PACK_1_DEPTH_2:.*]] = linalg.pack %[[PACK_1_SOURCE]] {{.*}} into %[[PACK_1_DEST]] : tensor<64x32xi32> -> tensor<1x1x64x32xi32>
// DEPTH-2:        %[[PACK_1_DEST_2:.*]] = tensor.extract_slice %{{.*}} : tensor<1x64x4x16x4x8xi32> to tensor<1x1x4x16x4x8xi32>
// DEPTH-2:        %[[PACK_1_DEPTH_1:.*]] = linalg.pack %[[PACK_1_DEPTH_2]] {{.*}} into %[[PACK_1_DEST_2]] : tensor<1x1x64x32xi32> -> tensor<1x1x4x16x4x8xi32>
// DEPTH-2:        %[[PACK_2_SOURCE:.*]] = tensor.extract_slice %{{.*}} : tensor<2048x64xi32> to tensor<32x64xi32>
// DEPTH-2:        %[[PACK_2_DEST:.*]] = tensor.extract_slice %{{.*}} : tensor<64x1x32x64xi32> to tensor<1x1x32x64xi32>
// DEPTH-2:        %[[PACK_2_DEPTH_2:.*]] = linalg.pack %[[PACK_2_SOURCE]] {{.*}} into %[[PACK_2_DEST]] : tensor<32x64xi32> -> tensor<1x1x32x64xi32>
// DEPTH-2:        %[[PACK_2_DEST_2:.*]] = tensor.extract_slice %{{.*}} : tensor<64x1x16x4x8x4xi32> to tensor<1x1x16x4x8x4xi32>
// DEPTH-2:        %[[PACK_2_DEPTH_1:.*]] = linalg.pack %[[PACK_2_DEPTH_2]] {{.*}} into %[[PACK_2_DEST_2]] : tensor<1x1x32x64xi32> -> tensor<1x1x16x4x8x4xi32>
// DEPTH-2:        linalg.generic {{.*}} ins(%[[PACK_1_DEPTH_1]], %[[PACK_2_DEPTH_1]] :
// DEPTH-2:  }

// -----

// Test to fuse multilevel pack ops into forall loop.

#map = affine_map<(d0) -> (d0 * 32)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
func.func @fuse_multilevel_pack_into_forall(%arg0: tensor<2048x2048xi32>, %arg1: tensor<2048x2048xi32>) -> tensor<2048x2048xi32> {
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<2048x2048xi32>
  %1 = scf.forall (%arg2, %arg3) = (0, 0) to (2048, 2048) step (64, 64) shared_outs(%arg4 = %0) -> (tensor<2048x2048xi32>) {
    %extracted_slice = tensor.extract_slice %arg0[%arg2, 0] [64, 2048] [1, 1] : tensor<2048x2048xi32> to tensor<64x2048xi32>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg3] [2048, 64] [1, 1] : tensor<2048x2048xi32> to tensor<2048x64xi32>
    %extracted_slice_1 = tensor.extract_slice %arg4[%arg2, %arg3] [64, 64] [1, 1] : tensor<2048x2048xi32> to tensor<64x64xi32>
    %2 = tensor.empty() : tensor<1x64x64x32xi32>
    %3 = tensor.empty() : tensor<64x1x32x64xi32>
    %alloc = memref.alloc() : memref<1x1x64x64xi32, 1 : i32>
    %4 = bufferization.to_tensor %alloc restrict writable : memref<1x1x64x64xi32, 1 : i32> to tensor<1x1x64x64xi32>
    %5 = tensor.empty() : tensor<1x64x4x16x4x8xi32>
    %6 = tensor.empty() : tensor<64x1x16x4x8x4xi32>
    %alloc_2 = memref.alloc() : memref<1x1x16x16x4x4xi32, 2 : i32>
    %7 = bufferization.to_tensor %alloc_2 restrict writable : memref<1x1x16x16x4x4xi32, 2 : i32> to tensor<1x1x16x16x4x4xi32>
    %8 = linalg.fill ins(%c0_i32 : i32) outs(%7 : tensor<1x1x16x16x4x4xi32>) -> tensor<1x1x16x16x4x4xi32>
    %9 = scf.for %arg5 = %c0 to %c64 step %c1 iter_args(%arg6 = %8) -> (tensor<1x1x16x16x4x4xi32>) {
      %10 = affine.apply #map(%arg5)
      %extracted_slice_4 = tensor.extract_slice %extracted_slice[0, %10] [64, 32] [1, 1] : tensor<64x2048xi32> to tensor<64x32xi32>
      %extracted_slice_5 = tensor.extract_slice %2[0, %arg5, 0, 0] [1, 1, 64, 32] [1, 1, 1, 1] : tensor<1x64x64x32xi32> to tensor<1x1x64x32xi32>
      %pack = linalg.pack %extracted_slice_4 inner_dims_pos = [0, 1] inner_tiles = [64, 32] into %extracted_slice_5 : tensor<64x32xi32> -> tensor<1x1x64x32xi32>
      %extracted_slice_6 = tensor.extract_slice %5[0, %arg5, 0, 0, 0, 0] [1, 1, 4, 16, 4, 8] [1, 1, 1, 1, 1, 1] : tensor<1x64x4x16x4x8xi32> to tensor<1x1x4x16x4x8xi32>
      %pack_7 = linalg.pack %pack outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %extracted_slice_6 : tensor<1x1x64x32xi32> -> tensor<1x1x4x16x4x8xi32>
      %extracted_slice_8 = tensor.extract_slice %extracted_slice_0[%10, 0] [32, 64] [1, 1] : tensor<2048x64xi32> to tensor<32x64xi32>
      %extracted_slice_9 = tensor.extract_slice %3[%arg5, 0, 0, 0] [1, 1, 32, 64] [1, 1, 1, 1] : tensor<64x1x32x64xi32> to tensor<1x1x32x64xi32>
      %pack_10 = linalg.pack %extracted_slice_8 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 64] into %extracted_slice_9 : tensor<32x64xi32> -> tensor<1x1x32x64xi32>
      %extracted_slice_11 = tensor.extract_slice %6[%arg5, 0, 0, 0, 0, 0] [1, 1, 16, 4, 8, 4] [1, 1, 1, 1, 1, 1] : tensor<64x1x16x4x8x4xi32> to tensor<1x1x16x4x8x4xi32>
      %pack_12 = linalg.pack %pack_10 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 4] into %extracted_slice_11 : tensor<1x1x32x64xi32> -> tensor<1x1x16x4x8x4xi32>
      %11 = scf.forall (%arg7, %arg8) in (1, 1) shared_outs(%arg9 = %arg6) -> (tensor<1x1x16x16x4x4xi32>) {
        %extracted_slice_13 = tensor.extract_slice %pack_7[%arg7, 0, 0, 0, 0, 0] [1, 1, 4, 16, 4, 8] [1, 1, 1, 1, 1, 1] : tensor<1x1x4x16x4x8xi32> to tensor<1x1x4x16x4x8xi32>
        %extracted_slice_14 = tensor.extract_slice %pack_12[0, %arg8, 0, 0, 0, 0] [1, 1, 16, 4, 8, 4] [1, 1, 1, 1, 1, 1] : tensor<1x1x16x4x8x4xi32> to tensor<1x1x16x4x8x4xi32>
        %extracted_slice_15 = tensor.extract_slice %arg9[%arg7, %arg8, 0, 0, 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : tensor<1x1x16x16x4x4xi32> to tensor<1x1x16x16x4x4xi32>
        %12 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%extracted_slice_13, %extracted_slice_14 : tensor<1x1x4x16x4x8xi32>, tensor<1x1x16x4x8x4xi32>) outs(%extracted_slice_15 : tensor<1x1x16x16x4x4xi32>) {
        ^bb0(%in: i32, %in_16: i32, %out: i32):
          %13 = arith.muli %in, %in_16 : i32
          %14 = arith.addi %out, %13 : i32
          linalg.yield %14 : i32
        } -> tensor<1x1x16x16x4x4xi32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %12 into %arg9[%arg7, %arg8, 0, 0, 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : tensor<1x1x16x16x4x4xi32> into tensor<1x1x16x16x4x4xi32>
        }
      } {mapping = [#gpu.block<y>, #gpu.block<x>]}
      scf.yield %11 : tensor<1x1x16x16x4x4xi32>
    }
    %unpack = linalg.unpack %9 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 4] into %4 : tensor<1x1x16x16x4x4xi32> -> tensor<1x1x64x64xi32>
    %unpack_3 = linalg.unpack %unpack inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %extracted_slice_1 : tensor<1x1x64x64xi32> -> tensor<64x64xi32>
    memref.dealloc %alloc : memref<1x1x64x64xi32, 1 : i32>
    memref.dealloc %alloc_2 : memref<1x1x16x16x4x4xi32, 2 : i32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %unpack_3 into %arg4[%arg2, %arg3] [64, 64] [1, 1] : tensor<64x64xi32> into tensor<2048x2048xi32>
    }
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  return %1 : tensor<2048x2048xi32>
}

// FORALL-DEPTH-1:  @fuse_multilevel_pack_into_forall
// FORALL-DEPTH-1:  scf.for
// FORALL-DEPTH-1:  {
// FORALL-DEPTH-1:     scf.forall
// FORALL-DEPTH-1:     {
// FORALL-DEPTH-1:        %[[PACK_1_SOURCE:.*]] = tensor.extract_slice %{{.*}} : tensor<1x1x64x32xi32> to tensor<1x1x64x32xi32>
// FORALL-DEPTH-1:        %[[PACK_1_DEST:.*]] = tensor.extract_slice %{{.*}} : tensor<1x1x4x16x4x8xi32> to tensor<1x1x4x16x4x8xi32>
// FORALL-DEPTH-1:        %[[PACK_1:.*]] = linalg.pack %[[PACK_1_SOURCE]] {{.*}} into %[[PACK_1_DEST]] : tensor<1x1x64x32xi32> -> tensor<1x1x4x16x4x8xi32>
// FORALL-DEPTH-1:        %[[PACK_2_SOURCE:.*]] = tensor.extract_slice %{{.*}} : tensor<1x1x32x64xi32> to tensor<1x1x32x64xi32>
// FORALL-DEPTH-1:        %[[PACK_2_DEST:.*]] = tensor.extract_slice %{{.*}} : tensor<1x1x16x4x8x4xi32> to tensor<1x1x16x4x8x4xi32>
// FORALL-DEPTH-1:        %[[PACK_2:.*]] = linalg.pack %[[PACK_2_SOURCE]] {{.*}} into %[[PACK_2_DEST]] : tensor<1x1x32x64xi32> -> tensor<1x1x16x4x8x4xi32>
// FORALL-DEPTH-1:        linalg.generic {{.*}} ins(%[[PACK_1]], %[[PACK_2]] :
// FORALL-DEPTH-1:      }
// FORALL-DEPTH-1:  }

// FORALL-DEPTH-2:  @fuse_multilevel_pack_into_forall
// FORALL-DEPTH-2:  scf.for
// FORALL-DEPTH-2:  {
// FORALL-DEPTH-2:     scf.forall
// FORALL-DEPTH-2:     {
// FORALL-DEPTH-2:        %[[PACK_1_SOURCE:.*]] = tensor.extract_slice %{{.*}} : tensor<64x32xi32> to tensor<64x32xi32>
// FORALL-DEPTH-2:        %[[PACK_1_DEST:.*]] = tensor.extract_slice %{{.*}} : tensor<1x1x64x32xi32> to tensor<1x1x64x32xi32>
// FORALL-DEPTH-2:        %[[PACK_1_DEPTH_2:.*]] = linalg.pack %[[PACK_1_SOURCE]] {{.*}} into %[[PACK_1_DEST]] : tensor<64x32xi32> -> tensor<1x1x64x32xi32>
// FORALL-DEPTH-2:        %[[PACK_1_DEST_2:.*]] = tensor.extract_slice %{{.*}} : tensor<1x1x4x16x4x8xi32> to tensor<1x1x4x16x4x8xi32>
// FORALL-DEPTH-2:        %[[PACK_1_DEPTH_1:.*]] = linalg.pack %[[PACK_1_DEPTH_2]] {{.*}} into %[[PACK_1_DEST_2]] : tensor<1x1x64x32xi32> -> tensor<1x1x4x16x4x8xi32>
// FORALL-DEPTH-2:        %[[PACK_2_SOURCE:.*]] = tensor.extract_slice %{{.*}} : tensor<32x64xi32> to tensor<32x64xi32>
// FORALL-DEPTH-2:        %[[PACK_2_DEST:.*]] = tensor.extract_slice %{{.*}} : tensor<1x1x32x64xi32> to tensor<1x1x32x64xi32>
// FORALL-DEPTH-2:        %[[PACK_2_DEPTH_2:.*]] = linalg.pack %[[PACK_2_SOURCE]] {{.*}} into %[[PACK_2_DEST]] : tensor<32x64xi32> -> tensor<1x1x32x64xi32>
// FORALL-DEPTH-2:        %[[PACK_2_DEST_2:.*]] = tensor.extract_slice %{{.*}} : tensor<1x1x16x4x8x4xi32> to tensor<1x1x16x4x8x4xi32>
// FORALL-DEPTH-2:        %[[PACK_2_DEPTH_1:.*]] = linalg.pack %[[PACK_2_DEPTH_2]] {{.*}} into %[[PACK_2_DEST_2]] : tensor<1x1x32x64xi32> -> tensor<1x1x16x4x8x4xi32>
// FORALL-DEPTH-2:        linalg.generic {{.*}} ins(%[[PACK_1_DEPTH_1]], %[[PACK_2_DEPTH_1]] :
// FORALL-DEPTH-2:      }
// FORALL-DEPTH-2:  }

// -----

// Test to fuse multilevel copy ops into for loop.

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @fuse_multilevel_copy_into_for(%arg0: tensor<8x16xi32>, %arg1: tensor<16x8xi32>) -> tensor<8x8xi32> {
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<8x8xi32>
  %1 = bufferization.alloc_tensor() : tensor<8x16xi32>
  %2 = linalg.copy ins(%arg0 : tensor<8x16xi32>) outs(%1 : tensor<8x16xi32>) -> tensor<8x16xi32>
  %3 = bufferization.alloc_tensor() : tensor<16x8xi32>
  %4 = linalg.copy ins(%arg1 : tensor<16x8xi32>) outs(%3 : tensor<16x8xi32>) -> tensor<16x8xi32>
  %5 = bufferization.alloc_tensor() : tensor<8x8xi32>
  %6 = bufferization.alloc_tensor() : tensor<8x16xi32>
  %7 = linalg.copy ins(%2 : tensor<8x16xi32>) outs(%6 : tensor<8x16xi32>) -> tensor<8x16xi32>
  %8 = bufferization.alloc_tensor() : tensor<16x8xi32>
  %9 = linalg.copy ins(%4 : tensor<16x8xi32>) outs(%8 : tensor<16x8xi32>) -> tensor<16x8xi32>
  %10 = bufferization.alloc_tensor() : tensor<8x8xi32>
  %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<8x8xi32>) -> tensor<8x8xi32>
  %12 = scf.for %arg2 = %c0 to %c16 step %c4 iter_args(%arg3 = %11) -> (tensor<8x8xi32>) {
    %extracted_slice = tensor.extract_slice %7[0, %arg2] [8, 4] [1, 1] : tensor<8x16xi32> to tensor<8x4xi32>
    %extracted_slice_0 = tensor.extract_slice %9[%arg2, 0] [4, 8] [1, 1] : tensor<16x8xi32> to tensor<4x8xi32>
    %15 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_0 : tensor<8x4xi32>, tensor<4x8xi32>) outs(%11 : tensor<8x8xi32>) {
      ^bb0(%in: i32, %in_1: i32, %out: i32):
        %21 = arith.muli %in, %in_1 : i32
        %22 = arith.addi %out, %21 : i32
        linalg.yield %22 : i32
      } -> tensor<8x8xi32>
    scf.yield %15 : tensor<8x8xi32>
  }
  %13 = linalg.copy ins(%12 : tensor<8x8xi32>) outs(%5 : tensor<8x8xi32>) -> tensor<8x8xi32>
  %14 = linalg.copy ins(%13 : tensor<8x8xi32>) outs(%0 : tensor<8x8xi32>) -> tensor<8x8xi32>
  return %14 : tensor<8x8xi32>
}

// DEPTH-1:  @fuse_multilevel_copy_into_for
// DEPTH-1:  scf.for
// DEPTH-1:  {
// DEPTH-1:        %[[COPY_1_SOURCE:.*]] = tensor.extract_slice %{{.*}} : tensor<8x16xi32> to tensor<8x4xi32>
// DEPTH-1:        %[[COPY_1_DEST:.*]] = tensor.extract_slice %{{.*}} : tensor<8x16xi32> to tensor<8x4xi32>
// DEPTH-1:        %[[COPY_1:.*]] = linalg.copy ins(%[[COPY_1_SOURCE]] : tensor<8x4xi32>) outs(%[[COPY_1_DEST]] : tensor<8x4xi32>) -> tensor<8x4xi32>
// DEPTH-1:        %[[COPY_2_SOURCE:.*]] = tensor.extract_slice %{{.*}} : tensor<16x8xi32> to tensor<4x8xi32>
// DEPTH-1:        %[[COPY_2_DEST:.*]] = tensor.extract_slice %{{.*}} : tensor<16x8xi32> to tensor<4x8xi32>
// DEPTH-1:        %[[COPY_2:.*]] = linalg.copy ins(%[[COPY_2_SOURCE]] : tensor<4x8xi32>) outs(%[[COPY_2_DEST]] : tensor<4x8xi32>) -> tensor<4x8xi32>
// DEPTH-1:        linalg.generic {{.*}} ins(%[[COPY_1]], %[[COPY_2]] :
// DEPTH-1:  }

// DEPTH-2:  @fuse_multilevel_copy_into_for
// DEPTH-2:  scf.for
// DEPTH-2:  {
// DEPTH-2:        %[[COPY_1_SOURCE:.*]] = tensor.extract_slice %{{.*}} : tensor<8x16xi32> to tensor<8x4xi32>
// DEPTH-2:        %[[COPY_1_DEST:.*]] = tensor.extract_slice %{{.*}} : tensor<8x16xi32> to tensor<8x4xi32>
// DEPTH-2:        %[[COPY_1_DEPTH_2:.*]] = linalg.copy ins(%[[COPY_1_SOURCE]] : tensor<8x4xi32>) outs(%[[COPY_1_DEST]] : tensor<8x4xi32>) -> tensor<8x4xi32>
// DEPTH-2:         tensor.extract_slice %{{.*}} : tensor<8x16xi32> to tensor<8x4xi32>
// DEPTH-2:        %[[COPY_1_DEST_2:.*]] = tensor.extract_slice %{{.*}} : tensor<8x16xi32> to tensor<8x4xi32>
// DEPTH-2:        %[[COPY_1_DEPTH_1:.*]] = linalg.copy ins(%[[COPY_1_DEPTH_2]] : tensor<8x4xi32>) outs(%[[COPY_1_DEST_2]] : tensor<8x4xi32>) -> tensor<8x4xi32>
// DEPTH-2:        %[[COPY_2_SOURCE:.*]] = tensor.extract_slice %{{.*}} : tensor<16x8xi32> to tensor<4x8xi32>
// DEPTH-2:        %[[COPY_2_DEST:.*]] = tensor.extract_slice %{{.*}} : tensor<16x8xi32> to tensor<4x8xi32>
// DEPTH-2:        %[[COPY_2_DEPTH_2:.*]] = linalg.copy ins(%[[COPY_2_SOURCE]] : tensor<4x8xi32>) outs(%[[COPY_2_DEST]] : tensor<4x8xi32>) -> tensor<4x8xi32>
// DEPTH-2:         tensor.extract_slice %{{.*}} : tensor<16x8xi32> to tensor<4x8xi32>
// DEPTH-2:        %[[COPY_2_DEST_2:.*]] = tensor.extract_slice %{{.*}} : tensor<16x8xi32> to tensor<4x8xi32>
// DEPTH-2:        %[[COPY_2_DEPTH_1:.*]] = linalg.copy ins(%[[COPY_2_DEPTH_2]] : tensor<4x8xi32>) outs(%[[COPY_2_DEST_2]] : tensor<4x8xi32>) -> tensor<4x8xi32>
// DEPTH-2:        linalg.generic {{.*}} ins(%[[COPY_1_DEPTH_1]], %[[COPY_2_DEPTH_1]] :
// DEPTH-2:  }

// -----

// Test to fuse multilevel copy ops into forall loop.

#map = affine_map<(d0) -> (d0 * 8)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @fuse_multilevel_copy_into_forall(%arg0: tensor<8x16xi32>, %arg1: tensor<16x8xi32>) -> tensor<8x8xi32> {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<8x8xi32>
  %1 = bufferization.alloc_tensor() : tensor<8x16xi32>
  %2 = linalg.copy ins(%arg0 : tensor<8x16xi32>) outs(%1 : tensor<8x16xi32>) -> tensor<8x16xi32>
  %3 = bufferization.alloc_tensor() : tensor<16x8xi32>
  %4 = linalg.copy ins(%arg1 : tensor<16x8xi32>) outs(%3 : tensor<16x8xi32>) -> tensor<16x8xi32>
  %5 = bufferization.alloc_tensor() : tensor<8x8xi32>
  %6 = bufferization.alloc_tensor() : tensor<8x16xi32>
  %7 = linalg.copy ins(%2 : tensor<8x16xi32>) outs(%6 : tensor<8x16xi32>) -> tensor<8x16xi32>
  %8 = bufferization.alloc_tensor() : tensor<16x8xi32>
  %9 = linalg.copy ins(%4 : tensor<16x8xi32>) outs(%8 : tensor<16x8xi32>) -> tensor<16x8xi32>
  %10 = bufferization.alloc_tensor() : tensor<8x8xi32>
  %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<8x8xi32>) -> tensor<8x8xi32>
  %12 = scf.forall (%arg2, %arg3) in (1, 1) shared_outs(%arg4 = %11) -> (tensor<8x8xi32>) {
    %15 = affine.apply #map(%arg2)
    %16 = affine.apply #map(%arg3)
    %extracted_slice = tensor.extract_slice %7[%15, 0] [8, 16] [1, 1] : tensor<8x16xi32> to tensor<8x16xi32>
    %extracted_slice_0 = tensor.extract_slice %9[0, %16] [16, 8] [1, 1] : tensor<16x8xi32> to tensor<16x8xi32>
    %extracted_slice_1 = tensor.extract_slice %arg4[%15, %16] [8, 8] [1, 1] : tensor<8x8xi32> to tensor<8x8xi32>
    %17 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_0 : tensor<8x16xi32>, tensor<16x8xi32>) outs(%extracted_slice_1 : tensor<8x8xi32>) {
    ^bb0(%in: i32, %in_2: i32, %out: i32):
      %18 = arith.muli %in, %in_2 : i32
      %19 = arith.addi %out, %18 : i32
      linalg.yield %19 : i32
    } -> tensor<8x8xi32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %17 into %arg4[%15, %16] [8, 8] [1, 1] : tensor<8x8xi32> into tensor<8x8xi32>
    }
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  %13 = linalg.copy ins(%12 : tensor<8x8xi32>) outs(%5 : tensor<8x8xi32>) -> tensor<8x8xi32>
  %14 = linalg.copy ins(%13 : tensor<8x8xi32>) outs(%0 : tensor<8x8xi32>) -> tensor<8x8xi32>
  return %14 : tensor<8x8xi32>
}

// FORALL-DEPTH-1:  @fuse_multilevel_copy_into_forall
// FORALL-DEPTH-1:  scf.forall
// FORALL-DEPTH-1:  {
// FORALL-DEPTH-1:        %[[COPY_1_SOURCE:.*]] = tensor.extract_slice %{{.*}} : tensor<8x16xi32> to tensor<8x16xi32>
// FORALL-DEPTH-1:        %[[COPY_1_DEST:.*]] = tensor.extract_slice %{{.*}} : tensor<8x16xi32> to tensor<8x16xi32>
// FORALL-DEPTH-1:        %[[COPY_1:.*]] = linalg.copy ins(%[[COPY_1_SOURCE]] : tensor<8x16xi32>) outs(%[[COPY_1_DEST]] : tensor<8x16xi32>) -> tensor<8x16xi32>
// FORALL-DEPTH-1:        %[[COPY_2_SOURCE:.*]] = tensor.extract_slice %{{.*}} : tensor<16x8xi32> to tensor<16x8xi32>
// FORALL-DEPTH-1:        %[[COPY_2_DEST:.*]] = tensor.extract_slice %{{.*}} : tensor<16x8xi32> to tensor<16x8xi32>
// FORALL-DEPTH-1:        %[[COPY_2:.*]] = linalg.copy ins(%[[COPY_2_SOURCE]] : tensor<16x8xi32>) outs(%[[COPY_2_DEST]] : tensor<16x8xi32>) -> tensor<16x8xi32>
// FORALL-DEPTH-1:        linalg.generic {{.*}} ins(%[[COPY_1]], %[[COPY_2]] :
// FORALL-DEPTH-1:  }

// FORALL-DEPTH-2:  @fuse_multilevel_copy_into_forall
// FORALL-DEPTH-2:  scf.forall
// FORALL-DEPTH-2:  {
// FORALL-DEPTH-2:        %[[COPY_1_SOURCE:.*]] = tensor.extract_slice %{{.*}} : tensor<8x16xi32> to tensor<8x16xi32>
// FORALL-DEPTH-2:        %[[COPY_1_DEST:.*]] = tensor.extract_slice %{{.*}} : tensor<8x16xi32> to tensor<8x16xi32>
// FORALL-DEPTH-2:        %[[COPY_1_DEPTH_2:.*]] = linalg.copy ins(%[[COPY_1_SOURCE]] : tensor<8x16xi32>) outs(%[[COPY_1_DEST]] : tensor<8x16xi32>) -> tensor<8x16xi32>
// FORALL-DEPTH-2:         tensor.extract_slice %{{.*}} : tensor<8x16xi32> to tensor<8x16xi32>
// FORALL-DEPTH-2:        %[[COPY_1_DEST_2:.*]] = tensor.extract_slice %{{.*}} : tensor<8x16xi32> to tensor<8x16xi32>
// FORALL-DEPTH-2:        %[[COPY_1_DEPTH_1:.*]] = linalg.copy ins(%[[COPY_1_DEPTH_2]] : tensor<8x16xi32>) outs(%[[COPY_1_DEST_2]] : tensor<8x16xi32>) -> tensor<8x16xi32>
// FORALL-DEPTH-2:        %[[COPY_2_SOURCE:.*]] = tensor.extract_slice %{{.*}} : tensor<16x8xi32> to tensor<16x8xi32>
// FORALL-DEPTH-2:        %[[COPY_2_DEST:.*]] = tensor.extract_slice %{{.*}} : tensor<16x8xi32> to tensor<16x8xi32>
// FORALL-DEPTH-2:        %[[COPY_2_DEPTH_2:.*]] = linalg.copy ins(%[[COPY_2_SOURCE]] : tensor<16x8xi32>) outs(%[[COPY_2_DEST]] : tensor<16x8xi32>) -> tensor<16x8xi32>
// FORALL-DEPTH-2:         tensor.extract_slice %{{.*}} : tensor<16x8xi32> to tensor<16x8xi32>
// FORALL-DEPTH-2:        %[[COPY_2_DEST_2:.*]] = tensor.extract_slice %{{.*}} : tensor<16x8xi32> to tensor<16x8xi32>
// FORALL-DEPTH-2:        %[[COPY_2_DEPTH_1:.*]] = linalg.copy ins(%[[COPY_2_DEPTH_2]] : tensor<16x8xi32>) outs(%[[COPY_2_DEST_2]] : tensor<16x8xi32>) -> tensor<16x8xi32>
// FORALL-DEPTH-2:        linalg.generic {{.*}} ins(%[[COPY_1_DEPTH_1]], %[[COPY_2_DEPTH_1]] :
// FORALL-DEPTH-2:  }

// -----

// A test with a linalg.generic which has a pack result as an operand.

#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
func.func @pack_without_slice(%arg0: tensor<1x1x32x512xi32>, %arg1: tensor<1x1x32x32xi32>) -> tensor<1x1x4x8x4x8xi32> {
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %15 = tensor.empty() : tensor<1x1x64x8x4x8xi32>
  %pack_8 = linalg.pack %arg0 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %15 : tensor<1x1x32x512xi32> -> tensor<1x1x64x8x4x8xi32>
  %16 = tensor.empty() : tensor<1x1x4x4x8x8xi32>
  %pack_10 = linalg.pack %arg1 outer_dims_perm = [0, 1, 2, 3] inner_dims_pos = [2, 3] inner_tiles = [8, 8] into %16 : tensor<1x1x32x32xi32> -> tensor<1x1x4x4x8x8xi32>

  %17 = tensor.empty() : tensor<1x1x4x8x4x8xi32>
  %18 = linalg.fill ins(%c0_i32 : i32) outs(%17 : tensor<1x1x4x8x4x8xi32>) -> tensor<1x1x4x8x4x8xi32>
  %19 = scf.for %arg6 = %c0 to %c64 step %c4 iter_args(%arg7 = %18) -> (tensor<1x1x4x8x4x8xi32>) {
    %extracted_slice_12 = tensor.extract_slice %pack_8[0, 0, %arg6, 0, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : tensor<1x1x64x8x4x8xi32> to tensor<1x1x4x8x4x8xi32>
    %20 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%extracted_slice_12, %pack_10 : tensor<1x1x4x8x4x8xi32>, tensor<1x1x4x4x8x8xi32>) outs(%arg7 : tensor<1x1x4x8x4x8xi32>) {
    ^bb0(%in: i32, %in_14: i32, %out: i32):
      %21 = arith.muli %in, %in_14 : i32
      %22 = arith.addi %out, %21 : i32
      linalg.yield %22 : i32
    } -> tensor<1x1x4x8x4x8xi32>
    scf.yield %20 : tensor<1x1x4x8x4x8xi32>
  }
  return %19 : tensor<1x1x4x8x4x8xi32>
}

// DEPTH-1-LABEL: pack_without_slice
// DEPTH-1:       scf.for
// DEPTH-1-DAG:   %[[PACK_1:.*]] = linalg.pack %{{.*}} into %{{.*}} : tensor<1x1x32x32xi32> -> tensor<1x1x4x4x8x8xi32>
// DEPTH-1-DAG:   %[[PACK_2:.*]] = linalg.pack %{{.*}} into %{{.*}} : tensor<1x1x32x32xi32> -> tensor<1x1x4x8x4x8xi32>
// DEPTH-1:       linalg.generic
// DEPTH-1-SAME:  ins(%[[PACK_2]], %[[PACK_1]]

// -----

// A test with linalg.pack in a different block compared to the linalg.generic.

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
module {
  func.func @fuse_diff_block_pack_into_forall(%arg0: tensor<8x16xi32>, %arg1: tensor<16x8xi32>) -> tensor<4x4x2x2xi32> {
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = bufferization.alloc_tensor() : tensor<4x4x2x2xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<4x4x2x2xi32>) -> tensor<4x4x2x2xi32>
    %2 = tensor.empty() : tensor<4x1x2x4xi32>
    %3 = tensor.empty() : tensor<4x1x4x2xi32>
    %4 = scf.for %arg2 = %c0 to %c16 step %c4 iter_args(%arg3 = %1) -> (tensor<4x4x2x2xi32>) {
      %extracted_slice = tensor.extract_slice %arg0[0, %arg2] [8, 4] [1, 1] : tensor<8x16xi32> to tensor<8x4xi32>
      %pack = linalg.pack %extracted_slice outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [2, 4] into %2 : tensor<8x4xi32> -> tensor<4x1x2x4xi32>
      %extracted_slice_0 = tensor.extract_slice %arg1[%arg2, 0] [4, 8] [1, 1] : tensor<16x8xi32> to tensor<4x8xi32>
      %pack_1 = linalg.pack %extracted_slice_0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [4, 2] into %3 : tensor<4x8xi32> -> tensor<4x1x4x2xi32>
      %5 = scf.forall (%arg4, %arg5) in (1, 1) shared_outs(%arg6 = %arg3) -> (tensor<4x4x2x2xi32>) {
        %6 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_1 : tensor<4x1x2x4xi32>, tensor<4x1x4x2xi32>) outs(%arg6 : tensor<4x4x2x2xi32>) {
        ^bb0(%in: i32, %in_2: i32, %out: i32):
          %7 = arith.muli %in, %in_2 : i32
          %8 = arith.addi %out, %7 : i32
          linalg.yield %8 : i32
        } -> tensor<4x4x2x2xi32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %6 into %arg6[0, 0, 0, 0] [4, 4, 2, 2] [1, 1, 1, 1] : tensor<4x4x2x2xi32> into tensor<4x4x2x2xi32>
        }
      }
      scf.yield %5 : tensor<4x4x2x2xi32>
    }
    return %4 : tensor<4x4x2x2xi32>
  }
}

// FORALL-DEPTH-1-LABEL: @fuse_diff_block_pack_into_forall
// FORALL-DEPTH-1:          scf.for
// FORALL-DEPTH-1:              scf.forall
// FORALL-DEPTH-1:                  %[[PACK_0:.*]] = linalg.pack
// FORALL-DEPTH-1:                  %[[PACK_1:.*]] = linalg.pack
// FORALL-DEPTH-1:                  linalg.generic
// FORALL-DEPTH-1-SAME:                  ins(%[[PACK_1]], %[[PACK_0]] :
