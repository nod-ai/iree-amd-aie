// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-bufferize-to-allocation{memory-space=2 bufferize-operand=linalg-input-output}))' --split-input-file %s | FileCheck %s --check-prefix=LINALG-INPUT-OUTPUT
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-bufferize-to-allocation{memory-space=2 bufferize-operand=linalg-input}))' --split-input-file %s | FileCheck %s --check-prefix=LINALG-INPUT
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-bufferize-to-allocation{memory-space=2 bufferize-operand=linalg-output}))' --split-input-file %s | FileCheck %s --check-prefix=LINALG-OUTPUT
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-bufferize-to-allocation{memory-space=1 bufferize-operand=pack-or-copy-input input-depth=2}))' --split-input-file %s | FileCheck %s --check-prefix=PACK-INPUT
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-bufferize-to-allocation{memory-space=2 bufferize-elementwise=true bufferize-operand=linalg-input}))' --split-input-file %s | FileCheck %s --check-prefix=ELEMENTWISE-INPUT
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-bufferize-to-allocation{memory-space=2 bufferize-elementwise=true bufferize-operand=linalg-input-output}))' --split-input-file %s | FileCheck %s --check-prefix=ELEMENTWISE-INPUT-OUTPUT

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d3, d5, d6, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d5, d4, d7, d8)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d3, d4, d6, d7)>
func.func @matmul_static(%arg0 : tensor<1024x2048xi32>, %arg1 : tensor<2048x512xi32>) -> tensor<1024x512xi32>
{
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %5 = tensor.empty() : tensor<1024x512xi32>
    %6 = tensor.empty() : tensor<16x32x64x64xi32>
    %pack = tensor.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %6 : tensor<1024x2048xi32> -> tensor<16x32x64x64xi32>
    %7 = tensor.empty() : tensor<32x8x64x64xi32>
    %pack_0 = tensor.pack %arg1 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %7 : tensor<2048x512xi32> -> tensor<32x8x64x64xi32>
    %8 = tensor.empty() : tensor<16x8x64x64xi32>
    %9 = tensor.empty() : tensor<16x32x16x8x4x8xi32>
    %pack_1 = tensor.pack %pack inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %9 : tensor<16x32x64x64xi32> -> tensor<16x32x16x8x4x8xi32>
    %10 = tensor.empty() : tensor<32x8x8x8x8x8xi32>
    %pack_2 = tensor.pack %pack_0 inner_dims_pos = [3, 2] inner_tiles = [8, 8] into %10 : tensor<32x8x64x64xi32> -> tensor<32x8x8x8x8x8xi32>
    %11 = tensor.empty() : tensor<16x8x16x8x4x8xi32>
    %12 = linalg.fill ins(%c0_i32 : i32) outs(%11 : tensor<16x8x16x8x4x8xi32>) -> tensor<16x8x16x8x4x8xi32>
    %13 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack_1, %pack_2 : tensor<16x32x16x8x4x8xi32>, tensor<32x8x8x8x8x8xi32>) outs(%12 : tensor<16x8x16x8x4x8xi32>) {
    ^bb0(%in: i32, %in_4: i32, %out: i32):
      %14 = arith.muli %in, %in_4 : i32
      %15 = arith.addi %out, %14 : i32
      linalg.yield %15 : i32
    } -> tensor<16x8x16x8x4x8xi32>
    %unpack = tensor.unpack %13 inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %8 : tensor<16x8x16x8x4x8xi32> -> tensor<16x8x64x64xi32>
    %unpack_3 = tensor.unpack %unpack inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %5 : tensor<16x8x64x64xi32> -> tensor<1024x512xi32>
    return %unpack_3 : tensor<1024x512xi32>
}

// LINALG-INPUT-OUTPUT-NOT:  memref.alloc
// LINALG-INPUT-OUTPUT:      tensor.pack
// LINALG-INPUT-OUTPUT-NOT:  memref.alloc
// LINALG-INPUT-OUTPUT:      tensor.pack
// LINALG-INPUT-OUTPUT:      memref.alloc() : memref<16x32x16x8x4x8xi32, 2 : i32>
// LINALG-INPUT-OUTPUT:      bufferization.to_tensor
// LINALG-INPUT-OUTPUT:      tensor.pack
// LINALG-INPUT-OUTPUT:      memref.alloc() : memref<32x8x8x8x8x8xi32, 2 : i32>
// LINALG-INPUT-OUTPUT:      bufferization.to_tensor
// LINALG-INPUT-OUTPUT:      tensor.pack
// LINALG-INPUT-OUTPUT:      memref.alloc() : memref<16x8x16x8x4x8xi32, 2 : i32>
// LINALG-INPUT-OUTPUT:      bufferization.to_tensor
// LINALG-INPUT-OUTPUT:      linalg.fill
// LINALG-INPUT-OUTPUT:      linalg.generic

// LINALG-INPUT-NOT:  memref.alloc
// LINALG-INPUT:      tensor.pack
// LINALG-INPUT-NOT:  memref.alloc
// LINALG-INPUT:      tensor.pack
// LINALG-INPUT:      memref.alloc() : memref<16x32x16x8x4x8xi32, 2 : i32>
// LINALG-INPUT:      bufferization.to_tensor
// LINALG-INPUT:      tensor.pack
// LINALG-INPUT:      memref.alloc() : memref<32x8x8x8x8x8xi32, 2 : i32>
// LINALG-INPUT:      bufferization.to_tensor
// LINALG-INPUT:      tensor.pack
// LINALG-INPUT-NOT:  memref.alloc
// LINALG-INPUT:      linalg.fill
// LINALG-INPUT:      linalg.generic

// LINALG-OUTPUT-NOT:  memref.alloc
// LINALG-OUTPUT:      tensor.pack
// LINALG-OUTPUT-NOT:  memref.alloc
// LINALG-OUTPUT:      tensor.pack
// LINALG-OUTPUT-NOT:  memref.alloc
// LINALG-OUTPUT:      tensor.pack
// LINALG-OUTPUT-NOT:  memref.alloc
// LINALG-OUTPUT:      tensor.pack
// LINALG-OUTPUT:      memref.alloc() : memref<16x8x16x8x4x8xi32, 2 : i32>
// LINALG-OUTPUT:      bufferization.to_tensor
// LINALG-OUTPUT:      linalg.fill
// LINALG-OUTPUT:      linalg.generic

// PACK-INPUT:      memref.alloc() : memref<16x32x64x64xi32, 1 : i32>
// PACK-INPUT:      bufferization.to_tensor
// PACK-INPUT:      tensor.pack
// PACK-INPUT:      memref.alloc() : memref<32x8x64x64xi32, 1 : i32>
// PACK-INPUT:      bufferization.to_tensor
// PACK-INPUT:      tensor.pack
// PACK-INPUT-NOT:  memref.alloc
// PACK-INPUT:      tensor.pack
// PACK-INPUT-NOT:  memref.alloc
// PACK-INPUT:      tensor.pack
// PACK-INPUT-NOT:  memref.alloc
// PACK-INPUT:      linalg.fill
// PACK-INPUT:      linalg.generic

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
func.func @matmul_elementwise(%arg0: tensor<1024x512xi8>, %arg1: tensor<512x1024xi8>, %arg2: tensor<1024x1024xi32>) -> tensor<1024x1024xi32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<1024x1024xi32>
  %1 = scf.forall (%arg3, %arg4) = (0, 0) to (1024, 1024) step (64, 64) shared_outs(%arg5 = %0) -> (tensor<1024x1024xi32>) {
    %extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [64, 512] [1, 1] : tensor<1024x512xi8> to tensor<64x512xi8>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg4] [512, 64] [1, 1] : tensor<512x1024xi8> to tensor<512x64xi8>
    %extracted_slice_1 = tensor.extract_slice %0[%arg3, %arg4] [64, 64] [1, 1] : tensor<1024x1024xi32> to tensor<64x64xi32>
    %2 = tensor.empty() : tensor<1x16x64x32xi8>
    %pack = tensor.pack %extracted_slice inner_dims_pos = [0, 1] inner_tiles = [64, 32] into %2 : tensor<64x512xi8> -> tensor<1x16x64x32xi8>
    %3 = tensor.empty() : tensor<16x1x32x64xi8>
    %pack_2 = tensor.pack %extracted_slice_0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 64] into %3 : tensor<512x64xi8> -> tensor<16x1x32x64xi8>
    %4 = tensor.empty() : tensor<1x1x64x64xi32>
    %5 = tensor.empty() : tensor<1x16x4x16x4x8xi8>
    %pack_3 = tensor.pack %pack outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %5 : tensor<1x16x64x32xi8> -> tensor<1x16x4x16x4x8xi8>
    %6 = tensor.empty() : tensor<16x1x8x4x8x8xi8>
    %pack_4 = tensor.pack %pack_2 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 8] into %6 : tensor<16x1x32x64xi8> -> tensor<16x1x8x4x8x8xi8>
    %7 = tensor.empty() : tensor<1x1x8x16x4x8xi32>
    %8 = linalg.fill ins(%c0_i32 : i32) outs(%7 : tensor<1x1x8x16x4x8xi32>) -> tensor<1x1x8x16x4x8xi32>
    %9 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack_3, %pack_4 : tensor<1x16x4x16x4x8xi8>, tensor<16x1x8x4x8x8xi8>) outs(%8 : tensor<1x1x8x16x4x8xi32>) {
    ^bb0(%in: i8, %in_12: i8, %out: i32):
      %11 = arith.extsi %in : i8 to i32
      %12 = arith.extsi %in_12 : i8 to i32
      %13 = arith.muli %11, %12 : i32
      %14 = arith.addi %out, %13 : i32
      linalg.yield %14 : i32
    } -> tensor<1x1x8x16x4x8xi32>
    %extracted_slice_5 = tensor.extract_slice %arg2[%arg3, %arg4] [64, 64] [1, 1] : tensor<1024x1024xi32> to tensor<64x64xi32>
    %extracted_slice_6 = tensor.extract_slice %arg5[%arg3, %arg4] [64, 64] [1, 1] : tensor<1024x1024xi32> to tensor<64x64xi32>
    %pack_7 = tensor.pack %extracted_slice_6 inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %4 : tensor<64x64xi32> -> tensor<1x1x64x64xi32>
    %pack_8 = tensor.pack %extracted_slice_5 inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %4 : tensor<64x64xi32> -> tensor<1x1x64x64xi32>
    %pack_9 = tensor.pack %pack_7 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %7 : tensor<1x1x64x64xi32> -> tensor<1x1x8x16x4x8xi32>
    %pack_10 = tensor.pack %pack_8 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %7 : tensor<1x1x64x64xi32> -> tensor<1x1x8x16x4x8xi32>
    %10 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%9, %pack_10 : tensor<1x1x8x16x4x8xi32>, tensor<1x1x8x16x4x8xi32>) outs(%pack_9 : tensor<1x1x8x16x4x8xi32>) {
    ^bb0(%in: i32, %in_12: i32, %out: i32):
      %11 = arith.addi %in, %in_12 : i32
      linalg.yield %11 : i32
    } -> tensor<1x1x8x16x4x8xi32>
    %unpack = tensor.unpack %10 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %4 : tensor<1x1x8x16x4x8xi32> -> tensor<1x1x64x64xi32>
    %unpack_11 = tensor.unpack %unpack inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %extracted_slice_1 : tensor<1x1x64x64xi32> -> tensor<64x64xi32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %unpack_11 into %arg5[%arg3, %arg4] [64, 64] [1, 1] : tensor<64x64xi32> into tensor<1024x1024xi32>
    }
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  return %1 : tensor<1024x1024xi32>
}

// ELEMENTWISE-INPUT-COUNT-4: tensor.pack
// ELEMENTWISE-INPUT:         linalg.fill
// ELEMENTWISE-INPUT:         linalg.generic
// ELEMENTWISE-INPUT-NOT:     memref.alloc
// ELEMENTWISE-INPUT:         tensor.pack
// ELEMENTWISE-INPUT-NOT:     memref.alloc
// ELEMENTWISE-INPUT:         tensor.pack
// ELEMENTWISE-INPUT-NOT:     memref.alloc
// ELEMENTWISE-INPUT:         tensor.pack
// ELEMENTWISE-INPUT:         memref.alloc() : memref<1x1x8x16x4x8xi32, 2 : i32>
// ELEMENTWISE-INPUT:         bufferization.to_tensor
// ELEMENTWISE-INPUT:         tensor.pack
// ELEMENTWISE-INPUT:         linalg.generic

// ELEMENTWISE-INPUT-OUTPUT-COUNT-4:  tensor.pack
// ELEMENTWISE-INPUT-OUTPUT:          linalg.fill
// ELEMENTWISE-INPUT-OUTPUT:          linalg.generic
// ELEMENTWISE-INPUT-OUTPUT-NOT:      memref.alloc
// ELEMENTWISE-INPUT-OUTPUT:          tensor.pack
// ELEMENTWISE-INPUT-OUTPUT-NOT:      memref.alloc
// ELEMENTWISE-INPUT-OUTPUT:          tensor.pack
// ELEMENTWISE-INPUT-OUTPUT:          memref.alloc() : memref<1x1x8x16x4x8xi32, 2 : i32>
// ELEMENTWISE-INPUT-OUTPUT:          bufferization.to_tensor
// ELEMENTWISE-INPUT-OUTPUT:          tensor.pack
// ELEMENTWISE-INPUT-OUTPUT:          memref.alloc() : memref<1x1x8x16x4x8xi32, 2 : i32>
// ELEMENTWISE-INPUT-OUTPUT:          bufferization.to_tensor
// ELEMENTWISE-INPUT-OUTPUT:          tensor.pack
// ELEMENTWISE-INPUT-OUTPUT:          linalg.generic
