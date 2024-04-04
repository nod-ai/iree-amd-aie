// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-bufferize-to-allocation{memory-space=2 bufferize-operand=input-output}))' --split-input-file %s | FileCheck %s --check-prefix=INPUT-OUTPUT
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-bufferize-to-allocation{memory-space=2 bufferize-operand=input}))' --split-input-file %s | FileCheck %s --check-prefix=INPUT
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-bufferize-to-allocation{memory-space=2 bufferize-operand=output}))' --split-input-file %s | FileCheck %s --check-prefix=OUTPUT
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-bufferize-to-allocation{memory-space=1 bufferize-operand=def-input}))' --split-input-file %s | FileCheck %s --check-prefix=DEF-INPUT

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

// INPUT-OUTPUT-NOT:  memref.alloc
//     INPUT-OUTPUT:  tensor.pack
// INPUT-OUTPUT-NOT:  memref.alloc
//     INPUT-OUTPUT:  tensor.pack
//     INPUT-OUTPUT:  memref.alloc() : memref<16x32x16x8x4x8xi32, 2 : i32>
//     INPUT-OUTPUT:  bufferization.to_tensor
//     INPUT-OUTPUT:  tensor.pack
//     INPUT-OUTPUT:  memref.alloc() : memref<32x8x8x8x8x8xi32, 2 : i32>
//     INPUT-OUTPUT:  bufferization.to_tensor
//     INPUT-OUTPUT:  tensor.pack
//     INPUT-OUTPUT:  memref.alloc() : memref<16x8x16x8x4x8xi32, 2 : i32>
//     INPUT-OUTPUT:  bufferization.to_tensor
//     INPUT-OUTPUT:  linalg.fill
//     INPUT-OUTPUT:  linalg.generic

// INPUT-NOT:  memref.alloc
//     INPUT:  tensor.pack
// INPUT-NOT:  memref.alloc
//     INPUT:  tensor.pack
//     INPUT:  memref.alloc() : memref<16x32x16x8x4x8xi32, 2 : i32>
//     INPUT:  bufferization.to_tensor
//     INPUT:  tensor.pack
//     INPUT:  memref.alloc() : memref<32x8x8x8x8x8xi32, 2 : i32>
//     INPUT:  bufferization.to_tensor
//     INPUT:  tensor.pack
// INPUT-NOT:  memref.alloc
//     INPUT:  linalg.fill
//     INPUT:  linalg.generic

// OUTPUT-NOT:  memref.alloc
//     OUTPUT:  tensor.pack
// OUTPUT-NOT:  memref.alloc
//     OUTPUT:  tensor.pack
// OUTPUT-NOT:  memref.alloc
//     OUTPUT:  tensor.pack
// OUTPUT-NOT:  memref.alloc
//     OUTPUT:  tensor.pack
//     OUTPUT:  memref.alloc() : memref<16x8x16x8x4x8xi32, 2 : i32>
//     OUTPUT:  bufferization.to_tensor
//     OUTPUT:  linalg.fill
//     OUTPUT:  linalg.generic

//     DEF-INPUT:  memref.alloc() : memref<16x32x64x64xi32, 1 : i32>
//     DEF-INPUT:  bufferization.to_tensor
//     DEF-INPUT:  tensor.pack
//     DEF-INPUT:  memref.alloc() : memref<32x8x64x64xi32, 1 : i32>
//     DEF-INPUT:  bufferization.to_tensor
//     DEF-INPUT:  tensor.pack
// DEF-INPUT-NOT:  memref.alloc
//     DEF-INPUT:  tensor.pack
// DEF-INPUT-NOT:  memref.alloc
//     DEF-INPUT:  tensor.pack
// DEF-INPUT-NOT:  memref.alloc
//     DEF-INPUT:  linalg.fill
//     DEF-INPUT:  linalg.generic
