// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-bufferize-to-allocation{memory-space=1 bufferize-operand=pack-or-copy-input input-depth=2}))' --split-input-file --verify-diagnostics %s | FileCheck %s

// Checks for packs with interleaved extract slices.
// CHECK-LABEL:  @matmul_tensor_extract_slice
// CHECK:        scf.forall
// CHECK:          %[[ALLOC_0:.+]] = memref.alloc() : memref<4x8x32x64xbf16, 1 : i32>
// CHECK:          %[[TO_TENSOR_0:.+]] = bufferization.to_tensor %[[ALLOC_0]]
// CHECK:          %[[PACK_0:.+]] = tensor.pack
// CHECK-SAME:     into %[[TO_TENSOR_0]]
// CHECK:          %[[ALLOC_1:.+]] = memref.alloc() : memref<4x8x64x32xbf16, 1 : i32>
// CHECK:          %[[TO_TENSOR_1:.+]] = bufferization.to_tensor %[[ALLOC_1]]
// CHECK:          %[[PACK_1:.+]] = tensor.pack
// CHECK-SAME:     into %[[TO_TENSOR_1]]
// CHECK:          scf.forall
// CHECK:            %[[SLICE_0:.+]] = tensor.extract_slice %[[PACK_0]]
// CHECK:            %[[SLICE_1:.+]] = tensor.extract_slice %[[PACK_1]]
// CHECK:            linalg.fill
// CHECK:            scf.for
// CHECK:              %[[SLICE_2:.+]] = tensor.extract_slice %[[SLICE_0]]
// CHECK:              %[[PACK_2:.+]] = tensor.pack %[[SLICE_2]]
// CHECK:              %[[SLICE_3:.+]] = tensor.extract_slice %[[SLICE_1]]
// CHECK:              %[[PACK_3:.+]] = tensor.pack %[[SLICE_3]]
// CHECK:              linalg.generic
func.func @matmul_tensor_extract_slice() {
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<512x512xbf16>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<512x4096xbf16>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<512x4096xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x512xbf16>> -> tensor<512x512xbf16>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x4096xbf16>> -> tensor<512x4096xbf16>
  %5 = tensor.empty() : tensor<512x4096xf32>
  %6 = scf.forall (%arg0, %arg1) = (0, 0) to (512, 4096) step (128, 128) shared_outs(%arg2 = %5) -> (tensor<512x4096xf32>) {
    %extracted_slice = tensor.extract_slice %3[%arg0, 0] [128, 512] [1, 1] : tensor<512x512xbf16> to tensor<128x512xbf16>
    %extracted_slice_0 = tensor.extract_slice %4[0, %arg1] [512, 128] [1, 1] : tensor<512x4096xbf16> to tensor<512x128xbf16>
    %extracted_slice_1 = tensor.extract_slice %arg2[%arg0, %arg1] [128, 128] [1, 1] : tensor<512x4096xf32> to tensor<128x128xf32>
    %7 = tensor.empty() : tensor<4x8x32x64xbf16>
    %pack = tensor.pack %extracted_slice inner_dims_pos = [0, 1] inner_tiles = [32, 64] into %7 : tensor<128x512xbf16> -> tensor<4x8x32x64xbf16>
    %8 = tensor.empty() : tensor<4x8x64x32xbf16>
    %pack_2 = tensor.pack %extracted_slice_0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [64, 32] into %8 : tensor<512x128xbf16> -> tensor<4x8x64x32xbf16>
    %alloc = memref.alloc() : memref<4x4x32x32xf32, 1 : i32>
    %9 = bufferization.to_tensor %alloc restrict writable : memref<4x4x32x32xf32, 1 : i32> to tensor<4x4x32x32xf32>
    %10 = tensor.empty() : tensor<4x4x8x8x4x4xf32>
    %11 = scf.forall (%arg3, %arg4) = (0, 0) to (4, 4) step (2, 2) shared_outs(%arg5 = %10) -> (tensor<4x4x8x8x4x4xf32>) {
      %extracted_slice_4 = tensor.extract_slice %pack[%arg3, 0, 0, 0] [2, 8, 32, 64] [1, 1, 1, 1] : tensor<4x8x32x64xbf16> to tensor<2x8x32x64xbf16>
      %12 = tensor.empty() : tensor<2x8x8x8x4x8xbf16>
      %extracted_slice_5 = tensor.extract_slice %pack_2[%arg4, 0, 0, 0] [2, 8, 64, 32] [1, 1, 1, 1] : tensor<4x8x64x32xbf16> to tensor<2x8x64x32xbf16>
      %13 = tensor.empty() : tensor<2x8x8x8x8x4xbf16>
      %extracted_slice_6 = tensor.extract_slice %arg5[%arg3, %arg4, 0, 0, 0, 0] [2, 2, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1] : tensor<4x4x8x8x4x4xf32> to tensor<2x2x8x8x4x4xf32>
      %14 = linalg.fill ins(%cst : f32) outs(%extracted_slice_6 : tensor<2x2x8x8x4x4xf32>) -> tensor<2x2x8x8x4x4xf32>
      %15 = scf.for %arg6 = %c0 to %c8 step %c1 iter_args(%arg7 = %14) -> (tensor<2x2x8x8x4x4xf32>) {
        %extracted_slice_7 = tensor.extract_slice %extracted_slice_4[0, %arg6, 0, 0] [2, 1, 32, 64] [1, 1, 1, 1] : tensor<2x8x32x64xbf16> to tensor<2x1x32x64xbf16>
        %extracted_slice_8 = tensor.extract_slice %12[0, %arg6, 0, 0, 0, 0] [2, 1, 8, 8, 4, 8] [1, 1, 1, 1, 1, 1] : tensor<2x8x8x8x4x8xbf16> to tensor<2x1x8x8x4x8xbf16>
        %pack_9 = tensor.pack %extracted_slice_7 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %extracted_slice_8 : tensor<2x1x32x64xbf16> -> tensor<2x1x8x8x4x8xbf16>
        %extracted_slice_10 = tensor.extract_slice %extracted_slice_5[0, %arg6, 0, 0] [2, 1, 64, 32] [1, 1, 1, 1] : tensor<2x8x64x32xbf16> to tensor<2x1x64x32xbf16>
        %extracted_slice_11 = tensor.extract_slice %13[0, %arg6, 0, 0, 0, 0] [2, 1, 8, 8, 8, 4] [1, 1, 1, 1, 1, 1] : tensor<2x8x8x8x8x4xbf16> to tensor<2x1x8x8x8x4xbf16>
        %pack_12 = tensor.pack %extracted_slice_10 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 4] into %extracted_slice_11 : tensor<2x1x64x32xbf16> -> tensor<2x1x8x8x8x4xbf16>
        %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d2, d4, d5, d8, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack_9, %pack_12 : tensor<2x1x8x8x4x8xbf16>, tensor<2x1x8x8x8x4xbf16>) outs(%arg7 : tensor<2x2x8x8x4x4xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 128], [2, 2, 0], [0, 0, 1], [1, 1, 0, 0, 0, 0]]>, packing_config = #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 64], transposePackIndices = [1], unpackEmpty = [false], innerPerm = [[1, 0]], outerPerm = [[1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>} {
        ^bb0(%in: bf16, %in_13: bf16, %out: f32):
          %17 = arith.extf %in : bf16 to f32
          %18 = arith.extf %in_13 : bf16 to f32
          %19 = arith.mulf %17, %18 : f32
          %20 = arith.addf %out, %19 : f32
          linalg.yield %20 : f32
        } -> tensor<2x2x8x8x4x4xf32>
        scf.yield %16 : tensor<2x2x8x8x4x4xf32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %15 into %arg5[%arg3, %arg4, 0, 0, 0, 0] [2, 2, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1] : tensor<2x2x8x8x4x4xf32> into tensor<4x4x8x8x4x4xf32>
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    %unpack = tensor.unpack %11 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 4] into %9 : tensor<4x4x8x8x4x4xf32> -> tensor<4x4x32x32xf32>
    %unpack_3 = tensor.unpack %unpack inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %extracted_slice_1 : tensor<4x4x32x32xf32> -> tensor<128x128xf32>
    memref.dealloc %alloc : memref<4x4x32x32xf32, 1 : i32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %unpack_3 into %arg2[%arg0, %arg1] [128, 128] [1, 1] : tensor<128x128xf32> into tensor<512x4096xf32>
    }
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [512, 4096], strides = [1, 1] : tensor<512x4096xf32> -> !flow.dispatch.tensor<writeonly:tensor<512x4096xf32>>
  return
}

// -----

// CHECK-LABEL:  @copy_pack_matmul
// CHECK:        memref.alloc() : memref<4x1x32x32xi32, 1 : i32>
// CHECK:        bufferization.to_tensor
// CHECK:        linalg.copy
// CHECK-NOT:    memref.alloc
// CHECK:        tensor.pack
// CHECK:        memref.alloc() : memref<4x1x32x32xi32, 1 : i32>
// CHECK:        bufferization.to_tensor
// CHECK:        linalg.copy
// CHECK-NOT:    memref.alloc
// CHECK:        tensor.pack
// CHECK:        linalg.generic
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d2, d5, d4, d7, d8)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
func.func @copy_pack_matmul(%arg0: tensor<4x1x32x32xi32>, %arg1: tensor<4x1x32x32xi32>) -> tensor<4x4x32x32xi32> {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<4x1x32x32xi32>
  %1 = tensor.empty() : tensor<4x1x32x32xi32>
  %2 = tensor.empty() : tensor<4x4x32x32xi32>
  %3 = tensor.empty() : tensor<4x1x4x8x4x8xi32>
  %4 = tensor.empty() : tensor<4x1x4x8x4x8xi32>
  %5 = tensor.empty() : tensor<4x4x8x8x4x4xi32>
  %6 = linalg.copy ins(%arg0 : tensor<4x1x32x32xi32>) outs(%0 : tensor<4x1x32x32xi32>) -> tensor<4x1x32x32xi32>
  %pack = tensor.pack %6 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %3 : tensor<4x1x32x32xi32> -> tensor<4x1x4x8x4x8xi32>
  %7 = linalg.copy ins(%arg1 : tensor<4x1x32x32xi32>) outs(%1 : tensor<4x1x32x32xi32>) -> tensor<4x1x32x32xi32>
  %pack_0 = tensor.pack %7 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %4 : tensor<4x1x32x32xi32> -> tensor<4x1x4x8x4x8xi32>
  %8 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_0 : tensor<4x1x4x8x4x8xi32>, tensor<4x1x4x8x4x8xi32>) outs(%5 : tensor<4x4x8x8x4x4xi32>) {
  ^bb0(%in: i32, %in_1: i32, %out: i32):
    %9 = arith.muli %in, %in_1 : i32
    %10 = arith.addi %out, %9 : i32
    linalg.yield %10 : i32
  } -> tensor<4x4x8x8x4x4xi32>
  %unpack = tensor.unpack %8 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 4] into %2 : tensor<4x4x8x8x4x4xi32> -> tensor<4x4x32x32xi32>
  return %unpack : tensor<4x4x32x32xi32>
}

// -----

#map = affine_map<(d3, d4, d5, d6, d7, d8) -> (d3, d5, d6, d8)>
#map1 = affine_map<(d3, d4, d5, d6, d7, d8) -> (d5, d4, d7, d8)>
#map2 = affine_map<(d3, d4, d5, d6, d7, d8) -> (d3, d4, d6, d7)>
func.func @pack_error(%arg0 : tensor<1024x2048xi32>, %arg1 : tensor<2048x512xi32>) -> tensor<1024x512xi32>
{
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %5 = tensor.empty() : tensor<1024x512xi32>
    %6 = tensor.empty() : tensor<16x32x64x64xi32>
    %pack = tensor.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %6 : tensor<1024x2048xi32> -> tensor<16x32x64x64xi32>
    %7 = tensor.empty() : tensor<32x8x64x64xi32>
    %pack_0 = tensor.pack %arg1 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %7 : tensor<2048x512xi32> -> tensor<32x8x64x64xi32>
    %8 = tensor.empty() : tensor<16x8x64x64xi32>
    %9 = linalg.fill ins(%c0_i32 : i32) outs(%8 : tensor<16x8x64x64xi32>) -> tensor<16x8x64x64xi32>
    // expected-error @+2 {{could not fetch operands to bufferize}}
    // expected-error @+1 {{'linalg.generic' op operand #0 only has pack/copy ops to depth 1, but request is for a depth 2 pack/copy op}}
    %10 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_0 : tensor<16x32x64x64xi32>, tensor<32x8x64x64xi32>) outs(%9 : tensor<16x8x64x64xi32>) {
    ^bb0(%in: i32, %in_4: i32, %out: i32):
      %14 = arith.muli %in, %in_4 : i32
      %15 = arith.addi %out, %14 : i32
      linalg.yield %15 : i32
    } -> tensor<16x8x64x64xi32>
    %unpack = tensor.unpack %10 inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %5 : tensor<16x8x64x64xi32> -> tensor<1024x512xi32>
    return %unpack : tensor<1024x512xi32>
}
