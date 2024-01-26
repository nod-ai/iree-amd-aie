// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-fuse-fill-into-forall))' %s | FileCheck %s

#map = affine_map<(d0) -> (d0 * 16)>
#map1 = affine_map<(d0) -> (d0 * 64)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d5, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
func.func @matmul_example_dispatch_0_matmul_16x256x256_i8xi8xi32() {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x256xi8>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xi8>>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x256xi32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [16, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x256xi8>> -> tensor<16x256xi8>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xi8>> -> tensor<256x256xi8>
  %5 = tensor.empty() : tensor<16x256xi32>
  %6 = scf.forall (%arg0, %arg1) in (1, 4) shared_outs(%arg2 = %5) -> (tensor<16x256xi32>) {
    %7 = affine.apply #map(%arg0)
    %8 = affine.apply #map1(%arg1)
    %extracted_slice = tensor.extract_slice %3[%7, 0] [16, 256] [1, 1] : tensor<16x256xi8> to tensor<16x256xi8>
    %extracted_slice_0 = tensor.extract_slice %4[0, %8] [256, 64] [1, 1] : tensor<256x256xi8> to tensor<256x64xi8>
    %extracted_slice_1 = tensor.extract_slice %arg2[%7, %8] [16, 64] [1, 1] : tensor<16x256xi32> to tensor<16x64xi32>
    %alloc = memref.alloc() : memref<1x4x16x64xi8, "shared">
    %9 = bufferization.to_tensor %alloc restrict writable : memref<1x4x16x64xi8, "shared">
    %pack = tensor.pack %extracted_slice inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %9 : tensor<16x256xi8> -> tensor<1x4x16x64xi8>
    %alloc_2 = memref.alloc() : memref<4x1x64x64xi8, "shared">
    %10 = bufferization.to_tensor %alloc_2 restrict writable : memref<4x1x64x64xi8, "shared">
    %pack_3 = tensor.pack %extracted_slice_0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %10 : tensor<256x64xi8> -> tensor<4x1x64x64xi8>
    %alloc_4 = memref.alloc() : memref<1x1x16x64xi32, "shared">
    %11 = bufferization.to_tensor %alloc_4 restrict writable : memref<1x1x16x64xi32, "shared">
    %12 = linalg.fill ins(%c0_i32 : i32) outs(%11 : tensor<1x1x16x64xi32>) -> tensor<1x1x16x64xi32>
    %13 = scf.forall (%arg3, %arg4) in (1, 1) shared_outs(%arg5 = %12) -> (tensor<1x1x16x64xi32>) {
      %extracted_slice_5 = tensor.extract_slice %pack[%arg3, 0, 0, 0] [1, 4, 16, 64] [1, 1, 1, 1] : tensor<1x4x16x64xi8> to tensor<1x4x16x64xi8>
      %extracted_slice_6 = tensor.extract_slice %pack_3[0, %arg4, 0, 0] [4, 1, 64, 64] [1, 1, 1, 1] : tensor<4x1x64x64xi8> to tensor<4x1x64x64xi8>
      %extracted_slice_7 = tensor.extract_slice %arg5[%arg3, %arg4, 0, 0] [1, 1, 16, 64] [1, 1, 1, 1] : tensor<1x1x16x64xi32> to tensor<1x1x16x64xi32>
      // CHECK: linalg.fill ins(%{{.*}}) outs(%{{.*}}) -> tensor<1x1x16x64xi32>
      %14 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%extracted_slice_5, %extracted_slice_6 : tensor<1x4x16x64xi8>, tensor<4x1x64x64xi8>) outs(%extracted_slice_7 : tensor<1x1x16x64xi32>) {
      ^bb0(%in: i8, %in_8: i8, %out: i32):
        %15 = arith.extsi %in : i8 to i32
        %16 = arith.extsi %in_8 : i8 to i32
        %17 = arith.muli %15, %16 : i32
        %18 = arith.addi %out, %17 : i32
        linalg.yield %18 : i32
      } -> tensor<1x1x16x64xi32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %14 into %arg5[%arg3, %arg4, 0, 0] [1, 1, 16, 64] [1, 1, 1, 1] : tensor<1x1x16x64xi32> into tensor<1x1x16x64xi32>
      }
    } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    %unpack = tensor.unpack %13 inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %extracted_slice_1 : tensor<1x1x16x64xi32> -> tensor<16x64xi32>
    memref.dealloc %alloc_2 : memref<4x1x64x64xi8, "shared">
    memref.dealloc %alloc : memref<1x4x16x64xi8, "shared">
    memref.dealloc %alloc_4 : memref<1x1x16x64xi32, "shared">
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %unpack into %arg2[%7, %8] [16, 64] [1, 1] : tensor<16x64xi32> into tensor<16x256xi32>
    }
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [16, 256], strides = [1, 1] : tensor<16x256xi32> -> !flow.dispatch.tensor<writeonly:tensor<16x256xi32>>
  return
}
