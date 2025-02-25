// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-amdaie-create-reference-to-allocation, iree-codegen-hoist-statically-bound-allocations))" %s | FileCheck %s

// CHECK-LABEL: func.func @single_alloc
// CHECK:  %[[ALLOC:.+]] = memref.alloc() : memref<8x16xi32, 2 : i32>
// CHECK:  %[[REFER:.+]] = amdaie.reference_to %[[ALLOC]] : memref<8x16xi32, 2 : i32>
// CHECK:  %[[USER:.+]] = bufferization.to_tensor %[[REFER]] restrict writable : memref<8x16xi32, 2 : i32>
// CHECK:  linalg.copy ins(%{{.*}}) outs(%[[USER:.+]])
// CHECK:  memref.dealloc %[[ALLOC]] : memref<8x16xi32, 2 : i32>
func.func @single_alloc(%arg0: tensor<8x16xi32>) -> tensor<8x16xi32> {
  %alloc = memref.alloc() : memref<8x16xi32, 2 : i32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<8x16xi32, 2 : i32> to tensor<8x16xi32>
  %1 = linalg.copy ins(%arg0 : tensor<8x16xi32>) outs(%0 : tensor<8x16xi32>) -> tensor<8x16xi32>
  memref.dealloc %alloc : memref<8x16xi32, 2 : i32>
  return %1 : tensor<8x16xi32>
}

// -----

// CHECK-LABEL: func.func @multiple_alloc
// CHECK:  %[[ALLOC0:.+]] = memref.alloc() : memref<8x16xi32, 2 : i32>
// CHECK:  %[[REFER0:.+]] = amdaie.reference_to %[[ALLOC0]]
// CHECK:  %[[USER0:.+]] = bufferization.to_tensor %[[REFER0]]
// CHECK:  linalg.copy ins(%{{.*}}) outs(%[[USER0:.+]])
// CHECK:  %[[ALLOC1:.+]] = memref.alloc() : memref<16x8xi32, 2 : i32>
// CHECK:  %[[REFER1:.+]] = amdaie.reference_to %[[ALLOC1]]
// CHECK:  %[[USER1:.+]] = bufferization.to_tensor %[[REFER1]]
// CHECK:  linalg.copy ins(%{{.*}}) outs(%[[USER1:.+]])
// CHECK:  %[[ALLOC2:.+]] = memref.alloc() : memref<8x8xi32, 2 : i32>
// CHECK:  %[[REFER2:.+]] = amdaie.reference_to %[[ALLOC2]]
// CHECK:  %[[USER2:.+]] = bufferization.to_tensor %[[REFER2]]
// CHECK:  linalg.fill ins(%{{.*}}) outs(%[[USER2:.+]])
// CHECK:  linalg.matmul
// CHECK:  linalg.copy
// CHECK:  memref.dealloc %[[ALLOC0]]
// CHECK:  memref.dealloc %[[ALLOC1]]
// CHECK:  memref.dealloc %[[ALLOC2]]
func.func @multiple_alloc(%arg0: tensor<8x16xi32>, %arg1: tensor<16x8xi32>) -> tensor<8x8xi32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<8x8xi32>
  %1 = bufferization.alloc_tensor() : tensor<8x16xi32>
  %alloc = memref.alloc() : memref<8x16xi32, 2 : i32>
  %2 = bufferization.to_tensor %alloc restrict writable : memref<8x16xi32, 2 : i32> to tensor<8x16xi32>
  %3 = linalg.copy ins(%arg0 : tensor<8x16xi32>) outs(%2 : tensor<8x16xi32>) -> tensor<8x16xi32>
  %4 = bufferization.alloc_tensor() : tensor<16x8xi32>
  %alloc_0 = memref.alloc() : memref<16x8xi32, 2 : i32>
  %5 = bufferization.to_tensor %alloc_0 restrict writable : memref<16x8xi32, 2 : i32> to tensor<16x8xi32>
  %6 = linalg.copy ins(%arg1 : tensor<16x8xi32>) outs(%5 : tensor<16x8xi32>) -> tensor<16x8xi32>
  %7 = bufferization.alloc_tensor() : tensor<8x8xi32>
  %alloc_1 = memref.alloc() : memref<8x8xi32, 2 : i32>
  %8 = bufferization.to_tensor %alloc_1 restrict writable : memref<8x8xi32, 2 : i32> to tensor<8x8xi32>
  %9 = linalg.fill ins(%c0_i32 : i32) outs(%8 : tensor<8x8xi32>) -> tensor<8x8xi32>
  %10 = linalg.matmul ins(%3, %6 : tensor<8x16xi32>, tensor<16x8xi32>) outs(%9 : tensor<8x8xi32>) -> tensor<8x8xi32>
  %11 = linalg.copy ins(%10 : tensor<8x8xi32>) outs(%0 : tensor<8x8xi32>) -> tensor<8x8xi32>
  memref.dealloc %alloc : memref<8x16xi32, 2 : i32>
  memref.dealloc %alloc_0 : memref<16x8xi32, 2 : i32>
  memref.dealloc %alloc_1 : memref<8x8xi32, 2 : i32>
  return %11 : tensor<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @alloc_in_L2
// CHECK-NOT:   amdaie.reference_to
func.func @alloc_in_L2(%arg0: tensor<8x16xi32>) -> tensor<8x16xi32> {
  %alloc = memref.alloc() : memref<8x16xi32, 1 : i32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<8x16xi32, 1 : i32> to tensor<8x16xi32>
  %1 = linalg.copy ins(%arg0 : tensor<8x16xi32>) outs(%0 : tensor<8x16xi32>) -> tensor<8x16xi32>
  memref.dealloc %alloc : memref<8x16xi32, 1 : i32>
  return %1 : tensor<8x16xi32>
}

// -----

// CHECK-LABEL: func.func @matmul_example
// CHECK:      %[[ALLOC0:.+]] = memref.alloc() : memref<1x1x8x4x8x4xi32, 2 : i32>
// CHECK:      %[[ALLOC1:.+]] = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
// CHECK:      %[[ALLOC2:.+]] = memref.alloc() : memref<1x2x32x32xi32, 1 : i32>
// CHECK:      %[[ALLOC3:.+]] = memref.alloc() : memref<2x1x32x32xi32, 1 : i32>
// CHECK:      %[[ALLOC4:.+]] = memref.alloc() : memref<2x2x8x8x4x4xi32, 2 : i32>
// CHECK:      %[[ALLOC5:.+]] = memref.alloc() : memref<2x2x32x32xi32, 1 : i32>
// CHECK:      scf.forall
// CHECK-NOT:    memref.alloc()
// CHECK:        bufferization.to_tensor %[[ALLOC5]]
// CHECK-NOT:    memref.alloc()
// CHECK:        %[[REFER4:.+]] = amdaie.reference_to %[[ALLOC4]]
// CHECK:        bufferization.to_tensor %[[REFER4]]
// CHECK:        scf.for
// CHECK-NOT:    memref.alloc()
// CHECK:        bufferization.to_tensor %[[ALLOC3]]
// CHECK-NOT:    memref.alloc()
// CHECK:        bufferization.to_tensor %[[ALLOC2]]
// CHECK:          scf.forall
// CHECK-NOT:      memref.alloc()
// CHECK:          %[[REFER1:.+]] = amdaie.reference_to %[[ALLOC1]]
// CHECK:          bufferization.to_tensor %[[REFER1]]
// CHECK-NOT:      memref.alloc()
// CHECK:          %[[REFER0:.+]] = amdaie.reference_to %[[ALLOC0]]
// CHECK:          bufferization.to_tensor %[[REFER0]]
// CHECK:      memref.dealloc %[[ALLOC5]]
// CHECK:      memref.dealloc %[[ALLOC4]]
// CHECK:      memref.dealloc %[[ALLOC3]]
// CHECK:      memref.dealloc %[[ALLOC2]]
// CHECK:      memref.dealloc %[[ALLOC1]]
// CHECK:      memref.dealloc %[[ALLOC0]]

#map = affine_map<(d0) -> (d0 * 32)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
func.func @matmul_example(%arg0: tensor<128x256xi32>, %arg1: tensor<256x128xi32>) -> tensor<128x128xi32> {
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<128x128xi32>
  %1 = scf.forall (%arg2, %arg3) = (0, 0) to (128, 128) step (64, 64) shared_outs(%arg4 = %0) -> (tensor<128x128xi32>) {
    %extracted_slice = tensor.extract_slice %arg0[%arg2, 0] [64, 256] [1, 1] : tensor<128x256xi32> to tensor<64x256xi32>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg3] [256, 64] [1, 1] : tensor<256x128xi32> to tensor<256x64xi32>
    %extracted_slice_1 = tensor.extract_slice %arg4[%arg2, %arg3] [64, 64] [1, 1] : tensor<128x128xi32> to tensor<64x64xi32>
    %alloc = memref.alloc() : memref<2x2x32x32xi32, 1 : i32>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<2x2x32x32xi32, 1 : i32> to tensor<2x2x32x32xi32>
    %alloc_2 = memref.alloc() : memref<2x2x8x8x4x4xi32, 2 : i32>
    %3 = bufferization.to_tensor %alloc_2 restrict writable : memref<2x2x8x8x4x4xi32, 2 : i32> to tensor<2x2x8x8x4x4xi32>
    %4 = linalg.fill ins(%c0_i32 : i32) outs(%3 : tensor<2x2x8x8x4x4xi32>) -> tensor<2x2x8x8x4x4xi32>
    %5 = tensor.empty() : tensor<2x1x4x8x4x8xi32>
    %6 = tensor.empty() : tensor<1x2x8x4x8x4xi32>
    %7 = scf.for %arg5 = %c0 to %c8 step %c1 iter_args(%arg6 = %4) -> (tensor<2x2x8x8x4x4xi32>) {
      %8 = affine.apply #map(%arg5)
      %extracted_slice_4 = tensor.extract_slice %extracted_slice[0, %8] [64, 32] [1, 1] : tensor<64x256xi32> to tensor<64x32xi32>
      %alloc_5 = memref.alloc() : memref<2x1x32x32xi32, 1 : i32>
      %9 = bufferization.to_tensor %alloc_5 restrict writable : memref<2x1x32x32xi32, 1 : i32> to tensor<2x1x32x32xi32>
      %pack = linalg.pack %extracted_slice_4 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %9 : tensor<64x32xi32> -> tensor<2x1x32x32xi32>
      %extracted_slice_6 = tensor.extract_slice %extracted_slice_0[%8, 0] [32, 64] [1, 1] : tensor<256x64xi32> to tensor<32x64xi32>
      %alloc_7 = memref.alloc() : memref<1x2x32x32xi32, 1 : i32>
      %10 = bufferization.to_tensor %alloc_7 restrict writable : memref<1x2x32x32xi32, 1 : i32> to tensor<1x2x32x32xi32>
      %pack_8 = linalg.pack %extracted_slice_6 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %10 : tensor<32x64xi32> -> tensor<1x2x32x32xi32>
      %11 = scf.forall (%arg7, %arg8) in (2, 2) shared_outs(%arg9 = %arg6) -> (tensor<2x2x8x8x4x4xi32>) {
        %extracted_slice_9 = tensor.extract_slice %pack[%arg7, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x1x32x32xi32> to tensor<1x1x32x32xi32>
        %extracted_slice_10 = tensor.extract_slice %5[%arg7, 0, 0, 0, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : tensor<2x1x4x8x4x8xi32> to tensor<1x1x4x8x4x8xi32>
        %alloc_11 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
        %12 = bufferization.to_tensor %alloc_11 restrict writable : memref<1x1x4x8x4x8xi32, 2 : i32> to tensor<1x1x4x8x4x8xi32>
        %pack_12 = linalg.pack %extracted_slice_9 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %12 : tensor<1x1x32x32xi32> -> tensor<1x1x4x8x4x8xi32>
        %extracted_slice_13 = tensor.extract_slice %pack_8[0, %arg8, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<1x2x32x32xi32> to tensor<1x1x32x32xi32>
        %extracted_slice_14 = tensor.extract_slice %6[0, %arg8, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [1, 1, 1, 1, 1, 1] : tensor<1x2x8x4x8x4xi32> to tensor<1x1x8x4x8x4xi32>
        %alloc_15 = memref.alloc() : memref<1x1x8x4x8x4xi32, 2 : i32>
        %13 = bufferization.to_tensor %alloc_15 restrict writable : memref<1x1x8x4x8x4xi32, 2 : i32> to tensor<1x1x8x4x8x4xi32>
        %pack_16 = linalg.pack %extracted_slice_13 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 4] into %13 : tensor<1x1x32x32xi32> -> tensor<1x1x8x4x8x4xi32>
        %extracted_slice_17 = tensor.extract_slice %arg9[%arg7, %arg8, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1] : tensor<2x2x8x8x4x4xi32> to tensor<1x1x8x8x4x4xi32>
        %14 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack_12, %pack_16 : tensor<1x1x4x8x4x8xi32>, tensor<1x1x8x4x8x4xi32>) outs(%extracted_slice_17 : tensor<1x1x8x8x4x4xi32>) {
        ^bb0(%in: i32, %in_18: i32, %out: i32):
          %15 = arith.muli %in, %in_18 : i32
          %16 = arith.addi %out, %15 : i32
          linalg.yield %16 : i32
        } -> tensor<1x1x8x8x4x4xi32>
        memref.dealloc %alloc_11 : memref<1x1x4x8x4x8xi32, 2 : i32>
        memref.dealloc %alloc_15 : memref<1x1x8x4x8x4xi32, 2 : i32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %14 into %arg9[%arg7, %arg8, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1] : tensor<1x1x8x8x4x4xi32> into tensor<2x2x8x8x4x4xi32>
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      memref.dealloc %alloc_5 : memref<2x1x32x32xi32, 1 : i32>
      memref.dealloc %alloc_7 : memref<1x2x32x32xi32, 1 : i32>
      scf.yield %11 : tensor<2x2x8x8x4x4xi32>
    }
    %unpack = linalg.unpack %7 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 4] into %2 : tensor<2x2x8x8x4x4xi32> -> tensor<2x2x32x32xi32>
    %unpack_3 = linalg.unpack %unpack inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %extracted_slice_1 : tensor<2x2x32x32xi32> -> tensor<64x64xi32>
    memref.dealloc %alloc : memref<2x2x32x32xi32, 1 : i32>
    memref.dealloc %alloc_2 : memref<2x2x8x8x4x4xi32, 2 : i32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %unpack_3 into %arg4[%arg2, %arg3] [64, 64] [1, 1] : tensor<64x64xi32> into tensor<128x128xi32>
    }
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  return %1 : tensor<128x128xi32>
}
