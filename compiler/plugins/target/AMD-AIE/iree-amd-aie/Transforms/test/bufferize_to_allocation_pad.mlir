// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-bufferize-to-allocation{memory-space=1 bufferize-level=0}))' --split-input-file %s | FileCheck %s --check-prefix=BUFFERIZE-LEVEL-0
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-bufferize-to-allocation{memory-space=2 bufferize-level=1}))' --split-input-file %s | FileCheck %s --check-prefix=BUFFERIZE-LEVEL-1

func.func @matmul_static(%arg0 : tensor<8x16xi32>, %arg1 : tensor<16x8xi32>) -> tensor<8x8xi32> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %5 = tensor.empty() : tensor<8x8xi32>
  %6 = scf.forall (%iv0, %iv1) = (0, 0) to (8, 8) step (8, 8) shared_outs(%arg2 = %5) -> (tensor<8x8xi32>) {
    %extracted_slice_0 = tensor.extract_slice %arg0[%iv0, 0] [8, 16] [1, 1] : tensor<8x16xi32> to tensor<8x16xi32>
    %extracted_slice_2 = tensor.extract_slice %arg1[0, %iv1] [16, 8] [1, 1] : tensor<16x8xi32> to tensor<16x8xi32>
    %11 = tensor.empty() : tensor<8x8xi32>
    %extracted_slice_3 = tensor.extract_slice %11[%iv0, %iv1] [8, 8] [1, 1] : tensor<8x8xi32> to tensor<8x8xi32>
    %extracted_slice_4 = tensor.extract_slice %arg2[%iv0, %iv1] [8, 8] [1, 1] : tensor<8x8xi32> to tensor<8x8xi32>
    %12 = linalg.fill ins(%c0_i32 : i32) outs(%extracted_slice_4 : tensor<8x8xi32>) -> tensor<8x8xi32>
    %extracted_slice_5 = tensor.extract_slice %arg2[%iv0, %iv1] [8, 8] [1, 1] : tensor<8x8xi32> to tensor<8x8xi32>
    %c0_i32_6 = arith.constant 0 : i32
    %13 = bufferization.alloc_tensor() : tensor<8x16xi32>
    %14 = linalg.copy ins(%extracted_slice_0 : tensor<8x16xi32>) outs(%13 : tensor<8x16xi32>) -> tensor<8x16xi32>
    %c0_i32_7 = arith.constant 0 : i32
    %15 = bufferization.alloc_tensor() : tensor<16x8xi32>
    %16 = linalg.copy ins(%extracted_slice_2 : tensor<16x8xi32>) outs(%15 : tensor<16x8xi32>) -> tensor<16x8xi32>
    %c0_i32_8 = arith.constant 0 : i32
    %17 = bufferization.alloc_tensor() : tensor<8x8xi32>
    %18 = linalg.copy ins(%12 : tensor<8x8xi32>) outs(%17 : tensor<8x8xi32>) -> tensor<8x8xi32>
    %19 = linalg.matmul ins(%14, %16 : tensor<8x16xi32>, tensor<16x8xi32>) outs(%18 : tensor<8x8xi32>) -> tensor<8x8xi32>
    %extracted_slice_9 = tensor.extract_slice %19[0, 0] [8, 8] [1, 1] : tensor<8x8xi32> to tensor<8x8xi32>
    %20 = linalg.copy ins(%extracted_slice_9 : tensor<8x8xi32>) outs(%12 : tensor<8x8xi32>) -> tensor<8x8xi32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %20 into %arg2[%iv0, %iv1] [8, 8] [1, 1] : tensor<8x8xi32> into tensor<8x8xi32>
    }
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  return %6 : tensor<8x8xi32>
}
//      BUFFERIZE-LEVEL-0: @matmul_static
//      BUFFERIZE-LEVEL-0:   scf.forall
// BUFFERIZE-LEVEL-0-SAME:   {
//      BUFFERIZE-LEVEL-0:       linalg.fill
//      BUFFERIZE-LEVEL-0:       memref.alloc() : memref<8x16xi32, 1>
//      BUFFERIZE-LEVEL-0:       bufferization.to_tensor %{{.*}} restrict writable
//      BUFFERIZE-LEVEL-0:       linalg.copy
//      BUFFERIZE-LEVEL-0:       memref.alloc() : memref<16x8xi32, 1>
//      BUFFERIZE-LEVEL-0:       bufferization.to_tensor %{{.*}} restrict writable
//      BUFFERIZE-LEVEL-0:       linalg.copy
//      BUFFERIZE-LEVEL-0:       memref.alloc() : memref<8x8xi32, 1>
//      BUFFERIZE-LEVEL-0:       bufferization.to_tensor %{{.*}} restrict writable
//      BUFFERIZE-LEVEL-0:       linalg.copy
//      BUFFERIZE-LEVEL-0:       linalg.matmul
//      BUFFERIZE-LEVEL-0:       linalg.copy
//      BUFFERIZE-LEVEL-0:       memref.dealloc
//      BUFFERIZE-LEVEL-0:       memref.dealloc
//      BUFFERIZE-LEVEL-0:       memref.dealloc
//      BUFFERIZE-LEVEL-0:   }

// -----

func.func @matmul_static(%arg0 : tensor<8x16xi32>, %arg1 : tensor<16x8xi32>) -> tensor<8x8xi32> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %5 = tensor.empty() : tensor<8x8xi32>
  %6 = scf.forall (%iv0, %iv1) = (0, 0) to (8, 8) step (8, 8) shared_outs(%arg2 = %5) -> (tensor<8x8xi32>) {
    %extracted_slice = tensor.extract_slice %arg0[%iv0, 0] [8, 16] [1, 1] : tensor<8x16xi32> to tensor<8x16xi32>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, %iv1] [16, 8] [1, 1] : tensor<16x8xi32> to tensor<16x8xi32>
    %extracted_slice_1 = tensor.extract_slice %arg2[%iv0, %iv1] [8, 8] [1, 1] : tensor<8x8xi32> to tensor<8x8xi32>
    %7 = bufferization.alloc_tensor() : tensor<8x16xi32>
    %alloc = memref.alloc() : memref<8x16xi32, 1>
    %8 = bufferization.to_tensor %alloc restrict writable : memref<8x16xi32, 1>
    %9 = linalg.copy ins(%extracted_slice : tensor<8x16xi32>) outs(%8 : tensor<8x16xi32>) -> tensor<8x16xi32>
    %10 = bufferization.alloc_tensor() : tensor<16x8xi32>
    %alloc_2 = memref.alloc() : memref<16x8xi32, 1>
    %11 = bufferization.to_tensor %alloc_2 restrict writable : memref<16x8xi32, 1>
    %12 = linalg.copy ins(%extracted_slice_0 : tensor<16x8xi32>) outs(%11 : tensor<16x8xi32>) -> tensor<16x8xi32>
    %13 = bufferization.alloc_tensor() : tensor<8x8xi32>
    %alloc_3 = memref.alloc() : memref<8x8xi32, 1>
    %14 = bufferization.to_tensor %alloc_3 restrict writable : memref<8x8xi32, 1>
    %15 = scf.forall (%iv3, %iv4) = (0, 0) to (8, 8) step (4, 4) shared_outs(%arg5 = %14) -> (tensor<8x8xi32>) {
      %extracted_slice_4 = tensor.extract_slice %9[%iv3, 0] [4, 16] [1, 1] : tensor<8x16xi32> to tensor<4x16xi32>
      %extracted_slice_5 = tensor.extract_slice %12[0, %iv4] [16, 4] [1, 1] : tensor<16x8xi32> to tensor<16x4xi32>
      %17 = bufferization.to_tensor %alloc_3 restrict writable : memref<8x8xi32, 1>
      %extracted_slice_6 = tensor.extract_slice %17[%iv3, %iv4] [4, 4] [1, 1] : tensor<8x8xi32> to tensor<4x4xi32>
      %extracted_slice_7 = tensor.extract_slice %arg5[%iv3, %iv4] [4, 4] [1, 1] : tensor<8x8xi32> to tensor<4x4xi32>
      %18 = linalg.fill ins(%c0_i32 : i32) outs(%extracted_slice_7 : tensor<4x4xi32>) -> tensor<4x4xi32>
      %extracted_slice_8 = tensor.extract_slice %arg5[%iv3, %iv4] [4, 4] [1, 1] : tensor<8x8xi32> to tensor<4x4xi32>
      %c0_i32_9 = arith.constant 0 : i32
      %19 = bufferization.alloc_tensor() : tensor<4x4xi32>
      %20 = linalg.copy ins(%18 : tensor<4x4xi32>) outs(%19 : tensor<4x4xi32>) -> tensor<4x4xi32>
      %21 = linalg.matmul ins(%extracted_slice_4, %extracted_slice_5 : tensor<4x16xi32>, tensor<16x4xi32>) outs(%20 : tensor<4x4xi32>) -> tensor<4x4xi32>
      %extracted_slice_10 = tensor.extract_slice %21[0, 0] [4, 4] [1, 1] : tensor<4x4xi32> to tensor<4x4xi32>
      %22 = linalg.copy ins(%extracted_slice_10 : tensor<4x4xi32>) outs(%18 : tensor<4x4xi32>) -> tensor<4x4xi32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %22 into %arg5[%iv3, %iv4] [4, 4] [1, 1] : tensor<4x4xi32> into tensor<8x8xi32>
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    %16 = linalg.copy ins(%15 : tensor<8x8xi32>) outs(%extracted_slice_1 : tensor<8x8xi32>) -> tensor<8x8xi32>
    memref.dealloc %alloc : memref<8x16xi32, 1>
    memref.dealloc %alloc_2 : memref<16x8xi32, 1>
    memref.dealloc %alloc_3 : memref<8x8xi32, 1>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %16 into %arg2[%iv0, %iv1] [8, 8] [1, 1] : tensor<8x8xi32> into tensor<8x8xi32>
    }
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  return %6 : tensor<8x8xi32>
}
//      BUFFERIZE-LEVEL-1: @matmul_static
//      BUFFERIZE-LEVEL-1:   scf.forall
// BUFFERIZE-LEVEL-1-SAME:   {
//      BUFFERIZE-LEVEL-1:       memref.alloc() : memref<8x16xi32, 1>
//      BUFFERIZE-LEVEL-1:       bufferization.to_tensor %{{.*}} restrict writable
//      BUFFERIZE-LEVEL-1:       linalg.copy
//      BUFFERIZE-LEVEL-1:       memref.alloc() : memref<16x8xi32, 1>
//      BUFFERIZE-LEVEL-1:       bufferization.to_tensor %{{.*}} restrict writable
//      BUFFERIZE-LEVEL-1:       linalg.copy
//      BUFFERIZE-LEVEL-1:       memref.alloc() : memref<8x8xi32, 1>
//      BUFFERIZE-LEVEL-1:       bufferization.to_tensor %{{.*}} restrict writable
//      BUFFERIZE-LEVEL-1:       scf.forall
// BUFFERIZE-LEVEL-1-SAME:       {
//      BUFFERIZE-LEVEL-1:            linalg.fill
//      BUFFERIZE-LEVEL-1:            memref.alloc() : memref<4x4xi32, 2>
//      BUFFERIZE-LEVEL-1:            bufferization.to_tensor %{{.*}} restrict writable
//      BUFFERIZE-LEVEL-1:            linalg.copy
//      BUFFERIZE-LEVEL-1:            linalg.matmul
//      BUFFERIZE-LEVEL-1:            linalg.copy
//      BUFFERIZE-LEVEL-1:            memref.dealloc
//      BUFFERIZE-LEVEL-1:       }
//      BUFFERIZE-LEVEL-1:       memref.dealloc
//      BUFFERIZE-LEVEL-1:       memref.dealloc
//      BUFFERIZE-LEVEL-1:       memref.dealloc
//      BUFFERIZE-LEVEL-1:   }
