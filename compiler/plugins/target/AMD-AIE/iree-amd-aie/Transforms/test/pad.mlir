// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-pad{padding-level=0}))' --split-input-file %s | FileCheck %s --check-prefix=PAD-LEVEL-0
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-pad{padding-level=1}))' --split-input-file %s | FileCheck %s --check-prefix=PAD-LEVEL-1
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-pad{padding-level=2}))' --split-input-file %s | FileCheck %s --check-prefix=PAD-LEVEL-2

func.func @matmul_static(%arg0 : tensor<8x16xi32>, %arg1 : tensor<16x8xi32>) -> tensor<8x8xi32> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %5 = tensor.empty() : tensor<8x8xi32>
  %6 = scf.forall (%iv0, %iv1) = (0, 0) to (8, 8) step (8, 8) shared_outs(%arg2 = %5) -> (tensor<8x8xi32>) {
    %extracted_slice = tensor.extract_slice %arg0[%iv0, 0] [8, 16] [1, 1] : tensor<8x16xi32> to tensor<8x16xi32>
    %extracted_slice_0 = tensor.extract_slice %arg0[%iv0, 0] [8, 16] [1, 1] : tensor<8x16xi32> to tensor<8x16xi32>
    %extracted_slice_1 = tensor.extract_slice %arg1[0, %iv1] [16, 8] [1, 1] : tensor<16x8xi32> to tensor<16x8xi32>
    %extracted_slice_2 = tensor.extract_slice %arg1[0, %iv1] [16, 8] [1, 1] : tensor<16x8xi32> to tensor<16x8xi32>
    %11 = tensor.empty() : tensor<8x8xi32>
    %extracted_slice_3 = tensor.extract_slice %11[%iv0, %iv1] [8, 8] [1, 1] : tensor<8x8xi32> to tensor<8x8xi32>
    %extracted_slice_4 = tensor.extract_slice %arg2[%iv0, %iv1] [8, 8] [1, 1] : tensor<8x8xi32> to tensor<8x8xi32>
    %12 = linalg.fill ins(%c0_i32 : i32) outs(%extracted_slice_4 : tensor<8x8xi32>) -> tensor<8x8xi32>
    %extracted_slice_5 = tensor.extract_slice %arg2[%iv0, %iv1] [8, 8] [1, 1] : tensor<8x8xi32> to tensor<8x8xi32>
    %13 = linalg.matmul ins(%extracted_slice_0, %extracted_slice_2 : tensor<8x16xi32>, tensor<16x8xi32>) outs(%12 : tensor<8x8xi32>) -> tensor<8x8xi32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %13 into %arg2[%iv0, %iv1] [8, 8] [1, 1] : tensor<8x8xi32> into tensor<8x8xi32>
    }
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  return %6 : tensor<8x8xi32>
}
//      PAD-LEVEL-0:   scf.forall
// PAD-LEVEL-0-SAME:   {
//      PAD-LEVEL-0:       linalg.fill
//      PAD-LEVEL-0:       bufferization.alloc_tensor
//      PAD-LEVEL-0:       linalg.copy
//      PAD-LEVEL-0:       bufferization.alloc_tensor
//      PAD-LEVEL-0:       linalg.copy
//      PAD-LEVEL-0:       bufferization.alloc_tensor
//      PAD-LEVEL-0:       linalg.copy
//      PAD-LEVEL-0:       linalg.matmul
//      PAD-LEVEL-0:       linalg.copy
//      PAD-LEVEL-0:       memref.dealloc
//      PAD-LEVEL-0:       memref.dealloc
//      PAD-LEVEL-0:       memref.dealloc
//      PAD-LEVEL-0:   }

// -----

func.func @matmul_static(%arg0: tensor<8x16xi32>, %arg1 : tensor<16x8xi32>) -> tensor<8x8xi32> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %5 = tensor.empty() : tensor<8x8xi32>
  %6 = scf.forall (%iv0, %iv1) = (0, 0) to (8, 8) step (8, 8) shared_outs(%arg2 = %5) -> (tensor<8x8xi32>) {
    %extracted_slice = tensor.extract_slice %arg0[%iv0, 0] [8, 16] [1, 1] : tensor<8x16xi32> to tensor<8x16xi32>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, %iv1] [16, 8] [1, 1] : tensor<16x8xi32> to tensor<16x8xi32>
    %extracted_slice_1 = tensor.extract_slice %arg2[%iv0, %iv1] [8, 8] [1, 1] : tensor<8x8xi32> to tensor<8x8xi32>
    %7 = bufferization.alloc_tensor() : tensor<8x16xi32>
    %alloc = memref.alloc() : memref<8x16xi32, 1 : i32>
    %8 = bufferization.to_tensor %alloc restrict writable : memref<8x16xi32, 1 : i32> to tensor<8x16xi32>
    %9 = linalg.copy ins(%extracted_slice : tensor<8x16xi32>) outs(%8 : tensor<8x16xi32>) -> tensor<8x16xi32>
    %10 = bufferization.alloc_tensor() : tensor<16x8xi32>
    %alloc_2 = memref.alloc() : memref<16x8xi32, 1 : i32>
    %11 = bufferization.to_tensor %alloc_2 restrict writable : memref<16x8xi32, 1 : i32> to tensor<16x8xi32>
    %12 = linalg.copy ins(%extracted_slice_0 : tensor<16x8xi32>) outs(%11 : tensor<16x8xi32>) -> tensor<16x8xi32>
    %13 = bufferization.alloc_tensor() : tensor<8x8xi32>
    %alloc_3 = memref.alloc() : memref<8x8xi32, 1 : i32>
    %14 = bufferization.to_tensor %alloc_3 restrict writable : memref<8x8xi32, 1 : i32> to tensor<8x8xi32>
    %17 = scf.forall (%arg3, %arg4) = (0, 0) to (8, 8) step (4, 4) shared_outs(%arg5 = %14) -> (tensor<8x8xi32>) {
      %extracted_slice_4 = tensor.extract_slice %9[%arg3, 0] [4, 16] [1, 1] : tensor<8x16xi32> to tensor<4x16xi32>
      %extracted_slice_5 = tensor.extract_slice %12[0, %arg4] [16, 4] [1, 1] : tensor<16x8xi32> to tensor<16x4xi32>
      %19 = bufferization.to_tensor %alloc_3 restrict writable : memref<8x8xi32, 1 : i32> to tensor<8x8xi32>
      %extracted_slice_6 = tensor.extract_slice %19[%arg3, %arg4] [4, 4] [1, 1] : tensor<8x8xi32> to tensor<4x4xi32>
      %extracted_slice_7 = tensor.extract_slice %arg5[%arg3, %arg4] [4, 4] [1, 1] : tensor<8x8xi32> to tensor<4x4xi32>
      %20 = linalg.fill ins(%c0_i32 : i32) outs(%extracted_slice_7 : tensor<4x4xi32>) -> tensor<4x4xi32>
      %extracted_slice_8 = tensor.extract_slice %arg5[%arg3, %arg4] [4, 4] [1, 1] : tensor<8x8xi32> to tensor<4x4xi32>
      %21 = linalg.matmul ins(%extracted_slice_4, %extracted_slice_5 : tensor<4x16xi32>, tensor<16x4xi32>) outs(%20 : tensor<4x4xi32>) -> tensor<4x4xi32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %21 into %arg5[%arg3, %arg4] [4, 4] [1, 1] : tensor<4x4xi32> into tensor<8x8xi32>
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    %18 = linalg.copy ins(%17 : tensor<8x8xi32>) outs(%extracted_slice_1 : tensor<8x8xi32>) -> tensor<8x8xi32>
    memref.dealloc %alloc : memref<8x16xi32, 1 : i32>
    memref.dealloc %alloc_2 : memref<16x8xi32, 1 : i32>
    memref.dealloc %alloc_3 : memref<8x8xi32, 1 : i32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %18 into %arg2[%iv0, %iv1] [8, 8] [1, 1] : tensor<8x8xi32> into tensor<8x8xi32>
    }
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  return %6 : tensor<8x8xi32>
}
//      PAD-LEVEL-1:   scf.forall
// PAD-LEVEL-1-SAME:   {
//      PAD-LEVEL-1:       linalg.fill
//      PAD-LEVEL-1:       memref.alloc() : memref<8x16xi32, 1 : i32>
//      PAD-LEVEL-1:       bufferization.to_tensor %{{.*}} restrict writable
//      PAD-LEVEL-1:       linalg.copy
//      PAD-LEVEL-1:       memref.alloc() : memref<16x8xi32, 1 : i32>
//      PAD-LEVEL-1:       bufferization.to_tensor %{{.*}} restrict writable
//      PAD-LEVEL-1:       linalg.copy
//      PAD-LEVEL-1:       memref.alloc() : memref<8x8xi32, 1 : i32>
//      PAD-LEVEL-1:       bufferization.to_tensor %{{.*}} restrict writable
//      PAD-LEVEL-1:       scf.forall
// PAD-LEVEL-1-SAME:       {
//      PAD-LEVEL-1:            linalg.fill
//      PAD-LEVEL-1:            bufferization.alloc_tensor
//      PAD-LEVEL-1:            linalg.copy
//      PAD-LEVEL-1:            linalg.matmul
//      PAD-LEVEL-1:            linalg.copy
//      PAD-LEVEL-1:       }
//      PAD-LEVEL-1:       memref.dealloc
//      PAD-LEVEL-1:       memref.dealloc
//      PAD-LEVEL-1:       memref.dealloc
//      PAD-LEVEL-1:   }

// -----

func.func @matmul_static(%arg0 : tensor<8x16xi32>, %arg1 : tensor<16x8xi32>) -> tensor<8x8xi32> {
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %5 = tensor.empty() : tensor<8x8xi32>
    %6 = scf.forall (%iv0, %iv1) = (0, 0) to (8, 8) step (8, 8) shared_outs(%arg2 = %5) -> (tensor<8x8xi32>) {
      %extracted_slice = tensor.extract_slice %arg0[%iv0, 0] [8, 16] [1, 1] : tensor<8x16xi32> to tensor<8x16xi32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %iv1] [16, 8] [1, 1] : tensor<16x8xi32> to tensor<16x8xi32>
      %extracted_slice_1 = tensor.extract_slice %arg2[%iv0, %iv1] [8, 8] [1, 1] : tensor<8x8xi32> to tensor<8x8xi32>
      %7 = bufferization.alloc_tensor() : tensor<8x16xi32>
      %alloc = memref.alloc() : memref<8x16xi32, 1 : i32>
      %8 = bufferization.to_tensor %alloc restrict writable : memref<8x16xi32, 1 : i32> to tensor<8x16xi32>
      %9 = linalg.copy ins(%extracted_slice : tensor<8x16xi32>) outs(%8 : tensor<8x16xi32>) -> tensor<8x16xi32>
      %10 = bufferization.alloc_tensor() : tensor<16x8xi32>
      %alloc_2 = memref.alloc() : memref<16x8xi32, 1 : i32>
      %11 = bufferization.to_tensor %alloc_2 restrict writable : memref<16x8xi32, 1 : i32> to tensor<16x8xi32>
      %12 = linalg.copy ins(%extracted_slice_0 : tensor<16x8xi32>) outs(%11 : tensor<16x8xi32>) -> tensor<16x8xi32>
      %13 = bufferization.alloc_tensor() : tensor<8x8xi32>
      %alloc_3 = memref.alloc() : memref<8x8xi32, 1 : i32>
      %14 = bufferization.to_tensor %alloc_3 restrict writable : memref<8x8xi32, 1 : i32> to tensor<8x8xi32>
      %15 = scf.forall (%arg3, %arg4) = (0, 0) to (8, 8) step (4, 4) shared_outs(%arg5 = %14) -> (tensor<8x8xi32>) {
        %extracted_slice_4 = tensor.extract_slice %9[%arg3, 0] [4, 16] [1, 1] : tensor<8x16xi32> to tensor<4x16xi32>
        %extracted_slice_5 = tensor.extract_slice %12[0, %arg4] [16, 4] [1, 1] : tensor<16x8xi32> to tensor<16x4xi32>
        %extracted_slice_6 = tensor.extract_slice %arg5[%arg3, %arg4] [4, 4] [1, 1] : tensor<8x8xi32> to tensor<4x4xi32>
        %17 = bufferization.alloc_tensor() : tensor<4x4xi32>
        %alloc_7 = memref.alloc() : memref<4x4xi32, 2 : i32>
        %18 = bufferization.to_tensor %alloc_7 restrict writable : memref<4x4xi32, 2 : i32> to tensor<4x4xi32>
        %19 = linalg.fill ins(%c0_i32 : i32) outs(%18 : tensor<4x4xi32>) -> tensor<4x4xi32>
        %20 = scf.for %arg6 = %c0 to %c16 step %c4 iter_args(%arg7 = %19) -> (tensor<4x4xi32>) {
          %extracted_slice_8 = tensor.extract_slice %extracted_slice_4[0, %arg6] [4, 4] [1, 1] : tensor<4x16xi32> to tensor<4x4xi32>
          %extracted_slice_9 = tensor.extract_slice %extracted_slice_5[%arg6, 0] [4, 4] [1, 1] : tensor<16x4xi32> to tensor<4x4xi32>
          %22 = linalg.matmul ins(%extracted_slice_8, %extracted_slice_9 : tensor<4x4xi32>, tensor<4x4xi32>) outs(%arg7 : tensor<4x4xi32>) -> tensor<4x4xi32>
          scf.yield %22 : tensor<4x4xi32>
        }
        %21 = linalg.copy ins(%20 : tensor<4x4xi32>) outs(%extracted_slice_6 : tensor<4x4xi32>) -> tensor<4x4xi32>
        memref.dealloc %alloc_7 : memref<4x4xi32, 2 : i32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %21 into %arg5[%arg3, %arg4] [4, 4] [1, 1] : tensor<4x4xi32> into tensor<8x8xi32>
        }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    %16 = linalg.copy ins(%15 : tensor<8x8xi32>) outs(%extracted_slice_1 : tensor<8x8xi32>) -> tensor<8x8xi32>
    memref.dealloc %alloc : memref<8x16xi32, 1 : i32>
    memref.dealloc %alloc_2 : memref<16x8xi32, 1 : i32>
    memref.dealloc %alloc_3 : memref<8x8xi32, 1 : i32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %16 into %arg2[%iv0, %iv1] [8, 8] [1, 1] : tensor<8x8xi32> into tensor<8x8xi32>
    }
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  return %6 : tensor<8x8xi32>
}
//      PAD-LEVEL-2:   scf.forall
// PAD-LEVEL-2-SAME:   {
//      PAD-LEVEL-2:       memref.alloc() : memref<8x16xi32, 1 : i32>
//      PAD-LEVEL-2:       bufferization.to_tensor %{{.*}} restrict writable
//      PAD-LEVEL-2:       linalg.copy
//      PAD-LEVEL-2:       memref.alloc() : memref<16x8xi32, 1 : i32>
//      PAD-LEVEL-2:       bufferization.to_tensor %{{.*}} restrict writable
//      PAD-LEVEL-2:       linalg.copy
//      PAD-LEVEL-2:       memref.alloc() : memref<8x8xi32, 1 : i32>
//      PAD-LEVEL-2:       bufferization.to_tensor %{{.*}} restrict writable
//      PAD-LEVEL-2:       scf.forall
// PAD-LEVEL-2-SAME:       {
//      PAD-LEVEL-2:            memref.alloc() : memref<4x4xi32, 2 : i32>
//      PAD-LEVEL-2:            bufferization.to_tensor %{{.*}} restrict writable
//      PAD-LEVEL-2:            scf.for
// PAD-LEVEL-2-SAME:            {
//      PAD-LEVEL-2:                bufferization.alloc_tensor
//      PAD-LEVEL-2:                linalg.copy
//      PAD-LEVEL-2:                bufferization.alloc_tensor
//      PAD-LEVEL-2:                linalg.copy
//      PAD-LEVEL-2:                linalg.matmul
//      PAD-LEVEL-2:                linalg.copy
//      PAD-LEVEL-2:            }
//      PAD-LEVEL-2:       }
//      PAD-LEVEL-2:       memref.dealloc
//      PAD-LEVEL-2:       memref.dealloc
//      PAD-LEVEL-2:       memref.dealloc
//      PAD-LEVEL-2:   }
