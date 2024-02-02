// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-pad{padding-level=0}))' --split-input-file %s | FileCheck %s --check-prefix=PAD-LEVEL-0
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-pad{padding-level=1}))' --split-input-file %s | FileCheck %s --check-prefix=PAD-LEVEL-1
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-pad{padding-level=2}))' --split-input-file %s | FileCheck %s --check-prefix=PAD-LEVEL-2

func.func @matmul_static(%arg0: tensor<8x16xi32>, %arg1: tensor<16x8xi32>) -> tensor<8x8xi32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<8x8xi32>
  %1 = scf.forall (%arg2, %arg3) = (0, 0) to (8, 8) step (8, 8) shared_outs(%arg4 = %0) -> (tensor<8x8xi32>) {
    %extracted_slice = tensor.extract_slice %arg0[%arg2, 0] [8, 16] [1, 1] : tensor<8x16xi32> to tensor<8x16xi32>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg3] [16, 8] [1, 1] : tensor<16x8xi32> to tensor<16x8xi32>
    %extracted_slice_1 = tensor.extract_slice %arg4[%arg2, %arg3] [8, 8] [1, 1] : tensor<8x8xi32> to tensor<8x8xi32>
    %2 = linalg.fill ins(%c0_i32 : i32) outs(%extracted_slice_1 : tensor<8x8xi32>) -> tensor<8x8xi32>
    %3 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<8x16xi32>, tensor<16x8xi32>) outs(%2 : tensor<8x8xi32>) -> tensor<8x8xi32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %3 into %arg4[%arg2, %arg3] [8, 8] [1, 1] : tensor<8x8xi32> into tensor<8x8xi32>
    }
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  return %1 : tensor<8x8xi32>
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

func.func @matmul_static(%arg0: tensor<8x16xi32>, %arg1: tensor<16x8xi32>) -> tensor<8x8xi32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<8x8xi32>
  %1 = scf.forall (%arg2, %arg3) = (0, 0) to (8, 8) step (8, 8) shared_outs(%arg4 = %0) -> (tensor<8x8xi32>) {
    %extracted_slice = tensor.extract_slice %arg0[%arg2, 0] [8, 16] [1, 1] : tensor<8x16xi32> to tensor<8x16xi32>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg3] [16, 8] [1, 1] : tensor<16x8xi32> to tensor<16x8xi32>
    %extracted_slice_1 = tensor.extract_slice %arg4[%arg2, %arg3] [8, 8] [1, 1] : tensor<8x8xi32> to tensor<8x8xi32>
    %2 = bufferization.alloc_tensor() : tensor<8x16xi32>
    %alloc = memref.alloc() : memref<8x16xi32, 1 : i32>
    %3 = bufferization.to_tensor %alloc restrict writable : memref<8x16xi32, 1 : i32>
    %4 = linalg.copy ins(%extracted_slice : tensor<8x16xi32>) outs(%3 : tensor<8x16xi32>) -> tensor<8x16xi32>
    %5 = bufferization.alloc_tensor() : tensor<16x8xi32>
    %alloc_2 = memref.alloc() : memref<16x8xi32, 1 : i32>
    %6 = bufferization.to_tensor %alloc_2 restrict writable : memref<16x8xi32, 1 : i32>
    %7 = linalg.copy ins(%extracted_slice_0 : tensor<16x8xi32>) outs(%6 : tensor<16x8xi32>) -> tensor<16x8xi32>
    %8 = bufferization.alloc_tensor() : tensor<8x8xi32>
    %alloc_3 = memref.alloc() : memref<8x8xi32, 1 : i32>
    %9 = bufferization.to_tensor %alloc_3 restrict writable : memref<8x8xi32, 1 : i32>
    %10 = scf.forall (%arg5, %arg6) = (0, 0) to (8, 8) step (4, 4) shared_outs(%arg7 = %9) -> (tensor<8x8xi32>) {
      %extracted_slice_4 = tensor.extract_slice %4[%arg5, 0] [4, 16] [1, 1] : tensor<8x16xi32> to tensor<4x16xi32>
      %extracted_slice_5 = tensor.extract_slice %7[0, %arg6] [16, 4] [1, 1] : tensor<16x8xi32> to tensor<16x4xi32>
      %extracted_slice_6 = tensor.extract_slice %arg7[%arg5, %arg6] [4, 4] [1, 1] : tensor<8x8xi32> to tensor<4x4xi32>
      %12 = linalg.fill ins(%c0_i32 : i32) outs(%extracted_slice_6 : tensor<4x4xi32>) -> tensor<4x4xi32>
      %13 = linalg.matmul ins(%extracted_slice_4, %extracted_slice_5 : tensor<4x16xi32>, tensor<16x4xi32>) outs(%12 : tensor<4x4xi32>) -> tensor<4x4xi32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %13 into %arg7[%arg5, %arg6] [4, 4] [1, 1] : tensor<4x4xi32> into tensor<8x8xi32>
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    %11 = linalg.copy ins(%10 : tensor<8x8xi32>) outs(%extracted_slice_1 : tensor<8x8xi32>) -> tensor<8x8xi32>
    memref.dealloc %alloc : memref<8x16xi32, 1 : i32>
    memref.dealloc %alloc_2 : memref<16x8xi32, 1 : i32>
    memref.dealloc %alloc_3 : memref<8x8xi32, 1 : i32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %11 into %arg4[%arg2, %arg3] [8, 8] [1, 1] : tensor<8x8xi32> into tensor<8x8xi32>
    }
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  return %1 : tensor<8x8xi32>
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

func.func @matmul_static(%arg0: tensor<8x16xi32>, %arg1: tensor<16x8xi32>) -> tensor<8x8xi32> {
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<8x8xi32>
    %1 = scf.forall (%arg2, %arg3) = (0, 0) to (8, 8) step (8, 8) shared_outs(%arg4 = %0) -> (tensor<8x8xi32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg2, 0] [8, 16] [1, 1] : tensor<8x16xi32> to tensor<8x16xi32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg3] [16, 8] [1, 1] : tensor<16x8xi32> to tensor<16x8xi32>
      %extracted_slice_1 = tensor.extract_slice %arg4[%arg2, %arg3] [8, 8] [1, 1] : tensor<8x8xi32> to tensor<8x8xi32>
      %2 = bufferization.alloc_tensor() : tensor<8x16xi32>
      %alloc = memref.alloc() : memref<8x16xi32, 1 : i32>
      %3 = bufferization.to_tensor %alloc restrict writable : memref<8x16xi32, 1 : i32>
      %4 = linalg.copy ins(%extracted_slice : tensor<8x16xi32>) outs(%3 : tensor<8x16xi32>) -> tensor<8x16xi32>
      %5 = bufferization.alloc_tensor() : tensor<16x8xi32>
      %alloc_2 = memref.alloc() : memref<16x8xi32, 1 : i32>
      %6 = bufferization.to_tensor %alloc_2 restrict writable : memref<16x8xi32, 1 : i32>
      %7 = linalg.copy ins(%extracted_slice_0 : tensor<16x8xi32>) outs(%6 : tensor<16x8xi32>) -> tensor<16x8xi32>
      %8 = bufferization.alloc_tensor() : tensor<8x8xi32>
      %alloc_3 = memref.alloc() : memref<8x8xi32, 1 : i32>
      %9 = bufferization.to_tensor %alloc_3 restrict writable : memref<8x8xi32, 1 : i32>
      %10 = scf.forall (%arg5, %arg6) = (0, 0) to (8, 8) step (4, 4) shared_outs(%arg7 = %9) -> (tensor<8x8xi32>) {
        %extracted_slice_4 = tensor.extract_slice %4[%arg5, 0] [4, 16] [1, 1] : tensor<8x16xi32> to tensor<4x16xi32>
        %extracted_slice_5 = tensor.extract_slice %7[0, %arg6] [16, 4] [1, 1] : tensor<16x8xi32> to tensor<16x4xi32>
        %extracted_slice_6 = tensor.extract_slice %arg7[%arg5, %arg6] [4, 4] [1, 1] : tensor<8x8xi32> to tensor<4x4xi32>
        %12 = bufferization.alloc_tensor() : tensor<4x4xi32>
        %alloc_7 = memref.alloc() : memref<4x4xi32, 2 : i32>
        %13 = bufferization.to_tensor %alloc_7 restrict writable : memref<4x4xi32, 2 : i32>
        %14 = linalg.fill ins(%c0_i32 : i32) outs(%13 : tensor<4x4xi32>) -> tensor<4x4xi32>
        %15 = scf.for %arg8 = %c0 to %c16 step %c4 iter_args(%arg9 = %14) -> (tensor<4x4xi32>) {
          %extracted_slice_8 = tensor.extract_slice %extracted_slice_4[0, %arg8] [4, 4] [1, 1] : tensor<4x16xi32> to tensor<4x4xi32>
          %extracted_slice_9 = tensor.extract_slice %extracted_slice_5[%arg8, 0] [4, 4] [1, 1] : tensor<16x4xi32> to tensor<4x4xi32>
          %17 = linalg.matmul ins(%extracted_slice_8, %extracted_slice_9 : tensor<4x4xi32>, tensor<4x4xi32>) outs(%arg9 : tensor<4x4xi32>) -> tensor<4x4xi32>
          scf.yield %17 : tensor<4x4xi32>
        }
        %16 = linalg.copy ins(%15 : tensor<4x4xi32>) outs(%extracted_slice_6 : tensor<4x4xi32>) -> tensor<4x4xi32>
        memref.dealloc %alloc_7 : memref<4x4xi32, 2 : i32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %16 into %arg7[%arg5, %arg6] [4, 4] [1, 1] : tensor<4x4xi32> into tensor<8x8xi32>
        }
      } {mapping = [#gpu.block<y>, #gpu.block<x>]}
      %11 = linalg.copy ins(%10 : tensor<8x8xi32>) outs(%extracted_slice_1 : tensor<8x8xi32>) -> tensor<8x8xi32>
      memref.dealloc %alloc : memref<8x16xi32, 1 : i32>
      memref.dealloc %alloc_2 : memref<16x8xi32, 1 : i32>
      memref.dealloc %alloc_3 : memref<8x8xi32, 1 : i32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %11 into %arg4[%arg2, %arg3] [8, 8] [1, 1] : tensor<8x8xi32> into tensor<8x8xi32>
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    return %1 : tensor<8x8xi32>
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
