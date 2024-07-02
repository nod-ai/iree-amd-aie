// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-amdaie-create-reference-to-allocation))" %s | FileCheck %s

// CHECK-LABEL: func.func @single_alloc
// CHECK:  %[[ALLOC:.+]] = memref.alloc() : memref<8x16xi32, 2 : i32>
// CHECK:  %[[REFER:.+]] = amdaie.reference_to %[[ALLOC]] : memref<8x16xi32, 2 : i32> -> memref<8x16xi32, 2 : i32>
// CHECK:  %[[USER:.+]] = bufferization.to_tensor %[[REFER]] restrict writable : memref<8x16xi32, 2 : i32>
// CHECK:  linalg.copy
// CHECK:  memref.dealloc %[[ALLOC]] : memref<8x16xi32, 2 : i32>
func.func @single_alloc(%arg0: tensor<8x16xi32>) -> tensor<8x16xi32> {
  %alloc = memref.alloc() : memref<8x16xi32, 2 : i32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<8x16xi32, 2 : i32>
  %1 = linalg.copy ins(%arg0 : tensor<8x16xi32>) outs(%0 : tensor<8x16xi32>) -> tensor<8x16xi32>
  memref.dealloc %alloc : memref<8x16xi32, 2 : i32>
  return %1 : tensor<8x16xi32>
}

// -----

// CHECK-LABEL: func.func @multiple_alloc
// CHECK:  %[[ALLOC0:.+]] = memref.alloc() : memref<8x16xi32, 2 : i32>
// CHECK:  %[[REFER0:.+]] = amdaie.reference_to %[[ALLOC0]]
// CHECK:  %[[USER0:.+]] = bufferization.to_tensor %[[REFER0]]
// CHECK:  linalg.copy
// CHECK:  %[[ALLOC1:.+]] = memref.alloc() : memref<16x8xi32, 2 : i32>
// CHECK:  %[[REFER1:.+]] = amdaie.reference_to %[[ALLOC1]]
// CHECK:  %[[USER1:.+]] = bufferization.to_tensor %[[REFER1]]
// CHECK:  linalg.copy
// CHECK:  %[[ALLOC2:.+]] = memref.alloc() : memref<8x8xi32, 2 : i32>
// CHECK:  %[[REFER2:.+]] = amdaie.reference_to %[[ALLOC2]]
// CHECK:  %[[USER2:.+]] = bufferization.to_tensor %[[REFER2]]
// CHECK:  linalg.fill
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
  %2 = bufferization.to_tensor %alloc restrict writable : memref<8x16xi32, 2 : i32>
  %3 = linalg.copy ins(%arg0 : tensor<8x16xi32>) outs(%2 : tensor<8x16xi32>) -> tensor<8x16xi32>
  %4 = bufferization.alloc_tensor() : tensor<16x8xi32>
  %alloc_0 = memref.alloc() : memref<16x8xi32, 2 : i32>
  %5 = bufferization.to_tensor %alloc_0 restrict writable : memref<16x8xi32, 2 : i32>
  %6 = linalg.copy ins(%arg1 : tensor<16x8xi32>) outs(%5 : tensor<16x8xi32>) -> tensor<16x8xi32>
  %7 = bufferization.alloc_tensor() : tensor<8x8xi32>
  %alloc_1 = memref.alloc() : memref<8x8xi32, 2 : i32>
  %8 = bufferization.to_tensor %alloc_1 restrict writable : memref<8x8xi32, 2 : i32>
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
  %0 = bufferization.to_tensor %alloc restrict writable : memref<8x16xi32, 1 : i32>
  %1 = linalg.copy ins(%arg0 : tensor<8x16xi32>) outs(%0 : tensor<8x16xi32>) -> tensor<8x16xi32>
  memref.dealloc %alloc : memref<8x16xi32, 1 : i32>
  return %1 : tensor<8x16xi32>
}
