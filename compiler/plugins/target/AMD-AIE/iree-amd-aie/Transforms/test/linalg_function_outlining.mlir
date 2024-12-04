// RUN: iree-opt --split-input-file --iree-amdaie-linalg-function-outlining --verify-diagnostics --split-input-file %s | FileCheck %s

// Test demonstrating multiple Matmul using different SSAs.

// CHECK-LABEL: func.func private @generic_matmul_0_outlined
// CHECK-SAME:  (%[[LHS:.*]]: memref<4x8xbf16>,
// CHECK-SAME:   %[[RHS:.*]]: memref<8x4xbf16>,
// CHECK-SAME:   %[[OUT:.*]]: memref<4x4xf32>) {
// CHECK:           linalg.generic
// CHECK-SAME:          ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:          outs(%[[OUT]] :
// CHECK:           return
// CHECK:        }
// CHECK-LABEL:  func.func @matmul_example
// CHECK-SAME:   (%[[A:.*]]: memref<4x8xbf16>,
// CHECK-SAME:    %[[B:.*]]: memref<8x4xbf16>,
// CHECK-SAME:    %[[C:.*]]: memref<4x4xf32>) {
// CHECK:            amdaie.core
// CHECK:               func.call @generic_matmul_0_outlined(%[[A]], %[[B]], %[[C]])
// CHECK-NOT:           linalg.generic
// CHECK:               amdaie.end
// CHECK:            }
// CHECK:            amdaie.core
// CHECK:               func.call @generic_matmul_0_outlined(%[[A]], %[[B]], %[[C]])
// CHECK-NOT:           linalg.generic
// CHECK:               amdaie.end
// CHECK:            }
// CHECK:            return
// CHECK:        }
func.func @matmul_example(%A: memref<4x8xbf16>, %B: memref<8x4xbf16>, %C: memref<4x4xf32>) {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %tile = amdaie.tile(%c1, %c2)
  %0 = amdaie.core(%tile, in : [], out : []) {
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%A, %B : memref<4x8xbf16>, memref<8x4xbf16>)
      outs(%C : memref<4x4xf32>) {
    ^bb0(%in: bf16, %in_17: bf16, %out: f32):
      %1 = arith.extf %in : bf16 to f32
      %2 = arith.extf %in_17 : bf16 to f32
      %3 = arith.mulf %1, %2 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4  : f32
    }
    amdaie.end
  }
  %1 = amdaie.core(%tile, in : [], out : []) {
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%A, %B : memref<4x8xbf16>, memref<8x4xbf16>)
      outs(%C : memref<4x4xf32>) {
    ^bb0(%in: bf16, %in_17: bf16, %out: f32):
      %2 = arith.extf %in : bf16 to f32
      %3 = arith.extf %in_17 : bf16 to f32
      %4 = arith.mulf %2, %3 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5  : f32
    }
    amdaie.end
  }
  return
}

// -----


// Test demonstrating different kind of matmul operations being mapped to a
// unique corresponding outlined function.

// CHECK-DAG:  func.func private @[[MATMUL_K6:.*]]({{.*}}memref<4x6xbf16>, {{.*}}memref<6x4xbf16>, {{.*}}memref<4x4xf32>)
// CHECK-DAG:  func.func private @[[MATMUL_K4:.*]]({{.*}}memref<4x4xbf16>, {{.*}}memref<4x4xbf16>, {{.*}}memref<4x4xf32>)
// CHECK-NOT:  func.func private
// CHECK:      func.func @matmul_example_2(
// CHECK-SAME:  %[[A0:.*]]: memref<4x4xbf16>, %[[B0:.*]]: memref<4x4xbf16>,
// CHECK-SAME:  %[[A1:.*]]: memref<4x6xbf16>, %[[B1:.*]]: memref<6x4xbf16>,
// CHECK-SAME:   %[[C:.*]]: memref<4x4xf32>) {
// CHECK:      amdaie.core
// CHECK-NEXT: func.call @[[MATMUL_K4]](%[[A0]], %[[B0]], %[[C]])
// CHECK-NEXT: func.call @[[MATMUL_K4]](%[[A0]], %[[B0]], %[[C]])
// CHECK-NEXT: func.call @[[MATMUL_K4]](%[[B0]], %[[A0]], %[[C]])
// CHECK-NEXT: func.call @[[MATMUL_K6]](%[[A1]], %[[B1]], %[[C]])
// CHECK-NEXT: amdaie.end
// CHECK:      return
func.func @matmul_example_2(%A0: memref<4x4xbf16>, %B0: memref<4x4xbf16>,
                            %A1: memref<4x6xbf16>, %B1: memref<6x4xbf16>,
                            %C: memref<4x4xf32>) {
  %c2 = arith.constant 2 : index
  %tile = amdaie.tile(%c2, %c2)
  %0 = amdaie.core(%tile, in : [], out : []) {
    linalg.matmul ins(%A0, %B0 : memref<4x4xbf16>, memref<4x4xbf16>)
                  outs(%C : memref<4x4xf32>)
    linalg.matmul ins(%A0, %B0 : memref<4x4xbf16>, memref<4x4xbf16>)
                  outs(%C : memref<4x4xf32>)
    linalg.matmul ins(%B0, %A0 : memref<4x4xbf16>, memref<4x4xbf16>)
                  outs(%C : memref<4x4xf32>)
    linalg.matmul ins(%A1, %B1 : memref<4x6xbf16>, memref<6x4xbf16>)
                  outs(%C : memref<4x4xf32>)
    amdaie.end
  }
  return
}

// -----

// CHECK-LABEL: @linalg_fill_copy
func.func @linalg_fill_copy(%A: memref<4xf32>, %B: memref<4xf32>) {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %tile = amdaie.tile(%c1, %c2)
  %0 = amdaie.core(%tile, in : [], out : []) {
    // CHECK:     linalg.fill
    // CHECK-NOT: func.call @fill_elementwise_0_outlined
    // CHECK:     linalg.copy
    // CHECK-NOT: func.call @copy_elementwise_1_outlined
    linalg.fill ins(%cst : f32) outs(%A : memref<4xf32>)
    linalg.copy ins(%A : memref<4xf32>) outs(%B : memref<4xf32>)
    amdaie.end
  }
  return
}

// -----

// Test demonstrating the outlining of a linalg.generic operation other than
// a matmul or elementwise operation. Specifically, one which has not been
// 'blacklisted' like linalg.copy has (see test linalg_fill_copy above).
// CHECK:       func.func private @generic_0_outlined
// CHECK-SAME:    memref<4xbf16>,
// CHECK-SAME:    memref<bf16>
// CHECK:       linalg.generic
// CHECK-SAME:    iterator_types = ["reduction"]
// CHECK:       return
// CHECK:       func.func @reduction
// CHECK-SAME:    memref<4xbf16>
// CHECK-SAME:    memref<bf16>
// CHECK:       func.call @generic_0_outlined
// CHECK-SAME:    (memref<4xbf16>, memref<bf16>) -> ()
// CHECK:       return
func.func @reduction(%A: memref<4xbf16>, %B: memref<bf16>) {
  %c2 = arith.constant 2 : index
  %tile = amdaie.tile(%c2, %c2)
  %1 = amdaie.core(%tile, in : [], out : []) {
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    } ins(%A: memref<4xbf16>) outs(%B : memref<bf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    }
    amdaie.end
  }
  return
}


// -----

// Test demonstrating the outlining of a linalg.generic where one
// operand has an unkown offset. The memref is still contiguous, however.
// CHECK:       func.func private @generic_0_outlined
// CHECK-SAME:    memref<4x8xbf16>,
// CHECK-SAME:    memref<bf16>
// CHECK:       linalg.generic
// CHECK-SAME:    iterator_types = ["reduction", "reduction"]
// CHECK:       return
// CHECK:       func.func @supported_linalg_op
// CHECK-SAME:    memref<4x8xbf16, strided<[8, 1], offset: ?>>
// CHECK-SAME:    memref<bf16>
// CHECK:       %[[CAST:.*]] = memref.cast
// CHECK-SAME:    memref<4x8xbf16, strided<[8, 1], offset: ?>>
// CHECK-SAME:    to memref<4x8xbf16>
// CHECK:       func.call @generic_0_outlined(%[[CAST]], %arg1) :
// CHECK-SAME:    (memref<4x8xbf16>, memref<bf16>) -> ()
// CHECK:       return
func.func @supported_linalg_op(%A: memref<4x8xbf16, strided<[8,1], offset:?>>, %B: memref<bf16>) {
  %c2 = arith.constant 2 : index
  %tile = amdaie.tile(%c2, %c2)
  %1 = amdaie.core(%tile, in : [], out : []) {
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> ()>],
      iterator_types = ["reduction", "reduction"]
    } ins(%A: memref<4x8xbf16, strided<[8,1], offset:?>>) outs(%B : memref<bf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    }
    amdaie.end
  }
  return
}


// -----

// Test illustrating the that when a linalg.generic operation has an
// operand that is not contiguous, it is not outlined.

// CHECK-COUNT-1: func.func
// CHECK-NOT:     func.func
func.func @unsupported_linalg_op(%A: memref<4x8xbf16, strided<[9,1]>>, %B: memref<bf16>) {
  %c2 = arith.constant 2 : index
  %tile = amdaie.tile(%c2, %c2)
  %1 = amdaie.core(%tile, in : [], out : []) {
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> ()>],
      iterator_types = ["reduction", "reduction"]
    } ins(%A: memref<4x8xbf16, strided<[9,1]>>) outs(%B : memref<bf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    }
    amdaie.end
  }
  return
}

