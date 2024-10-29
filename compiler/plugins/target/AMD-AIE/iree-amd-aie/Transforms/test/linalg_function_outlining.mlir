// RUN: iree-opt --split-input-file --iree-amdaie-linalg-function-outlining --verify-diagnostics --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func private @generic_matmul_outlined
// CHECK-SAME:  (%[[LHS:.*]]: memref<1x1x4x8x4x8xbf16>,
// CHECK-SAME:   %[[RHS:.*]]: memref<1x1x8x4x8x4xbf16>,
// CHECK-SAME:   %[[OUT:.*]]: memref<1x1x8x8x4x4xf32>) {
// CHECK:           linalg.generic
// CHECK-SAME:          ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:          outs(%[[OUT]] :
// CHECK:           return
// CHECK:        }
// CHECK-LABEL:  func.func @matmul_example
// CHECK-SAME:   (%[[A:.*]]: memref<1x1x4x8x4x8xbf16>,
// CHECK-SAME:    %[[B:.*]]: memref<1x1x8x4x8x4xbf16>,
// CHECK-SAME:    %[[C:.*]]: memref<1x1x8x8x4x4xf32>) {
// CHECK:            amdaie.core
// CHECK:               func.call @generic_matmul_outlined(%[[A]], %[[B]], %[[C]])
// CHECK-NOT:           linalg.generic
// CHECK:               amdaie.end
// CHECK:            }
// CHECK:            return
// CHECK:        }
func.func @matmul_example(%A: memref<1x1x4x8x4x8xbf16>, %B: memref<1x1x8x4x8x4xbf16>, %C: memref<1x1x8x8x4x4xf32>) {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %tile = amdaie.tile(%c1, %c2)
  %0 = amdaie.core(%tile, in : [], out : []) {
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>,
                       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>,
                       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
                      ],
      iterator_types = ["parallel", "parallel", "reduction",
                        "parallel", "parallel", "reduction",
                        "parallel", "parallel", "reduction"
                       ]
    } ins(%A, %B : memref<1x1x4x8x4x8xbf16>, memref<1x1x8x4x8x4xbf16>)
      outs(%C : memref<1x1x8x8x4x4xf32>) {
    ^bb0(%in: bf16, %in_17: bf16, %out: f32):
      %1 = arith.extf %in : bf16 to f32
      %2 = arith.extf %in_17 : bf16 to f32
      %3 = arith.mulf %1, %2 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4  : f32
    }
    amdaie.end
  }
  return
}

// -----

// CHECK-LABEL: func.func private @generic_elementwise_outlined
// CHECK-SAME:  (%[[INPUT:.*]]: memref<1x1x8x8x4x4xf32>,
// CHECK-SAME:   %[[OUTPUT:.*]]: memref<1x1x8x8x4x4xbf16>) {
// CHECK:           linalg.generic
// CHECK-SAME:          ins(%[[INPUT]] :
// CHECK-SAME:          outs(%[[OUTPUT]] :
// CHECK:           return
// CHECK:        }
// CHECK-LABEL:  func.func @elemwise_example
// CHECK-SAME:   (%[[A:.*]]: memref<1x1x8x8x4x4xf32>,
// CHECK-SAME:    %[[C:.*]]: memref<1x1x8x8x4x4xbf16>) {
// CHECK:            amdaie.core
// CHECK:               func.call @generic_elementwise_outlined(%[[A]], %[[C]])
// CHECK-NOT:           linalg.generic
// CHECK:               amdaie.end
// CHECK:            }
// CHECK:            return
// CHECK:        }
func.func @elemwise_example(%A: memref<1x1x8x8x4x4xf32>, %C: memref<1x1x8x8x4x4xbf16>) {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %tile = amdaie.tile(%c1, %c2)
  %0 = amdaie.core(%tile, in : [], out : []) {
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>,
                       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>],
      iterator_types = ["parallel", "parallel", "parallel",
                        "parallel", "parallel", "parallel"
                       ]
    } ins(%A : memref<1x1x8x8x4x4xf32>)
      outs(%C : memref<1x1x8x8x4x4xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %1 = arith.truncf %in : f32 to bf16
      linalg.yield %1 : bf16
    }
    amdaie.end
  }
  return
}
