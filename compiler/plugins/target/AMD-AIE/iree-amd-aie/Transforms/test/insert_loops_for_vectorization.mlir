// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-insert-loops-for-vectorization))" --split-input-file %s | FileCheck %s

!t2_bf16 = tensor<64x64xbf16>
!t3_bf16 = tensor<64x64x64xbf16>
!t4_bf16 = tensor<64x64x64x64xbf16>

!t2_f32  = tensor<64x64xf32>
!t3_f32  = tensor<64x64x64xf32>
!t4_f32  = tensor<64x64x64x64xf32>


#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
   // A generic that corresponds to a simple matmul (2 rank-2 operands)
   // does NOT get tiled.
   // CHECK-LABEL: vanilla
   // CHECK-NOT:      scf.for
  func.func @vanilla(%arg0: !t2_bf16, %arg1: !t2_bf16, %arg2: !t2_f32) -> !t2_f32 {
    %0 = linalg.generic {indexing_maps =
                          [
                           affine_map<(d0, d1, d2) -> (d0, d2)>,
                           affine_map<(d0, d1, d2) -> (d2, d1)>,
                           affine_map<(d0, d1, d2) -> (d0, d1)>
                          ],
                         iterator_types = ["parallel", "parallel", "reduction"]}
                         ins(%arg0, %arg1 : !t2_bf16, !t2_bf16) outs(%arg2 : !t2_f32) {
    ^bb0(%in_0_bf16: bf16, %in_1_bf16: bf16, %out: f32):
      %in_0 = arith.extf %in_0_bf16: bf16 to f32
      %in_1 = arith.extf %in_1_bf16: bf16 to f32
      %1 = arith.mulf %in_0, %in_1 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> !t2_f32
    return %0 : !t2_f32
  }

  // A batched matmul gets the batch dimension converted to a single scf.for
  // CHECK-LABEL: batched0
  // CHECK:         scf.for
  // CHECK:           linalg.generic
  // CHECK-SAME:        iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  // CHECK-SAME:        ins
  // CHECK-SAME:        tensor<1x64x64xbf16>, tensor<1x64x64xbf16>
  // CHECK-SAME:        outs
  // CHECK-SAME:        tensor<1x64x64xf32>
  // CHECK-NOT:     scf.for
  func.func @batched0(%arg0: !t3_bf16, %arg1: !t3_bf16, %arg2: !t3_f32) -> !t3_f32 {
    %0 = linalg.generic {indexing_maps =
                          [
                           affine_map<(b0, d0, d1, d2) -> (b0, d0, d2)>,
                           affine_map<(b0, d0, d1, d2) -> (b0, d2, d1)>,
                           affine_map<(b0, d0, d1, d2) -> (b0, d0, d1)>
                          ],
                         iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
                         ins(%arg0, %arg1 : !t3_bf16, !t3_bf16) outs(%arg2 : !t3_f32) {
    ^bb0(%in_0_bf16: bf16, %in_1_bf16: bf16, %out: f32):
      %in_0 = arith.extf %in_0_bf16: bf16 to f32
      %in_1 = arith.extf %in_1_bf16: bf16 to f32
      %1 = arith.mulf %in_0, %in_1 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> !t3_f32
    return %0 : !t3_f32
  }

  // A batched matmul where the element types are not supported for
  // vectorization on AIE, does not get tiles:
  // CHECK-LABEL: batched_bad_element_types
  // CHECK-NOT:     scf.for
  func.func @batched_bad_element_types(%arg0: !t3_f32, %arg1: !t3_f32, %arg2: !t3_f32) -> !t3_f32 {
    %0 = linalg.generic {indexing_maps =
                          [
                           affine_map<(b0, d0, d1, d2) -> (b0, d0, d2)>,
                           affine_map<(b0, d0, d1, d2) -> (b0, d2, d1)>,
                           affine_map<(b0, d0, d1, d2) -> (b0, d0, d1)>
                          ],
                         iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
                         ins(%arg0, %arg1 : !t3_f32, !t3_f32) outs(%arg2 : !t3_f32) {
    ^bb0(%in_0: f32, %in_1: f32, %out: f32):
      %1 = arith.mulf %in_0, %in_1 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> !t3_f32
    return %0 : !t3_f32
  }

  // A test like the above, but with a matmul_tranpose_b instead of a matmul
  // CHECK-LABEL: batched_transpose_b
  // CHECK:         scf.for
  // CHECK:           linalg.generic
  // CHECK-SAME:        iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  // CHECK-SAME:        ins
  // CHECK-SAME:        tensor<1x64x64xbf16>, tensor<1x64x64xbf16>
  // CHECK-SAME:        outs
  // CHECK-SAME:        tensor<1x64x64xf32>
  // CHECK-NOT:     scf.for
  func.func @batched_transpose_b(%arg0: !t3_bf16, %arg1: !t3_bf16, %arg2: !t3_f32) -> !t3_f32 {
    %0 = linalg.generic {indexing_maps =
                          [
                           affine_map<(b0, d0, d1, d2) -> (b0, d0, d2)>,
                           affine_map<(b0, d0, d1, d2) -> (b0, d1, d2)>,
                           affine_map<(b0, d0, d1, d2) -> (b0, d0, d1)>
                          ],
                         iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
                         ins(%arg0, %arg1 : !t3_bf16, !t3_bf16) outs(%arg2 : !t3_f32) {
    ^bb0(%in_0_bf16: bf16, %in_1_bf16: bf16, %out: f32):
      %in_0 = arith.extf %in_0_bf16: bf16 to f32
      %in_1 = arith.extf %in_1_bf16: bf16 to f32
      %1 = arith.mulf %in_0, %in_1 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> !t3_f32
    return %0 : !t3_f32
  }

  // Another test with a transposed matmul, but in this case A is transposed.
  // Currently the pass does not modify this (so no scf.for should appear).
  // We'll probably add support for this in the future, but for now we don't.
  // CHECK-LABEL: batched_transpose_a
  // CHECK-NOT:      scf.for

  func.func @batched_transpose_a(%arg0: !t3_bf16, %arg1: !t3_bf16, %arg2: !t3_f32) -> !t3_f32 {
    %0 = linalg.generic {indexing_maps =
                          [
                           affine_map<(b0, d0, d1, d2) -> (b0, d2, d1)>,
                           affine_map<(b0, d0, d1, d2) -> (b0, d2, d0)>,
                           affine_map<(b0, d0, d1, d2) -> (b0, d0, d1)>
                          ],
                         iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
                         ins(%arg0, %arg1 : !t3_bf16, !t3_bf16) outs(%arg2 : !t3_f32) {
    ^bb0(%in_0_bf16: bf16, %in_1_bf16: bf16, %out: f32):
      %in_0 = arith.extf %in_0_bf16: bf16 to f32
      %in_1 = arith.extf %in_1_bf16: bf16 to f32
      %1 = arith.mulf %in_0, %in_1 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> !t3_f32
    return %0 : !t3_f32
  }

  // Check that the final 3 dimensions do have the pattern of a matmul (or matmul transpose)
  // CHECK-LABEL: batched1
  // CHECK-NOT:     scf.for
  func.func @batched1(%arg0: !t3_bf16, %arg1: !t3_bf16, %arg2: !t3_f32) -> !t3_f32 {
    %0 = linalg.generic {indexing_maps =
                          [
                           // This is like a matmul but the first operand is
                           // transposed:
                           affine_map<(b0, d0, d1, d2) -> (b0, d2, d0)>,
                           affine_map<(b0, d0, d1, d2) -> (b0, d2, d1)>,
                           affine_map<(b0, d0, d1, d2) -> (b0, d0, d1)>
                          ],
                         iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
                         ins(%arg0, %arg1 : !t3_bf16, !t3_bf16) outs(%arg2 : !t3_f32) {
    ^bb0(%in_0_bf16: bf16, %in_1_bf16: bf16, %out: f32):
      %in_0 = arith.extf %in_0_bf16: bf16 to f32
      %in_1 = arith.extf %in_1_bf16: bf16 to f32
      %1 = arith.mulf %in_0, %in_1 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> !t3_f32
    return %0 : !t3_f32
  }

  // Check for a batched matmul where operand 0 is broadcast:
  // CHECK-LABEL: batched2
  // CHECK:         scf.for
  // CHECK:             linalg.generic
  // CHECK-SAME:            iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  // CHECK-SAME:            ins
  // CHECK-SAME:            tensor<64x64xbf16>, tensor<1x64x64xbf16>
  // CHECK-SAME:            outs
  // CHECK-SAME:            tensor<1x64x64xf32>
  // CHECK-NOT:     scf.for
  func.func @batched2(%arg0: !t2_bf16, %arg1: !t3_bf16, %arg2: !t3_f32) -> !t3_f32 {
    %0 = linalg.generic {indexing_maps =
                          [
                           affine_map<(b0, d0, d1, d2) -> (d0, d2)>,
                           affine_map<(b0, d0, d1, d2) -> (b0, d2, d1)>,
                           affine_map<(b0, d0, d1, d2) -> (b0, d0, d1)>
                          ],
                         iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
                         ins(%arg0, %arg1 : !t2_bf16, !t3_bf16) outs(%arg2 : !t3_f32) {
    ^bb0(%in_0_bf16: bf16, %in_1_bf16: bf16, %out: f32):
      %in_0 = arith.extf %in_0_bf16: bf16 to f32
      %in_1 = arith.extf %in_1_bf16: bf16 to f32
      %1 = arith.mulf %in_0, %in_1 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> !t3_f32
    return %0 : !t3_f32
  }

  // A function which arises from the pack-based pipeline in iree-amd-aie,
  // we verify it gets tiled to 6 nested for loops (3 of which get canonicalized
  // away as they have loop count 1) and an inner linalg.generic for
  // matmul of size LHS: 4x8 and RHS: 8x4.

  // CHECK-LABEL: packBasedPipeline
  // CHECK-COUNT-3: scf.for
  // CHECK-NOT:     scf.for
  // CHECK:           linalg.generic
  // CHECK-SAME:            tensor<1x1x4x8xbf16>
  // CHECK-SAME:            tensor<1x1x8x4xbf16>
  // CHECK-SAME:            tensor<1x1x4x4xf32>

  func.func @packBasedPipeline(
               %arg0: tensor<1x1x4x8x4x8xbf16>,
               %arg1: tensor<1x1x8x4x8x4xbf16>,
               %arg2: tensor<1x1x8x8x4x4xf32>) -> tensor<1x1x8x8x4x4xf32> {
     %0 = linalg.generic {indexing_maps =
         [
           affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>,
           affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>,
           affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
         ],
   iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
         ins(%arg0, %arg1 : tensor<1x1x4x8x4x8xbf16>, tensor<1x1x8x4x8x4xbf16>)
         outs(%arg2 : tensor<1x1x8x8x4x4xf32>) {
    ^bb0(%in: bf16, %in_16: bf16, %out: f32):
      %18 = arith.extf %in : bf16 to f32
      %19 = arith.extf %in_16 : bf16 to f32
      %20 = arith.mulf %18, %19 : f32
      %21 = arith.addf %out, %20 : f32
      linalg.yield %21 : f32
    } -> tensor<1x1x8x8x4x4xf32>
    return %0 : tensor<1x1x8x8x4x4xf32>
  }
}

// -----

// CHECK-LABEL: @element_wise
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<4x6x8xf32>, %[[ARG1:.*]]: tensor<4x6x8xbf16>)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @element_wise(%arg0: tensor<4x6x8xf32>, %arg1: tensor<4x6x8xbf16>) -> tensor<4x6x8xbf16>{
    %cst = arith.constant 0.000000e+00 : bf16
    // CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
    // CHECK:       scf.for %[[IV:.*]] = %{{.*}} to %[[C4]]
    // CHECK-SAME:                      iter_args(%[[ARG3:.*]] = %[[ARG1]])
    // CHECK-NOT:     scf.for
    // CHECK:         tensor.extract_slice %[[ARG0]][%[[IV]], 0, 0] [1, 6, 8] [1, 1, 1]
    // CHECK:         tensor.extract_slice %[[ARG3]][%[[IV]], 0, 0] [1, 6, 8] [1, 1, 1]
    // CHECK:         %[[RES:.*]] = linalg.generic
    // CHECK:         tensor.insert_slice %[[RES]] into %[[ARG3]][%[[IV]], 0, 0] [1, 6, 8] [1, 1, 1]
    %0 = linalg.generic {
              indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                               affine_map<(d0, d1, d2) -> (d0, d1, d2)>
                              ],
              iterator_types = ["parallel", "parallel", "parallel"]
          } ins(%arg0 : tensor<4x6x8xf32>)
            outs(%arg1 : tensor<4x6x8xbf16>) {
      ^bb0(%in: f32, %out: bf16):
        %1 = arith.truncf %in : f32 to bf16
        %2 = arith.maximumf %1, %cst : bf16
        linalg.yield %2 : bf16
    } -> tensor<4x6x8xbf16>
    return %0 : tensor<4x6x8xbf16>
  }
}

// -----

// CHECK-LABEL: @element_wise_bufferized
// CHECK-SAME:  (%[[ARG0:.*]]: memref<1x1x4x6x8xf32>, %[[ARG1:.*]]: memref<1x1x4x6x8xbf16>)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @element_wise_bufferized(%arg0: memref<1x1x4x6x8xf32>, %arg1: memref<1x1x4x6x8xbf16>) -> memref<1x1x4x6x8xbf16>{
    %cst = arith.constant 0.000000e+00 : bf16
    // CHECK:       %[[COLLAPSE_UNIT_DIM_0:.*]] = memref.subview %[[ARG0]]
    // CHECK-SAME:          memref<1x1x4x6x8xf32> to memref<4x6x8xf32, strided<[48, 8, 1]>>
    // CHECK:       %[[COLLAPSE_UNIT_DIM_1:.*]] = memref.subview %[[ARG1]]
    // CHECK-SAME:          memref<1x1x4x6x8xbf16> to memref<4x6x8xbf16, strided<[48, 8, 1]>>
    // CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
    // CHECK:       scf.for %[[IV:.*]] = %{{.*}} to %[[C4]]
    // CHECK-NOT:     scf.for
    // CHECK:         %[[SLICE_0:.*]] = memref.subview %[[COLLAPSE_UNIT_DIM_0]]
    // CHECK:         %[[SLICE_1:.*]] = memref.subview %[[COLLAPSE_UNIT_DIM_1]]
    // CHECK:         linalg.generic
    // CHECK-SAME:        ins(%[[SLICE_0]] :
    // CHECK-SAME:        outs(%[[SLICE_1]] :
    linalg.generic {
              indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                               affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
                              ],
              iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
          } ins(%arg0 : memref<1x1x4x6x8xf32>)
            outs(%arg1 : memref<1x1x4x6x8xbf16>) {
      ^bb0(%in: f32, %out: bf16):
        %1 = arith.truncf %in : f32 to bf16
        %2 = arith.maximumf %1, %cst : bf16
        linalg.yield %2 : bf16
    }
    return %arg1 : memref<1x1x4x6x8xbf16>
  }
}

// -----

// CHECK-LABEL: @matmul_bufferized
// CHECK-SAME:  (%[[ARG0:.*]]: memref<1x1x4x8x4x8xbf16>,
// CHECK-SAME:   %[[ARG1:.*]]: memref<1x1x8x4x8x4xbf16>
// CHECK-SAME:   %[[ARG2:.*]]: memref<1x1x8x8x4x4xf32>)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @matmul_bufferized(
               %arg0: memref<1x1x4x8x4x8xbf16>,
               %arg1: memref<1x1x8x4x8x4xbf16>,
               %arg2: memref<1x1x8x8x4x4xf32>) -> memref<1x1x8x8x4x4xf32> {
    // CHECK-COUNT-3: scf.for
    // CHECK-NOT:     scf.for
    // CHECK:           linalg.generic
    // CHECK-SAME:            memref<1x1x4x8xbf16
    // CHECK-SAME:            memref<1x1x8x4xbf16
    // CHECK-SAME:            memref<1x1x4x4xf32
    linalg.generic {indexing_maps =
         [
           affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>,
           affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>,
           affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
         ],
   iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
         ins(%arg0, %arg1 : memref<1x1x4x8x4x8xbf16>, memref<1x1x8x4x8x4xbf16>)
         outs(%arg2 : memref<1x1x8x8x4x4xf32>) {
    ^bb0(%in: bf16, %in_16: bf16, %out: f32):
      %18 = arith.extf %in : bf16 to f32
      %19 = arith.extf %in_16 : bf16 to f32
      %20 = arith.mulf %18, %19 : f32
      %21 = arith.addf %out, %20 : f32
      linalg.yield %21 : f32
    }
    return %arg2 : memref<1x1x8x8x4x4xf32>
  }
}

// -----

module {
  func.func @no_amdaie_device(
               %arg0: memref<1x1x4x8x4x8xbf16>,
               %arg1: memref<1x1x8x4x8x4xbf16>,
               %arg2: memref<1x1x8x8x4x4xf32>) -> memref<1x1x8x8x4x4xf32> {
    // expected-error @+1 {{has no AMDAIEDevice in the target attribute configuration. This device-specific information is required to determine what vector sizes are supported.}}
    linalg.generic {indexing_maps =
         [
           affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>,
           affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>,
           affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
         ],
   iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
         ins(%arg0, %arg1 : memref<1x1x4x8x4x8xbf16>, memref<1x1x8x4x8x4xbf16>)
         outs(%arg2 : memref<1x1x8x8x4x4xf32>) {
    ^bb0(%in: bf16, %in_16: bf16, %out: f32):
      %18 = arith.extf %in : bf16 to f32
      %19 = arith.extf %in_16 : bf16 to f32
      %20 = arith.mulf %18, %19 : f32
      %21 = arith.addf %out, %20 : f32
      linalg.yield %21 : f32
    }
    return %arg2 : memref<1x1x8x8x4x4xf32>
  }
}
