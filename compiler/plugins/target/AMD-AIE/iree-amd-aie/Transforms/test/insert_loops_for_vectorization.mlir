// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-insert-loops-for-vectorization))" %s | FileCheck %s

!t2 = tensor<64x64xf32>
!t3 = tensor<64x64x64xf32>
!t4 = tensor<64x64x64x64xf32>
module {
   // A generic that corresponds to a simple matmul (2 rank-2 operands)
   // does NOT get tiled.
   // CHECK-LABEL: vanilla
   // CHECK-NOT: scf.for
  func.func @vanilla(%arg0: !t2, %arg1: !t2, %arg2: !t2) -> !t2 {
    %0 = linalg.generic {indexing_maps =
                          [
                           affine_map<(d0, d1, d2) -> (d0, d2)>,
                           affine_map<(d0, d1, d2) -> (d2, d1)>,
                           affine_map<(d0, d1, d2) -> (d0, d1)>
                          ],
                         iterator_types = ["parallel", "parallel", "reduction"]}
                         ins(%arg0, %arg1 : !t2, !t2) outs(%arg2 : !t2) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> !t2
    return %0 : !t2
  }

  // A batched matmul gets the batch dimension converted to a single scf.for
  // CHECK-LABEL: batched0
  // CHECK: scf.for
  // CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  // CHECK-SAME: ins
  // CHECK-SAME: tensor<1x64x64xf32>, tensor<1x64x64xf32>
  // CHECK-SAME: outs
  // CHECK-SAME: tensor<1x64x64xf32>
  // CHECK-NOT: scf.for
  func.func @batched0(%arg0: !t3, %arg1: !t3, %arg2: !t3) -> !t3 {
    %0 = linalg.generic {indexing_maps =
                          [
                           affine_map<(b0, d0, d1, d2) -> (b0, d0, d2)>,
                           affine_map<(b0, d0, d1, d2) -> (b0, d2, d1)>,
                           affine_map<(b0, d0, d1, d2) -> (b0, d0, d1)>
                          ],
                         iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
                         ins(%arg0, %arg1 : !t3, !t3) outs(%arg2 : !t3) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> !t3
    return %0 : !t3
  }

  // A test like the above, but with a matmul_tranpose_b instead of a matmul
  // CHECK-LABEL: batched_transpose_b
  // CHECK: scf.for
  // CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  // CHECK-SAME: ins
  // CHECK-SAME: tensor<1x64x64xf32>, tensor<1x64x64xf32>
  // CHECK-SAME: outs
  // CHECK-SAME: tensor<1x64x64xf32>
  // CHECK-NOT: scf.for
  func.func @batched_transpose_b(%arg0: !t3, %arg1: !t3, %arg2: !t3) -> !t3 {
    %0 = linalg.generic {indexing_maps =
                          [
                           affine_map<(b0, d0, d1, d2) -> (b0, d0, d2)>,
                           affine_map<(b0, d0, d1, d2) -> (b0, d1, d2)>,
                           affine_map<(b0, d0, d1, d2) -> (b0, d0, d1)>
                          ],
                         iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
                         ins(%arg0, %arg1 : !t3, !t3) outs(%arg2 : !t3) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> !t3
    return %0 : !t3
  }

  // Another test with a transposed matmul, but in this case A is transposed.
  // Currently the pass does not modify this (so no scf.for should appear).
  // We'll probably add support for this in the future, but for now we don't.
  // CHECK-LABEL: batched_transpose_a
  // CHECK-NOT: scf.for

  func.func @batched_transpose_a(%arg0: !t3, %arg1: !t3, %arg2: !t3) -> !t3 {
    %0 = linalg.generic {indexing_maps =
                          [
                           affine_map<(b0, d0, d1, d2) -> (b0, d2, d1)>,
                           affine_map<(b0, d0, d1, d2) -> (b0, d2, d0)>,
                           affine_map<(b0, d0, d1, d2) -> (b0, d0, d1)>
                          ],
                         iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
                         ins(%arg0, %arg1 : !t3, !t3) outs(%arg2 : !t3) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> !t3
    return %0 : !t3
  }


  // A check that a linalg.generic where the number of operands is not 3, does
  // not get transformed to have an scf.for
  // CHECK-LABEL: funcWithTwoOperands
  // CHECK-NOT: scf.for
  func.func @funcWithTwoOperands(%arg0: !t4, %arg1: !t4) -> !t4 {
    %0 = linalg.generic {indexing_maps =
                          [
                           affine_map<(b0, d0, d1, d2) -> (b0, d0, d1, d2)>,
                           affine_map<(b0, d0, d1, d2) -> (d0, d1, d2, b0)>
                          ],
                         iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
                         ins(%arg0 : !t4) outs(%arg1 : !t4) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> !t4
    return %0 : !t4
  }

  // Check that the final 3 dimensions do have the pattern of a matmul (or matmul transpose)
  // CHECK-LABEL: batched1
  // CHECK-NOT: scf.for
  func.func @batched1(%arg0: !t3, %arg1: !t3, %arg2: !t3) -> !t3 {
    %0 = linalg.generic {indexing_maps =
                          [
                           // This is like a matmul but the first operand is
                           // transposed:
                           affine_map<(b0, d0, d1, d2) -> (b0, d2, d0)>,
                           affine_map<(b0, d0, d1, d2) -> (b0, d2, d1)>,
                           affine_map<(b0, d0, d1, d2) -> (b0, d0, d1)>
                          ],
                         iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
                         ins(%arg0, %arg1 : !t3, !t3) outs(%arg2 : !t3) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> !t3
    return %0 : !t3
  }

  // Check for a batched matmul where operand 0 is broadcast:
  // CHECK-LABEL: batched2
  // CHECK: scf.for
  // CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  // CHECK-SAME: ins
  // CHECK-SAME: tensor<64x64xf32>, tensor<1x64x64xf32>
  // CHECK-SAME: outs
  // CHECK-SAME: tensor<1x64x64xf32>
  // CHECK-NOT: scf.for
  func.func @batched2(%arg0: !t2, %arg1: !t3, %arg2: !t3) -> !t3 {
    %0 = linalg.generic {indexing_maps =
                          [
                           affine_map<(b0, d0, d1, d2) -> (d0, d2)>,
                           affine_map<(b0, d0, d1, d2) -> (b0, d2, d1)>,
                           affine_map<(b0, d0, d1, d2) -> (b0, d0, d1)>
                          ],
                         iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
                         ins(%arg0, %arg1 : !t2, !t3) outs(%arg2 : !t3) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> !t3
    return %0 : !t3
  }

  // A function which arises from the pack-based pipeline in iree-amd-aie,
  // we verify it gets tiled to 6 nested for loops (3 of which get canonicalized
  // away as they have loop count 1) and an inner linalg.generic for
  // matmul of size LHS: 4x8 and RHS: 8x4.

  // CHECK-LABEL: packBasedPipeline
  // CHECK-COUNT-6: scf.for
  // CHECK-NOT: scf.for
  // CHECK: linalg.generic
  // CHECK-SAME: tensor<1x1x1x1x4x8xbf16>
  // CHECK-SAME: tensor<1x1x1x1x8x4xbf16>
  // CHECK-SAME: tensor<1x1x1x1x4x4xf32>

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


