// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-tile-and-fuse{tiling-level=0}))" --split-input-file %s | FileCheck %s

func.func @matmul_static(%lhs : tensor<8x16xi32>,
    %rhs : tensor<16x8xi32>) -> tensor<8x8xi32> {
  %empty = tensor.empty() : tensor<8x8xi32>
  %cst = arith.constant 0 : i32
  %fill = linalg.fill ins(%cst : i32) outs(%empty : tensor<8x8xi32>) -> tensor<8x8xi32>
  %2 = linalg.matmul ins(%lhs, %rhs : tensor<8x16xi32>, tensor<16x8xi32>)
      outs(%fill : tensor<8x8xi32>) -> tensor<8x8xi32>
  return %2 : tensor<8x8xi32>
}
// CHECK-LABEL: @matmul_static
// CHECK-SAME: (%[[LHS:.+]]: tensor<8x16xi32>, %[[RHS:.+]]: tensor<16x8xi32>)
// CHECK-SAME: {
// CHECK:         %[[EMPTY_TENSOR:.+]] = tensor.empty() : tensor<8x8xi32>
// CHECK:         %[[ZERO_CST:.+]] = arith.constant 0 : i32
// CHECK:         %[[OUTPUT:.+]] = scf.forall (%[[I:.+]], %[[J:.+]]) = (0, 0) to (8, 8) step (8, 8) shared_outs(%[[SHARED_EMPTY_TENSOR:.+]] = %[[EMPTY_TENSOR]]) -> (tensor<8x8xi32>) {
// CHECK:            %[[TILED_LHS:.+]] = tensor.extract_slice %[[LHS]][%[[I]], 0] [8, 16] [1, 1] : tensor<8x16xi32> to tensor<8x16xi32>
// CHECK:            %[[TILED_RHS:.+]] = tensor.extract_slice %[[RHS]][0, %[[J]]] [16, 8] [1, 1] : tensor<16x8xi32> to tensor<16x8xi32>
// CHECK:            %[[TILED_EMPTY_TENSOR:.+]] = tensor.extract_slice %[[SHARED_EMPTY_TENSOR]][%[[I]], %[[J]]] [8, 8] [1, 1] : tensor<8x8xi32> to tensor<8x8xi32>
// CHECK:            %[[FILL:.+]] = linalg.fill ins(%[[ZERO_CST]] : i32) outs(%[[TILED_EMPTY_TENSOR]] : tensor<8x8xi32>) -> tensor<8x8xi32>
// CHECK:            %[[MATMUL:.+]] = linalg.matmul ins(%[[TILED_LHS]], %[[TILED_RHS]] : tensor<8x16xi32>, tensor<16x8xi32>) outs(%[[FILL]] : tensor<8x8xi32>) -> tensor<8x8xi32>
// CHECK:            scf.forall.in_parallel {
// CHECK:              tensor.parallel_insert_slice %[[MATMUL]] into %[[SHARED_EMPTY_TENSOR]][%[[I]], %[[J]]] [8, 8] [1, 1] : tensor<8x8xi32> into tensor<8x8xi32>
// CHECK:            }
// CHECK:         } {mapping = [#gpu.block<y>, #gpu.block<x>]}
// CHECK:         return %[[OUTPUT]] : tensor<8x8xi32>
// CHECK:      }

// -----

func.func @matmul_bias_add(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %0 = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[10, 20, 30]]>}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1)-> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%1, %arg2 : tensor<?x?xf32>, tensor<?xf32>)
    outs(%init : tensor<?x?xf32>) {
      ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
        %3 = arith.addf %arg3, %arg4 : f32
        linalg.yield %3 : f32
    } -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}
//      CHECK: @matmul_bias_add
//      CHECK:   scf.forall
// CHECK-SAME:   {
//      CHECK:       linalg.fill
//      CHECK:       linalg.matmul
//      CHECK:       linalg.generic
//      CHECK:   }