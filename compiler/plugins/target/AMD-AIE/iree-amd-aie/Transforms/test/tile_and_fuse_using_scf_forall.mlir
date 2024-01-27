// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-tile-and-fuse{tiling-level=0}))' --split-input-file %s | FileCheck %s --check-prefix=TILE-LEVEL-0
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-tile-and-fuse{tiling-level=1}))' --split-input-file %s | FileCheck %s --check-prefix=TILE-LEVEL-1

func.func @matmul_static(%arg0: tensor<8x16xi32>, %arg1 : tensor<16x8xi32>) -> tensor<8x8xi32> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %5 = tensor.empty() : tensor<8x8xi32>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<8x8xi32>) -> tensor<8x8xi32>
  %7 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[8, 8], [4, 4], [0, 0, 4]]>} ins(%arg0, %arg1 : tensor<8x16xi32>, tensor<16x8xi32>) outs(%6 : tensor<8x8xi32>) -> tensor<8x8xi32>
  return %7 : tensor<8x8xi32>
}
//      TILE-LEVEL-0: @matmul_static
//      TILE-LEVEL-0:   scf.forall
// TILE-LEVEL-0-SAME:   {
//      TILE-LEVEL-0:       linalg.fill
//      TILE-LEVEL-0:       linalg.matmul
//      TILE-LEVEL-0:   }

// -----


//      TILE-LEVEL-1: @matmul_static
//      TILE-LEVEL-1:   scf.forall
// TILE-LEVEL-1-SAME:   {
//      TILE-LEVEL-1:       scf.forall
// TILE-LEVEL-1-SAME:       {
//      TILE-LEVEL-1:           linalg.fill
//      TILE-LEVEL-1:           linalg.matmul
//      TILE-LEVEL-1:       }
//      TILE-LEVEL-1:   }

// -----

func.func @matmul_bias_add(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %0 = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[8, 8], [4, 4], [0, 0, 4]]>}
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
//      TILE-LEVEL-0: @matmul_bias_add
//      TILE-LEVEL-0:   scf.forall
// TILE-LEVEL-0-SAME:   {
//      TILE-LEVEL-0:       linalg.fill
//      TILE-LEVEL-0:       linalg.matmul
//      TILE-LEVEL-0:       linalg.generic
//      TILE-LEVEL-0:   }
