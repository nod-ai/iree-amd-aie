// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-amdaie-fuse-consumer-into-loop{use-scf-for=true}))' --verify-diagnostics %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
module {
  func.func @fuse_consumer_into_scffor_matmul(%arg0: tensor<1x1x4x8x4x8xi8>, %arg1: tensor<1x1x4x4x8x8xi8>, %arg2: tensor<1x1x8x16x4x8xi32>, %arg3: tensor<1024x1024xi32>) -> tensor<1024x1024xi32> {
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %0 = scf.forall (%arg4, %arg5) in (2, 2) shared_outs(%arg6 = %arg3) -> (tensor<1024x1024xi32>) {
      %1 = scf.forall (%arg7, %arg8) in (2, 2) shared_outs(%arg9 = %arg2) -> (tensor<1x1x8x16x4x8xi32>) {
        %extracted_slice = tensor.extract_slice %arg9[0, 0, %arg8, %arg7, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : tensor<1x1x8x16x4x8xi32> to tensor<1x1x4x8x4x8xi32>
        %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1x1x4x8x4x8xi8>, tensor<1x1x4x4x8x8xi8>) outs(%extracted_slice : tensor<1x1x4x8x4x8xi32>) {
        ^bb0(%in: i8, %in_1: i8, %out: i32):
          %8 = arith.extsi %in : i8 to i32
          %9 = arith.extsi %in_1 : i8 to i32
          %10 = arith.muli %8, %9 : i32
          %11 = arith.addi %out, %10 : i32
          linalg.yield %11 : i32
        } -> tensor<1x1x4x8x4x8xi32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %7 into %arg9[0, 0, %arg8, %arg7, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : tensor<1x1x4x8x4x8xi32> into tensor<1x1x8x16x4x8xi32>
        }
      }
      %2 = scf.for %arg7 = %c0 to %c64 step %c4 iter_args(%arg9 = %1) -> (tensor<1x1x8x16x4x8xi32>) {
        %extracted_slice = tensor.extract_slice %arg9[0, 0, %arg7, %arg7, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : tensor<1x1x8x16x4x8xi32> to tensor<1x1x4x8x4x8xi32>
        %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1x1x4x8x4x8xi8>, tensor<1x1x4x4x8x8xi8>) outs(%extracted_slice : tensor<1x1x4x8x4x8xi32>) {
        ^bb0(%in: i8, %in_1: i8, %out: i32):
          %8 = arith.extsi %in : i8 to i32
          %9 = arith.extsi %in_1 : i8 to i32
          %10 = arith.muli %8, %9 : i32
          %11 = arith.addi %out, %10 : i32
          linalg.yield %11 : i32
        } -> tensor<1x1x4x8x4x8xi32>
        %8 = tensor.insert_slice %7 into %arg9[0, 0, %arg7, %arg7, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : tensor<1x1x4x8x4x8xi32> into tensor<1x1x8x16x4x8xi32>
        scf.yield %8 : tensor<1x1x8x16x4x8xi32>
      }
      %3 = tensor.empty() : tensor<64x64xi32>
      %4 = tensor.empty() : tensor<1x1x64x64xi32>
      %unpack = tensor.unpack %2 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %4 : tensor<1x1x8x16x4x8xi32> -> tensor<1x1x64x64xi32>
      %unpack_0 = tensor.unpack %unpack inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %3 : tensor<1x1x64x64xi32> -> tensor<64x64xi32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %unpack_0 into %arg6[%arg4, %arg5] [64, 64] [1, 1] : tensor<64x64xi32> into tensor<1024x1024xi32>
      }
    }
    return %0 : tensor<1024x1024xi32>
  }
}
//      CHECK: #[[UNPACK_RESULT_MAP0:.*]] = affine_map<(d0) -> (d0 * 4)>
//      CHECK: #[[UNPACK_RESULT_MAP1:.*]] = affine_map<(d0) -> (d0 * 8)>
//      CHECK:   %[[FINAL:.*]] = scf.forall
// CHECK-SAME:                         shared_outs(%[[ITER_ARG_FINAL:.*]] = %{{.*}})
//      CHECK:   {
//      CHECK:       %[[FIRST_LOOP:.*]] = scf.forall
//      CHECK:       {
//      CHECK:       }
//      CHECK:       %[[SECOND_UNPACK_OUT:.*]] = tensor.empty() : tensor<64x64xi32>
//      CHECK:       %[[UNPACK_OUT:.*]] = tensor.empty() : tensor<1x1x64x64xi32>

//      CHECK:       %[[SECOND_LOOP:.*]]:2 = scf.for %[[IV0:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER_ARG_1:.*]] = %[[FIRST_LOOP]], %[[ITER_ARG_3:.*]] = %[[UNPACK_OUT]])
//      CHECK:       {
//      CHECK:            %[[MATMUL:.*]] = linalg.generic
//      CHECK:            affine.apply
//      CHECK:            affine.apply
//      CHECK:            %[[iv0:.*]] = affine.apply #[[UNPACK_RESULT_MAP0]](%[[IV0]])
//      CHECK:            %[[iv1:.*]] = affine.apply #[[UNPACK_RESULT_MAP1]](%[[IV0]])
//      CHECK:            %[[TILED_UNPACK_DEST:.*]] = tensor.extract_slice %[[ITER_ARG_3]][0, 0, %[[iv0]], %[[iv1]]] [1, 1, 32, 32] [1, 1, 1, 1]
//      CHECK:            %[[TILED_UNPACK:.*]] = tensor.unpack %[[MATMUL]] outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %[[TILED_UNPACK_DEST]]
//      CHECK:            %[[YIELD_MATMUL:.*]] = tensor.insert_slice %[[MATMUL]] into %[[ITER_ARG_1]]
//      CHECK:            %[[YIELD_UNPACK:.*]] = tensor.insert_slice %[[TILED_UNPACK]] into %[[ITER_ARG_3]]
//      CHECK:            scf.yield %[[YIELD_MATMUL]], %[[YIELD_UNPACK]]
//      CHECK:       }
//      CHECK:       %[[SECOND_UNPACK:.*]] = tensor.unpack %[[SECOND_LOOP]]#1 inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %[[SECOND_UNPACK_OUT]] :
//      CHECK:       scf.forall.in_parallel
//      CHECK:            tensor.parallel_insert_slice %[[SECOND_UNPACK]] into %[[ITER_ARG_FINAL]]
//      CHECK:       }
//      CHECK:   }
//      CHECK:   return %[[FINAL]]

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
module {
  func.func @fuse_consumer_into_scffor_matmul_elemwise_fusion(%arg0: tensor<1x1x4x8x4x8xi8>, %arg1: tensor<1x1x4x4x8x8xi8>, %arg2: tensor<1x1x8x16x4x8xi32>, %arg3: tensor<1024x1024xi32>) -> tensor<1024x1024xi32> {
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %0 = scf.forall (%arg4, %arg5) in (2, 2) shared_outs(%arg6 = %arg3) -> (tensor<1024x1024xi32>) {
      %1 = scf.forall (%arg7, %arg8) in (2, 2) shared_outs(%arg9 = %arg2) -> (tensor<1x1x8x16x4x8xi32>) {
        %extracted_slice = tensor.extract_slice %arg9[0, 0, %arg8, %arg7, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : tensor<1x1x8x16x4x8xi32> to tensor<1x1x4x8x4x8xi32>
        %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1x1x4x8x4x8xi8>, tensor<1x1x4x4x8x8xi8>) outs(%extracted_slice : tensor<1x1x4x8x4x8xi32>) {
        ^bb0(%in: i8, %in_1: i8, %out: i32):
          %8 = arith.extsi %in : i8 to i32
          %9 = arith.extsi %in_1 : i8 to i32
          %10 = arith.muli %8, %9 : i32
          %11 = arith.addi %out, %10 : i32
          linalg.yield %11 : i32
        } -> tensor<1x1x4x8x4x8xi32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %7 into %arg9[0, 0, %arg8, %arg7, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : tensor<1x1x4x8x4x8xi32> into tensor<1x1x8x16x4x8xi32>
        }
      }
      %2 = scf.for %arg7 = %c0 to %c64 step %c4 iter_args(%arg9 = %1) -> (tensor<1x1x8x16x4x8xi32>) {
        %extracted_slice = tensor.extract_slice %arg9[0, 0, %arg7, %arg7, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : tensor<1x1x8x16x4x8xi32> to tensor<1x1x4x8x4x8xi32>
        %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1x1x4x8x4x8xi8>, tensor<1x1x4x4x8x8xi8>) outs(%extracted_slice : tensor<1x1x4x8x4x8xi32>) {
        ^bb0(%in: i8, %in_1: i8, %out: i32):
          %8 = arith.extsi %in : i8 to i32
          %9 = arith.extsi %in_1 : i8 to i32
          %10 = arith.muli %8, %9 : i32
          %11 = arith.addi %out, %10 : i32
          linalg.yield %11 : i32
        } -> tensor<1x1x4x8x4x8xi32>
        %8 = tensor.insert_slice %7 into %arg9[0, 0, %arg7, %arg7, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : tensor<1x1x4x8x4x8xi32> into tensor<1x1x8x16x4x8xi32>
        scf.yield %8 : tensor<1x1x8x16x4x8xi32>
      }
      %3 = tensor.empty() : tensor<1x1x8x16x4x8xi32>
      %4 = tensor.empty() : tensor<64x64xi32>
      %5 = tensor.empty() : tensor<1x1x64x64xi32>
      %6 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2, %3 : tensor<1x1x8x16x4x8xi32>, tensor<1x1x8x16x4x8xi32>) outs(%3 : tensor<1x1x8x16x4x8xi32>) {
      ^bb0(%in: i32, %in_1: i32, %out: i32):
        %7 = arith.addi %in, %in_1 : i32
        linalg.yield %7 : i32
      } -> tensor<1x1x8x16x4x8xi32>
      %unpack = tensor.unpack %6 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %5 : tensor<1x1x8x16x4x8xi32> -> tensor<1x1x64x64xi32>
      %unpack_0 = tensor.unpack %unpack inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %4 : tensor<1x1x64x64xi32> -> tensor<64x64xi32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %unpack_0 into %arg6[%arg4, %arg5] [64, 64] [1, 1] : tensor<64x64xi32> into tensor<1024x1024xi32>
      }
    }
    return %0 : tensor<1024x1024xi32>
  }
}
//      CHECK: #[[UNPACK_RESULT_MAP0:.*]] = affine_map<(d0) -> (d0 * 4)>
//      CHECK: #[[UNPACK_RESULT_MAP1:.*]] = affine_map<(d0) -> (d0 * 8)>
//      CHECK:   %[[FINAL:.*]] = scf.forall
// CHECK-SAME:                         shared_outs(%[[ITER_ARG_FINAL:.*]] = %{{.*}})
//      CHECK:   {
//      CHECK:       %[[FIRST_LOOP:.*]] = scf.forall
//      CHECK:       {
//      CHECK:       }
//      CHECK:       %[[ELEM_OUT:.*]] = tensor.empty() : tensor<1x1x8x16x4x8xi32>
//      CHECK:       %[[SECOND_UNPACK_OUT:.*]] = tensor.empty() : tensor<64x64xi32>
//      CHECK:       %[[UNPACK_OUT:.*]] = tensor.empty() : tensor<1x1x64x64xi32>

//      CHECK:       %[[SECOND_LOOP:.*]]:3 = scf.for %[[IV0:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER_ARG_1:.*]] = %[[FIRST_LOOP]], %[[ITER_ARG_2:.*]] = %[[ELEM_OUT]], %[[ITER_ARG_3:.*]] = %[[UNPACK_OUT]])
//      CHECK:       {
//      CHECK:            %[[MATMUL:.*]] = linalg.generic
//      CHECK:            %[[OPERAND2:.*]] = tensor.extract_slice %[[ELEM_OUT]][0, 0, %[[IV0]], %[[IV0]], 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1]
//      CHECK:            %[[OPERAND3:.*]] = tensor.extract_slice %[[ITER_ARG_2]][0, 0, %[[IV0]], %[[IV0]], 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1]
//      CHECK:            %[[FUSED_CONSUMER:.*]] = linalg.generic
// CHECK-SAME:                                         ins(%[[MATMUL]], %[[OPERAND2]] :
// CHECK-SAME:                                         outs(%[[OPERAND3]] :
//      CHECK:                                    {
//      CHECK:                                         arith.addi  
//      CHECK:                                    }
//      CHECK:            %[[YIELD_MATMUL:.*]] = tensor.insert_slice %[[MATMUL]] into %[[ITER_ARG_1]]
//      CHECK:            affine.apply
//      CHECK:            affine.apply
//      CHECK:            %[[iv0:.*]] = affine.apply #[[UNPACK_RESULT_MAP0]](%[[IV0]])
//      CHECK:            %[[iv1:.*]] = affine.apply #[[UNPACK_RESULT_MAP1]](%[[IV0]])
//      CHECK:            %[[TILED_UNPACK_DEST:.*]] = tensor.extract_slice %[[ITER_ARG_3]][0, 0, %[[iv0]], %[[iv1]]] [1, 1, 32, 32] [1, 1, 1, 1]
//      CHECK:            %[[TILED_UNPACK:.*]] = tensor.unpack %[[FUSED_CONSUMER]] outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %[[TILED_UNPACK_DEST]]
//      CHECK:            %[[YIELD_ELEM:.*]] = tensor.insert_slice %[[FUSED_CONSUMER]] into %[[ITER_ARG_2]]
//      CHECK:            %[[YIELD_UNPACK:.*]] = tensor.insert_slice %[[TILED_UNPACK]] into %[[ITER_ARG_3]]
//      CHECK:            scf.yield %[[YIELD_MATMUL]], %[[YIELD_ELEM]], %[[YIELD_UNPACK]]
//      CHECK:       }
//      CHECK:       %[[SECOND_UNPACK:.*]] = tensor.unpack %[[SECOND_LOOP]]#2 inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %[[SECOND_UNPACK_OUT]] :
//      CHECK:       scf.forall.in_parallel
//      CHECK:            tensor.parallel_insert_slice %[[SECOND_UNPACK]] into %[[ITER_ARG_FINAL]]
//      CHECK:       }
//      CHECK:   }
//      CHECK:   return %[[FINAL]]

// -----

func.func @no_user_of_producer(%arg0: tensor<64xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %arg1 = %c0 to %c8 step %c1 {
    // expected-error @+1 {{Expected only one user of the compute op}}
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg0 : tensor<64xf32>, tensor<64xf32>) outs(%arg0 : tensor<64xf32>) -> tensor<64xf32>
  }
  return
}

// -----

func.func @insert_slice_not_found(%arg0: tensor<64xf32>) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %0 = scf.for %arg1 = %c0 to %c8 step %c1 iter_args(%out = %arg0) -> tensor<64xf32> {
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg0 : tensor<64xf32>, tensor<64xf32>) outs(%arg0 : tensor<64xf32>) -> tensor<64xf32>
    // expected-error @+1 {{Expected either tensor.insert_slice OR tensor.parallel_insert_slice to be the only user of the compute op}}
    scf.yield %1 : tensor<64xf32>
  }
  return %0 : tensor<64xf32>
}

// -----

func.func @no_consumer_fusion(%arg0: tensor<64xf32>) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %0 = scf.for %arg1 = %c0 to %c8 step %c1 iter_args(%out = %arg0) -> tensor<64xf32> {
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg0 : tensor<64xf32>, tensor<64xf32>) outs(%arg0 : tensor<64xf32>) -> tensor<64xf32>
    // expected-error @+1 {{Failed to fuse any consumer op into the producer}}
    %2 = tensor.insert_slice %1 into %out[%arg1] [64] [1] : tensor<64xf32> into tensor<64xf32>
    scf.yield %2 : tensor<64xf32>
  }
  return %0 : tensor<64xf32>
}
