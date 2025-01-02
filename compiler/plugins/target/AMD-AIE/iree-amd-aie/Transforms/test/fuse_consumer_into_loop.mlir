// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-amdaie-fuse-consumer-into-loop))' %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-amdaie-fuse-consumer-into-loop{max-iterations=1}))' --verify-diagnostics %s

// CHECK-DAG:  #[[UNPACK_RESULT_MAP0:.*]] = affine_map<(d0) -> (d0 * 4)>
// CHECK-DAG:  #[[UNPACK_RESULT_MAP1:.*]] = affine_map<(d0) -> (d0 * 8)>
// CHECK-DAG:  #[[EXTRACT_SLICE_MAP0:.*]] = affine_map<(d0) -> (32, d0 * -4 + 64)>
// CHECK-DAG:  #[[EXTRACT_SLICE_MAP1:.*]] = affine_map<(d0) -> (32, d0 * -8 + 64)>
// CHECK:        %[[FINAL:.*]] = scf.forall
// CHECK-SAME:                         shared_outs(%[[ITER_ARG_FINAL:.*]] = %{{.*}})
// CHECK:        {
// CHECK:            %[[FIRST_LOOP:.*]] = scf.forall
// CHECK:            {
// CHECK:            }
// CHECK:            %[[SECOND_UNPACK_OUT:.*]] = tensor.empty() : tensor<64x64xi32>
// CHECK:            %[[UNPACK_OUT:.*]] = tensor.empty() : tensor<1x1x64x64xi32>
// CHECK:            %[[SECOND_LOOP:.*]]:2 = scf.for %[[IV0:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER_ARG_1:.*]] = %[[FIRST_LOOP]], %[[ITER_ARG_3:.*]] = %[[UNPACK_OUT]])
// CHECK:            {
// CHECK:                 %[[MATMUL:.*]] = linalg.generic
// CHECK-DAG:             %[[YIELD_MATMUL:.*]] = tensor.insert_slice %[[MATMUL]] into %[[ITER_ARG_1]]
// CHECK-DAG:             %[[iv0:.*]] = affine.apply #[[UNPACK_RESULT_MAP0]](%[[IV0]])
// CHECK-DAG:             %[[iv1:.*]] = affine.apply #[[UNPACK_RESULT_MAP1]](%[[IV0]])
// CHECK-DAG:             %[[iv2:.*]] = affine.min #[[EXTRACT_SLICE_MAP0]](%[[IV0]])
// CHECK-DAG:             %[[iv3:.*]] = affine.min #[[EXTRACT_SLICE_MAP1]](%[[IV0]])
// CHECK-DAG:             %[[TILED_UNPACK_DEST:.*]] = tensor.extract_slice %[[ITER_ARG_3]][0, 0, %[[iv0]], %[[iv1]]] [1, 1, %[[iv2]], %[[iv3]]] [1, 1, 1, 1]
// CHECK-DAG:             %[[TILED_UNPACK:.*]] = tensor.unpack %[[MATMUL]] outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %[[TILED_UNPACK_DEST]]
// CHECK:                 %[[YIELD_UNPACK:.*]] = tensor.insert_slice %[[TILED_UNPACK]] into %[[ITER_ARG_3]]
// CHECK:                 scf.yield %[[YIELD_MATMUL]], %[[YIELD_UNPACK]]
// CHECK:            }
// CHECK:            %[[SECOND_UNPACK:.*]] = tensor.unpack %[[SECOND_LOOP]]#1 inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %[[SECOND_UNPACK_OUT]] :
// CHECK:            scf.forall.in_parallel
// CHECK:                 tensor.parallel_insert_slice %[[SECOND_UNPACK]] into %[[ITER_ARG_FINAL]]
// CHECK:            }
// CHECK:        }
// CHECK:        return %[[FINAL]]
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
module {
  // expected-error @+1 {{Maximum number of iterations reached, consumer fusion is likely stuck in an infinite loop.}}
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

// -----

// CHECK-DAG:  #[[UNPACK_RESULT_MAP0:.*]] = affine_map<(d0) -> (d0 * 4)>
// CHECK-DAG:  #[[UNPACK_RESULT_MAP1:.*]] = affine_map<(d0) -> (d0 * 8)>
// CHECK-DAG:  #[[EXTRACT_SLICE_MAP0:.*]] = affine_map<(d0) -> (32, d0 * -4 + 64)>
// CHECK-DAG:  #[[EXTRACT_SLICE_MAP1:.*]] = affine_map<(d0) -> (32, d0 * -8 + 64)>
// CHECK:        %[[FINAL:.*]] = scf.forall
// CHECK-SAME:    shared_outs(%[[ITER_ARG_FINAL:.*]] = %{{.*}})
// CHECK:        {
// CHECK:            %[[FIRST_LOOP:.*]] = scf.forall
// CHECK:            {
// CHECK:            }
// CHECK:            %[[ELEM_OUT:.*]] = tensor.empty() : tensor<1x1x8x16x4x8xi32>
// CHECK:            %[[SECOND_UNPACK_OUT:.*]] = tensor.empty() : tensor<64x64xi32>
// CHECK:            %[[UNPACK_OUT:.*]] = tensor.empty() : tensor<1x1x64x64xi32>
// CHECK:            %[[SECOND_LOOP:.*]]:3 = scf.for %[[IV0:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER_ARG_1:.*]] = %[[FIRST_LOOP]], %[[ITER_ARG_2:.*]] = %[[ELEM_OUT]], %[[ITER_ARG_3:.*]] = %[[UNPACK_OUT]])
// CHECK:            {
// CHECK:                 %[[MATMUL:.*]] = linalg.generic
// CHECK-DAG:             %[[OPERAND2:.*]] = tensor.extract_slice %[[ELEM_OUT]][0, 0, %[[IV0]], %[[IV0]], 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1]
// CHECK-DAG:             %[[OPERAND3:.*]] = tensor.extract_slice %[[ITER_ARG_2]][0, 0, %[[IV0]], %[[IV0]], 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1]
// CHECK:                 %[[FUSED_CONSUMER:.*]] = linalg.generic
// CHECK-SAME:              ins(%[[MATMUL]], %[[OPERAND2]] :
// CHECK-SAME:              outs(%[[OPERAND3]] :
// CHECK:                 {
// CHECK:                   arith.addi  
// CHECK:                 }
// CHECK-DAG:             %[[YIELD_MATMUL:.*]] = tensor.insert_slice %[[MATMUL]] into %[[ITER_ARG_1]]
// CHECK-DAG:             %[[YIELD_ELEM:.*]] = tensor.insert_slice %[[FUSED_CONSUMER]] into %[[ITER_ARG_2]]
// CHECK-DAG:             %[[iv0:.*]] = affine.apply #[[UNPACK_RESULT_MAP0]](%[[IV0]])
// CHECK-DAG:             %[[iv1:.*]] = affine.apply #[[UNPACK_RESULT_MAP1]](%[[IV0]])
// CHECK-DAG:             %[[iv2:.*]] = affine.min #[[EXTRACT_SLICE_MAP0]](%[[IV0]])
// CHECK-DAG:             %[[iv3:.*]] = affine.min #[[EXTRACT_SLICE_MAP1]](%[[IV0]])
// CHECK-DAG:             %[[TILED_UNPACK_DEST:.*]] = tensor.extract_slice %[[ITER_ARG_3]][0, 0, %[[iv0]], %[[iv1]]] [1, 1, %[[iv2]], %[[iv3]]] [1, 1, 1, 1]
// CHECK-DAG:             %[[TILED_UNPACK:.*]] = tensor.unpack %[[FUSED_CONSUMER]] outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %[[TILED_UNPACK_DEST]]
// CHECK-DAG:             %[[YIELD_UNPACK:.*]] = tensor.insert_slice %[[TILED_UNPACK]] into %[[ITER_ARG_3]]
// CHECK:                 scf.yield %[[YIELD_MATMUL]], %[[YIELD_ELEM]], %[[YIELD_UNPACK]]
// CHECK:            }
// CHECK:            %[[SECOND_UNPACK:.*]] = tensor.unpack %[[SECOND_LOOP]]#2 inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %[[SECOND_UNPACK_OUT]] :
// CHECK:            scf.forall.in_parallel
// CHECK:                 tensor.parallel_insert_slice %[[SECOND_UNPACK]] into %[[ITER_ARG_FINAL]]
// CHECK:            }
// CHECK:        }
// CHECK:        return %[[FINAL]]
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
module {
  // expected-error @+1 {{Maximum number of iterations reached, consumer fusion is likely stuck in an infinite loop.}}
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

// -----

// CHECK-LABEL: @no_consumer_fusion
// CHECK:       linalg.elemwise_binary
// CHECK:       tensor.insert_slice

// expected-error @+1 {{Maximum number of iterations reached, consumer fusion is likely stuck in an infinite loop.}}
func.func @no_consumer_fusion(%arg0: tensor<64xf32>) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %0 = scf.for %arg1 = %c0 to %c8 step %c1 iter_args(%out = %arg0) -> tensor<64xf32> {
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg0 : tensor<64xf32>, tensor<64xf32>) outs(%arg0 : tensor<64xf32>) -> tensor<64xf32>
    %2 = tensor.insert_slice %1 into %out[%arg1] [64] [1] : tensor<64xf32> into tensor<64xf32>
    scf.yield %2 : tensor<64xf32>
  }
  return %0 : tensor<64xf32>
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
module {
  // expected-error @+1 {{Maximum number of iterations reached, consumer fusion is likely stuck in an infinite loop.}}
  func.func @fuse_consumer_into_scfforall_matmul(%arg0: tensor<1x1x4x8x4x8xi8>, %arg1: tensor<1x1x4x4x8x8xi8>, %arg2: tensor<1x1x8x16x4x8xi32>, %arg3: tensor<1024x1024xi32>) -> tensor<1024x1024xi32> {
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
      %2 = scf.forall (%arg7, %arg8) in (2, 2) shared_outs(%arg9 = %1) -> (tensor<1x1x8x16x4x8xi32>) {
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
//  CHECK-DAG: #[[UNPACK_RESULT_MAP0:.*]] = affine_map<(d0) -> (d0 * 4)>
//  CHECK-DAG: #[[UNPACK_RESULT_MAP1:.*]] = affine_map<(d0) -> (d0 * 8)>
//  CHECK-DAG: #[[EXTRACT_SLICE_MAP0:.*]] = affine_map<(d0) -> (32, d0 * -4 + 64)>
//  CHECK-DAG: #[[EXTRACT_SLICE_MAP1:.*]] = affine_map<(d0) -> (32, d0 * -8 + 64)>
//      CHECK:   %[[FINAL:.*]] = scf.forall
// CHECK-SAME:                         shared_outs(%[[ITER_ARG_FINAL:.*]] = %{{.*}})
//      CHECK:   {
//      CHECK:       %[[FIRST_LOOP:.*]] = scf.forall
//      CHECK:       {
//      CHECK:       }
//      CHECK:       %[[SECOND_UNPACK_OUT:.*]] = tensor.empty() : tensor<64x64xi32>
//      CHECK:       %[[UNPACK_OUT:.*]] = tensor.empty() : tensor<1x1x64x64xi32>
//      CHECK:       %[[SECOND_LOOP:.*]]:2 = scf.forall (%[[IV0:.*]], %[[IV1:.*]]) in (2, 2) shared_outs(%[[ITER_ARG_1:.*]] = %[[FIRST_LOOP]], %[[ITER_ARG_3:.*]] = %[[UNPACK_OUT]])
//      CHECK:       {
//      CHECK:            %[[MATMUL:.*]] = linalg.generic
//  CHECK-DAG:            %[[iv0:.*]] = affine.apply #[[UNPACK_RESULT_MAP0]](%[[IV0]])
//  CHECK-DAG:            %[[iv1:.*]] = affine.apply #[[UNPACK_RESULT_MAP1]](%[[IV1]])
//  CHECK-DAG:            %[[iv2:.*]] = affine.min #[[EXTRACT_SLICE_MAP0]](%[[IV0]])
//  CHECK-DAG:            %[[iv3:.*]] = affine.min #[[EXTRACT_SLICE_MAP1]](%[[IV1]])
//      CHECK:            %[[TILED_UNPACK_DEST:.*]] = tensor.extract_slice %[[ITER_ARG_3]][0, 0, %[[iv0]], %[[iv1]]] [1, 1, %[[iv2]], %[[iv3]]] [1, 1, 1, 1]
//      CHECK:            %[[TILED_UNPACK:.*]] = tensor.unpack %[[MATMUL]] outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %[[TILED_UNPACK_DEST]]
//      CHECK:            scf.forall.in_parallel {
//      CHECK:                 tensor.parallel_insert_slice %[[MATMUL]] into %[[ITER_ARG_1]][0, 0, %[[IV1]], %[[IV0]], 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1]
//      CHECK:                 tensor.parallel_insert_slice %[[TILED_UNPACK]] into %[[ITER_ARG_3]][0, 0, %[[iv0]], %[[iv1]]] [1, 1,  %[[iv2]], %[[iv3]]] [1, 1, 1, 1]
//      CHECK:            }
//      CHECK:        }
//      CHECK:        %[[SECOND_UNPACK:.*]] = tensor.unpack %[[SECOND_LOOP]]#1 inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %[[SECOND_UNPACK_OUT]] :
//      CHECK:        scf.forall.in_parallel
//      CHECK:             tensor.parallel_insert_slice %[[SECOND_UNPACK]] into %[[ITER_ARG_FINAL]]
//      CHECK:        }
//      CHECK:   }
//      CHECK:   return %[[FINAL]]

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
module {
  // expected-error @+1 {{Maximum number of iterations reached, consumer fusion is likely stuck in an infinite loop.}}
  func.func @fuse_consumer_into_scfforall_matmul_elemwise_fusion(%arg0: tensor<1x1x4x8x4x8xi8>, %arg1: tensor<1x1x4x4x8x8xi8>, %arg2: tensor<1x1x8x16x4x8xi32>, %arg3: tensor<1024x1024xi32>) -> tensor<1024x1024xi32> {
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
      %2 = scf.forall (%arg7, %arg8) in (2, 2) shared_outs(%arg9 = %1) -> (tensor<1x1x8x16x4x8xi32>) {
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
//  CHECK-DAG: #[[UNPACK_RESULT_MAP0:.*]] = affine_map<(d0) -> (d0 * 4)>
//  CHECK-DAG: #[[UNPACK_RESULT_MAP1:.*]] = affine_map<(d0) -> (d0 * 8)>
//  CHECK-DAG: #[[EXTRACT_SLICE_MAP0:.*]] = affine_map<(d0) -> (32, d0 * -4 + 64)>
//  CHECK-DAG: #[[EXTRACT_SLICE_MAP1:.*]] = affine_map<(d0) -> (32, d0 * -8 + 64)>
//      CHECK:   %[[FINAL:.*]] = scf.forall
// CHECK-SAME:                         shared_outs(%[[ITER_ARG_FINAL:.*]] = %{{.*}})
//      CHECK:   {
//      CHECK:       %[[FIRST_LOOP:.*]] = scf.forall
//      CHECK:       {
//      CHECK:       }
//      CHECK:       %[[ELEM_OUT:.*]] = tensor.empty() : tensor<1x1x8x16x4x8xi32>
//      CHECK:       %[[SECOND_UNPACK_OUT:.*]] = tensor.empty() : tensor<64x64xi32>
//      CHECK:       %[[UNPACK_OUT:.*]] = tensor.empty() : tensor<1x1x64x64xi32>
//      CHECK:       %[[SECOND_LOOP:.*]]:3 = scf.forall (%[[IV0:.*]], %[[IV1:.*]]) in (2, 2) shared_outs(%[[ITER_ARG_1:.*]] = %[[FIRST_LOOP]], %[[ITER_ARG_2:.*]] = %[[ELEM_OUT]], %[[ITER_ARG_3:.*]] = %[[UNPACK_OUT]])
//      CHECK:       {
//      CHECK:            %[[MATMUL:.*]] = linalg.generic
//      CHECK:            %[[OPERAND2:.*]] = tensor.extract_slice %[[ELEM_OUT]][0, 0, %[[IV1]], %[[IV0]], 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1]
//      CHECK:            %[[OPERAND3:.*]] = tensor.extract_slice %[[ITER_ARG_2]][0, 0, %[[IV1]], %[[IV0]], 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1]
//      CHECK:            %[[FUSED_CONSUMER:.*]] = linalg.generic
// CHECK-SAME:                                         ins(%[[MATMUL]], %[[OPERAND2]] :
// CHECK-SAME:                                         outs(%[[OPERAND3]] :
//      CHECK:                                    {
//      CHECK:                                         arith.addi  
//      CHECK:                                    }
//  CHECK-DAG:            %[[iv0:.*]] = affine.apply #[[UNPACK_RESULT_MAP0]](%[[IV0]])
//  CHECK-DAG:            %[[iv1:.*]] = affine.apply #[[UNPACK_RESULT_MAP1]](%[[IV1]])
//  CHECK-DAG:            %[[iv2:.*]] = affine.min #[[EXTRACT_SLICE_MAP0]](%[[IV0]])
//  CHECK-DAG:            %[[iv3:.*]] = affine.min #[[EXTRACT_SLICE_MAP1]](%[[IV1]])
//      CHECK:            %[[TILED_UNPACK_DEST:.*]] = tensor.extract_slice %[[ITER_ARG_3]][0, 0, %[[iv0]], %[[iv1]]] [1, 1, %[[iv2]], %[[iv3]]] [1, 1, 1, 1]
//      CHECK:            %[[TILED_UNPACK:.*]] = tensor.unpack %[[FUSED_CONSUMER]] outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %[[TILED_UNPACK_DEST]]
//      CHECK:            scf.forall.in_parallel {
//      CHECK:                 tensor.parallel_insert_slice %[[MATMUL]] into %[[ITER_ARG_1]][0, 0, %[[IV1]], %[[IV0]], 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1]
//      CHECK:                 tensor.parallel_insert_slice %[[FUSED_CONSUMER]] into %[[ITER_ARG_2]][0, 0, %[[IV1]], %[[IV0]], 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1]
//      CHECK:                 tensor.parallel_insert_slice %[[TILED_UNPACK]] into %[[ITER_ARG_3]][0, 0, %[[iv0]], %[[iv1]]] [1, 1, %[[iv2]], %[[iv3]]] [1, 1, 1, 1]
//      CHECK:            }
//      CHECK:        }
//      CHECK:        %[[SECOND_UNPACK:.*]] = tensor.unpack %[[SECOND_LOOP]]#2 inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %[[SECOND_UNPACK_OUT]] :
//      CHECK:        scf.forall.in_parallel
//      CHECK:             tensor.parallel_insert_slice %[[SECOND_UNPACK]] into %[[ITER_ARG_FINAL]]
//      CHECK:        }
//      CHECK:   }
//      CHECK:   return %[[FINAL]]

// -----

// CHECK-LABEL: @no_consumer_fusion
// CHECK:       linalg.elemwise_binary
// CHECK:       scf.forall.in_parallel
// CHECK:         tensor.parallel_insert_slice

// expected-error @+1 {{Maximum number of iterations reached, consumer fusion is likely stuck in an infinite loop.}}
func.func @no_consumer_fusion(%arg0: tensor<64xf32>) -> tensor<64xf32> {
  %0 = scf.forall (%arg1, %arg2) in (1,2) shared_outs(%out = %arg0) -> tensor<64xf32> {
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg0 : tensor<64xf32>, tensor<64xf32>) outs(%arg0 : tensor<64xf32>) -> tensor<64xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %1 into %out[%arg1] [64] [1] : tensor<64xf32> into tensor<64xf32>
    }
  }
  return %0 : tensor<64xf32>
}

// -----

// CHECK-LABEL: @fuse_consumer_into_mix_scf_forall_for
// CHECK:         scf.forall
// CHECK:           %[[FORALL:.*]]:2 = scf.forall
// CHECK:             %[[FOR:.*]] = scf.for
// CHECK:               %[[MATMUL:.*]] = linalg.generic
// CHECK:               scf.yield %[[MATMUL]]
// CHECK:             }
// CHECK:             %[[FUSED_UNPACK:.*]] = tensor.unpack %[[FOR]]
// CHECK-SAME:                               tensor<1x1x8x8x4x4xi32> -> tensor<1x1x32x32xi32>
// CHECK:             scf.forall.in_parallel {
// CHECK:                 tensor.parallel_insert_slice %[[FOR]]
// CHECK:                 tensor.parallel_insert_slice %[[FUSED_UNPACK]]
// CHECK:             }
// CHECK:           }
// CHECK:           %[[UNPACK:.*]] = tensor.unpack %[[FORALL]]#1
// CHECK:           scf.forall.in_parallel {
// CHECK:               tensor.parallel_insert_slice %[[UNPACK]]
// CHECK:           }
// CHECK:         }
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
module {
  // expected-error @+1 {{Maximum number of iterations reached, consumer fusion is likely stuck in an infinite loop.}}
  func.func @fuse_consumer_into_mix_scf_forall_for(%arg0: tensor<1x1x4x8x4x8xi32>, %arg1: tensor<1x1x8x4x8x4xi32>, %arg2: tensor<4x4x8x8x4x4xi32>, %arg3: tensor<128x128xi32>) -> tensor<128x128xi32> {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = scf.forall (%arg4, %arg5) in (2, 2) shared_outs(%arg6 = %arg3) -> (tensor<128x128xi32>) {
      %1 = scf.forall (%arg7, %arg8) in (2, 2) shared_outs(%arg9 = %arg2) -> (tensor<4x4x8x8x4x4xi32>) {
        %extracted_slice = tensor.extract_slice %arg9[%arg7, %arg8, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1] : tensor<4x4x8x8x4x4xi32> to tensor<1x1x8x8x4x4xi32>
        %4 = scf.for %arg10 = %c0 to %c2 step %c1 iter_args(%arg11 = %extracted_slice) -> (tensor<1x1x8x8x4x4xi32>) {
          %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1x1x4x8x4x8xi32>, tensor<1x1x8x4x8x4xi32>) outs(%arg11 : tensor<1x1x8x8x4x4xi32>) {
          ^bb0(%in: i32, %in_1: i32, %out: i32):
            %6 = arith.muli %in, %in_1 : i32
            %7 = arith.addi %out, %6 : i32
            linalg.yield %7 : i32
          } -> tensor<1x1x8x8x4x4xi32>
          scf.yield %5 : tensor<1x1x8x8x4x4xi32>
        }
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %4 into %arg9[%arg7, %arg8, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1] : tensor<1x1x8x8x4x4xi32> into tensor<4x4x8x8x4x4xi32>
        }
      }
      %2 = tensor.empty() : tensor<4x4x32x32xi32>
      %3 = tensor.empty() : tensor<128x128xi32>
      %unpack = tensor.unpack %1 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 4] into %2 : tensor<4x4x8x8x4x4xi32> -> tensor<4x4x32x32xi32>
      %unpack_0 = tensor.unpack %unpack inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %3 : tensor<4x4x32x32xi32> -> tensor<128x128xi32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %unpack_0 into %arg6[0, 0] [128, 128] [1, 1] : tensor<128x128xi32> into tensor<128x128xi32>
      }
    }
    return %0 : tensor<128x128xi32>
  }
}

// -----

// The following tests relies on consumer fusion being applied for a few iterations to fuse the penultimate unpack into the last innermost scf.forall.
// CHECK-LABEL: @matmul_multiple_fusion_iterations
// CHECK:       %[[EXTRACT_SLICE_0:.+]] = tensor.extract_slice %{{.+}}[%{{.+}}, %{{.+}}] [256, 256] [1, 1]
// CHECK:       %[[FORALL_0:.+]]:2 = scf.forall (%[[ARG0:.+]], %[[ARG1:.+]]) = (0, 0) to (8, 8) step (4, 4) shared_outs(%[[MATMUL_OUT:.+]] = %{{.*}}, %[[UNPACK_OUT:.+]] = %{{.*}})
// CHECK:         %[[FORALL_1:.+]]:2 = scf.forall (%[[ARG2:.+]], %[[ARG3:.+]]) in (4, 4) shared_outs(%[[MATMUL_LOCAL_OUT:.+]] = %{{.*}}, %[[UNPACK_LOCAL_OUT:.+]] = %{{.*}})
// CHECK-SAME:    {
// CHECK:           %[[MATMUL:.+]] = linalg.generic
// CHECK-DAG:       %[[EXTRACT_SLICE_1:.+]] = tensor.extract_slice %[[UNPACK_LOCAL_OUT]][%[[ARG2]], %[[ARG3]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1]
// CHECK-DAG:       %[[UNPACK:.+]] = tensor.unpack %[[MATMUL]] outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 4] into %[[EXTRACT_SLICE_1]]
// CHECK:           scf.forall.in_parallel {
// CHECK-DAG:         tensor.parallel_insert_slice %[[MATMUL]] into %[[MATMUL_LOCAL_OUT]][%[[ARG2]], %[[ARG3]], 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1]
// CHECK-DAG:         tensor.parallel_insert_slice %[[UNPACK]] into %[[UNPACK_LOCAL_OUT]][%[[ARG2]], %[[ARG3]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1]
// CHECK:           }
// CHECK:         }
// CHECK:       scf.forall.in_parallel {
// CHECK-DAG:     tensor.parallel_insert_slice %[[FORALL_1]]#0 into %[[MATMUL_OUT]][%[[ARG0]], %[[ARG1]], 0, 0, 0, 0] [4, 4, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1]
// CHECK-DAG:     tensor.parallel_insert_slice %[[FORALL_1]]#1 into %[[UNPACK_OUT]][%[[ARG0]], %[[ARG1]], 0, 0] [4, 4, 32, 32] [1, 1, 1, 1]
// CHECK:       }
// CHECK:       tensor.unpack %[[FORALL_0]]#1 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[EXTRACT_SLICE_0]]
#config = #iree_codegen.lowering_config<tile_sizes = [[256, 256], [4, 4, 0], [0, 0, 1], [1, 1, 0, 0, 0, 0]]>
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d2, d4, d5, d8, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
#packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 64], transposePackIndices = [0, 1], unpackEmpty = [false, false], innerPerm = [[0, 1], [1, 0]], outerPerm = [[0, 1], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module {
  // expected-error @+1 {{Maximum number of iterations reached, consumer fusion is likely stuck in an infinite loop.}}
  func.func @matmul_multiple_fusion_iterations() {
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<1x1x8x8x8x4xbf16, 2 : i32>
    %alloc_0 = memref.alloc() : memref<1x1x8x8x4x8xbf16, 2 : i32>
    %alloc_1 = memref.alloc() : memref<8x8x32x32xf32, 1 : i32>
    %alloc_2 = memref.alloc() : memref<8x8x64x32xbf16, 1 : i32>
    %alloc_3 = memref.alloc() : memref<8x8x32x64xbf16, 1 : i32>
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<512x512xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<512x4096xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<512x4096xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x512xbf16>> -> tensor<512x512xbf16>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x4096xbf16>> -> tensor<512x4096xbf16>
    %5 = tensor.empty() : tensor<512x4096xf32>
    %6 = scf.forall (%arg0, %arg1) = (0, 0) to (512, 4096) step (256, 256) shared_outs(%arg2 = %5) -> (tensor<512x4096xf32>) {
      %extracted_slice = tensor.extract_slice %3[%arg0, 0] [256, 512] [1, 1] : tensor<512x512xbf16> to tensor<256x512xbf16>
      %extracted_slice_4 = tensor.extract_slice %4[0, %arg1] [512, 256] [1, 1] : tensor<512x4096xbf16> to tensor<512x256xbf16>
      %extracted_slice_5 = tensor.extract_slice %arg2[%arg0, %arg1] [256, 256] [1, 1] : tensor<512x4096xf32> to tensor<256x256xf32>
      %7 = bufferization.to_tensor %alloc_3 restrict writable : memref<8x8x32x64xbf16, 1 : i32> to tensor<8x8x32x64xbf16>
      %pack = tensor.pack %extracted_slice outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 64] into %7 : tensor<256x512xbf16> -> tensor<8x8x32x64xbf16>
      %8 = bufferization.to_tensor %alloc_2 restrict writable : memref<8x8x64x32xbf16, 1 : i32> to tensor<8x8x64x32xbf16>
      %pack_6 = tensor.pack %extracted_slice_4 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [64, 32] into %8 : tensor<512x256xbf16> -> tensor<8x8x64x32xbf16>
      %9 = bufferization.to_tensor %alloc_1 restrict writable : memref<8x8x32x32xf32, 1 : i32> to tensor<8x8x32x32xf32>
      %10 = tensor.empty() : tensor<8x8x8x8x4x4xf32>
      %11 = scf.forall (%arg3, %arg4) = (0, 0) to (8, 8) step (4, 4) shared_outs(%arg5 = %10) -> (tensor<8x8x8x8x4x4xf32>) {
        %extracted_slice_8 = tensor.extract_slice %pack[%arg3, 0, 0, 0] [4, 8, 32, 64] [1, 1, 1, 1] : tensor<8x8x32x64xbf16> to tensor<4x8x32x64xbf16>
        %extracted_slice_9 = tensor.extract_slice %pack_6[%arg4, 0, 0, 0] [4, 8, 64, 32] [1, 1, 1, 1] : tensor<8x8x64x32xbf16> to tensor<4x8x64x32xbf16>
        %extracted_slice_10 = tensor.extract_slice %extracted_slice_8[0, 7, 0, 0] [4, 1, 32, 64] [1, 1, 1, 1] : tensor<4x8x32x64xbf16> to tensor<4x1x32x64xbf16>
        %extracted_slice_11 = tensor.extract_slice %extracted_slice_9[0, 7, 0, 0] [4, 1, 64, 32] [1, 1, 1, 1] : tensor<4x8x64x32xbf16> to tensor<4x1x64x32xbf16>
        %12 = tensor.empty() : tensor<4x4x8x8x4x4xf32>
        %13 = scf.forall (%arg6, %arg7) in (4, 4) shared_outs(%arg8 = %12) -> (tensor<4x4x8x8x4x4xf32>) {
          %extracted_slice_12 = tensor.extract_slice %extracted_slice_10[%arg6, 0, 0, 0] [1, 1, 32, 64] [1, 1, 1, 1] : tensor<4x1x32x64xbf16> to tensor<1x1x32x64xbf16>
          %14 = bufferization.to_tensor %alloc_0 restrict writable : memref<1x1x8x8x4x8xbf16, 2 : i32> to tensor<1x1x8x8x4x8xbf16>
          %pack_13 = tensor.pack %extracted_slice_12 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %14 : tensor<1x1x32x64xbf16> -> tensor<1x1x8x8x4x8xbf16>
          %extracted_slice_14 = tensor.extract_slice %extracted_slice_11[%arg7, 0, 0, 0] [1, 1, 64, 32] [1, 1, 1, 1] : tensor<4x1x64x32xbf16> to tensor<1x1x64x32xbf16>
          %15 = bufferization.to_tensor %alloc restrict writable : memref<1x1x8x8x8x4xbf16, 2 : i32> to tensor<1x1x8x8x8x4xbf16>
          %pack_15 = tensor.pack %extracted_slice_14 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 4] into %15 : tensor<1x1x64x32xbf16> -> tensor<1x1x8x8x8x4xbf16>
          %extracted_slice_16 = tensor.extract_slice %arg8[%arg6, %arg7, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1] : tensor<4x4x8x8x4x4xf32> to tensor<1x1x8x8x4x4xf32>
          %16 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack_13, %pack_15 : tensor<1x1x8x8x4x8xbf16>, tensor<1x1x8x8x8x4xbf16>) outs(%extracted_slice_16 : tensor<1x1x8x8x4x4xf32>) attrs =  {lowering_config = #config, packing_config = #packingConfig} {
          ^bb0(%in: bf16, %in_17: bf16, %out: f32):
            %17 = arith.extf %in : bf16 to f32
            %18 = arith.extf %in_17 : bf16 to f32
            %19 = arith.mulf %17, %18 : f32
            %20 = arith.addf %out, %19 : f32
            linalg.yield %20 : f32
          } -> tensor<1x1x8x8x4x4xf32>
          scf.forall.in_parallel {
            tensor.parallel_insert_slice %16 into %arg8[%arg6, %arg7, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1] : tensor<1x1x8x8x4x4xf32> into tensor<4x4x8x8x4x4xf32>
          }
        } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %13 into %arg5[%arg3, %arg4, 0, 0, 0, 0] [4, 4, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1] : tensor<4x4x8x8x4x4xf32> into tensor<8x8x8x8x4x4xf32>
        }
      } {mapping = [#gpu.block<y>, #gpu.block<x>]}
      %unpack = tensor.unpack %11 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 4] into %9 : tensor<8x8x8x8x4x4xf32> -> tensor<8x8x32x32xf32>
      %unpack_7 = tensor.unpack %unpack inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %extracted_slice_5 : tensor<8x8x32x32xf32> -> tensor<256x256xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %unpack_7 into %arg2[%arg0, %arg1] [256, 256] [1, 1] : tensor<256x256xf32> into tensor<512x4096xf32>
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [512, 4096], strides = [1, 1] : tensor<512x4096xf32> -> !flow.dispatch.tensor<writeonly:tensor<512x4096xf32>>
    memref.dealloc %alloc_3 : memref<8x8x32x64xbf16, 1 : i32>
    memref.dealloc %alloc_2 : memref<8x8x64x32xbf16, 1 : i32>
    memref.dealloc %alloc_1 : memref<8x8x32x32xf32, 1 : i32>
    memref.dealloc %alloc_0 : memref<1x1x8x8x4x8xbf16, 2 : i32>
    memref.dealloc %alloc : memref<1x1x8x8x8x4xbf16, 2 : i32>
    return
  }
}
