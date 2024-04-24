// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-amdaie-fuse-consumer-into-loop))' %s | FileCheck %s --check-prefix=FORALL
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-amdaie-fuse-consumer-into-loop{use-scf-for=true}))' %s | FileCheck %s --check-prefix=FOR

// FORALL-LABEL: @fuse_consumer_into_scfforall
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @fuse_consumer_into_scfforall(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    //      FORALL:   %[[FINAL:.*]] = scf.forall
    // FORALL-SAME:                      shared_outs(%[[ITER_ARG_FINAL:.*]] = %{{.*}})
    %0 = scf.forall (%arg3, %arg4) in (2, 2) shared_outs(%arg5 = %arg2) -> (tensor<64x64xf32>) {
        //      FORALL:   %[[FIRST_LOOP:.*]] = scf.forall
        %1 = scf.forall (%arg6, %arg7) in (2, 2) shared_outs(%arg8 = %arg5) -> (tensor<64x64xf32>) {
            %extracted_slice = tensor.extract_slice %arg8[%arg6, %arg7] [32, 32] [1, 1] : tensor<64x64xf32> to tensor<32x32xf32>
            %6 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%extracted_slice : tensor<32x32xf32>) -> tensor<32x32xf32>
            scf.forall.in_parallel {
                tensor.parallel_insert_slice %6 into %arg8[%arg6, %arg7] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<64x64xf32>
            }
        }
        //      FORALL:   %[[ELEM_OPERAND_2:.*]] = tensor.empty() : tensor<64x64xf32>
        //      FORALL:   %[[ELEM_OUT:.*]] = tensor.empty() : tensor<64x64xf32>
        //      FORALL:   %[[SECOND_LOOP:.*]]:2 = scf.forall (%[[IV0:.*]], %[[IV1:.*]]) in (2, 2) shared_outs(%[[ITER_ARG_1:.*]] = %[[FIRST_LOOP]], %[[ITER_ARG_2:.*]] = %[[ELEM_OUT]])
        %2 = scf.forall (%arg6, %arg7) in (2, 2) shared_outs(%arg8 = %1) -> (tensor<64x64xf32>) {
            //      FORALL:   %[[MATMUL:.*]] = linalg.matmul
            //      FORALL:   %[[OPERAND1:.*]] = tensor.extract_slice %[[MATMUL]][%[[IV0]], %[[IV1]]] [32, 32] [1, 1]
            //      FORALL:   %[[OPERAND2:.*]] = tensor.extract_slice %[[ELEM_OPERAND_2]][%[[IV0]], %[[IV1]]] [32, 32] [1, 1]
            //      FORALL:   %[[OPERAND3:.*]] = tensor.extract_slice %[[ITER_ARG_2]][%[[IV0]], %[[IV1]]] [32, 32] [1, 1]
            //      FORALL:   %[[FUSED_CONSUMER:.*]] = linalg.generic
            // FORALL-SAME:                      ins(%[[OPERAND1]], %[[OPERAND2]] :
            // FORALL-SAME:                      outs(%[[OPERAND3]] :
            //      FORALL:   {
            //      FORALL:         arith.addf   
            //      FORALL:   }
            //      FORALL:   scf.forall.in_parallel {
            //      FORALL:      tensor.parallel_insert_slice %[[FUSED_CONSUMER]] into %[[ITER_ARG_2]][%[[IV0]], %[[IV1]]] [32, 32] [1, 1]
            //      FORALL:      tensor.parallel_insert_slice %[[MATMUL]] into %[[ITER_ARG_1]][%[[IV0]], %[[IV1]]] [32, 32] [1, 1]
            //      FORALL:   }
            %extracted_slice = tensor.extract_slice %arg8[%arg6, %arg7] [32, 32] [1, 1] : tensor<64x64xf32> to tensor<32x32xf32>
            %6 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%extracted_slice : tensor<32x32xf32>) -> tensor<32x32xf32>
            scf.forall.in_parallel {
                tensor.parallel_insert_slice %6 into %arg8[%arg6, %arg7] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<64x64xf32>
            }
        }
        %3 = tensor.empty() : tensor<64x64xf32>
        %4 = tensor.empty() : tensor<64x64xf32>
        %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%4 : tensor<64x64xf32>) {
        ^bb0(%in: f32, %in_25: f32, %out: f32):
            %21 = arith.addf %in, %in_25 : f32
            linalg.yield %21 : f32
        } -> tensor<64x64xf32>
        //      FORALL: scf.forall.in_parallel
        //      FORALL:      tensor.parallel_insert_slice %[[SECOND_LOOP]]#1 into %[[ITER_ARG_FINAL]]
        //      FORALL: }
        scf.forall.in_parallel {
            tensor.parallel_insert_slice %5 into %arg5[%arg3, %arg4] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<64x64xf32>
        }
    }
    //      FORALL: return %[[FINAL]]
    return %0 : tensor<64x64xf32>
}

// -----

// FOR-LABEL: @fuse_consumer_into_scffor
func.func @fuse_consumer_into_scffor(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    //      FOR:   %[[FINAL:.*]] = scf.for %{{.*}} iter_args(%[[ITER_ARG_FINAL:.*]] = %{{.*}}) ->
    %0 = scf.for %arg3 = %c0 to %c64 step %c4 iter_args(%arg4 = %arg2) -> (tensor<64x64xf32>) {
        //      FOR:   %[[FIRST_LOOP:.*]] = scf.for
        %1 = scf.for %arg5 = %c0 to %c64 step %c4 iter_args(%arg6 = %arg4) -> (tensor<64x64xf32>) {
            %extracted_slice = tensor.extract_slice %arg6[%arg5, %arg5] [32, 32] [1, 1] : tensor<64x64xf32> to tensor<32x32xf32>
            %6 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%extracted_slice : tensor<32x32xf32>) -> tensor<32x32xf32>
            %inserted_slice = tensor.insert_slice %6 into %arg6[%arg5, %arg5] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<64x64xf32>
            scf.yield %inserted_slice : tensor<64x64xf32>
        }
        //      FOR:   %[[ELEM_OPERAND_2:.*]] = tensor.empty() : tensor<64x64xf32>
        //      FOR:   %[[ELEM_OUT:.*]] = tensor.empty() : tensor<64x64xf32>
        //      FOR:   %[[SECOND_LOOP:.*]]:2 = scf.for %[[IV0:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER_ARG_1:.*]] = %[[FIRST_LOOP]], %[[ITER_ARG_2:.*]] = %[[ELEM_OUT]])
        %2 = scf.for %arg5 = %c0 to %c64 step %c4 iter_args(%arg6 = %1) -> (tensor<64x64xf32>) {
            //      FOR:   %[[MATMUL:.*]] = linalg.matmul
            //      FOR:   %[[YIELD_MATMUL:.*]] = tensor.insert_slice %[[MATMUL]] into %[[ITER_ARG_1]][%[[IV0]], %[[IV0]]] [32, 32] [1, 1]
            //      FOR:   %[[OPERAND1:.*]] = tensor.extract_slice %[[YIELD_MATMUL]][%[[IV0]], %[[IV0]]] [32, 32] [1, 1]
            //      FOR:   %[[OPERAND2:.*]] = tensor.extract_slice %[[ELEM_OPERAND_2]][%[[IV0]], %[[IV0]]] [32, 32] [1, 1]
            //      FOR:   %[[OPERAND3:.*]] = tensor.extract_slice %[[ITER_ARG_2]][%[[IV0]], %[[IV0]]] [32, 32] [1, 1]
            //      FOR:   %[[FUSED_CONSUMER:.*]] = linalg.elemwise_binary
            // FOR-SAME:                      ins(%[[OPERAND1]], %[[OPERAND2]] :
            // FOR-SAME:                      outs(%[[OPERAND3]] :
            //      FOR:   %[[YIELD_ELEM:.*]] = tensor.insert_slice %[[FUSED_CONSUMER]] into %[[ITER_ARG_2]][%[[IV0]], %[[IV0]]] [32, 32] [1, 1]
            //      FOR:   scf.yield %[[YIELD_MATMUL]], %[[YIELD_ELEM]]
            %extracted_slice = tensor.extract_slice %arg6[%arg5, %arg5] [32, 32] [1, 1] : tensor<64x64xf32> to tensor<32x32xf32>
            %6 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%extracted_slice : tensor<32x32xf32>) -> tensor<32x32xf32>
            %inserted_slice = tensor.insert_slice %6 into %arg6[%arg5, %arg5] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<64x64xf32>
            scf.yield %inserted_slice : tensor<64x64xf32>
        }
        %3 = tensor.empty() : tensor<64x64xf32>
        %4 = tensor.empty() : tensor<64x64xf32>
        %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%2, %3 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
        //      FOR:  scf.yield %[[SECOND_LOOP]]#1
        scf.yield %5 : tensor<64x64xf32>
    }
    //      FOR: return %[[FINAL]]
    return %0 : tensor<64x64xf32>
}
