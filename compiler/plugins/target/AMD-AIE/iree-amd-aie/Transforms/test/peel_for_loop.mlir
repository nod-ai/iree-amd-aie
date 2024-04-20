// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-peel-for-loop{peeling-type=first}))"  --split-input-file %s | FileCheck %s --check-prefix=PEEL-FIRST
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-peel-for-loop{peeling-type=last}))"  --split-input-file %s | FileCheck %s --check-prefix=PEEL-LAST
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-peel-for-loop{peeling-type=first-last}))"  --split-input-file %s | FileCheck %s --check-prefix=FIRST-LAST
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-peel-for-loop))"  --split-input-file %s | FileCheck %s --check-prefix=DEFAULT-FLAG
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-peel-for-loop))"  --split-input-file %s | FileCheck %s --check-prefix=OVERWRITE-FLAG

#map = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
func.func @peel_indivisible_example() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %lb = arith.constant 0 : index
  %step = arith.constant 4 : index
  %ub = arith.constant 17 : index
  %r = scf.for %iv = %lb to %ub step %step iter_args(%arg = %c0_i32) -> i32 {
    %s = affine.min #map(%ub, %iv)[%step]
    %casted = arith.index_cast %s : index to i32
    %0 = arith.addi %arg, %casted : i32
    scf.yield %0 : i32
  }
  return %r : i32
}

// PEEL-FIRST-LABEL: func.func @peel_indivisible_example()
//       PEEL-FIRST:   %[[C0_I32:.*]] = arith.constant 0 : i32
//       PEEL-FIRST:   %[[C0:.*]] = arith.constant 0 : index
//       PEEL-FIRST:   %[[C4:.*]] = arith.constant 4 : index
//       PEEL-FIRST:   %[[C17:.*]] = arith.constant 17 : index
//       PEEL-FIRST:   %[[C4_0:.*]] = arith.constant 4 : index
//       PEEL-FIRST:   %[[FIRST:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C4_0]]
//  PEEL-FIRST-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32) {
//       PEEL-FIRST:   }
//       PEEL-FIRST:   %[[RESULT:.*]] = scf.for %[[IV1:.*]] = %[[C4_0]] to %[[C17]]
//  PEEL-FIRST-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[FIRST]]) -> (i32) {
//       PEEL-FIRST:   }
//       PEEL-FIRST:   return %[[RESULT]]

// PEEL-LAST-LABEL: func.func @peel_indivisible_example()
//       PEEL-LAST:   %[[C0_I32:.*]] = arith.constant 0 : i32
//       PEEL-LAST:   %[[C0:.*]] = arith.constant 0 : index
//       PEEL-LAST:   %[[C4:.*]] = arith.constant 4 : index
//       PEEL-LAST:   %[[C17:.*]] = arith.constant 17 : index
//       PEEL-LAST:   %[[C16:.*]] = arith.constant 16 : index
//       PEEL-LAST:   %[[MAIN:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C16]]
//  PEEL-LAST-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32) {
//       PEEL-LAST:   }
//       PEEL-LAST:   %[[LAST:.*]] = scf.for %[[IV1:.*]] = %[[C16]] to %[[C17]]
//  PEEL-LAST-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[MAIN]]) -> (i32) {
//       PEEL-LAST:   }
//       PEEL-LAST:   return %[[LAST]]

// FIRST-LAST-LABEL: func.func @peel_indivisible_example()
//       FIRST-LAST:   %[[C0_I32:.*]] = arith.constant 0 : i32
//       FIRST-LAST:   %[[C0:.*]] = arith.constant 0 : index
//       FIRST-LAST:   %[[C4:.*]] = arith.constant 4 : index
//       FIRST-LAST:   %[[C17:.*]] = arith.constant 17 : index
//       FIRST-LAST:   %[[C4_0:.*]] = arith.constant 4 : index
//       FIRST-LAST:   %[[FIRST:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C4_0]]
//  FIRST-LAST-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32) {
//       FIRST-LAST:   }
//       FIRST-LAST:   %[[C16:.*]] = arith.constant 16 : index
//       FIRST-LAST:   %[[MAIN:.*]] = scf.for %[[IV1:.*]] = %[[C4_0]] to %[[C16]]
//  FIRST-LAST-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[FIRST]]) -> (i32) {
//       FIRST-LAST:   }
//       FIRST-LAST:   %[[LAST:.*]] = scf.for %[[IV2:.*]] = %[[C16]] to %[[C17]]
//  FIRST-LAST-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[MAIN]]) -> (i32) {
//       FIRST-LAST:   }
//       FIRST-LAST:   return %[[LAST]]

// DEFAULT-FLAG-LABEL: func.func @peel_indivisible_example()
//       DEFAULT-FLAG:   %[[C0_I32:.*]] = arith.constant 0 : i32
//       DEFAULT-FLAG:   %[[C0:.*]] = arith.constant 0 : index
//       DEFAULT-FLAG:   %[[C4:.*]] = arith.constant 4 : index
//       DEFAULT-FLAG:   %[[C17:.*]] = arith.constant 17 : index
//       DEFAULT-FLAG:   %[[C4_0:.*]] = arith.constant 4 : index
//       DEFAULT-FLAG:   %[[FIRST:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C4_0]]
//  DEFAULT-FLAG-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32) {
//       DEFAULT-FLAG:   }
//       DEFAULT-FLAG:   %[[RESULT:.*]] = scf.for %[[IV1:.*]] = %[[C4_0]] to %[[C17]]
//  DEFAULT-FLAG-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[FIRST]]) -> (i32) {
//       DEFAULT-FLAG:   }
//       DEFAULT-FLAG:   return %[[RESULT]]

// -----

#map = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
func.func @peel_divisible_example() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %lb = arith.constant 0 : index
  %step = arith.constant 4 : index
  %ub = arith.constant 20 : index
  %r = scf.for %iv = %lb to %ub step %step iter_args(%arg = %c0_i32) -> i32 {
    %s = affine.min #map(%ub, %iv)[%step]
    %casted = arith.index_cast %s : index to i32
    %0 = arith.addi %arg, %casted : i32
    scf.yield %0 : i32
  }
  return %r : i32
}

// PEEL-FIRST-LABEL: func.func @peel_divisible_example()
//       PEEL-FIRST:   %[[C0_I32:.*]] = arith.constant 0 : i32
//       PEEL-FIRST:   %[[C0:.*]] = arith.constant 0 : index
//       PEEL-FIRST:   %[[C4:.*]] = arith.constant 4 : index
//       PEEL-FIRST:   %[[C20:.*]] = arith.constant 20 : index
//       PEEL-FIRST:   %[[C4_0:.*]] = arith.constant 4 : index
//       PEEL-FIRST:   %[[FIRST:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C4_0]]
//  PEEL-FIRST-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32) {
//       PEEL-FIRST:   }
//       PEEL-FIRST:   %[[RESULT:.*]] = scf.for %[[IV1:.*]] = %[[C4_0]] to %[[C20]]
//  PEEL-FIRST-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[FIRST]]) -> (i32) {
//       PEEL-FIRST:   }
//       PEEL-FIRST:   return %[[RESULT]]

// PEEL-LAST-LABEL: func.func @peel_divisible_example()
//       PEEL-LAST:   %[[C0_I32:.*]] = arith.constant 0 : i32
//       PEEL-LAST:   %[[C0:.*]] = arith.constant 0 : index
//       PEEL-LAST:   %[[C4:.*]] = arith.constant 4 : index
//       PEEL-LAST:   %[[C20:.*]] = arith.constant 20 : index
//       PEEL-LAST:   %[[C16:.*]] = arith.constant 16 : index
//       PEEL-LAST:   %[[MAIN:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C16]]
//  PEEL-LAST-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32) {
//       PEEL-LAST:   }
//       PEEL-LAST:   %[[LAST:.*]] = scf.for %[[IV1:.*]] = %[[C16]] to %[[C20]]
//  PEEL-LAST-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[MAIN]]) -> (i32) {
//       PEEL-LAST:   }
//       PEEL-LAST:   return %[[LAST]]

// FIRST-LAST-LABEL: func.func @peel_divisible_example()
//       FIRST-LAST:   %[[C0_I32:.*]] = arith.constant 0 : i32
//       FIRST-LAST:   %[[C0:.*]] = arith.constant 0 : index
//       FIRST-LAST:   %[[C4:.*]] = arith.constant 4 : index
//       FIRST-LAST:   %[[C20:.*]] = arith.constant 20 : index
//       FIRST-LAST:   %[[C4_0:.*]] = arith.constant 4 : index
//       FIRST-LAST:   %[[FIRST:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C4_0]]
//  FIRST-LAST-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32) {
//       FIRST-LAST:   }
//       FIRST-LAST:   %[[C16:.*]] = arith.constant 16 : index
//       FIRST-LAST:   %[[MAIN:.*]] = scf.for %[[IV1:.*]] = %[[C4_0]] to %[[C16]]
//  FIRST-LAST-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[FIRST]]) -> (i32) {
//       FIRST-LAST:   }
//       FIRST-LAST:   %[[LAST:.*]] = scf.for %[[IV2:.*]] = %[[C16]] to %[[C20]]
//  FIRST-LAST-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[MAIN]]) -> (i32) {
//       FIRST-LAST:   }
//       FIRST-LAST:   return %[[LAST]]

// -----

#map = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
func.func @no_peeling_example() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %lb = arith.constant 0 : index
  %step = arith.constant 4 : index
  %ub = arith.constant 4 : index
  %r = scf.for %iv = %lb to %ub step %step iter_args(%arg = %c0_i32) -> i32 {
    %s = affine.min #map(%ub, %iv)[%step]
    %casted = arith.index_cast %s : index to i32
    %0 = arith.addi %arg, %casted : i32
    scf.yield %0 : i32
  }
  return %r : i32
}

//      PEEL-FIRST: #[[MAP:.*]] = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
//      PEEL-FIRST: func @no_peeling_example(
//      PEEL-FIRST:   %[[C0_I32:.*]] = arith.constant 0 : i32
//      PEEL-FIRST:   %[[C0:.*]] = arith.constant 0 : index
//      PEEL-FIRST:   %[[C4:.*]] = arith.constant 4 : index
//      PEEL-FIRST:   %[[C4_0:.*]] = arith.constant 4 : index
//      PEEL-FIRST:   %[[RESULT:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C4_0]]
// PEEL-FIRST-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32) {
//      PEEL-FIRST:     %[[MIN:.*]] = affine.min #[[MAP]](%[[C4_0]], %[[IV]])[%[[C4]]]
//      PEEL-FIRST:     %[[CAST:.*]] = arith.index_cast %[[MIN]] : index to i32
//      PEEL-FIRST:     %[[ADD:.*]] = arith.addi %[[ACC]], %[[CAST]] : i32
//      PEEL-FIRST:     scf.yield %[[ADD]]
//      PEEL-FIRST:   }
//      PEEL-FIRST:   return %[[RESULT]]

// -----

#map = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
func.func @two_iteration_example() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %lb = arith.constant 2 : index
  %step = arith.constant 4 : index
  %ub = arith.constant 10 : index
  %r = scf.for %iv = %lb to %ub step %step iter_args(%arg = %c0_i32) -> i32 {
    %s = affine.min #map(%ub, %iv)[%step]
    %casted = arith.index_cast %s : index to i32
    %0 = arith.addi %arg, %casted : i32
    scf.yield %0 : i32
  }
  return %r : i32
}

// FIRST-LAST-LABEL: func.func @two_iteration_example()
//       FIRST-LAST:   %[[C0_I32:.*]] = arith.constant 0 : i32
//       FIRST-LAST:   %[[C2:.*]] = arith.constant 2 : index
//       FIRST-LAST:   %[[C4:.*]] = arith.constant 4 : index
//       FIRST-LAST:   %[[C10:.*]] = arith.constant 10 : index
//       FIRST-LAST:   %[[C6:.*]] = arith.constant 6 : index
//       FIRST-LAST:   %[[FIRST:.*]] = scf.for %[[IV:.*]] = %[[C2]] to %[[C6]]
//  FIRST-LAST-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32) {
//       FIRST-LAST:   }
//       FIRST-LAST:   %[[LAST:.*]] = scf.for %[[IV1:.*]] = %[[C6]] to %[[C10]]
//  FIRST-LAST-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[FIRST]]) -> (i32) {
//       FIRST-LAST:   }
//       FIRST-LAST:   return %[[LAST]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @matmul_elementwise_i32(%arg0: tensor<1024x512xi8>, %arg1: tensor<512x1024xi8>, %arg2: tensor<1024x1024xi32>) -> tensor<1024x1024xi32> {
  %c32 = arith.constant 32 : index
  %c512 = arith.constant 512 : index
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<1024x1024xi32>
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<1024x1024xi32>) -> tensor<1024x1024xi32>
  %2 = scf.for %arg3 = %c0 to %c512 step %c32 iter_args(%arg4 = %1) -> (tensor<1024x1024xi32>) {
    %extracted_slice = tensor.extract_slice %arg0[0, %arg3] [1024, 32] [1, 1] : tensor<1024x512xi8> to tensor<1024x32xi8>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg3, 0] [32, 1024] [1, 1] : tensor<512x1024xi8> to tensor<32x1024xi8>
    %4 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<1024x32xi8>, tensor<32x1024xi8>) outs(%arg4 : tensor<1024x1024xi32>) -> tensor<1024x1024xi32>
    scf.yield %4 : tensor<1024x1024xi32>
  }
  %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %arg2 : tensor<1024x1024xi32>, tensor<1024x1024xi32>) outs(%0 : tensor<1024x1024xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %4 = arith.addi %in, %in_0 : i32
    linalg.yield %4 : i32
  } -> tensor<1024x1024xi32>
  return %3 : tensor<1024x1024xi32>
}

// OVERWRITE-FLAG-LABEL: func.func @matmul_elementwise_i32
//       OVERWRITE-FLAG:   %[[C32:.*]] = arith.constant 32 : index
//       OVERWRITE-FLAG:   %[[C512:.*]] = arith.constant 512 : index
//       OVERWRITE-FLAG:   %[[C0_I32:.*]] = arith.constant 0 : i32
//       OVERWRITE-FLAG:   %[[C0:.*]] = arith.constant 0 : index
//       OVERWRITE-FLAG:   %[[FILL:.*]] = linalg.fill
//       OVERWRITE-FLAG:   %[[C32_0:.*]] = arith.constant 32 : index
//       OVERWRITE-FLAG:   %[[FIRST:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C32_0]]
//  OVERWRITE-FLAG-SAME:       step %[[C32]] iter_args(%[[ACC:.*]] = %[[FILL]]) -> (tensor<1024x1024xi32>) {
//       OVERWRITE-FLAG:   }
//       OVERWRITE-FLAG:   %[[C480:.*]] = arith.constant 480 : index
//       OVERWRITE-FLAG:   %[[MAIN:.*]] = scf.for %[[IV1:.*]] = %[[C32_0]] to %[[C480]]
//  OVERWRITE-FLAG-SAME:       step %[[C32]] iter_args(%[[ACC_1:.*]] = %[[FIRST]]) -> (tensor<1024x1024xi32>) {
//       OVERWRITE-FLAG:   }
//       OVERWRITE-FLAG:   %[[LAST:.*]] = scf.for %[[IV2:.*]] = %[[C480]] to %[[C512]]
//  OVERWRITE-FLAG-SAME:       step %[[C32]] iter_args(%[[ACC_2:.*]] = %[[MAIN]]) -> (tensor<1024x1024xi32>) {
//       OVERWRITE-FLAG:   }
