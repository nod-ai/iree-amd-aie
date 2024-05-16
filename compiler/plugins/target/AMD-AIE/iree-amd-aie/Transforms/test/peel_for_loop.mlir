// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-peel-for-loop{peeling-type=first}))"  --split-input-file %s | FileCheck %s --check-prefix=PEEL-FIRST
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-peel-for-loop{peeling-type=last}))"  --split-input-file %s | FileCheck %s --check-prefix=PEEL-LAST
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-peel-for-loop{peeling-type=first-last}))"  --split-input-file %s | FileCheck %s --check-prefix=FIRST-LAST

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
