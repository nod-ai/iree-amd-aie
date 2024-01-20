// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-peel-for-loop))"  --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
func.func @peel_example() -> i32 {
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

//      CHECK: func.func @peel_example()
//      CHECK:   %[[C0_I32:.*]] = arith.constant 0 : i32
//      CHECK:   %[[C0:.*]] = arith.constant 0 : index
//      CHECK:   %[[C4:.*]] = arith.constant 4 : index
//      CHECK:   %[[C17:.*]] = arith.constant 17 : index
//      CHECK:   %[[C4_0:.*]] = arith.constant 4 : index
//      CHECK:   %[[FIRST:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C4_0]]
// CHECK-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32) {
//      CHECK:   }
//      CHECK:   %[[RESULT:.*]] = scf.for %[[IV:.*]] = %[[C4_0]] to %[[C17]]
// CHECK-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[FIRST]]) -> (i32) {
//      CHECK:   }
//      CHECK:   return %[[RESULT]]
