// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-convert-core-forall-to-for,canonicalize)" --split-input-file %s | FileCheck %s

// CHECK-LABEL: @test_single
// CHECK-DAG:   amdaie.core
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C4:.+]] = arith.constant 4 : index
// CHECK:         scf.for %[[ARG0:.+]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK-DAG:       %[[D:.+]]:2 = affine.delinearize_index %[[ARG0]] into (%[[C2]], %[[C2]]) : index, index
// CHECK-DAG:       func.call @callee(%[[D]]#0, %[[D]]#1) : (index, index) -> ()
module @test_single {
  func.func private @callee(%i: index, %j: index)
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c7 = arith.constant 7 : index
  %c11 = arith.constant 11 : index
  %tile_7_11 = amdaie.tile(%c7, %c11)
  %core_7_11 = amdaie.core(%tile_7_11, in : [], out : []) {
    scf.forall (%i, %j) in (2, 2) {
      func.call @callee(%i, %j) : (index, index) -> ()
    }
    amdaie.end
  }
}

// -----

// CHECK-LABEL: @test_multi
// CHECK-DAG:   amdaie.core
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK:         scf.for %[[ARG0:.+]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK-DAG:       %[[D:.+]]:2 = affine.delinearize_index %[[ARG0]] into (%[[C2]], %[[C2]]) : index, index
// CHECK-DAG:       func.call @callee(%[[D]]#0, %[[D]]#1) : (index, index) -> ()
// CHECK-DAG:     scf.for %[[ARG1:.+]] = %[[C0]] to %[[C16]] step %[[C1]] {
// CHECK-DAG:       %[[D:.+]]:2 = affine.delinearize_index %[[ARG0]] into (%[[C4]], %[[C4]]) : index, index
// CHECK-DAG:       func.call @callee(%[[D]]#0, %[[D]]#1) : (index, index) -> ()
module @test_multi {
  func.func private @callee(%i: index, %j: index)
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c7 = arith.constant 7 : index
  %c11 = arith.constant 11 : index
  %tile_7_11 = amdaie.tile(%c7, %c11)
  %core_7_11 = amdaie.core(%tile_7_11, in : [], out : []) {
    scf.forall (%i, %j) in (2, 2) {
      func.call @callee(%i, %j) : (index, index) -> ()
    }
    scf.forall (%i, %j) in (4, 4) {
      func.call @callee(%i, %j) : (index, index) -> ()
    }
    amdaie.end
  }
}

// -----

// CHECK-LABEL: @test_nested
// CHECK-DAG:   amdaie.core
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK:         scf.for %[[ARG0:.+]] = %[[C0]] to %[[C16]] step %[[C1]] {
// CHECK-DAG:       %[[D1:.+]]:2 = affine.delinearize_index %[[ARG0]] into (%[[C4]], %[[C4]]) : index, index
// CHECK-DAG:       scf.for %[[ARG1:.+]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK-DAG:         %[[D2:.+]]:2 = affine.delinearize_index %[[ARG1]] into (%[[C2]], %[[C2]]) : index, index
// CHECK-DAG:         func.call @callee(%[[D1]]#0, %[[D1]]#1, %[[D2]]#0, %[[D2]]#1) : (index, index, index, index) -> ()
module @test_nested {
  func.func private @callee(%i: index, %j: index, %k: index, %l: index)
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c7 = arith.constant 7 : index
  %c11 = arith.constant 11 : index
  %tile_7_11 = amdaie.tile(%c7, %c11)
  %core_7_11 = amdaie.core(%tile_7_11, in : [], out : []) {
    scf.forall (%i, %j) in (4, 4) {
      scf.forall (%k, %l) in (2, 2) {
        func.call @callee(%i, %j, %k, %l) : (index, index, index, index) -> ()
      }
    }
    amdaie.end
  }
}
