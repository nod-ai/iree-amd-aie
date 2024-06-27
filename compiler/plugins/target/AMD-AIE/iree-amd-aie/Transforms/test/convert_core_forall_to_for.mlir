// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-convert-core-forall-to-for,canonicalize)" --split-input-file %s | FileCheck %s

// CHECK-LABEL: @test_single
// CHECK:       aie.core
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:   scf.for %[[ARG0:.+]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK-DAG:     %[[REM:.+]] = arith.remsi %[[ARG0]], %[[C2]] : index
// CHECK-DAG:     %[[DIV:.+]] = arith.divsi %[[ARG0]], %[[C2]] : index
// CHECK-DAG:     func.call @callee(%[[DIV]], %[[REM]]) : (index, index) -> ()
module @test_single {
  func.func private @callee(%i: index, %j: index)
  %tile_0_2 = aie.tile(0, 2)
  %core_0_2 = aie.core(%tile_0_2) {
    scf.forall (%i, %j) in (2, 2) {
      func.call @callee(%i, %j) : (index, index) -> ()
    }
    aie.end
  }
}

// -----

// CHECK-LABEL: @test_multi
// CHECK:       aie.core
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:   scf.for %[[ARG0:.+]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK-DAG:     %[[REM:.+]] = arith.remsi %[[ARG0]], %[[C2]] : index
// CHECK-DAG:     %[[DIV:.+]] = arith.divsi %[[ARG0]], %[[C2]] : index
// CHECK-DAG:     func.call @callee(%[[DIV]], %[[REM]]) : (index, index) -> ()
// CHECK-DAG:   scf.for %[[ARG1:.+]] = %[[C0]] to %[[C16]] step %[[C1]] {
// CHECK-DAG:     %[[REM1:.+]] = arith.remsi %[[ARG1]], %[[C4]] : index
// CHECK-DAG:     %[[DIV1:.+]] = arith.divsi %[[ARG1]], %[[C4]] : index
// CHECK-DAG:     func.call @callee(%[[DIV1]], %[[REM1]]) : (index, index) -> ()
module @test_multi {
  func.func private @callee(%i: index, %j: index)
  %tile_0_2 = aie.tile(0, 2)
  %core_0_2 = aie.core(%tile_0_2) {
    scf.forall (%i, %j) in (2, 2) {
      func.call @callee(%i, %j) : (index, index) -> ()
    }
    scf.forall (%i, %j) in (4, 4) {
      func.call @callee(%i, %j) : (index, index) -> ()
    }
    aie.end
  }
}

// -----

// CHECK-LABEL: @test_nested
// CHECK:       aie.core
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:   scf.for %[[ARG0:.+]] = %[[C0]] to %[[C16]] step %[[C1]] {
// CHECK-DAG:     %[[REM0:.+]] = arith.remsi %[[ARG0]], %[[C4]] : index
// CHECK-DAG:     %[[DIV0:.+]] = arith.divsi %[[ARG0]], %[[C4]] : index
// CHECK-DAG:     scf.for %[[ARG1:.+]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK-DAG:       %[[REM1:.+]] = arith.remsi %[[ARG1]], %[[C2]] : index
// CHECK-DAG:       %[[DIV1:.+]] = arith.divsi %[[ARG1]], %[[C2]] : index
// CHECK-DAG:       func.call @callee(%[[DIV0]], %[[REM0]], %[[DIV1]], %[[REM1]]) : (index, index, index, index) -> ()
module @test_nested {
  func.func private @callee(%i: index, %j: index, %k: index, %l: index)
  %tile_0_2 = aie.tile(0, 2)
  %core_0_2 = aie.core(%tile_0_2) {
    scf.forall (%i, %j) in (4, 4) {
      scf.forall (%k, %l) in (2, 2) {
        func.call @callee(%i, %j, %k, %l) : (index, index, index, index) -> ()
      }
    }
    aie.end
  }
}
