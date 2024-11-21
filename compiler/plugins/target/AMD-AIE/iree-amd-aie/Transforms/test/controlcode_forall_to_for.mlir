// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-controlcode-forall-to-for,canonicalize)" --split-input-file %s | FileCheck %s

// CHECK-LABEL: @test_promotion
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   amdaie.controlcode
// CHECK:         func.call @callee(%[[C0]], %[[C0]]) : (index, index) -> ()
module @test_promotion {
  func.func private @callee(%i: index, %j: index)
  amdaie.workgroup {
    amdaie.controlcode {
      scf.forall (%i, %j) in (1, 1) {
        func.call @callee(%i, %j) : (index, index) -> ()
      }
      amdaie.end
    }
  }
}

// -----

// CHECK-LABEL: @test_single
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG:   amdaie.controlcode
// CHECK:         scf.for %[[ARG0:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:           scf.for %[[ARG1:.+]] = %[[C0]] to %[[C3]] step %[[C1]] {
// CHECK:             func.call @callee(%[[ARG0]], %[[ARG1]]) : (index, index) -> ()
module @test_single {
  func.func private @callee(%i: index, %j: index)
  amdaie.workgroup {
    amdaie.controlcode {
      scf.forall (%i, %j) in (2, 3) {
        func.call @callee(%i, %j) : (index, index) -> ()
      }
      amdaie.end
    }
  }
}

// -----

// CHECK-LABEL: @test_multi
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG:   amdaie.controlcode
// CHECK:         scf.for %[[ARG0:.+]] = %[[C0]] to %[[C3]] step %[[C1]] {
// CHECK:           scf.for %[[ARG1:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:             func.call @callee(%[[ARG0]], %[[ARG1]]) : (index, index) -> ()
// CHECK:         scf.for %[[ARG0:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:           scf.for %[[ARG1:.+]] = %[[C0]] to %[[C3]] step %[[C1]] {
// CHECK:             func.call @callee(%[[ARG0]], %[[ARG1]]) : (index, index) -> ()
module @test_multi {
  func.func private @callee(%i: index, %j: index)
  amdaie.workgroup {
    amdaie.controlcode {
      scf.forall (%i, %j) in (3, 2) {
        func.call @callee(%i, %j) : (index, index) -> ()
      }
      scf.forall (%i, %j) in (2, 3) {
        func.call @callee(%i, %j) : (index, index) -> ()
      }
      amdaie.end
    }
  }
}

// -----

// CHECK-LABEL: @test_nested
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG:   amdaie.controlcode
// CHECK:         scf.for %[[ARG0:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:           scf.for %[[ARG1:.+]] = %[[C0]] to %[[C3]] step %[[C1]] {
// CHECK:             scf.for %[[ARG2:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:               scf.for %[[ARG3:.+]] = %[[C0]] to %[[C3]] step %[[C1]] {
// CHECK:                 func.call @callee(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]) : (index, index, index, index) -> ()
module @test_nested {
  func.func private @callee(%i: index, %j: index, %k: index, %l: index)
  amdaie.workgroup {
    amdaie.controlcode {
      scf.forall (%i, %j) in (2, 3) {
        scf.forall (%k, %l) in (2, 3) {
          func.call @callee(%i, %j, %k, %l) : (index, index, index, index) -> ()
        }
      }
      amdaie.end
    }
  }
}

// -----

// CHECK-LABEL: @test_affine_apply
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG:   amdaie.controlcode
// CHECK:         scf.for %[[ARG0:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:           %[[APPLY0:.+]] = affine.apply #map(%[[ARG0]])
// CHECK:           scf.for %[[ARG1:.+]] = %[[C0]] to %[[C3]] step %[[C1]] {
// CHECK:             %[[APPLY1:.+]] = affine.apply #map(%[[ARG1]])
// CHECK:             scf.for %[[ARG2:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:               %[[APPLY2:.+]] = affine.apply #map(%[[ARG2]])
// CHECK:               scf.for %[[ARG3:.+]] = %[[C0]] to %[[C3]] step %[[C1]] {
// CHECK:                 %[[APPLY3:.+]] = affine.apply #map(%[[ARG3]])
// CHECK:                 func.call @callee(%[[APPLY0]], %[[APPLY1]], %[[APPLY2]], %[[APPLY3]]) : (index, index, index, index) -> ()
#map = affine_map<(d0) -> (d0 * 32)>
module @test_affine_apply {
  func.func private @callee(%i: index, %j: index, %k: index, %l: index)
  amdaie.workgroup {
    amdaie.controlcode {
      scf.forall (%i, %j) in (2, 3) {
        scf.forall (%k, %l) in (2, 3) {
          %0 = affine.apply #map(%i)
          %1 = affine.apply #map(%j)
          %2 = affine.apply #map(%k)
          %3 = affine.apply #map(%l)
          func.call @callee(%0, %1, %2, %3) : (index, index, index, index) -> ()
        }
      }
      amdaie.end
    }
  }
}
