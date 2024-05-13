// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-controlcode-loop-unroll,canonicalize))" --split-input-file %s | FileCheck %s

func.func private @callee(%i: index)

// CHECK-LABEL: @unroll_for
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
// CHECK:       amdaie.controlcode
// CHECK:         func.call @callee(%[[C0]])
// CHECK:         func.call @callee(%[[C2]])
// CHECK:         func.call @callee(%[[C4]])
func.func @unroll_for() {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c6 = arith.constant 6 : index
  amdaie.workgroup {
    amdaie.controlcode {
      scf.for %arg0 = %c0 to %c6 step %c2 {
        func.call @callee(%arg0) : (index) -> ()
      }
      amdaie.end
    }
  }
  return
}

// -----

func.func private @callee(%i: index, %j: index)

// CHECK-LABEL: @unroll_forall
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK:       amdaie.controlcode
// CHECK:         func.call @callee(%[[C0]], %[[C0]])
// CHECK:         func.call @callee(%[[C0]], %[[C1]])
func.func @unroll_forall() {
  amdaie.workgroup {
    amdaie.controlcode {
      scf.forall (%i, %j) in (1, 2) {
        func.call @callee(%i, %j) : (index, index) -> ()
      }
      amdaie.end
    }
  }
  return
}

// -----

func.func private @callee(%i: index, %j: index, %k: index, %l: index)
func.func private @callee2(%i: index, %j: index, %k: index)

// CHECK-LABEL: @unroll_nested
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK:       amdaie.controlcode
// CHECK:         func.call @callee(%[[C0]], %[[C0]], %[[C0]], %[[C0]])
// CHECK:         func.call @callee(%[[C0]], %[[C0]], %[[C1]], %[[C0]])
// CHECK:         func.call @callee(%[[C0]], %[[C0]], %[[C2]], %[[C0]])
// CHECK:         func.call @callee2(%[[C0]], %[[C0]], %[[C0]])
// CHECK:         func.call @callee2(%[[C0]], %[[C0]], %[[C1]])
// CHECK:         func.call @callee(%[[C0]], %[[C1]], %[[C0]], %[[C0]])
// CHECK:         func.call @callee(%[[C0]], %[[C1]], %[[C1]], %[[C0]])
// CHECK:         func.call @callee(%[[C0]], %[[C1]], %[[C2]], %[[C0]])
// CHECK:         func.call @callee2(%[[C0]], %[[C1]], %[[C0]])
// CHECK:         func.call @callee2(%[[C0]], %[[C1]], %[[C1]])
func.func @unroll_nested() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  amdaie.workgroup {
    amdaie.controlcode {
      scf.forall (%i, %j) in (1, 2) {
        scf.forall (%k, %l) in (3, 1) {
          func.call @callee(%i, %j, %k, %l) : (index, index, index, index) -> ()
        }
        scf.for %arg0 = %c0 to %c2 step %c1 {
          func.call @callee2(%i, %j, %arg0) : (index, index, index) -> ()
        }
      }
      amdaie.end
    }
  }
  return
}

// -----

func.func private @callee(%i: index)

// CHECK-LABEL: @no_for_unroll_outside_controlcode
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK:       scf.for %[[ARG0:.+]] = %[[C0]] to %[[C6]] step %[[C2]]
// CHECK:         func.call @callee(%[[ARG0]])
// CHECK:       amdaie.controlcode
func.func @no_for_unroll_outside_controlcode() {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c6 = arith.constant 6 : index
  amdaie.workgroup {
    scf.for %arg0 = %c0 to %c6 step %c2 {
      func.call @callee(%arg0) : (index) -> ()
    }
    amdaie.controlcode {
      amdaie.end
    }
  }
  return
}

// -----

func.func private @callee(%i: index, %j: index)

// CHECK-LABEL: @no_forall_unroll_outside_controlcode
// CHECK:       scf.forall (%[[ARG0:.+]], %[[ARG1:.+]]) in (2, 2)
// CHECK:         func.call @callee(%[[ARG0]], %[[ARG1]])
// CHECK:       amdaie.controlcode
func.func @no_forall_unroll_outside_controlcode() {
  amdaie.workgroup {
    scf.forall (%i, %j) in (2, 2) {
      func.call @callee(%i, %j) : (index, index) -> ()
    }
    amdaie.controlcode {
      amdaie.end
    }
  }
  return
}
