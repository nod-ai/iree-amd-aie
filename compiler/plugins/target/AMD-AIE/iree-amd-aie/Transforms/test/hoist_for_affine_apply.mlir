// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-amdaie-hoist-for-affine-apply))" %s | FileCheck %s

// CHECK-LABEL: @hoist_affine_map
#map = affine_map<(d0) -> (d0 * 32)>
#map1 = affine_map<(d0) -> (d0 * 64)>
func.func @hoist_affine_map(%lb1: index, %ub1: index, %step1: index, %lb2: index, %ub2: index, %step2: index) {
  scf.for %i = %lb1 to %ub1 step %step1 {
    scf.for %j = %lb2 to %ub2 step %step2 {
      %0 = affine.apply #map(%i)
      %1 = affine.apply #map1(%i)
    }
  }
  // CHECK:       scf.for %[[IV1:.+]] =
  // CHECK-DAG:       %[[MAP0:.+]] = affine.apply #map(%[[IV1]])
  // CHECK-DAG:       %[[MAP1:.+]] = affine.apply #map1(%[[IV1]])
  // CHECK:         scf.for
  // CHECK-NOT:       %[[MAP2:.+]] = affine.apply #map(%[[IV1]])
  // CHECK-NOT:       %[[MAP3:.+]] = affine.apply #map1(%[[IV1]])
  return
}

// -----

// CHECK-LABEL: @no_hoist
#map = affine_map<(d0) -> (d0 * 32)>
#map1 = affine_map<(d0) -> (d0 * 64)>
func.func @no_hoist(%lb1: index, %ub1: index, %step1: index, %lb2: index, %ub2: index, %step2: index) {
  scf.for %i = %lb1 to %ub1 step %step1 {
    scf.for %j = %lb2 to %ub2 step %step2 {
      %0 = affine.apply #map(%j)
      %1 = affine.apply #map1(%j)
    }
  }
  // CHECK:       scf.for %[[IV1:.+]] =
  // CHECK-NOT:     %[[MAP0:.+]] = affine.apply #map(%[[IV1]])
  // CHECK-NOT:     %[[MAP1:.+]] = affine.apply #map1(%[[IV1]])
  // CHECK:         scf.for %[[IV2:.+]] =
  // CHECK-DAG:       %[[MAP2:.+]] = affine.apply #map(%[[IV2]])
  // CHECK-DAG:       %[[MAP3:.+]] = affine.apply #map1(%[[IV2]])
  return
}
