// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-hoist-logical-objectfifo)" %s | FileCheck %s

// CHECK-LABEL: @func_hoist
// CHECK-SAME:  %[[ARG0:.+]]: memref<32x64xi32>
// CHECK:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:       %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:       amdaie.logicalobjectfifo.from_memref %[[ARG0]], {%[[TILE_0_0]]}
// CHECK:       scf.forall
// CHECK-NOT:     amdaie.logicalobjectfifo.from_memref
module {
  func.func @func_hoist(%arg0: memref<32x64xi32>) {
    %c0 = arith.constant 0 : index
    %tile_0_0 = amdaie.tile(%c0, %c0)
    scf.forall (%arg1, %arg2) in (1, 2) {
      %obj0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<32x64xi32>>
    }
    return
  }
}


// -----

// CHECK-LABEL: @func_no_hoist
// CHECK-SAME:  %[[ARG0:.+]]: memref<32x64xi32>
// CHECK:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:       scf.forall
// CHECK:         %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:         amdaie.logicalobjectfifo.from_memref %[[ARG0]], {%[[TILE_0_0]]}
module {
  func.func @func_no_hoist(%arg0: memref<32x64xi32>) {
    %c0 = arith.constant 0 : index
    scf.forall (%arg1, %arg2) in (1, 2) {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %obj0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<32x64xi32>>
    }
    return
  }
}

// -----

// CHECK-LABEL: @workgroup_hoist
// CHECK-SAME:  %[[ARG0:.+]]: memref<32x64xi32>
// CHECK:       amdaie.workgroup
// CHECK:         %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:         amdaie.logicalobjectfifo.from_memref %[[ARG0]], {%[[TILE_0_0]]}
// CHECK:         scf.forall
// CHECK-NOT:       amdaie.logicalobjectfifo.from_memref
func.func @workgroup_hoist(%arg0: memref<32x64xi32>) {
  amdaie.workgroup {
    %c0 = arith.constant 0 : index
    %tile_0_0 = amdaie.tile(%c0, %c0)
    scf.forall (%arg1, %arg2) in (1, 2) {
      %obj0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<32x64xi32>>
    }
    amdaie.controlcode {
      amdaie.end
    }
  }
  return
}

// -----

// CHECK-LABEL: @workgroup_no_hoist
// CHECK-SAME:  %[[ARG0:.+]]: memref<32x64xi32>
// CHECK:       amdaie.workgroup
// CHECK:         %[[C0:.+]] = arith.constant 0 : index
// CHECK:         scf.forall
// CHECK:           %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:           amdaie.logicalobjectfifo.from_memref %[[ARG0]], {%[[TILE_0_0]]}
func.func @workgroup_no_hoist(%arg0: memref<32x64xi32>) {
  amdaie.workgroup {
    %c0 = arith.constant 0 : index
    scf.forall (%arg1, %arg2) in (1, 2) {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %obj0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<32x64xi32>>
    }
    amdaie.controlcode {
      amdaie.end
    }
  }
  return
}
// -----

// CHECK-LABEL: @workgroup_no_hoist_outside
// CHECK-SAME:  %[[ARG0:.+]]: memref<32x64xi32>
// CHECK:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:       %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:       amdaie.workgroup
// CHECK:         amdaie.logicalobjectfifo.from_memref %[[ARG0]], {%[[TILE_0_0]]}
func.func @workgroup_no_hoist_outside(%arg0: memref<32x64xi32>) {
  %c0 = arith.constant 0 : index
  %tile_0_0 = amdaie.tile(%c0, %c0)
  amdaie.workgroup {
    %obj0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<32x64xi32>>
    amdaie.controlcode {
      amdaie.end
    }
  }
  return
}

// -----

// CHECK-LABEL: @controlcode_hoist
// CHECK-SAME:  %[[ARG0:.+]]: memref<32x64xi32>
// CHECK:       amdaie.controlcode
// CHECK:         %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:         amdaie.logicalobjectfifo.from_memref %[[ARG0]], {%[[TILE_0_0]]}
// CHECK:         scf.forall
// CHECK:           scf.forall
// CHECK-NOT:         amdaie.logicalobjectfifo.from_memref
func.func @controlcode_hoist(%arg0: memref<32x64xi32>) {
  amdaie.workgroup {
    amdaie.controlcode {
      %c0 = arith.constant 0 : index
      %tile_0_0 = amdaie.tile(%c0, %c0)
      scf.forall (%arg1, %arg2) in (1, 2) {
        scf.forall (%arg3, %arg4) in (1, 2) {
          %obj0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<32x64xi32>>
        }
      }
      amdaie.end
    }
  }
  return
}

// -----

// CHECK-LABEL: @controlcode_partial_hoist
// CHECK-SAME:  %[[ARG0:.+]]: memref<32x64xi32>
// CHECK:       amdaie.controlcode
// CHECK:         %[[C0:.+]] = arith.constant 0 : index
// CHECK:         scf.forall
// CHECK:           %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:           amdaie.logicalobjectfifo.from_memref %[[ARG0]], {%[[TILE_0_0]]}
// CHECK:           scf.forall
// CHECK-NOT:         amdaie.logicalobjectfifo.from_memref
func.func @controlcode_partial_hoist(%arg0: memref<32x64xi32>) {
  amdaie.workgroup {
    amdaie.controlcode {
      %c0 = arith.constant 0 : index
      scf.forall (%arg1, %arg2) in (1, 2) {
        %tile_0_0 = amdaie.tile(%c0, %c0)
        scf.forall (%arg3, %arg4) in (1, 2) {
          %obj0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<32x64xi32>>
        }
      }
      amdaie.end
    }
  }
  return
}

// -----

// CHECK-LABEL: @controlcode_no_hoist
// CHECK-SAME:  %[[ARG0:.+]]: memref<32x64xi32>
// CHECK:       amdaie.controlcode
// CHECK:         %[[C0:.+]] = arith.constant 0 : index
// CHECK:         scf.forall
// CHECK:           scf.forall
// CHECK:             %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:             amdaie.logicalobjectfifo.from_memref %[[ARG0]], {%[[TILE_0_0]]}
func.func @controlcode_no_hoist(%arg0: memref<32x64xi32>) {
  amdaie.workgroup {
    amdaie.controlcode {
      %c0 = arith.constant 0 : index
      scf.forall (%arg1, %arg2) in (1, 2) {
        scf.forall (%arg3, %arg4) in (1, 2) {
          %tile_0_0 = amdaie.tile(%c0, %c0)
          %obj0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<32x64xi32>>
        }
      }
      amdaie.end
    }
  }
  return
}

// -----

// CHECK-LABEL: @controlcode_no_hoist_outside
// CHECK-SAME:  %[[ARG0:.+]]: memref<32x64xi32>
// CHECK:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:       %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:       amdaie.controlcode
// CHECK:         amdaie.logicalobjectfifo.from_memref %[[ARG0]], {%[[TILE_0_0]]}
func.func @controlcode_no_hoist_outside(%arg0: memref<32x64xi32>) {
  amdaie.workgroup {
    %c0 = arith.constant 0 : index
    %tile_0_0 = amdaie.tile(%c0, %c0)
    amdaie.controlcode {
      %obj0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<32x64xi32>>
      amdaie.end
    }
  }
  return
}
