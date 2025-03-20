// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-replicate-calls{replication=0})" --verify-diagnostics --split-input-file  %s | FileCheck %s --check-prefix=EMPTY
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-replicate-calls{replication=1})" --verify-diagnostics --split-input-file  %s | FileCheck %s --check-prefix=NOT_EMPTY
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-replicate-calls{replication=2})" --verify-diagnostics --split-input-file  %s | FileCheck %s --check-prefix=REPLICATED


#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module {
  func.func private @foo(%arg0: memref<4xbf16>, %arg1: memref<bf16>) {
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg0 : memref<4xbf16>) outs(%arg1 : memref<bf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    }
    return
  }
  func.func @bar(%arg0: memref<4xbf16>, %arg1: memref<bf16>) {
    %c2 = arith.constant 2 : index
    %tile_2_2 = amdaie.tile(%c2, %c2)
    %0 = amdaie.core(%tile_2_2, in : [], out : []) {
      func.call @foo(%arg0, %arg1) : (memref<4xbf16>, memref<bf16>) -> ()
      amdaie.end
    }
    return
  }
}

// When replication=0, there is one call to an empty version of the callee.
// EMPTY:       func.func private @foo_empty([[ARG0:%.*]]: memref<4xbf16>, [[ARG1:%.*]]: memref<bf16>) {
// EMPTY-NEXT:    return
// EMPTY-NEXT:  }
// EMPTY:       func.func @bar
// EMPTY:       amdaie.core
// EMPTY-NEXT:  func.call @foo_empty
// EMPTY-NEXT:  amdaie.end

// The (default) case where replication=1, the outlined function should be called once.
// NOT_EMPTY:     func.func private
// NOT_EMPTY:     linalg.generic
// NOT_EMPTY:     return
// NOT_EMPTY-NOT: scf.for
// NOT_EMPTY:     func.call @foo

// When replication=2, the outlined function should be called twice.
// REPLICATED:      func.func private
// REPLICATED:      linalg.generic
// REPLICATED:      return
// REPLICATED:      scf.for
// REPLICATED-SAME: %c0 to %c2_0 step %c1
// REPLICATED:      func.call @foo
