// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-linalg-function-outlining{call-replication=0})" --verify-diagnostics --split-input-file  %s | FileCheck %s --check-prefix=EMPTY
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-linalg-function-outlining{call-replication=1})" --verify-diagnostics --split-input-file  %s | FileCheck %s --check-prefix=NOT_EMPTY
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-linalg-function-outlining{call-replication=2})" --verify-diagnostics --split-input-file  %s | FileCheck %s --check-prefix=REPLICATED

func.func @reduction(%A: memref<4xbf16>, %B: memref<bf16>) {
  %c2 = arith.constant 2 : index
  %tile = amdaie.tile(%c2, %c2)
  %1 = amdaie.core(%tile, in : [], out : []) {
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    } ins(%A: memref<4xbf16>) outs(%B : memref<bf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    }
    amdaie.end
  }
  return
}

// When call-replication=0, the outlined function should not be called at all.
// EMPTY: func.func private
// EMPTY-NOT: linalg.generic
// EMPTY: return

// The (default) case where call-replication=1, the outlined function should be called once.
// NOT_EMPTY: func.func private
// NOT_EMPTY: linalg.generic
// NOT_EMPTY: return
// NOT_EMPTY-NOT: scf.for
// NOT_EMPTY: func.call @generic_0_outlined

// When call-replication=2, the outlined function should be called twice.
// REPLICATED: func.func private
// REPLICATED: linalg.generic
// REPLICATED: return
// REPLICATED: scf.for
// REPLICATED-SAME: %c0 to %c2_0 step %c1
// REPLICATED: func.call @generic_0_outlined
