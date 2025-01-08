// RUN: iree-opt --split-input-file  --pass-pipeline="builtin.module(iree-amdaie-linalg-function-outlining{empty-functions=true})" --verify-diagnostics --split-input-file  %s | FileCheck %s --check-prefix=EMPTY
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-linalg-function-outlining{empty-functions=false})" --verify-diagnostics --split-input-file  %s | FileCheck %s --check-prefix=NOT_EMPTY

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

// The (default) case where empty-functions is false, outlining works as usual.
// NOT_EMPTY: func.func private
// NOT_EMPTY: linalg.generic
// NOT_EMPTY: return

// When empty-functions=true, the outlined function shouldn't contain compute.
// EMPTY: func.func private
// EMPTY-NOT: linalg.generic
// EMPTY: return
