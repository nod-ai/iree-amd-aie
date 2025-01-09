// Currently this is the default:
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-linalg-function-outlining{empty-functions=false no-alias-final-arg=true})" --verify-diagnostics --split-input-file  %s | FileCheck %s --check-prefix=CHECK_NOTEMPTY_NOALIAS

// CHECK_NOTEMPTY_NOALIAS:       func.func private @
// CHECK_NOTEMPTY_NOALIAS-SAME:    memref<4xbf16>,
// CHECK_NOTEMPTY_NOALIAS-SAME:    memref<bf16> {llvm.noalias}) {
// CHECK_NOTEMPTY_NOALIAS:       linalg.generic
// CHECK_NOTEMPTY_NOALIAS:       return
// CHECK_NOTEMPTY_NOALIAS:       func.func @reduction

// A run to check the option empty-functions=true:
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-linalg-function-outlining{empty-functions=true no-alias-final-arg=true})"  --verify-diagnostics --split-input-file  %s | FileCheck %s --check-prefix=CHECK_EMPTY_NOALIAS

// CHECK_EMPTY_NOALIAS:       func.func private @
// CHECK_EMPTY_NOALIAS-SAME:    memref<4xbf16>,
// CHECK_EMPTY_NOALIAS-SAME:    memref<bf16> {llvm.noalias}) {
// CHECK_EMPTY_NOALIAS-NOT:       linalg.generic
// CHECK_EMPTY_NOALIAS:       return
// CHECK_EMPTY_NOALIAS:       func.func @reduction

// A run to check the option no-alias-final-arg=false:
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-linalg-function-outlining{empty-functions=false no-alias-final-arg=false})" --verify-diagnostics --split-input-file  %s | FileCheck %s --check-prefix=CHECK_NOTEMPTY_ALIAS

// CHECK_NOTEMPTY_ALIAS:       func.func private @
// CHECK_NOTEMPTY_ALIAS-SAME:    memref<4xbf16>,
// CHECK_NOTEMPTY_ALIAS-SAME:    memref<bf16>) {
// CHECK_NOTEMPTY_ALIAS:       linalg.generic
// CHECK_NOTEMPTY_ALIAS:       return
// CHECK_NOTEMPTY_ALIAS:       func.func @reduction

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



