// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-remove-wrap-flag-from-gep))" %s | FileCheck %s

// Check that nuw and nusw are removed.
// CHECK-LABEL: optimized_func
llvm.func @optimized_func(%ptr: !llvm.ptr, %idx: i64) {
  // CHECK: llvm.getelementptr %
 llvm.getelementptr nuw %ptr[%idx, 0, %idx] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<(array<10 x f32>)>
  // CHECK: llvm.getelementptr %
 llvm.getelementptr nusw %ptr[%idx, 0, %idx] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<(array<10 x f32>)>
 llvm.return
}
