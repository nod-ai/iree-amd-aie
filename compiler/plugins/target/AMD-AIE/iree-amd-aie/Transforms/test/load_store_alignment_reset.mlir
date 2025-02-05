// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-load-store-alignment-reset))" %s | FileCheck %s


// CHECK-LABEL: func @alignmentsWillBeRemoved(%arg0: !llvm.ptr)
func.func  @alignmentsWillBeRemoved(%arg0: !llvm.ptr) {

  // CHECK:      %[[L1:.+]] = llvm.load %arg0 : !llvm.ptr -> vector<32xi8>
  // CHECK-NEXT: %[[L2:.+]] = llvm.load %arg0 : !llvm.ptr -> vector<32xi8>
  // CHECK-NEXT: %[[L4:.+]] = llvm.load %arg0 : !llvm.ptr -> vector<32xi8>
  %l1 = llvm.load %arg0 {alignment = 1 : i64} : !llvm.ptr -> vector<32xi8>
  %l2 = llvm.load %arg0 {alignment = 2 : i64} : !llvm.ptr -> vector<32xi8>
  %l4 = llvm.load %arg0 {alignment = 4 : i64} : !llvm.ptr -> vector<32xi8>

  // CHECK-NEXT: llvm.store %[[L1]], %arg0 : vector<32xi8>, !llvm.ptr
  // CHECK-NEXT: llvm.store %[[L1]], %arg0 : vector<32xi8>, !llvm.ptr
  // CHECK-NEXT: llvm.store %[[L1]], %arg0 : vector<32xi8>, !llvm.ptr
  llvm.store %l1, %arg0 {alignment = 1 : i64} : vector<32xi8>, !llvm.ptr
  llvm.store %l1, %arg0 {alignment = 2 : i64} : vector<32xi8>, !llvm.ptr
  llvm.store %l1, %arg0 {alignment = 4 : i64} : vector<32xi8>, !llvm.ptr

  // CHECK-NEXT: return
  return
}
