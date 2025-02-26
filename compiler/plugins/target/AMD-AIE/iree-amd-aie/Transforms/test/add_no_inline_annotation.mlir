// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-add-no-inline-annotation)" %s | FileCheck %s

module {
  // CHECK: llvm.func @foo() attributes {no_inline}
  llvm.func @foo() {
    llvm.return
  }
}
