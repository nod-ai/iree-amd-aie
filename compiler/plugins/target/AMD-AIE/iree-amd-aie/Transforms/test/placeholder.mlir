// RUN: iree-opt --iree-amdaie-placeholder %s | FileCheck %s

// CHECK-LABEL: @foobar
func.func @foobar() {
  return
}