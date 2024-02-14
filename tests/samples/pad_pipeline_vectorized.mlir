// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-sources %s | iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" --iree-amdaie-use-pipeline=pad | FileCheck %s 


// linalg.matmul (vector.contract) appears 8 times ... 4 tiles with ping-pong? 
// CHECK-LABEL:   aie.device
// CHECK-COUNT-8: vector.contract{{.*}}vector<4x4x4xf32>, vector<4x4x4xf32> into vector<4x4xf32>
// CHECK-NOT:     vector.contract



!lhs = tensor<8x16xf32>
!rhs = tensor<16x8xf32>
!out = tensor<8x8xf32>

func.func @matmul_static(%lhs : !lhs, %rhs : !rhs) -> !out {
  %empty = tensor.empty() : !out
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%empty : !out) -> !out
  %out = linalg.matmul ins(%lhs, %rhs : !lhs, !rhs) outs(%fill : !out) -> !out
  return %out : !out
}

