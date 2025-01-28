// This test demonstrates enabling / disabling function outlining in the default
// pipeline. We check 3 paths:
//
// 1) Explicitly disabling linalg function outlining with
//              --iree-amdaie-enable-function-outlining=none
//
// 2) Explicitly enabling linalg function outlining for all linalg ops with
//              --iree-amdaie-enable-function-outlining=all
//
// 3) Not specifying the flag at all, which should use the default value (balanced).


// 1) Explicitly disabled:
// RUN: iree-compile --iree-hal-target-backends=amd-aie \
// RUN:   --compile-to=executable-targets --iree-amdaie-enable-function-outlining=none %s | FileCheck %s -check-prefix=CHECK-DISABLED

// 2) Explicitly enabled:
// RUN: iree-compile --iree-hal-target-backends=amd-aie \
// RUN:   --compile-to=executable-targets --iree-amdaie-enable-function-outlining=all %s | FileCheck %s -check-prefix=CHECK-ENABLED

// 3) Default value (balanced):
// RUN: iree-compile --iree-hal-target-backends=amd-aie \
// RUN:   --compile-to=executable-targets %s | FileCheck %s -check-prefix=CHECK-DEFAULT

func.func @matmul(%lhs: tensor<64x64xbf16>,
                              %rhs: tensor<64x64xbf16>) -> tensor<64x64xf32> {
  %empty = tensor.empty() : tensor<64x64xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32)
                      outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %res = linalg.matmul ins(%lhs, %rhs: tensor<64x64xbf16>, tensor<64x64xbf16>)
                       outs(%fill: tensor<64x64xf32>) -> tensor<64x64xf32>
  return %res : tensor<64x64xf32>
}

// CHECK-DISABLED-NOT:    func.call
// CHECK-ENABLED-COUNT-2: func.call
// CHECK-DEFAULT-COUNT-1: func.call
