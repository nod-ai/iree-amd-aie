// This test demonstrates enabling / disabling function outlining in the default
// pipeline (note that below the pipeline is not specified explicitly with
// the flag --iree-amdaie-tile-pipeline). We check 3 paths:
//
// 1) Explicitly disabling linalg function outlining with
//             --iree-amdaie-enable-function-outlining=0
//
// 2) Explicitly enabling linalg function outlining with
//             --iree-amdaie-enable-function-outlining=1
//
// 3) Not specifying the flag at all, which should use the default value (1).


// 1) Explicitly disabled:
// RUN: iree-compile --iree-hal-target-backends=amd-aie \
// RUN:   --compile-to=executable-targets --iree-amdaie-enable-function-outlining=0 %s | FileCheck %s -check-prefix=CHECK-DISABLED

// 2) Explicitly enabled:
// RUN: iree-compile --iree-hal-target-backends=amd-aie \
// RUN:   --compile-to=executable-targets --iree-amdaie-enable-function-outlining=1 %s | FileCheck %s -check-prefix=CHECK-ENABLED

// 3) Default value:
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

// CHECK-DISABLED-NOT: func.call
// CHECK-ENABLED:      func.call
// CHECK-DEFAULT-NOT:  func.call
