// This test demonstrates enabling / disabling vectorization in the default
// pipeline (note that below the pipeline is not specified explicitly with
// the flag --iree-amdaie-use-pipeline). We check 3 paths:
//
// 1) Explicitly disabling vectorization with
//             --iree-amdaie-enable-vectorization-in-pass-pipeline=0
//
// 2) Explicitly enabling vectorization with
//             --iree-amdaie-enable-vectorization-in-pass-pipeline=1
//
// 3) Not specifying the flag at all, which should use the default value (1).
//
// We first perform the step which is common to all 3 paths: we compile the
// free function to an 'executable sources' file.
// RUN: iree-compile --iree-hal-target-backends=amd-aie \
// RUN:   --compile-to=executable-sources %s > exe-sources.mlir

// 1) Explicitly disabled:
// RUN: iree-opt \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(\
// RUN:   iree-hal-translate-target-executable-variants{target=amd-aie})))" \
// RUN:   --iree-amdaie-enable-vectorization-in-pass-pipeline=0 exe-sources.mlir \
// RUN:   | FileCheck %s -check-prefix=CHECK-DISABLED

// 2) Explicitly enabled:
// RUN: iree-opt \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(\
// RUN:   iree-hal-translate-target-executable-variants{target=amd-aie})))" \
// RUN:   --iree-amdaie-enable-vectorization-in-pass-pipeline=1 exe-sources.mlir \
// RUN:   | FileCheck %s -check-prefix=CHECK-ENABLED

// 3) Default value:
// RUN: iree-opt \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(\
// RUN:   iree-hal-translate-target-executable-variants{target=amd-aie})))" \
// RUN:   exe-sources.mlir | FileCheck %s -check-prefix=CHECK-DEFAULT


func.func @mm_in_bf16_out_f32(%lhs: tensor<64x64xbf16>,
                              %rhs: tensor<64x64xbf16>) -> tensor<64x64xf32> {
  %empty = tensor.empty() : tensor<64x64xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32)
                      outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %res = linalg.matmul ins(%lhs, %rhs: tensor<64x64xbf16>, tensor<64x64xbf16>)
                       outs(%fill: tensor<64x64xf32>) -> tensor<64x64xf32>
  return %res : tensor<64x64xf32>
}

// CHECK-DISABLED-NOT: vector.contract
// CHECK-ENABLED: vector.contract
// CHECK-DEFAULT: vector.contract
