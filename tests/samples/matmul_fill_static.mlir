// Example using a transform dialect script.

// Compile to mlir-aie:

// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-sources %s | iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" --iree-codegen-transform-dialect-library=%S/matmul_fill_spec_pad.mlir | FileCheck %s

// To compile to just after the transform dialect pass: // RUN: iree-opt %s   --iree-hal-target-backends=amd-aie --iree-abi-transformation-pipeline --iree-flow-transformation-pipeline --iree-stream-transformation-pipeline --iree-hal-configuration-pipeline | cat - %S/matmul_fill_spec_pad.mlir | iree-opt --iree-transform-dialect-interpreter | FileCheck %s


// CHECK: scf.for
// CHECK: scf.for

func.func @matmul_static(%lhs : tensor<128x256xi32>,
    %rhs : tensor<256x512xi32>) -> tensor<128x512xi32> {
  %empty = tensor.empty() : tensor<128x512xi32>
  %cst = arith.constant 0 : i32
  %fill = linalg.fill ins(%cst : i32) outs(%empty : tensor<128x512xi32>) -> tensor<128x512xi32>
  %2 = linalg.matmul ins(%lhs, %rhs : tensor<128x256xi32>, tensor<256x512xi32>)
      outs(%fill : tensor<128x512xi32>) -> tensor<128x512xi32>
  return %2 : tensor<128x512xi32>
}

