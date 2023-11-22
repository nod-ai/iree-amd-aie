// RUN: iree-compile %s --iree-hal-target-backends=amd-aie --iree-codegen-transform-dialect-library=%p/matmul_fill_spec_pad.mlir| FileCheck %s

// CHECK-LABEL: @matmul_static
func.func @matmul_static(%lhs : tensor<128x256xi32>,
    %rhs : tensor<256x512xi32>) -> tensor<128x512xi32> {
  %empty = tensor.empty() : tensor<128x512xi32>
  %cst = arith.constant 0 : i32
  %fill = linalg.fill ins(%cst : i32) outs(%empty : tensor<128x512xi32>) -> tensor<128x512xi32>
  %2 = linalg.matmul ins(%lhs, %rhs : tensor<128x256xi32>, tensor<256x512xi32>)
      outs(%fill : tensor<128x512xi32>) -> tensor<128x512xi32>
  return %2 : tensor<128x512xi32>
}
