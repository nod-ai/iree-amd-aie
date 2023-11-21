func.func @matmul_static(%lhs : tensor<32x32xi32>,
    %rhs : tensor<32x32xi32>) -> tensor<32x32xi32> {
  %empty = tensor.empty() : tensor<32x32xi32>
  %cst = arith.constant 0 : i32
  %fill = linalg.fill ins(%cst : i32) outs(%empty : tensor<32x32xi32>) -> tensor<32x32xi32>
  %2 = linalg.matmul ins(%lhs, %rhs : tensor<32x32xi32>, tensor<32x32xi32>)
      outs(%fill : tensor<32x32xi32>) -> tensor<32x32xi32>
  return %2 : tensor<32x32xi32>
}
