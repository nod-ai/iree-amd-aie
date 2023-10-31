func.func @matmul_static(%lhs : tensor<100x200xf32>,
    %rhs : tensor<200x300xf32>) -> tensor<100x300xf32> {
  %empty = tensor.empty() : tensor<100x300xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<100x300xf32>) -> tensor<100x300xf32>
  %2 = linalg.matmul ins(%lhs, %rhs : tensor<100x200xf32>, tensor<200x300xf32>)
      outs(%fill : tensor<100x300xf32>) -> tensor<100x300xf32>
  return %2 : tensor<100x300xf32>
}
