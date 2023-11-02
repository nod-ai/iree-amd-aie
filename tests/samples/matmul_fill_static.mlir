func.func @matmul_static(%lhs : tensor<128x256xf32>,
    %rhs : tensor<256x512xf32>) -> tensor<128x512xf32> {
  %empty = tensor.empty() : tensor<128x512xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x512xf32>) -> tensor<128x512xf32>
  %2 = linalg.matmul ins(%lhs, %rhs : tensor<128x256xf32>, tensor<256x512xf32>)
      outs(%fill : tensor<128x512xf32>) -> tensor<128x512xf32>
  return %2 : tensor<128x512xf32>
}
