func.func @matmul_static(%lhs : tensor<128x256xf32>,
    %rhs : tensor<256x512xf32>, %init: tensor<128x512xf32>) -> tensor<128x512xf32> {
  %1 = linalg.matmul ins(%lhs, %rhs : tensor<128x256xf32>, tensor<256x512xf32>)
      outs(%init : tensor<128x512xf32>) -> tensor<128x512xf32>
  return %1 : tensor<128x512xf32>
}
