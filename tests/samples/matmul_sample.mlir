func.func @matmul_i32(%lhs: tensor<128x256xi32>, %rhs: tensor<256x128xi32>) -> tensor<128x128xi32>
{
  %cst = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<128x128xi32>
  %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<128x128xi32>) -> tensor<128x128xi32>
  %res = linalg.matmul ins(%lhs, %rhs: tensor<128x256xi32>, tensor<256x128xi32>)
                    outs(%1: tensor<128x128xi32>) -> tensor<128x128xi32>
  return %res : tensor<128x128xi32>
}