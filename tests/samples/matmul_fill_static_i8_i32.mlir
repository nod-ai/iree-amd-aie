func.func @matmul_example(%lhs: tensor<16x256xi8>, %rhs: tensor<256x256xi8>) -> tensor<16x256xi32>
{
  %empty = tensor.empty() : tensor<16x256xi32>
  %cst = arith.constant 0 : i32
  %fill = linalg.fill ins(%cst : i32) outs(%empty : tensor<16x256xi32>) -> tensor<16x256xi32>
  %res = linalg.matmul ins(%lhs, %rhs: tensor<16x256xi8>, tensor<256x256xi8>)
                    outs(%fill: tensor<16x256xi32>) -> tensor<16x256xi32>
  return %res : tensor<16x256xi32>
}