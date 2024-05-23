func.func @matmul_i32(%lhs: tensor<128x256xi32>, %rhs: tensor<256x128xi32>, %add: tensor<128x128xi32>) -> tensor<128x128xi32>
{
  %cst = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<128x128xi32>
  %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<128x128xi32>) -> tensor<128x128xi32>
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<128x256xi32>, tensor<256x128xi32>)
                    outs(%1: tensor<128x128xi32>) -> tensor<128x128xi32>
  %res = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%matmul, %add : tensor<128x128xi32>, tensor<128x128xi32>) outs(%0 : tensor<128x128xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %11 = arith.addi %in, %in_0 : i32
    linalg.yield %11 : i32
  } -> tensor<128x128xi32>
  return %res : tensor<128x128xi32>
}