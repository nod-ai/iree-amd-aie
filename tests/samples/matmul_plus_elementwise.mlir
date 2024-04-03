func.func @matmul_plus_elementwise(%lhs: tensor<8x2048xf32>, %rhs: tensor<2048x2048xf32>, %elem_operand2: tensor<2048xf32>) -> tensor<8x2048xf32> {
  %empty = tensor.empty() : tensor<8x2048xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8x2048xf32>) -> tensor<8x2048xf32>
  %matmul = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%lhs, %rhs : tensor<8x2048xf32>, tensor<2048x2048xf32>) outs(%fill : tensor<8x2048xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %33 = arith.mulf %in, %in_0 : f32
    %34 = arith.addf %out, %33 : f32
    linalg.yield %34 : f32
  } -> tensor<8x2048xf32>
  %elementwise = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%lhs, %elem_operand2, %matmul : tensor<8x2048xf32>, tensor<2048xf32>, tensor<8x2048xf32>) outs(%empty : tensor<8x2048xf32>) {
  ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
    %33 = arith.addf %in_0, %in_1 : f32
    %34 = arith.addf %in, %33 : f32
    linalg.yield %34 : f32
  } -> tensor<8x2048xf32>
  return %elementwise : tensor<8x2048xf32>
}