func.func @matmul_elementwise(%arg0: tensor<${M}x${K}xbf16>, %arg1: tensor<${K}x${N}xbf16>, %arg2: tensor<${N}xf32>) -> tensor<${M}x${N}xf32>
{
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<${M}x${N}xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<${M}x${N}xf32>) -> tensor<${M}x${N}xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<${M}x${K}xbf16>, tensor<${K}x${N}xbf16>) outs(%1: tensor<${M}x${N}xf32>) -> tensor<${M}x${N}xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %arg2 : tensor<1024x1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024x1024xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
  %4 = arith.addf %in, %in_0 : f32
  linalg.yield %4 : f32
  } -> tensor<${M}x${N}xf32>
  return %3: tensor<${M}x${N}xf32>
}
