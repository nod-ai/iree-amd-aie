// input ${M}x${K}x32x64xbf16
// input ${N}x${K}x64x32xbf16

func.func @matmul4d(%arg0: tensor<${M}x${K}x32x64xbf16>, %arg1: tensor<${N}x${K}x64x32xbf16>) -> tensor<${N}x${M}x32x32xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<${N}x${M}x32x32xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<${N}x${M}x32x32xf32>) -> tensor<${N}x${M}x32x32xf32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<${M}x${K}x32x64xbf16>, tensor<${N}x${K}x64x32xbf16>) outs(%1 : tensor<${N}x${M}x32x32xf32>) {
    ^bb0(%in: bf16, %in_1: bf16, %out: f32):
      %12 = arith.extf %in : bf16 to f32
      %13 = arith.extf %in_1 : bf16 to f32
      %14 = arith.mulf %12, %13 : f32
      %15 = arith.addf %out, %14 : f32
      linalg.yield %15 : f32
    } -> tensor<${N}x${M}x32x32xf32>
  return %2 : tensor<${N}x${M}x32x32xf32>
}
