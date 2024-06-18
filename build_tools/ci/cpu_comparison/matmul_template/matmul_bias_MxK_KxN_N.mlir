// input ${M}x${K}x${TYPE1}
// input ${K}x${N}x${TYPE1}
// input ${N}x${TYPE2}

func.func @matmul_bias(%arg0: tensor<${M}x${K}x${TYPE1}>, %arg1: tensor<${K}x${N}x${TYPE1}>, %arg2: tensor<${N}x${TYPE2}>) -> tensor<${M}x${N}x${TYPE2}>
{
  %cst = arith.constant ${ZERO} : ${TYPE2}
  %0 = tensor.empty() : tensor<${M}x${N}x${TYPE2}>
  %1 = linalg.fill ins(%cst : ${TYPE2}) outs(%0 : tensor<${M}x${N}x${TYPE2}>) -> tensor<${M}x${N}x${TYPE2}>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<${M}x${K}x${TYPE1}>, tensor<${K}x${N}x${TYPE1}>) outs(%1: tensor<${M}x${N}x${TYPE2}>) -> tensor<${M}x${N}x${TYPE2}>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %arg2 : tensor<1024x1024x${TYPE2}>, tensor<1024x${TYPE2}>) outs(%0 : tensor<1024x1024x${TYPE2}>) {
  ^bb0(%in: ${TYPE2}, %in_0: ${TYPE2}, %out: ${TYPE2}):
  %4 = ${ADD} %in, %in_0 : ${TYPE2}
  linalg.yield %4 : ${TYPE2}
  } -> tensor<${M}x${N}x${TYPE2}>
  return %3: tensor<${M}x${N}x${TYPE2}>
}
