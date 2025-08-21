// input ${K}x${M}x${TYPE1}
// input ${K}x${N}x${TYPE1}

func.func @matmul_transpose_a(%arg0: tensor<${K}x${M}x${TYPE1}>, %arg1: tensor<${K}x${N}x${TYPE1}>) -> tensor<${M}x${N}x${TYPE2}>
{
  %cst = arith.constant ${ZERO} : ${TYPE2}
  %0 = tensor.empty() : tensor<${M}x${N}x${TYPE2}>
  %1 = linalg.fill ins(%cst : ${TYPE2}) outs(%0 : tensor<${M}x${N}x${TYPE2}>) -> tensor<${M}x${N}x${TYPE2}>
  %2 = linalg.matmul
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d2, d0)>,
      affine_map<(d0, d1, d2) -> (d2, d1)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ]
    ins(%arg0, %arg1 : tensor<${K}x${M}x${TYPE1}>, tensor<${K}x${N}x${TYPE1}>)
    outs(%1: tensor<${M}x${N}x${TYPE2}>) -> tensor<${M}x${N}x${TYPE2}>
  return %2: tensor<${M}x${N}x${TYPE2}>
}
