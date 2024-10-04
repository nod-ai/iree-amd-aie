// input ${M}x${K}x${TYPE1}
// input ${K}x${N}x${TYPE1}

func.func @matmul_truncf(%arg0: tensor<${M}x${K}x${TYPE1}>, %arg1: tensor<${K}x${N}x${TYPE1}>) -> tensor<${M}x${N}x${TYPE1}>
{
  %cst = arith.constant ${ZERO} : ${TYPE2}
  %0 = tensor.empty() : tensor<${M}x${N}x${TYPE2}>
  %1 = linalg.fill ins(%cst : ${TYPE2}) outs(%0 : tensor<${M}x${N}x${TYPE2}>) -> tensor<${M}x${N}x${TYPE2}>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<${M}x${K}x${TYPE1}>, tensor<${K}x${N}x${TYPE1}>)
    outs(%1: tensor<${M}x${N}x${TYPE2}>) -> tensor<${M}x${N}x${TYPE2}>
  %3 = arith.truncf %2 : tensor<${M}x${N}x${TYPE2}> to tensor<${M}x${N}x${TYPE1}>
  return %2: tensor<${M}x${N}x${TYPE1}>
}
