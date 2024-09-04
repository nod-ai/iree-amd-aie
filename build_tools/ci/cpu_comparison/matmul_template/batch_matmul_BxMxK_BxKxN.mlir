// input ${B}x${M}x${K}x${TYPE1}
// input ${B}x${K}x${N}x${TYPE1}

func.func @batch_matmul(%arg0: tensor<${B}x${M}x${K}x${TYPE1}>, %arg1: tensor<${B}x${K}x${N}x${TYPE1}>) -> tensor<${B}x${M}x${N}x${TYPE2}>
{
  %cst = arith.constant ${ZERO} : ${TYPE2}
  %0 = tensor.empty() : tensor<${B}x${M}x${N}x${TYPE2}>
  %1 = linalg.fill ins(%cst : ${TYPE2}) outs(%0 : tensor<${B}x${M}x${N}x${TYPE2}>) -> tensor<${B}x${M}x${N}x${TYPE2}>
  %2 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<${B}x${M}x${K}x${TYPE1}>, tensor<${B}x${K}x${N}x${TYPE1}>)
    outs(%1: tensor<${B}x${M}x${N}x${TYPE2}>) -> tensor<${B}x${M}x${N}x${TYPE2}>
  return %2: tensor<${B}x${M}x${N}x${TYPE2}>
}
