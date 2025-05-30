// These lines are required for e2e numerical testing:
// input ${M}x${N}x${TYPE}
// output ${M}x${N}x${TYPE}

func.func @softmax(%in0: tensor<${M}x${N}x${TYPE}>) -> tensor<${M}x${N}x${TYPE}> {
  %cst = arith.constant ${ZERO} : ${TYPE}
  %0 = tensor.empty() : tensor<${M}x${N}x${TYPE}>
  %1 = linalg.fill ins(%cst : ${TYPE}) outs(%0 : tensor<${M}x${N}x${TYPE}>) -> tensor<${M}x${N}x${TYPE}>
  %2 = linalg.softmax dimension(1) ins(%in0 : tensor<${M}x${N}x${TYPE}>) outs(%1 : tensor<${M}x${N}x${TYPE}>) -> tensor<${M}x${N}x${TYPE}>
  return %2 : tensor<${M}x${N}x${TYPE}>
}
