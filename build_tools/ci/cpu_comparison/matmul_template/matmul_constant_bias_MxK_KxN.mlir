// input ${M}x${K}x${TYPE1}
// input ${K}x${N}x${TYPE1}

func.func @matmul_constant_bias(%arg0: tensor<${M}x${K}x${TYPE1}>, %arg1: tensor<${K}x${N}x${TYPE1}>) -> tensor<${M}x${N}x${TYPE2}> {
  %cst_zero = arith.constant ${ZERO} : ${TYPE2}
  %cst_bias = arith.constant ${CONSTANT} : ${TYPE2}
  %0 = tensor.empty() : tensor<${M}x${N}x${TYPE2}>
  %1 = linalg.fill ins(%cst_zero : ${TYPE2}) outs(%0 : tensor<${M}x${N}x${TYPE2}>) -> tensor<${M}x${N}x${TYPE2}>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<${M}x${K}x${TYPE1}>, tensor<${K}x${N}x${TYPE1}>)
    outs(%1: tensor<${M}x${N}x${TYPE2}>) -> tensor<${M}x${N}x${TYPE2}>
  %3 = linalg.generic {indexing_maps = [
                              affine_map<(d0, d1) -> (d0, d1)>,
                              affine_map<(d0, d1) -> (d0, d1)>
                       ],
                       iterator_types = ["parallel", "parallel"]
                      } ins(%2 : tensor<${M}x${N}x${TYPE2}>) outs(%0 : tensor<${M}x${N}x${TYPE2}>) {
    ^bb0(%in: ${TYPE2}, %out: ${TYPE2}):
      %4 = arith.addi %in, %cst_bias : ${TYPE2}
      linalg.yield %4 : ${TYPE2}
    } -> tensor<${M}x${N}x${TYPE2}>
  return %3: tensor<${M}x${N}x${TYPE2}>
}
