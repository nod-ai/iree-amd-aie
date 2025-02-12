// input ${M}x${K}x${TYPE1}
// input ${K}x${N}x${TYPE1}

func.func @matmul_trunci(%arg0: tensor<${M}x${K}x${TYPE1}>, %arg1: tensor<${K}x${N}x${TYPE1}>) -> tensor<${M}x${N}x${TYPE1}>
{
  %cst = arith.constant ${ZERO} : ${TYPE2}
  %cst_mul = arith.constant 10 : ${TYPE2}
  %cst_div = arith.constant 137 : ${TYPE2}
  %0 = tensor.empty() : tensor<${M}x${N}x${TYPE2}>
  %i8out = tensor.empty() : tensor<${M}x${N}x${TYPE1}>
  %1 = linalg.fill ins(%cst : ${TYPE2}) outs(%0 : tensor<${M}x${N}x${TYPE2}>) -> tensor<${M}x${N}x${TYPE2}>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<${M}x${K}x${TYPE1}>, tensor<${K}x${N}x${TYPE1}>)
    outs(%1: tensor<${M}x${N}x${TYPE2}>) -> tensor<${M}x${N}x${TYPE2}>
  %3 = linalg.generic {indexing_maps = [
                              affine_map<(d0, d1) -> (d0, d1)>,
                              affine_map<(d0, d1) -> (d0, d1)>
                       ],
                       iterator_types = ["parallel", "parallel"]
                      } ins(%2 : tensor<${M}x${N}x${TYPE2}>) outs(%i8out : tensor<${M}x${N}x${TYPE1}>) {
    ^bb0(%in: ${TYPE2}, %out: ${TYPE1}):
      %26 = arith.muli %in, %cst_mul : ${TYPE2}
      %27 = arith.divsi %26, %cst_div : ${TYPE2}
      %28 = arith.trunci %27 : ${TYPE2} to ${TYPE1}
      linalg.yield %28 : ${TYPE1}
    } -> tensor<${M}x${N}x${TYPE1}>
  return %3: tensor<${M}x${N}x${TYPE1}>
}
