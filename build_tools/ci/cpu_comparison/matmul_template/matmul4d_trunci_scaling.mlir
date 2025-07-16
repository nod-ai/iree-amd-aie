// input ${M}x${K}x${TYPE1}
// input ${K}x${N}x${TYPE1}

// Matmul + Trunci variant with scaling.
// In an actual quantized model, truncating from a higher bitwidth to a lower precision bitwidth
// won't work and we need to scale.
// Since the output of the Matmul here is an integer cannot be multiplied with a floating point
// scale factor, we need to represent the scale factor with a multiplier and a shift operator instead.
func.func @matmul4d_trunci(%arg0: tensor<${M}x${K}x${TYPE1}>, %arg1: tensor<${K}x${N}x${TYPE1}>) -> tensor<${M}x${N}x${TYPE1}>
{
  %cst = arith.constant ${ZERO} : ${TYPE2}
  %cst_mul = arith.constant 10 : ${TYPE_MUL_RESULT}
  %cst_shift = arith.constant 7 : ${TYPE_MUL_RESULT}
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
      // %4 = arith.extsi %in : ${TYPE2} to ${TYPE_MUL_RESULT}
      // %5 = arith.muli %4, %cst_mul : ${TYPE_MUL_RESULT}
      // %6 = arith.shrsi %5, %cst_shift : ${TYPE_MUL_RESULT}
      %7 = arith.trunci %in : ${TYPE2} to ${TYPE1}
      linalg.yield %7 : ${TYPE1}
    } -> tensor<${M}x${N}x${TYPE1}>
  return %3: tensor<${M}x${N}x${TYPE1}>
}
