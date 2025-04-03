// input ${M1}x${K1}x${M0}x${K0}x${TYPE1}
// input ${N1}x${K1}x${K0}x${N0}x${TYPE1}

func.func @matmul4d_trunci(%arg0: tensor<${M1}x${K1}x${M0}x${K0}x${TYPE1}>, %arg1: tensor<${N1}x${K1}x${K0}x${N0}x${TYPE1}>) -> tensor<${N1}x${M1}x${M0}x${N0}x${TYPE1}> {
  %r = flow.dispatch.region -> (tensor<${N1}x${M1}x${M0}x${N0}x${TYPE1}>) {
    %cst = arith.constant ${ZERO} : ${TYPE2}
    %cst_shift = arith.constant 5 : ${TYPE2}
    %0 = tensor.empty() : tensor<${N1}x${M1}x${M0}x${N0}x${TYPE2}>
    %i8out = tensor.empty() : tensor<${N1}x${M1}x${M0}x${N0}x${TYPE1}>
    %1 = linalg.fill ins(%cst : ${TYPE2}) outs(%0 : tensor<${N1}x${M1}x${M0}x${N0}x${TYPE2}>) -> tensor<${N1}x${M1}x${M0}x${N0}x${TYPE2}>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>],
                         iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
                         } ins(%arg0, %arg1 : tensor<${M1}x${K1}x${M0}x${K0}x${TYPE1}>, tensor<${N1}x${K1}x${K0}x${N0}x${TYPE1}>) outs(%1 : tensor<${N1}x${M1}x${M0}x${N0}x${TYPE2}>) {
      ^bb0(%in: ${TYPE1}, %in_1: ${TYPE1}, %out: ${TYPE2}):
        %12 = ${EXT} %in : ${TYPE1} to ${TYPE2}
        %13 = ${EXT} %in_1 : ${TYPE1} to ${TYPE2}
        %14 = ${MUL} %12, %13 : ${TYPE2}
        %15 = ${ADD} %out, %14 : ${TYPE2}
        linalg.yield %15 : ${TYPE2}
      } -> tensor<${N1}x${M1}x${M0}x${N0}x${TYPE2}>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                         iterator_types = ["parallel", "parallel", "parallel", "parallel"]
                        } ins(%2 : tensor<${N1}x${M1}x${M0}x${N0}x${TYPE2}>) outs(%i8out : tensor<${N1}x${M1}x${M0}x${N0}x${TYPE1}>) {
      ^bb0(%in: ${TYPE2}, %out: ${TYPE1}):
        %4 = arith.shrsi %in, %cst_shift : ${TYPE2}
        %5 = arith.trunci %4 : ${TYPE2} to ${TYPE1}
        linalg.yield %5 : ${TYPE1}
    } -> tensor<${N1}x${M1}x${M0}x${N0}x${TYPE1}>
    flow.return %3 : tensor<${N1}x${M1}x${M0}x${N0}x${TYPE1}>
  }
  return %r : tensor<${N1}x${M1}x${M0}x${N0}x${TYPE1}>
}
