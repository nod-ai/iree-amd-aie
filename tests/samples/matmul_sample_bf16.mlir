func.func @matmul_bf16(%lhs: tensor<128x256xbf16>, %rhs: tensor<256x128xbf16>) -> tensor<128x128xbf16>
{
  %cst = arith.constant 0.0 : bf16
  %0 = tensor.empty() : tensor<128x128xbf16>
  %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<128x128xbf16>) -> tensor<128x128xbf16>
  %res = linalg.matmul ins(%lhs, %rhs: tensor<128x256xbf16>, tensor<256x128xbf16>)
                    outs(%1: tensor<128x128xbf16>) -> tensor<128x128xbf16>
  return %res : tensor<128x128xbf16>
}