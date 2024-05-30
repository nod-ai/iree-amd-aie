// These lines are strictly required by the script which generates input data:
//
// input 128x128xbf16
// input 128x128xbf16

!lhs = tensor<128x128xbf16>
!rhs = tensor<128x128xbf16>
!res = tensor<128x128xf32>

// The function name must match the filename:
func.func @matmul_bf16(%lhs : !lhs, %rhs : !rhs) -> !res {
  %empty = tensor.empty() : !res
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%empty : !res) -> !res
  %2 = linalg.matmul ins(%lhs, %rhs : !lhs, !rhs)
      outs(%fill : !res) -> !res
  return %2 : !res
}
