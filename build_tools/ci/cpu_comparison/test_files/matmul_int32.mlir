// These lines are required for e2e numerical testing:
// input 128x128xi32
// input 128x128xi32
// output 128x128xi32

!lhs = tensor<128x128xi32>
!rhs = tensor<128x128xi32>
!res = tensor<128x128xi32>

// The function name must match the filename:
func.func @matmul_int32(%lhs : !lhs, %rhs : !rhs) -> !res {
  %empty = tensor.empty() : !res
  %cst = arith.constant 0 : i32
  %fill = linalg.fill ins(%cst : i32) outs(%empty : !res) -> !res
  %2 = linalg.matmul ins(%lhs, %rhs : !lhs, !rhs)
      outs(%fill : !res) -> !res
  return %2 : !res
}
