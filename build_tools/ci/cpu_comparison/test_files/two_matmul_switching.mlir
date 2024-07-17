// This test shows switching between two matmuls and is useful to model the switching cost

// These 2 lines are required by the script which generates input data:
//
// input 8x4xf32
// input 4x8xf32

!A_TYPE = tensor<8x4xf32>
!B_TYPE = tensor<4x8xf32>
!C_TYPE = tensor<8x8xf32>
func.func @matmul_small(%lhs : !A_TYPE,
    %rhs : !B_TYPE) -> !A_TYPE {
  %empty = tensor.empty() : !C_TYPE
  %empty2 = tensor.empty() : !A_TYPE
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%empty : !C_TYPE) -> !C_TYPE
  %fill2 = linalg.fill ins(%cst : f32) outs(%empty2 : !A_TYPE) -> !A_TYPE
  %1 = linalg.matmul ins(%lhs, %rhs : !A_TYPE, !B_TYPE)
      outs(%fill : !C_TYPE) -> !C_TYPE
  %2 = linalg.matmul ins(%1, %lhs : !C_TYPE, !A_TYPE)
      outs(%fill2 : !A_TYPE) -> !A_TYPE
  %3 = linalg.matmul ins(%2, %rhs : !A_TYPE, !B_TYPE)
      outs(%fill : !C_TYPE) -> !C_TYPE
  %4 = linalg.matmul ins(%3, %lhs : !C_TYPE, !A_TYPE)
      outs(%fill2 : !A_TYPE) -> !A_TYPE
  %5 = linalg.matmul ins(%4, %rhs : !A_TYPE, !B_TYPE)
      outs(%fill : !C_TYPE) -> !C_TYPE
  %6 = linalg.matmul ins(%5, %lhs : !C_TYPE, !A_TYPE)
      outs(%fill2 : !A_TYPE) -> !A_TYPE
  %7 = linalg.matmul ins(%6, %rhs : !A_TYPE, !B_TYPE)
      outs(%fill : !C_TYPE) -> !C_TYPE
  %8 = linalg.matmul ins(%7, %lhs : !C_TYPE, !A_TYPE)
      outs(%fill2 : !A_TYPE) -> !A_TYPE
  return %8 : !A_TYPE
}
