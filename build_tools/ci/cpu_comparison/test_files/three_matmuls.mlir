// This test shows arbitrary matmuls that would have producer consumer relationships
// across different dispatches running on CI.

// These 4 lines are required by the script which generates input data:
//
// input 32x32xf32
// input 32x32xf32
// input 32x4xf32
// input 4x32xf32

!A_TYPE = tensor<32x32xf32>
!B_TYPE = tensor<32x4xf32>
!C_TYPE = tensor<4x32xf32>
!D_TYPE = tensor<4x4xf32>
func.func @two_mm(%lhs : !A_TYPE,
    %rhs : !A_TYPE, %rhs_2 : !B_TYPE, %lhs_2 : !C_TYPE) -> !D_TYPE {
  %empty = tensor.empty() : !A_TYPE
  %empty_2 = tensor.empty() : !B_TYPE
  %empty_3 = tensor.empty() : !D_TYPE
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%empty : !A_TYPE) -> !A_TYPE
  %fill_2 = linalg.fill ins(%cst : f32) outs(%empty_2 : !B_TYPE) -> !B_TYPE
  %fill_3 = linalg.fill ins(%cst : f32) outs(%empty_3 : !D_TYPE) -> !D_TYPE
  %2 = linalg.matmul ins(%lhs, %rhs : !A_TYPE, !A_TYPE)
      outs(%fill : !A_TYPE) -> !A_TYPE
  %3 = linalg.matmul ins(%2, %rhs_2 : !A_TYPE, !B_TYPE)
      outs(%fill_2 : !B_TYPE) -> !B_TYPE
  %4 = linalg.matmul ins(%lhs_2, %3 : !C_TYPE, !B_TYPE)
      outs(%fill_3 : !D_TYPE) -> !D_TYPE
  return %4 : !D_TYPE
}
