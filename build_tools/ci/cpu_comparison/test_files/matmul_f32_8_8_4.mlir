// This test is useful to compare against the `two_matmul_switching` when no switching happens
// and we successively call the same matmul

// These 2 lines are required by the script which generates input data:
//
// input 8x4xf32
// input 4x8xf32

!A_TYPE = tensor<8x4xf32>
!B_TYPE = tensor<4x8xf32>
!C_TYPE = tensor<8x8xf32>
func.func @matmul_8_8_4(%lhs : !A_TYPE,
    %rhs : !B_TYPE) -> !C_TYPE {
  %empty = tensor.empty() : !C_TYPE
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%empty : !C_TYPE) -> !C_TYPE
  %1 = linalg.matmul ins(%lhs, %rhs : !A_TYPE, !B_TYPE)
      outs(%fill : !C_TYPE) -> !C_TYPE
  %slice1 = tensor.extract_slice %1[0, 0][4, 8][1, 1] :
  !C_TYPE to !B_TYPE
  %2 = linalg.matmul ins(%lhs, %slice1 : !A_TYPE, !B_TYPE)
      outs(%fill : !C_TYPE) -> !C_TYPE
  %slice2 = tensor.extract_slice %2[0, 0][4, 8][1, 1] :
  !C_TYPE to !B_TYPE
  %3 = linalg.matmul ins(%lhs, %slice2 : !A_TYPE, !B_TYPE)
      outs(%fill : !C_TYPE) -> !C_TYPE
  %slice3 = tensor.extract_slice %3[0, 0][4, 8][1, 1] :
  !C_TYPE to !B_TYPE
  %4 = linalg.matmul ins(%lhs, %slice3 : !A_TYPE, !B_TYPE)
      outs(%fill : !C_TYPE) -> !C_TYPE
  return %4 : !C_TYPE
}
