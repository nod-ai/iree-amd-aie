// This test is useful to compare against the `two_matmul_switching` when no switching happens
// and we successively call the same matmul.
//
// This test is a part of three test files that developers can look at together.
// 1. two_matmul_switching.mlir => switches calls between two matmuls M,N,K size (8,4,8) and (8,8,4) 
      4 times each (8 calls total). The graph ((8,4,8) -> (8,8,4)) x 4
// 2. matmul_f32_8_4_8.mlir => calls the (8,4,8) matmul 4 times hence doesnt have a switching cost so we
      have a baseline for it. The graph is (8,4,8) x 4
// 3. matmul_f32_8_8_4.mlir => calls the (8,8,4) matmul 4 times hence doesnt have a switching cost so we
      have a baseline for it. The graph is (8,8,4) x 4

// These 2 lines are required by the script which generates input data:
//
// input 8x8xf32
// input 8x4xf32

!A_TYPE = tensor<8x8xf32>
!B_TYPE = tensor<8x4xf32>
!C_TYPE = tensor<8x4xf32>
func.func @matmul_8_4_8(%lhs : !A_TYPE,
    %rhs : !B_TYPE) -> !C_TYPE {
  %empty = tensor.empty() : !C_TYPE
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%empty : !C_TYPE) -> !C_TYPE
  %1 = linalg.matmul ins(%lhs, %rhs : !A_TYPE, !B_TYPE)
      outs(%fill : !C_TYPE) -> !C_TYPE
  %2 = linalg.matmul ins(%lhs, %1 : !A_TYPE, !B_TYPE)
      outs(%fill : !C_TYPE) -> !C_TYPE
  %3 = linalg.matmul ins(%lhs, %2 : !A_TYPE, !B_TYPE)
      outs(%fill : !C_TYPE) -> !C_TYPE
  %4 = linalg.matmul ins(%lhs, %3 : !A_TYPE, !B_TYPE)
      outs(%fill : !C_TYPE) -> !C_TYPE
  return %4 : !C_TYPE
}
