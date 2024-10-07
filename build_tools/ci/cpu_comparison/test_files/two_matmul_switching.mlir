// This test shows switching between two matmuls and is useful to model the switching cost.
// This test is a part of three test files that developers can look at together.
// 1. two_matmul_switching.mlir => switches calls between two matmuls M,N,K size (8,4,8) and (8,8,4)
//    4 times each (8 calls total). The graph ((8,4,8) -> (8,8,4)) x 4
// 2. matmul_f32_8_4_8.mlir => calls the (8,4,8) matmul 4 times hence doesnt have a switching cost so we
//    have a baseline for it. The graph is (8,4,8) x 4
// 3. matmul_f32_8_8_4.mlir => calls the (8,8,4) matmul 4 times hence doesnt have a switching cost so we
//    have a baseline for it. The graph is (8,8,4) x 4

// These lines are required for e2e numerical testing:
//
// input 8x4xf32
// input 4x8xf32
// output 8x4xf32

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
