// Input to printing test:
!t = tensor<64x64xi32>
func.func @matmul(%lhs : !t, %rhs : !t) -> !t {
  %init_acc = tensor.empty() : !t
  %c0_acc_t = arith.constant 0 : i32
  %acc = linalg.fill ins(%c0_acc_t : i32) outs(%init_acc : !t) -> !t
  %result = linalg.matmul ins(%lhs, %rhs: !t, !t) outs(%acc: !t) -> !t
  return %result: !t
}

