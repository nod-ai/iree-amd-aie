// C = 7 + A @ B (The '@' symbol denotes matrix multiplication)
func.func @matmul(%A : tensor<64x16xi32>, 
                  %B : tensor<16x64xi32>) -> tensor<64x64xi32> {

  // Initialize output tensor with '7'
  %init_acc = tensor.empty() : tensor<64x64xi32>
  %c0_acc_type = arith.constant 7 : i32

  %acc = linalg.fill ins(%c0_acc_type : i32) 
                     outs(%init_acc : tensor<64x64xi32>) -> tensor<64x64xi32>

  %C = linalg.matmul ins(%A, %B: tensor<64x16xi32>, tensor<16x64xi32>) 
                     outs(%acc: tensor<64x64xi32>) -> tensor<64x64xi32>

  return %C: tensor<64x64xi32>
}



