// These lines are required for e2e numerical testing:
// input 32x32xbf16
// input 32x32xbf16
// output 32x32xbf16

func.func @matmul_truncf(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16>
{
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<32x32xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xbf16>, tensor<32x32xbf16>)
    outs(%1: tensor<32x32xf32>) -> tensor<32x32xf32>
  %3 = arith.truncf %2 : tensor<32x32xf32> to tensor<32x32xbf16>
  return %3: tensor<32x32xbf16>
}
