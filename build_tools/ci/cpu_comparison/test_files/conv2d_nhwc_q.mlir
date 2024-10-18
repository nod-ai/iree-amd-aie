// These lines are required for e2e numerical testing:
// input 2x14x14x32xi8
// input 3x3x32x64xi8

func.func @conv_2d_nhwc_hwcf_q(%arg0: tensor<2x14x14x32xi8>, %arg1: tensor<3x3x32x64xi8>) -> tensor<2x12x12x64xi32> {
  %cst = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<2x12x12x64xi32>
  %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<2x12x12x64xi32>) -> tensor<2x12x12x64xi32>
  %2 = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %arg1, %cst, %cst : tensor<2x14x14x32xi8>, tensor<3x3x32x64xi8>, i32, i32) outs(%1 : tensor<2x12x12x64xi32>) -> tensor<2x12x12x64xi32>
  return %2 : tensor<2x12x12x64xi32>
}
