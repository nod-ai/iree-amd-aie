// These 2 lines are required by the script which generates input data:
// input 2x14x14x32xbf16
// input 3x3x32x64xbf16

func.func @conv_2d_nhwc_hwcf(%arg0: tensor<2x14x14x32xbf16>, %arg1: tensor<3x3x32x64xbf16>) -> tensor<2x12x12x64xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<2x12x12x64xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x12x12x64xf32>) -> tensor<2x12x12x64xf32>
  %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %arg1 : tensor<2x14x14x32xbf16>, tensor<3x3x32x64xbf16>) outs(%1 : tensor<2x12x12x64xf32>) -> tensor<2x12x12x64xf32>
  return %2 : tensor<2x12x12x64xf32>
}
