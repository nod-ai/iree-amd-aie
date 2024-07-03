// These 2 lines are required by the script which generates input data:
// input 2x14x14x32xbf16
// input 3x3x32x64xbf16

func.func @conv_2d_nhwc_hwcf(%arg0: tensor<2x14x14x32xbf16>, %arg1: tensor<3x3x32x64xbf16>) -> tensor<2x12x12x64xbf16> {
  %cst = arith.constant 0.0 : bf16
  %0 = tensor.empty() : tensor<2x12x12x64xbf16>
  %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<2x12x12x64xbf16>) -> tensor<2x12x12x64xbf16>
  %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %arg1 : tensor<2x14x14x32xbf16>, tensor<3x3x32x64xbf16>) outs(%1 : tensor<2x12x12x64xbf16>) -> tensor<2x12x12x64xbf16>
  return %2 : tensor<2x12x12x64xbf16>
}
