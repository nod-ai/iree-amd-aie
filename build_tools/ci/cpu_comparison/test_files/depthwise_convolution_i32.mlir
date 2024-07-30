// These 2 lines are required by the script which generates input data:
// input 1x14x14x64xi32
// input 3x3x64xi32

func.func @depthwise_conv_2d_nhwc_hwc(%arg0: tensor<1x14x14x64xi32>, %arg1: tensor<3x3x64xi32>) -> tensor<1x12x12x64xi32> {
  %cst = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<1x12x12x64xi32>
  %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<1x12x12x64xi32>) -> tensor<1x12x12x64xi32>
  %2 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %arg1 : tensor<1x14x14x64xi32>, tensor<3x3x64xi32>) outs(%1 : tensor<1x12x12x64xi32>) -> tensor<1x12x12x64xi32>
  return %2 : tensor<1x12x12x64xi32>
}

