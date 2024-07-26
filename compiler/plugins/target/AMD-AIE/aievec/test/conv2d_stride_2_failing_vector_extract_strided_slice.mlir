// RUN: iree-compile %s --iree-hal-target-backends=amd-aie \
// RUN:   --iree-amdaie-tile-pipeline=conv-decompose \
// RUN:   --iree-amdaie-matmul-elementwise-fusion \
// RUN:   --iree-hal-dump-executable-files-to=%S \
// RUN:   --iree-amd-aie-show-invoked-commands \
// RUN:   --iree-scheduling-optimize-bindings=false \
// RUN:   --mlir-disable-threading -o %t.vmfb


func.func @conv_2d_nhwc_hwcf(%arg0: tensor<1x129x129x16xi8>, %arg1: tensor<3x3x16x32xi8>) -> tensor<1x64x64x32xi32> {
  %cst = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<1x64x64x32xi32>
  %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<1x64x64x32xi32>) -> tensor<1x64x64x32xi32>
  %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %arg1 : tensor<1x129x129x16xi8>, tensor<3x3x16x32xi8>) outs(%1 : tensor<1x64x64x32xi32>) -> tensor<1x64x64x32xi32>
  return %2 : tensor<1x64x64x32xi32>
}
