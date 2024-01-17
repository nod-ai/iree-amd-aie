// RUN: iree-compile \
// RUN: --iree-hal-target-backends=amd-aie \
// RUN: --compile-to=executable-sources %s | FileCheck %s

// CHECK-LABEL: conv_2d_example
func.func @conv_2d_example(%arg0: tensor<1x16x16x4xf32>,
                      %arg1: tensor<3x2x4x16xf32>,
                      %arg2: tensor<1x14x15x16xf32>) -> tensor<1x14x15x16xf32> {

// CHECK-NOT: linalg.conv
// CHECK: linalg.matmul
// CHECK-NOT: linalg.conv
  %0 = linalg.conv_2d_nhwc_hwcf
        {dilations = dense<1> : tensor<2xi64>,
         strides = dense<1> : tensor<2xi64> }
        ins(%arg0, %arg1: tensor<1x16x16x4xf32>, tensor<3x2x4x16xf32>)
        outs(%arg2: tensor<1x14x15x16xf32>) -> tensor<1x14x15x16xf32>

// CHECK: return
  return %0 : tensor<1x14x15x16xf32>
}




