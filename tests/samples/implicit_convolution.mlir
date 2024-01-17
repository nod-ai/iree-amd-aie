// RUN: iree-compile \
// RUN: --iree-preprocessing-pass-pipeline='builtin.module(func.func(iree-global-opt-convert-1x1-filter-conv2d-to-matmul,iree-preprocessing-convert-conv2d-to-img2col))' \
// RUN: --iree-hal-target-backends=amd-aie \
// RUN: --compile-to=executable-sources %s | FileCheck %s

// An illustration of how we can replace a linalg.conv_* operation with a
// linalg.matmul operation, using existing IREE infrastucture and passes.

// Using --iree-preprocessing-pass-pipline='...' in iree-compile makes
// it possible to insert passes just after the lowering from the ML dialects
// (torch-mlir, etc.) to the linalg dialect. It is important to do the
// decomposition of convolution before dispatch creation, so that the IREE
// compiler can choose to separate data rearrangement (im2col) and
// computation (matmul in this case) into different dispatches.

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




