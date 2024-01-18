// RUN: iree-compile --iree-plugin=-amd_aie \
// RUN: --iree-hal-target-backends=rocm \
// RUN: --compile-to=executable-sources %s | FileCheck %s

// Without the compiler flag --iree-plugin=-amd_aie to disable the modifications 
// to the pipeline that the amd-aie plugin makes, the convolution in turned into 
// a matmul even when the target backend is rocm. 

// CHECK-LABEL: conv_2d_example
func.func @conv_2d_example(%arg0: tensor<1x16x16x4xf32>,
                      %arg1: tensor<3x2x4x16xf32>,
                      %arg2: tensor<1x14x15x16xf32>) -> tensor<1x14x15x16xf32> {

// CHECK-NOT: linalg.matmul
// CHECK: linalg.conv
// CHECK-NOT: linalg.matmul
  %0 = linalg.conv_2d_nhwc_hwcf
        {dilations = dense<1> : tensor<2xi64>,
         strides = dense<1> : tensor<2xi64> }
        ins(%arg0, %arg1: tensor<1x16x16x4xf32>, tensor<3x2x4x16xf32>)
        outs(%arg2: tensor<1x14x15x16xf32>) -> tensor<1x14x15x16xf32>

// CHECK: return
  return %0 : tensor<1x14x15x16xf32>
}




