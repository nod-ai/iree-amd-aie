// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-targets --iree-amdaie-num-rows=2 --iree-amdaie-num-cols=2 --iree-amdaie-lower-to-aie-pipeline=air --iree-amdaie-tile-pipeline=pack-peel --iree-amdaie-matmul-elementwise-fusion --split-input-file %s | FileCheck %s

func.func @matmul_elementwise_i32(%lhs: tensor<1024x512xi32>, %rhs: tensor<512x1024xi32>, %ele: tensor<1024x1024xi32>) -> tensor<1024x1024xi32>
{
  %cst = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<1024x1024xi32>
  %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<1024x1024xi32>) -> tensor<1024x1024xi32>
  %res = linalg.matmul ins(%lhs, %rhs: tensor<1024x512xi32>, tensor<512x1024xi32>)
                    outs(%1: tensor<1024x1024xi32>) -> tensor<1024x1024xi32>
  %add = linalg.generic
        {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
         affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
         ins(%res, %ele : tensor<1024x1024xi32>, tensor<1024x1024xi32>)
         outs(%0 : tensor<1024x1024xi32>) {
           ^bb0(%in: i32, %in_0: i32, %out: i32):
           %2 = arith.addi %in, %in_0 : i32
           linalg.yield %2 : i32
         } -> tensor<1024x1024xi32>
  return %add : tensor<1024x1024xi32>
}

// CHECK-LABEL: hal.executable.export public @matmul_elementwise_i32_dispatch_0_matmul_1024x1024x512_i32
// CHECK:           aie.device(npu1_4col)
// CHECK-COUNT-3:   aie.shim_dma_allocation

// -----

func.func @matmul_elementwise_bf16_f32(%arg0: tensor<1024x512xbf16>, %arg1: tensor<512x1024xbf16>, %arg2: tensor<1024xf32>) -> tensor<1024x1024xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1024x1024xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<1024x512xbf16>, tensor<512x1024xbf16>) outs(%1 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %arg2 : tensor<1024x1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024x1024xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %5 = arith.addf %in, %in_0 : f32
    linalg.yield %5 : f32
  } -> tensor<1024x1024xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<1024x1024xf32>) outs(%0 : tensor<1024x1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    %5 = arith.cmpf ugt, %in, %cst : f32
    %6 = arith.select %5, %in, %cst : f32
    linalg.yield %6 : f32
  } -> tensor<1024x1024xf32>
  return %4 : tensor<1024x1024xf32>
}

// CHECK-LABEL: hal.executable.export public @matmul_elementwise_bf16_f32_dispatch_0_matmul_1024x1024x512_bf16xbf16xf32
// CHECK:           aie.device(npu1_4col)
// CHECK-COUNT-3:   aie.shim_dma_allocation

// -----

func.func @matmul_elementwise_bf16(%arg0: tensor<512x512xbf16>, %arg1: tensor<512x8192xbf16>, %arg2: tensor<512xf32>) -> tensor<512x8192xbf16> {
  %cst = arith.constant 0.000000e+00 : f32
  %7 = tensor.empty() : tensor<512x8192xbf16>
  %8 = tensor.empty() : tensor<512x8192xf32>
  %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<512x8192xf32>) -> tensor<512x8192xf32>
  %10 = linalg.matmul ins(%arg0, %arg1 : tensor<512x512xbf16>, tensor<512x8192xbf16>) outs(%9 : tensor<512x8192xf32>) -> tensor<512x8192xf32>
  %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%10, %arg2 : tensor<512x8192xf32>, tensor<512xf32>) outs(%7 : tensor<512x8192xbf16>) {
  ^bb0(%in: f32, %in_0: f32, %out: bf16):
    %12 = arith.addf %in, %in_0 : f32
    %13 = arith.truncf %12 : f32 to bf16
    linalg.yield %13 : bf16
  } -> tensor<512x8192xbf16>
  return %11 : tensor<512x8192xbf16>
}

// CHECK-LABEL:    hal.executable.export public @matmul_elementwise_bf16_dispatch_0_matmul_512x8192x512_bf16xbf16xf32
// CHECK:          aie.device(npu1_4col)
// CHECK-COUNT-3:  aie.shim_dma_allocation
