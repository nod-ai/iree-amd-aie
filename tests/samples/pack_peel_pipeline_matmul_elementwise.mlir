// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-sources %s | iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" --iree-amdaie-use-pipeline=pack-peel --iree-amdaie-matmul-elementwise-fusion --iree-amdaie-enable-vectorization-passes=false --split-input-file | FileCheck %s

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
//       CHECK:    aie.device(npu1_4col)
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    func.func @matmul_elementwise_i32_dispatch_0_matmul_1024x1024x512_i32(%arg0: memref<1024x512xi32>, %arg1: memref<512x1024xi32>, %arg2: memref<1024x1024xi32>, %arg3: memref<1024x1024xi32>)
//       CHECK:      aiex.npu.dma_memcpy_nd
//       CHECK:      aiex.npu.dma_memcpy_nd
//       CHECK:      aiex.npu.dma_memcpy_nd
//       CHECK:      aiex.npu.sync

// -----

func.func @matmul_elementwise_bf16(%arg0: tensor<1024x512xbf16>, %arg1: tensor<512x1024xbf16>, %arg2: tensor<1024xbf16>) -> tensor<1024x1024xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = tensor.empty() : tensor<1024x1024xbf16>
  %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<1024x512xbf16>, tensor<512x1024xbf16>) outs(%1 : tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %arg2 : tensor<1024x1024xbf16>, tensor<1024xbf16>) outs(%0 : tensor<1024x1024xbf16>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
    %5 = arith.addf %in, %in_0 : bf16
    linalg.yield %5 : bf16
  } -> tensor<1024x1024xbf16>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<1024x1024xbf16>) outs(%0 : tensor<1024x1024xbf16>) {
  ^bb0(%in: bf16, %out: bf16):
    %5 = arith.cmpf ugt, %in, %cst : bf16
    %6 = arith.select %5, %in, %cst : bf16
    linalg.yield %6 : bf16
  } -> tensor<1024x1024xbf16>
  return %4 : tensor<1024x1024xbf16>
}

// CHECK-LABEL: hal.executable.export public @matmul_elementwise_bf16_dispatch_0_matmul_1024x1024x512_bf16
//       CHECK:    aie.device(npu1_4col)
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    func.func @matmul_elementwise_bf16_dispatch_0_matmul_1024x1024x512_bf16(%arg0: memref<262144xi32>, %arg1: memref<262144xi32>, %arg2: memref<512xi32>, %arg3: memref<524288xi32>)
//       CHECK:      aiex.npu.dma_memcpy_nd
//       CHECK:      aiex.npu.dma_memcpy_nd
//       CHECK:      aiex.npu.dma_memcpy_nd
//       CHECK:      aiex.npu.sync

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
func.func @matmul_elementwise_transpose_b(%arg0: tensor<256x512xbf16>, %arg1: tensor<1024x512xbf16>, %arg2: tensor<1024xbf16>) -> tensor<256x1024xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = tensor.empty() : tensor<512x1024xbf16>
  %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<1024x512xbf16>) outs(%0 : tensor<512x1024xbf16>) {
  ^bb0(%in: bf16, %out: bf16):
    linalg.yield %in : bf16
  } -> tensor<512x1024xbf16>
  %2 = tensor.empty() : tensor<256x1024xbf16>
  %3 = linalg.fill ins(%cst : bf16) outs(%2 : tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
  %4 = linalg.matmul ins(%arg0, %1 : tensor<256x512xbf16>, tensor<512x1024xbf16>) outs(%3 : tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
  %5 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %arg2 : tensor<256x1024xbf16>, tensor<1024xbf16>) outs(%2 : tensor<256x1024xbf16>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
    %7 = arith.addf %in, %in_0 : bf16
    linalg.yield %7 : bf16
  } -> tensor<256x1024xbf16>
  %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<256x1024xbf16>) outs(%2 : tensor<256x1024xbf16>) {
  ^bb0(%in: bf16, %out: bf16):
    %7 = arith.cmpf ugt, %in, %cst : bf16
    %8 = arith.select %7, %in, %cst : bf16
    linalg.yield %8 : bf16
  } -> tensor<256x1024xbf16>
  return %6 : tensor<256x1024xbf16>
}

// CHECK-LABEL: hal.executable.export public @matmul_elementwise_transpose_b_dispatch_0_matmul_transpose_b_256x1024x512_bf16
//       CHECK:    aie.device(npu1_4col)
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    func.func @matmul_elementwise_transpose_b_dispatch_0_matmul_transpose_b_256x1024x512_bf16(%arg0: memref<65536xi32>, %arg1: memref<262144xi32>, %arg2: memref<512xi32>, %arg3: memref<131072xi32>)
//       CHECK:      aiex.npu.dma_memcpy_nd
//       CHECK:      aiex.npu.dma_memcpy_nd
//       CHECK:      aiex.npu.dma_memcpy_nd
//       CHECK:      aiex.npu.sync
