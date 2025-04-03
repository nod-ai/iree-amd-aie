// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy{use-lower-to-aie-pipeline=air use-tile-pipeline=pack-peel})' %s | FileCheck %s --check-prefix=CHECK-PACK-PEEL

// CHECK-PACK-PEEL{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[512, 512, 0], [0, 0, 1], [1, 1, 0]]>
// CHECK-PACK-PEEL{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [128, 128, 128], transposePackIndices = [0, 1], unpackEmpty = [false, false], innerPerm = [[0, 1], [1, 0]], outerPerm = [[0, 1], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 8, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
builtin.module {
  func.func @matmul_large_dispatch_0_matmul_2048x2048x2048_i8_i32() {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x2048xi8>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x2048xi8>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x2048xi32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x2048xi8>> -> tensor<2048x2048xi8>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x2048xi8>> -> tensor<2048x2048xi8>
    %5 = tensor.empty() : tensor<2048x2048xi32>
    %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<2048x2048xi32>) -> tensor<2048x2048xi32>
    // CHECK:  linalg.matmul {lowering_config = #config, packing_config = #packingConfig}
    %7 = linalg.matmul ins(%3, %4 : tensor<2048x2048xi8>, tensor<2048x2048xi8>) outs(%6 : tensor<2048x2048xi32>) -> tensor<2048x2048xi32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : tensor<2048x2048xi32> -> !flow.dispatch.tensor<writeonly:tensor<2048x2048xi32>>
    return
  }
}

// -----

// CHECK-PACK-PEEL{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[44, 128, 0], [0, 0, 1], [1, 1, 0]]>
// CHECK-PACK-PEEL{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [44, 32, 64], transposePackIndices = [0, 1], unpackEmpty = [false, false], innerPerm = [[0, 1], [1, 0]], outerPerm = [[0, 1], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
module {
  func.func @matmul_large_dispatch_0_matmul_308x2432x9728_bf16() {
    %cst = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<308x9728xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<9728x2432xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<308x2432xbf16>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [308, 9728], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<308x9728xbf16>> -> tensor<308x9728xbf16>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [9728, 2432], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<9728x2432xbf16>> -> tensor<9728x2432xbf16>
    %5 = tensor.empty() : tensor<308x2432xbf16>
    %6 = linalg.fill ins(%cst : bf16) outs(%5 : tensor<308x2432xbf16>) -> tensor<308x2432xbf16>
    %7 = linalg.matmul ins(%3, %4 : tensor<308x9728xbf16>, tensor<9728x2432xbf16>) outs(%6 : tensor<308x2432xbf16>) -> tensor<308x2432xbf16>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [308, 2432], strides = [1, 1] : tensor<308x2432xbf16> -> !flow.dispatch.tensor<writeonly:tensor<308x2432xbf16>>
    return
  }
}

// -----

// CHECK-PACK-PEEL{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 0], [0, 0, 1], [1, 1, 0]]>
// CHECK-PACK-PEEL{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 32], transposePackIndices = [0, 1], unpackEmpty = [false, false], innerPerm = [[0, 1], [0, 1]], outerPerm = [[0, 1], [0, 1]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [0, 1], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
builtin.module {
  func.func @matmul_transpose_b_dispatch_0_matmul_transpose_b_256x1024x512_i32() {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x512xi32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x512xi32>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<256x1024xi32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x512xi32>> -> tensor<256x512xi32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x512xi32>> -> tensor<1024x512xi32>
    %5 = tensor.empty() : tensor<256x1024xi32>
    %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<256x1024xi32>) -> tensor<256x1024xi32>
    // CHECK:  linalg.matmul_transpose_b {lowering_config = #config, packing_config = #packingConfig}
    %7 = linalg.matmul_transpose_b ins(%3, %4 : tensor<256x512xi32>, tensor<1024x512xi32>) outs(%6 : tensor<256x1024xi32>) -> tensor<256x1024xi32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1] : tensor<256x1024xi32> -> !flow.dispatch.tensor<writeonly:tensor<256x1024xi32>>
    return
  }
}
