// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy{target-device=npu1_4col num-rows=2 num-cols=2})' %s | FileCheck %s --check-prefix=CHECK-2x2
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy{target-device=npu1_4col num-rows=4 num-cols=2})' %s | FileCheck %s --check-prefix=CHECK-4x2
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy{target-device=npu1_4col})' %s | FileCheck %s --check-prefix=CHECK-4x4
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy{target-device=npu1_4col use-tile-pipeline=pack-peel-4-level-tiling})' %s | FileCheck %s --check-prefix=PACK-PEEL-4-LEVEL

// CHECK-2x2{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0], [0, 0, 1], [1, 1, 0]]>
// CHECK-2x2{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 128], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1], [1, 0], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>

// CHECK-4x2{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[128, 64, 0], [0, 0, 1], [1, 1, 0]]>
// CHECK-4x2{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 128], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1], [1, 0], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>

// CHECK-4x4{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 0], [0, 0, 1], [1, 1, 0]]>
// CHECK-4x4{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 128], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1], [1, 0], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>

// PACK-PEEL-4-LEVEL{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 0], [4, 4, 0], [0, 0, 1], [1, 1, 0]]
// PACK-PEEL-4-LEVEL{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 128], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1], [1, 0], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
module {
  func.func @matmul_128x128x256_bf16xbf16xf32() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x128xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xbf16>> -> tensor<128x256xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x128xbf16>> -> tensor<256x128xbf16>
    %5 = tensor.empty() : tensor<128x128xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x128xf32>) -> tensor<128x128xf32>
    // CHECK:  linalg.matmul {lowering_config = #config, packing_config = #packingConfig}
    %7 = linalg.matmul ins(%3, %4 : tensor<128x256xbf16>, tensor<256x128xbf16>) outs(%6 : tensor<128x128xf32>) -> tensor<128x128xf32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>
    return
  }
}

// -----

// CHECK-2x2{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0], [0, 0, 1], [1, 1, 0]]>
// CHECK-2x2{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 64], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1], [1, 0], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>

// CHECK-4x2{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[128, 64, 0], [0, 0, 1], [1, 1, 0]]>
// CHECK-4x2{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 64], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1], [1, 0], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>

// CHECK-4x4{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 0], [0, 0, 1], [1, 1, 0]]>
// CHECK-4x4{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 64], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1], [1, 0], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>

// PACK-PEEL-4-LEVEL{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 0], [4, 4, 0], [0, 0, 1], [1, 1, 0]]>
// PACK-PEEL-4-LEVEL{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 64], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1], [1, 0], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
module {
  func.func @matmul_128x128x256_i32() {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xi32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x128xi32>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xi32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xi32>> -> tensor<128x256xi32>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x128xi32>> -> tensor<256x128xi32>
    %5 = tensor.empty() : tensor<128x128xi32>
    %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<128x128xi32>) -> tensor<128x128xi32>
    // CHECK:  linalg.matmul {lowering_config = #config, packing_config = #packingConfig}
    %7 = linalg.matmul ins(%3, %4 : tensor<128x256xi32>, tensor<256x128xi32>) outs(%6 : tensor<128x128xi32>) -> tensor<128x128xi32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xi32>>
    return
  }
}

// -----

// CHECK-2x2{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[128, 64, 0], [0, 0, 1], [1, 1, 0]]>
// CHECK-2x2{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [64, 32, 128], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1], [1, 0], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 8, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>

// CHECK-4x2{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[128, 64, 0], [0, 0, 1], [1, 1, 0]]>
// CHECK-4x2{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 128], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1], [1, 0], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 8, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>

// CHECK-4x4{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 0], [0, 0, 1], [1, 1, 0]]>
// CHECK-4x4{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 128], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1], [1, 0], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 8, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>

// PACK-PEEL-4-LEVEL{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 0], [4, 4, 0], [0, 0, 1], [1, 1, 0]]>
// PACK-PEEL-4-LEVEL{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 128], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1], [1, 0], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 8, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
module {
  func.func @matmul_128x128x256_i8xi8xi32() {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xi8>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x128xi8>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xi32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xi8>> -> tensor<128x256xi8>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x128xi8>> -> tensor<256x128xi8>
    %5 = tensor.empty() : tensor<128x128xi32>
    %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<128x128xi32>) -> tensor<128x128xi32>
    // CHECK:  linalg.matmul {lowering_config = #config, packing_config = #packingConfig}
    %7 = linalg.matmul ins(%3, %4 : tensor<128x256xi8>, tensor<256x128xi8>) outs(%6 : tensor<128x128xi32>) -> tensor<128x128xi32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xi32>>
    return
  }
}

// -----

// CHECK-2x2{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 64, 0], [0, 0, 0, 1], [0, 1, 1, 0]]>
// CHECK-2x2{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [0, 32, 32, 64], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 2], [0, 2, 1], [0, 2, 1]]}, {packedSizes = [0, 0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 2, 4, 3], [0, 1, 2, 4, 3], [0, 1, 2, 4, 3]]}]>

// CHECK-4x2{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[1, 128, 64, 0], [0, 0, 0, 1], [0, 1, 1, 0]]>
// CHECK-4x2{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [0, 32, 32, 64], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 2], [0, 2, 1], [0, 2, 1]]}, {packedSizes = [0, 0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 2, 4, 3], [0, 1, 2, 4, 3], [0, 1, 2, 4, 3]]}]>

// CHECK-4x4{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[1, 128, 128, 0], [0, 0, 0, 1], [0, 1, 1, 0]]>
// CHECK-4x4{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [0, 32, 32, 64], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 2], [0, 2, 1], [0, 2, 1]]}, {packedSizes = [0, 0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 2, 4, 3], [0, 1, 2, 4, 3], [0, 1, 2, 4, 3]]}]>

// PACK-PEEL-4-LEVEL{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[1, 128, 128, 0], [0, 4, 4, 0], [0, 0, 0, 1], [0, 1, 1, 0]]>
// PACK-PEEL-4-LEVEL{LITERAL}: #packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [0, 32, 32, 64], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 2], [0, 2, 1], [0, 2, 1]]}, {packedSizes = [0, 0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 2, 4, 3], [0, 1, 2, 4, 3], [0, 1, 2, 4, 3]]}]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
module {
  func.func @batch_matmul_1x128x128x256_i32() {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x128x256xi32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x256x128xi32>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x128x128xi32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [1, 128, 256], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x128x256xi32>> -> tensor<1x128x256xi32>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [1, 256, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x256x128xi32>> -> tensor<1x256x128xi32>
    %5 = tensor.empty() : tensor<1x128x128xi32>
    %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<1x128x128xi32>) -> tensor<1x128x128xi32>
    // CHECK:  linalg.batch_matmul {lowering_config = #config, packing_config = #packingConfig}
    %7 = linalg.batch_matmul ins(%3, %4 : tensor<1x128x256xi32>, tensor<1x256x128xi32>) outs(%6 : tensor<1x128x128xi32>) -> tensor<1x128x128xi32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [1, 128, 128], strides = [1, 1, 1] : tensor<1x128x128xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x128x128xi32>>
    return
  }
}

// -----

// CHECK-2x2{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0], [0, 0, 1], [1, 1, 0]]>
// CHECK-2x2{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 128], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [0, 1], [0, 1]], outerPerm = [[0, 1], [0, 1], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [0, 1], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>

// CHECK-4x2{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[128, 64, 0], [0, 0, 1], [1, 1, 0]]>
// CHECK-4x2{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 128], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [0, 1], [0, 1]], outerPerm = [[0, 1], [0, 1], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [0, 1], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>

// CHECK-4x4{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 0], [0, 0, 1], [1, 1, 0]]>
// CHECK-4x4{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 128], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [0, 1], [0, 1]], outerPerm = [[0, 1], [0, 1], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [0, 1], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>

// PACK-PEEL-4-LEVEL{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 0], [4, 4, 0], [0, 0, 1], [1, 1, 0]]>
// PACK-PEEL-4-LEVEL{LITERAL}: #packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 128], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [0, 1], [0, 1]], outerPerm = [[0, 1], [0, 1], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [0, 1], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
module {
  func.func @matmul_transpose_b_128x128x256_bf16xbf16xf32() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xbf16>> -> tensor<128x256xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xbf16>> -> tensor<128x256xbf16>
    %5 = tensor.empty() : tensor<128x128xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x128xf32>) -> tensor<128x128xf32>
    // CHECK:  linalg.matmul_transpose_b {lowering_config = #config, packing_config = #packingConfig}
    %7 = linalg.matmul_transpose_b ins(%3, %4 : tensor<128x256xbf16>, tensor<128x256xbf16>) outs(%6 : tensor<128x128xf32>) -> tensor<128x128xf32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>
    return
  }
}

// -----

// CHECK-2x2{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0], [0, 0, 1], [1, 1, 0]]>
// CHECK-2x2{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 128], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[1, 0], [1, 0], [0, 1]], outerPerm = [[1, 0], [1, 0], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[1, 0], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>

// CHECK-4x2{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[128, 64, 0], [0, 0, 1], [1, 1, 0]]>
// CHECK-4x2{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 128], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[1, 0], [1, 0], [0, 1]], outerPerm = [[1, 0], [1, 0], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[1, 0], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>

// CHECK-4x4{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 0], [0, 0, 1], [1, 1, 0]]>
// CHECK-4x4{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 128], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[1, 0], [1, 0], [0, 1]], outerPerm = [[1, 0], [1, 0], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[1, 0], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>

// PACK-PEEL-4-LEVEL{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 0], [4, 4, 0], [0, 0, 1], [1, 1, 0]]>
// PACK-PEEL-4-LEVEL{LITERAL}: #packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 128], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[1, 0], [1, 0], [0, 1]], outerPerm = [[1, 0], [1, 0], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[1, 0], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
module {
  func.func @matmul_transpose_a_128x128x256_bf16xbf16xf32() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x128xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x128xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x128xbf16>> -> tensor<256x128xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x128xbf16>> -> tensor<256x128xbf16>
    %5 = tensor.empty() : tensor<128x128xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x128xf32>) -> tensor<128x128xf32>
    // CHECK:  linalg.matmul_transpose_a {lowering_config = #config, packing_config = #packingConfig}
    %7 = linalg.matmul_transpose_a ins(%3, %4 : tensor<256x128xbf16>, tensor<256x128xbf16>) outs(%6 : tensor<128x128xf32>) -> tensor<128x128xf32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>
    return
  }
}

// -----

// Tests for pack-peel-4-level-tiling pipeline with large input sizes.

// PACK-PEEL-4-LEVEL{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[256, 256, 0], [4, 4, 0], [0, 0, 1], [1, 1, 0]]>
// PACK-PEEL-4-LEVEL{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 128], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1], [1, 0], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
module {
  func.func @matmul_2048x2048x512_bf16xbf16xf32() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x512xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x2048xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2048xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x512xbf16>> -> tensor<2048x512xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x2048xbf16>> -> tensor<512x2048xbf16>
    %5 = tensor.empty() : tensor<2048x2048xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
    // CHECK:  linalg.matmul {lowering_config = #config, packing_config = #packingConfig}
    %7 = linalg.matmul ins(%3, %4 : tensor<2048x512xbf16>, tensor<512x2048xbf16>) outs(%6 : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : tensor<2048x2048xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2048xf32>>
    return
  }
}

// -----

// PACK-PEEL-4-LEVEL{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[512, 256, 0], [4, 4, 0], [0, 0, 1], [1, 1, 0]]>
// PACK-PEEL-4-LEVEL{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [64, 32, 128], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1], [1, 0], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 8, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
module {
  func.func @matmul_2048x2048x512_i8xi8xi32() {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x512xi8>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x2048xi8>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2048xi32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x512xi8>> -> tensor<2048x512xi8>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x2048xi8>> -> tensor<512x2048xi8>
    %5 = tensor.empty() : tensor<2048x2048xi32>
    %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<2048x2048xi32>) -> tensor<2048x2048xi32>
    // CHECK:  linalg.matmul {lowering_config = #config, packing_config = #packingConfig}
    %7 = linalg.matmul ins(%3, %4 : tensor<2048x512xi8>, tensor<512x2048xi8>) outs(%6 : tensor<2048x2048xi32>) -> tensor<2048x2048xi32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : tensor<2048x2048xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2048xi32>>
    return
  }
}
