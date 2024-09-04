// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy{use-pass-pipeline=pack-peel use-lower-to-aie-pipeline=objectFifo})' %s | FileCheck %s

// CHECK-{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [0, 0, 1], [1, 1, 0, 0, 0, 0]]>
// CHECK-{LITERAL}: #packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 32], transposePackIndices = [1], unpackEmpty = [false], innerPerm = [[1, 0]], outerPerm = [[0, 1]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
module {
  func.func @matmul_i32_dispatch_0_matmul_128x128x256_bf16xbf16xf32() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x256xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x128xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x128xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x256xbf16>> -> tensor<128x256xbf16>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x128xbf16>> -> tensor<256x128xbf16>
    %5 = tensor.empty() : tensor<128x128xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x128xf32>) -> tensor<128x128xf32>
    // CHECK:  linalg.matmul {lowering_config = #config, packing_config = #packingConfig}
    %7 = linalg.matmul ins(%3, %4 : tensor<128x256xbf16>, tensor<256x128xbf16>) outs(%6 : tensor<128x128xf32>) -> tensor<128x128xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x128xf32>>
    return
  }
}

// -----

// CHECK-{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [0, 0, 1], [1, 1, 0, 0, 0, 0]]>
// CHECK-{LITERAL}: #packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 32], transposePackIndices = [1], unpackEmpty = [false], innerPerm = [[1, 0]], outerPerm = [[0, 1]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
module {
  func.func @matmul_i32_dispatch_0_matmul_128x128x256_i32() {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x256xi32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x128xi32>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x128xi32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x256xi32>> -> tensor<128x256xi32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x128xi32>> -> tensor<256x128xi32>
    %5 = tensor.empty() : tensor<128x128xi32>
    %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<128x128xi32>) -> tensor<128x128xi32>
    // CHECK:  linalg.matmul {lowering_config = #config, packing_config = #packingConfig}
    %7 = linalg.matmul ins(%3, %4 : tensor<128x256xi32>, tensor<256x128xi32>) outs(%6 : tensor<128x128xi32>) -> tensor<128x128xi32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xi32> -> !flow.dispatch.tensor<writeonly:tensor<128x128xi32>>
    return
  }
}

// -----

// CHECK-{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [0, 0, 1], [1, 1, 0, 0, 0, 0]]>
// CHECK-{LITERAL}: #packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 32], transposePackIndices = [1], unpackEmpty = [false], innerPerm = [[1, 0]], outerPerm = [[0, 1]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
module {
  func.func @matmul_i32_dispatch_0_matmul_128x128x256_i8xi8xi32() {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x256xi8>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x128xi8>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x128xi32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x256xi8>> -> tensor<128x256xi8>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x128xi8>> -> tensor<256x128xi8>
    %5 = tensor.empty() : tensor<128x128xi32>
    %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<128x128xi32>) -> tensor<128x128xi32>
    // CHECK:  linalg.matmul {lowering_config = #config, packing_config = #packingConfig}
    %7 = linalg.matmul ins(%3, %4 : tensor<128x256xi8>, tensor<256x128xi8>) outs(%6 : tensor<128x128xi32>) -> tensor<128x128xi32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xi32> -> !flow.dispatch.tensor<writeonly:tensor<128x128xi32>>
    return
  }
}

// -----

// CHECK-{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 64], [0, 0, 0, 1], [0, 1, 1, 0, 0, 0, 0]]>
// CHECK-{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [0, 32, 32, 32], transposePackIndices = [1], unpackEmpty = [false], innerPerm = [[1, 0]], outerPerm = [[0, 1, 2]]}, {packedSizes = [0, 0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 2, 4, 3], [0, 1, 2, 4, 3], [0, 1, 2, 4, 3]]}]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
module {
  func.func @batch_matmul_1x128x128x256_i32() {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x128x256xi32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x256x128xi32>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x128x128xi32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [1, 128, 256], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x128x256xi32>> -> tensor<1x128x256xi32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [1, 256, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x256x128xi32>> -> tensor<1x256x128xi32>
    %5 = tensor.empty() : tensor<1x128x128xi32>
    %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<1x128x128xi32>) -> tensor<1x128x128xi32>
    // CHECK:  linalg.batch_matmul {lowering_config = #config, packing_config = #packingConfig}
    %7 = linalg.batch_matmul ins(%3, %4 : tensor<1x128x256xi32>, tensor<1x256x128xi32>) outs(%6 : tensor<1x128x128xi32>) -> tensor<1x128x128xi32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [1, 128, 128], strides = [1, 1, 1] : tensor<1x128x128xi32> -> !flow.dispatch.tensor<writeonly:tensor<1x128x128xi32>>
    return
  }
}
