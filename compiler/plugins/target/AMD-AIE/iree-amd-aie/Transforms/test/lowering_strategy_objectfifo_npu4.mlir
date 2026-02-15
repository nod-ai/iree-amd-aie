// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy{target-device=npu4})' %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy{target-device=npu4 use-tile-pipeline=pack-peel-4-level-tiling})' %s | FileCheck %s --check-prefix=PACK-PEEL-4-LEVEL
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy{target-device=npu4 use-tile-pipeline=pack-peel-4-level-tiling num-rows=1 num-cols=1})' %s | FileCheck %s --check-prefix=PACK-PEEL-4-LEVEL-1-CORE

// CHECK:       #config = #iree_codegen.lowering_config<tile_sizes = [
// CHECK-SAME:                [128, 128, 0], [0, 0, 1], [1, 1, 0]
// CHECK-SAME:            ]>
// CHECK:       #packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 128], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true],
// CHECK-SAME:                      innerPerm = [
// CHECK-SAME:                              [0, 1], [1, 0], [0, 1]
// CHECK-SAME:                   ], outerPerm = [
// CHECK-SAME:                              [0, 1], [1, 0], [1, 0]
// CHECK-SAME:                   ]}, {packedSizes = [0, 0, 0, 8, 8, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true],
// CHECK-SAME:                      innerPerm = [
// CHECK-SAME:                              [0, 1], [1, 0], [0, 1]
// CHECK-SAME:                   ], outerPerm = [
// CHECK-SAME:                              [0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]
// CHECK-SAME:                   ]}]>

// PACK-PEEL-4-LEVEL{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 0], [4, 4, 0], [0, 0, 1], [1, 1, 0]]>
// PACK-PEEL-4-LEVEL{LITERAL}: #packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 128], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1], [1, 0], [1, 0]]}, {packedSizes = [0, 0, 0, 8, 8, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
module {
  func.func @matmul_i32_dispatch_0_matmul_128x128x256_bf16xbf16xf32() {
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

// Tests a matmul shape for pack-peel-4-level-tiling in which the tile size generated
// should ideally be equal to the M, N, K dimension size of the matmul - but it won't work
// until support for DMA ops' reconfiguration is added. The workaround therefore halves
// the tile size for N as N/2.

// Pack-peel-4-level tiling on 4x4 cores : the tile size remains maximum in this case.
// PACK-PEEL-4-LEVEL{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[32, 512, 0], [4, 4, 0], [0, 0, 1], [1, 1, 0]]>
// PACK-PEEL-4-LEVEL{LITERAL}: #packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [8, 32, 64], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1], [1, 0], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>

// Pack-peel-4-level tiling on 1x1 core : the tile size for N gets halved in this case.
// PACK-PEEL-4-LEVEL-1-CORE{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[32, 256, 0], [1, 1, 0], [0, 0, 1], [1, 1, 0]]>
// PACK-PEEL-4-LEVEL-1-CORE{LITERAL}: #packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 64], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1], [1, 0], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
func.func @matmul_32x512x64_i32() {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x64xi32>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x512xi32>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x512xi32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x64xi32>> -> tensor<32x64xi32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [64, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x512xi32>> -> tensor<64x512xi32>
  %5 = tensor.empty() : tensor<32x512xi32>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<32x512xi32>) -> tensor<32x512xi32>
  %7 = linalg.matmul ins(%3, %4 : tensor<32x64xi32>, tensor<64x512xi32>) outs(%6 : tensor<32x512xi32>) -> tensor<32x512xi32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [32, 512], strides = [1, 1] : tensor<32x512xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x512xi32>>
  return
}

// -----

// Based on above workaround this test shows the packing size of N also being halved
// in case the tile size for N dimension becomes less than the corresponding packing size.

// PACK-PEEL-4-LEVEL-1-CORE{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[32, 16, 0], [1, 1, 0], [0, 0, 1], [1, 1, 0]]>
// PACK-PEEL-4-LEVEL-1-CORE{LITERAL}: #packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [32, 16, 64], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1], [1, 0], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
func.func @matmul_dispatch_0_matmul_32x32x128_i32() {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x128xi32>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x32xi32>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x32xi32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x128xi32>> -> tensor<32x128xi32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [128, 32], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x32xi32>> -> tensor<128x32xi32>
  %5 = tensor.empty() : tensor<32x32xi32>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<32x32xi32>) -> tensor<32x32xi32>
  %7 = linalg.matmul ins(%3, %4 : tensor<32x128xi32>, tensor<128x32xi32>) outs(%6 : tensor<32x32xi32>) -> tensor<32x32xi32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [32, 32], strides = [1, 1] : tensor<32x32xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x32xi32>>
  return
}
