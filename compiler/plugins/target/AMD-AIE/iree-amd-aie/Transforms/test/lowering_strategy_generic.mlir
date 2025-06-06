// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy)' %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy{use-tile-pipeline=pack-peel-4-level-tiling})' %s | FileCheck %s --check-prefix=PACK-PEEL-4-LEVEL

// Test generic version of matmul.

// CHECK{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 0], [0, 0, 1], [1, 1, 0]]>
// CHECK{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 64], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1], [1, 0], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
module {
  func.func @matmul_generic_128x128x256_i32() {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xi32>>
    %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x128xi32>>
    %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xi32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xi32>> -> tensor<128x256xi32>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x128xi32>> -> tensor<256x128xi32>
    %5 = tensor.empty() : tensor<128x128xi32>
    %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<128x128xi32>) -> tensor<128x128xi32>
    //      CHECK:  linalg.generic
    // CHECK-SAME:  attrs = {lowering_config = #config, packing_config = #packingConfig}
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<128x256xi32>, tensor<256x128xi32>) outs(%6 : tensor<128x128xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %8 = arith.muli %in, %in_0 : i32
      %9 = arith.addi %out, %8 : i32
      linalg.yield %9 : i32
    } -> tensor<128x128xi32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xi32>>
    return
  }
}

// -----

// Test generic version of matmul_transpose_b.

// CHECK{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 0], [0, 0, 1], [1, 1, 0]]>
// CHECK{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 64], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [0, 1], [0, 1]], outerPerm = [[0, 1], [0, 1], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [0, 1], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
module {
  func.func @matmul_transpose_b_generic_128x128x256_i32() {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xi32>>
    %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xi32>>
    %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xi32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xi32>> -> tensor<128x256xi32>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xi32>> -> tensor<128x256xi32>
    %5 = tensor.empty() : tensor<128x128xi32>
    %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<128x128xi32>) -> tensor<128x128xi32>
    //      CHECK:  linalg.generic
    // CHECK-SAME:  attrs = {lowering_config = #config, packing_config = #packingConfig}
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<128x256xi32>, tensor<128x256xi32>) outs(%6 : tensor<128x128xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %8 = arith.muli %in, %in_0 : i32
      %9 = arith.addi %out, %8 : i32
      linalg.yield %9 : i32
    } -> tensor<128x128xi32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xi32>>
    return
  }
}

// -----

// Test generic version of matmul_transpose_a.

// CHECK{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 0], [0, 0, 1], [1, 1, 0]]>
// CHECK{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 64], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[1, 0], [1, 0], [0, 1]], outerPerm = [[1, 0], [1, 0], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[1, 0], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
module {
  func.func @matmul_transpose_a_generic_128x128x256_i32() {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x128xi32>>
    %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x128xi32>>
    %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xi32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x128xi32>> -> tensor<256x128xi32>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x128xi32>> -> tensor<256x128xi32>
    %5 = tensor.empty() : tensor<128x128xi32>
    %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<128x128xi32>) -> tensor<128x128xi32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<256x128xi32>, tensor<256x128xi32>) outs(%6 : tensor<128x128xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %8 = arith.muli %in, %in_0 : i32
      %9 = arith.addi %out, %8 : i32
      linalg.yield %9 : i32
    } -> tensor<128x128xi32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xi32>>
    return
  }
}

// -----

// Test generic version of matmul with reduction loop at first, i.e, (d0, d1, d2) = (k, m, n).

// CHECK{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[0, 128, 128], [1, 0, 0], [0, 1, 1]]>
// CHECK{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [64, 32, 32], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1], [1, 0], [1, 0]]}, {packedSizes = [0, 0, 0, 8, 4, 4], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
module {
  func.func @matmul_generic_128x128x256_i32() {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xi32>>
    %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x128xi32>>
    %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xi32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xi32>> -> tensor<128x256xi32>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x128xi32>> -> tensor<256x128xi32>
    %5 = tensor.empty() : tensor<128x128xi32>
    %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<128x128xi32>) -> tensor<128x128xi32>
    //      CHECK:  linalg.generic
    // CHECK-SAME:  attrs = {lowering_config = #config, packing_config = #packingConfig}
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d0)>, affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%3, %4 : tensor<128x256xi32>, tensor<256x128xi32>) outs(%6 : tensor<128x128xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %8 = arith.muli %in, %in_0 : i32
      %9 = arith.addi %out, %8 : i32
      linalg.yield %9 : i32
    } -> tensor<128x128xi32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xi32>>
    return
  }
}

// -----

// Test generic version of matmul with 4d inputs and output.

// PACK-PEEL-4-LEVEL{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[8, 8, 0, 0, 0, 0], [4, 4, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [1, 1, 0, 0, 0, 0]]>
// PACK-PEEL-4-LEVEL{LITERAL}: #amdaie.packing_config<packing_config = [{packedSizes = [0, 0, 4, 4, 0, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
module {
  func.func @mmt4d_dispatch_0_matmul_like_16x128x32x32x8x64_bf16xbf16xf32() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x8x32x64xbf16>>
    %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x8x64x32xbf16>>
    %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x16x32x32xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [16, 8, 32, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x8x32x64xbf16>> -> tensor<16x8x32x64xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [128, 8, 64, 32], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x8x64x32xbf16>> -> tensor<128x8x64x32xbf16>
    %5 = tensor.empty() : tensor<128x16x32x32xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xf32>
    //      PACK-PEEL-4-LEVEL:  linalg.generic
    // PACK-PEEL-4-LEVEL-SAME:  attrs = {lowering_config = #config, packing_config = #packingConfig}
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d2, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d4, d5, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%3, %4 : tensor<16x8x32x64xbf16>, tensor<128x8x64x32xbf16>) outs(%6 : tensor<128x16x32x32xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %8 = arith.extf %in : bf16 to f32
      %9 = arith.extf %in_0 : bf16 to f32
      %10 = arith.mulf %8, %9 : f32
      %11 = arith.addf %out, %10 : f32
      linalg.yield %11 : f32
    } -> tensor<128x16x32x32xf32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [128, 16, 32, 32], strides = [1, 1, 1, 1] : tensor<128x16x32x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x16x32x32xf32>>
    return
  }
}
