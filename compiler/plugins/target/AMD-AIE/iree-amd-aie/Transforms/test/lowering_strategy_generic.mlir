// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy)' %s | FileCheck %s

// Test generic version of matmul.

// CHECK{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[128, 128], [0, 0, 1], [1, 1, 0, 0, 0, 0]]>
// CHECK{LITERAL}: #packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 32], transposePackIndices = [0, 1], unpackEmpty = [false, false], innerPerm = [[0, 1], [1, 0]], outerPerm = [[0, 1], [0, 1]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
module {
  func.func @matmul_generic_128x128x256_i32() {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<128x256xi32>>
    %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<256x128xi32>>
    %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<128x128xi32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x256xi32>> -> tensor<128x256xi32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x128xi32>> -> tensor<256x128xi32>
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
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xi32> -> !flow.dispatch.tensor<writeonly:tensor<128x128xi32>>
    return
  }
}

// -----

// Test generic version of matmul_transpose_b.

// CHECK{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[128, 128], [0, 0, 1], [1, 1, 0, 0, 0, 0]]>
// CHECK{LITERAL}: #packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 32], transposePackIndices = [0, 1], unpackEmpty = [false, false], innerPerm = [[0, 1], [0, 1]], outerPerm = [[0, 1], [0, 1]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [0, 1], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
module {
  func.func @matmul_transpose_b_generic_128x128x256_i32() {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<128x256xi32>>
    %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<128x256xi32>>
    %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<128x128xi32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x256xi32>> -> tensor<128x256xi32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x256xi32>> -> tensor<128x256xi32>
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
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xi32> -> !flow.dispatch.tensor<writeonly:tensor<128x128xi32>>
    return
  }
}

// -----

// Test generic version of matmul_transpose_a.

// CHECK{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[128, 128], [0, 0, 1], [1, 1, 0, 0, 0, 0]]>
// CHECK{LITERAL}: #packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 32], transposePackIndices = [0, 1], unpackEmpty = [false, false], innerPerm = [[1, 0], [1, 0]], outerPerm = [[0, 1], [0, 1]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[1, 0], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
module {
  func.func @matmul_transpose_a_generic_128x128x256_i32() {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<256x128xi32>>
    %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<256x128xi32>>
    %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<128x128xi32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x128xi32>> -> tensor<256x128xi32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x128xi32>> -> tensor<256x128xi32>
    %5 = tensor.empty() : tensor<128x128xi32>
    %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<128x128xi32>) -> tensor<128x128xi32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<256x128xi32>, tensor<256x128xi32>) outs(%6 : tensor<128x128xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %8 = arith.muli %in, %in_0 : i32
      %9 = arith.addi %out, %8 : i32
      linalg.yield %9 : i32
    } -> tensor<128x128xi32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xi32> -> !flow.dispatch.tensor<writeonly:tensor<128x128xi32>>
    return
  }
}
