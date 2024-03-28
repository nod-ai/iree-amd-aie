// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-amdaie-lowering-strategy{use-pass-pipeline=pad-pack})))' %s | FileCheck %s

// CHECK{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [0, 0, 256], [16, 16], [0, 0, 2]]>
// CHECK{LITERAL}: #packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[1, 0], [1, 0], [1, 0]]}]>
hal.executable private @matmul_pad_pack_large_i64 {
  hal.executable.variant public @amdaie_xclbin_fb target(<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "none"}>) {
    hal.executable.export public @matmul_large_dispatch_0_matmul_2048x2048x2048_i64 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_large_dispatch_0_matmul_2048x2048x2048_i64() {
        %c0_i64 = arith.constant 0 : i64
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x2048xi64>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x2048xi64>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x2048xi64>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x2048xi64>> -> tensor<2048x2048xi64>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x2048xi64>> -> tensor<2048x2048xi64>
        %5 = tensor.empty() : tensor<2048x2048xi64>
        %6 = linalg.fill ins(%c0_i64 : i64) outs(%5 : tensor<2048x2048xi64>) -> tensor<2048x2048xi64>
        // CHECK:  linalg.matmul {lowering_config = #config, packing_config = #packingConfig} 
        %7 = linalg.matmul ins(%3, %4 : tensor<2048x2048xi64>, tensor<2048x2048xi64>) outs(%6 : tensor<2048x2048xi64>) -> tensor<2048x2048xi64>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : tensor<2048x2048xi64> -> !flow.dispatch.tensor<writeonly:tensor<2048x2048xi64>>
        return
      }
    }
  }
}

// -----

// CHECK{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[128, 128], [0, 0, 256], [32, 32], [0, 0, 4]]>
// CHECK{LITERAL}: #packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[1, 0], [1, 0], [1, 0]]}]>
hal.executable private @matmul_pad_pack_large_i32 {
  hal.executable.variant public @amdaie_xclbin_fb target(<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "none"}>) {
    hal.executable.export public @matmul_large_dispatch_0_matmul_2048x2048x2048_i32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_large_dispatch_0_matmul_2048x2048x2048_i32() {
        %c0_i32 = arith.constant 0 : i32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x2048xi32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x2048xi32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x2048xi32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x2048xi32>> -> tensor<2048x2048xi32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x2048xi32>> -> tensor<2048x2048xi32>
        %5 = tensor.empty() : tensor<2048x2048xi32>
        %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<2048x2048xi32>) -> tensor<2048x2048xi32>
        // CHECK:  linalg.matmul {lowering_config = #config, packing_config = #packingConfig} 
        %7 = linalg.matmul ins(%3, %4 : tensor<2048x2048xi32>, tensor<2048x2048xi32>) outs(%6 : tensor<2048x2048xi32>) -> tensor<2048x2048xi32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : tensor<2048x2048xi32> -> !flow.dispatch.tensor<writeonly:tensor<2048x2048xi32>>
        return
      }
    }
  }
}

// -----

// CHECK{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[256, 256], [0, 0, 256], [64, 64], [0, 0, 8]]>
// CHECK{LITERAL}: #packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[1, 0], [1, 0], [1, 0]]}]>
hal.executable private @matmul_pad_pack_large_bf16 {
  hal.executable.variant public @amdaie_xclbin_fb target(<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "none"}>) {
    hal.executable.export public @matmul_large_dispatch_0_matmul_2048x2048x2048_bf16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_large_dispatch_0_matmul_2048x2048x2048_bf16() {
        %c0_bf16 = arith.constant 0.0 : bf16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x2048xbf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x2048xbf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x2048xbf16>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x2048xbf16>> -> tensor<2048x2048xbf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x2048xbf16>> -> tensor<2048x2048xbf16>
        %5 = tensor.empty() : tensor<2048x2048xbf16>
        %6 = linalg.fill ins(%c0_bf16 : bf16) outs(%5 : tensor<2048x2048xbf16>) -> tensor<2048x2048xbf16>
        // CHECK:  linalg.matmul {lowering_config = #config, packing_config = #packingConfig} 
        %7 = linalg.matmul ins(%3, %4 : tensor<2048x2048xbf16>, tensor<2048x2048xbf16>) outs(%6 : tensor<2048x2048xbf16>) -> tensor<2048x2048xbf16>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : tensor<2048x2048xbf16> -> !flow.dispatch.tensor<writeonly:tensor<2048x2048xbf16>>
        return
      }
    }
  }
}

// -----

// CHECK{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[8, 32], [0, 0, 16], [8, 16], [0, 0, 2]]>
// CHECK{LITERAL}: #packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[1, 0], [1, 0], [1, 0]]}]>
hal.executable private @matmul_pad_pack_small_i64 {
  hal.executable.variant public @amdaie_xclbin_fb target(<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "none"}>) {
    hal.executable.export public @matmul_small_dispatch_0_matmul_8x32x16_i64 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_small_dispatch_0_matmul_8x32x16_i64() {
        %c0_i64 = arith.constant 0 : i64
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8x16xi64>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x32xi64>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8x32xi64>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 16], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x16xi64>> -> tensor<8x16xi64>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x32xi64>> -> tensor<16x32xi64>
        %5 = tensor.empty() : tensor<8x32xi64>
        %6 = linalg.fill ins(%c0_i64 : i64) outs(%5 : tensor<8x32xi64>) -> tensor<8x32xi64>
        // CHECK:  linalg.matmul {lowering_config = #config, packing_config = #packingConfig} 
        %7 = linalg.matmul ins(%3, %4 : tensor<8x16xi64>, tensor<16x32xi64>) outs(%6 : tensor<8x32xi64>) -> tensor<8x32xi64>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [8, 32], strides = [1, 1] : tensor<8x32xi64> -> !flow.dispatch.tensor<writeonly:tensor<8x32xi64>>
        return
      }
    }
  }
}

// -----

// CHECK{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[8, 32], [0, 0, 16], [8, 32], [0, 0, 2]]>
// CHECK{LITERAL}: #packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[1, 0], [1, 0], [1, 0]]}]>
hal.executable private @matmul_pad_pack_small_i32 {
  hal.executable.variant public @amdaie_xclbin_fb target(<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "none"}>) {
    hal.executable.export public @matmul_small_dispatch_0_matmul_8x32x16_i32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_small_dispatch_0_matmul_8x32x16_i32() {
        %c0_i32 = arith.constant 0 : i32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8x16xi32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x32xi32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8x32xi32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 16], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x16xi32>> -> tensor<8x16xi32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x32xi32>> -> tensor<16x32xi32>
        %5 = tensor.empty() : tensor<8x32xi32>
        %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<8x32xi32>) -> tensor<8x32xi32>
        // CHECK:  linalg.matmul {lowering_config = #config, packing_config = #packingConfig} 
        %7 = linalg.matmul ins(%3, %4 : tensor<8x16xi32>, tensor<16x32xi32>) outs(%6 : tensor<8x32xi32>) -> tensor<8x32xi32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [8, 32], strides = [1, 1] : tensor<8x32xi32> -> !flow.dispatch.tensor<writeonly:tensor<8x32xi32>>
        return
      }
    }
  }
}

// -----

// CHECK{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[8, 32], [0, 0, 16], [8, 32], [0, 0, 2]]>
// CHECK{LITERAL}: #packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[1, 0], [1, 0], [1, 0]]}]>
hal.executable private @matmul_pad_pack_small_bf16 {
  hal.executable.variant public @amdaie_xclbin_fb target(<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "none"}>) {
    hal.executable.export public @matmul_small_dispatch_0_matmul_8x32x16_bf16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_small_dispatch_0_matmul_8x32x16_bf16() {
        %c0_bf16 = arith.constant 0.0 : bf16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8x16xbf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x32xbf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8x32xbf16>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 16], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x16xbf16>> -> tensor<8x16xbf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x32xbf16>> -> tensor<16x32xbf16>
        %5 = tensor.empty() : tensor<8x32xbf16>
        %6 = linalg.fill ins(%c0_bf16 : bf16) outs(%5 : tensor<8x32xbf16>) -> tensor<8x32xbf16>
        // CHECK:  linalg.matmul {lowering_config = #config, packing_config = #packingConfig} 
        %7 = linalg.matmul ins(%3, %4 : tensor<8x16xbf16>, tensor<16x32xbf16>) outs(%6 : tensor<8x32xbf16>) -> tensor<8x32xbf16>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [8, 32], strides = [1, 1] : tensor<8x32xbf16> -> !flow.dispatch.tensor<writeonly:tensor<8x32xbf16>>
        return
      }
    }
  }
}
