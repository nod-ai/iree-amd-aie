// RUN: iree-opt %s --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" | FileCheck %s

// This test case is just to demonstrate that TransformDialectCodegen is not being set
// and that the Pad/Pack pipeline is selected based on `translation_info`.

// CHECK-LABEL: hal.executable.export public @matmul_static_dispatch_0_matmul_8x8x16_i32
//       CHECK:    aie.device(ipu)
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    func.func @matmul_static_dispatch_0_matmul_8x8x16_i32(%arg0: memref<8x16xi32>, %arg1: memref<16x8xi32>, %arg2: memref<8x8xi32>)
//       CHECK:      aiex.ipu.dma_memcpy_nd
//       CHECK:      aiex.ipu.dma_memcpy_nd
//       CHECK:      aiex.ipu.dma_memcpy_nd
//       CHECK:      aiex.ipu.sync
#config_pad = #iree_codegen.lowering_config<tile_sizes = [[8, 8], [4, 4], [0, 0, 4]]>
#translation_pad = #iree_codegen.translation_info<CPUDefault>
hal.executable private @matmul_static_pad {
  hal.executable.variant public @amdaie_xclbin_fb target(<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd"}>) {
    hal.executable.export public @matmul_static_dispatch_0_matmul_8x8x16_i32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {translation_info = #translation_pad} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_static_dispatch_0_matmul_8x8x16_i32() {
        %c0 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8x16xi32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x8xi32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8x8xi32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 16], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x16xi32>> -> tensor<8x16xi32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 8], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x8xi32>> -> tensor<16x8xi32>
        %5 = tensor.empty() : tensor<8x8xi32>
        %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<8x8xi32>) -> tensor<8x8xi32>
        %7 = linalg.matmul {lowering_config = #config_pad} ins(%3, %4 : tensor<8x16xi32>, tensor<16x8xi32>) outs(%6 : tensor<8x8xi32>) -> tensor<8x8xi32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [8, 8], strides = [1, 1] : tensor<8x8xi32> -> !flow.dispatch.tensor<writeonly:tensor<8x8xi32>>
        return
      }
    }
  }
}

// -----

// CHECK-LABEL: hal.executable.export public @matmul_static_dispatch_0_matmul_8x32x16_i32
//       CHECK:    aie.device(ipu)
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    aie.shim_dma_allocation
//       CHECK:    func.func @matmul_static_dispatch_0_matmul_8x32x16_i32(%arg0: memref<8x16xi32>, %arg1: memref<16x32xi32>, %arg2: memref<8x32xi32>)
//       CHECK:      aiex.ipu.dma_memcpy_nd
//       CHECK:      aiex.ipu.dma_memcpy_nd
//       CHECK:      aiex.ipu.dma_memcpy_nd
//       CHECK:      aiex.ipu.sync
#config_pack = #iree_codegen.lowering_config<tile_sizes = [[8, 16], [1, 1], [0, 0, 1]]>
#translation_pack = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @matmul_static_pack {
  hal.executable.variant public @amdaie_xclbin_fb target(<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd"}>) {
    hal.executable.export public @matmul_static_dispatch_0_matmul_8x32x16_i32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {translation_info = #translation_pack} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_static_dispatch_0_matmul_8x32x16_i32() {
        %c0 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8x16xi32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x32xi32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8x32xi32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 16], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x16xi32>> -> tensor<8x16xi32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x32xi32>> -> tensor<16x32xi32>
        %5 = tensor.empty() : tensor<8x32xi32>
        %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<8x32xi32>) -> tensor<8x32xi32>
        %7 = linalg.matmul {lowering_config = #config_pack} ins(%3, %4 : tensor<8x16xi32>, tensor<16x32xi32>) outs(%6 : tensor<8x32xi32>) -> tensor<8x32xi32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [8, 32], strides = [1, 1] : tensor<8x32xi32> -> !flow.dispatch.tensor<writeonly:tensor<8x32xi32>>
        return
      }
    }
  }
}
