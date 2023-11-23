// For now this isnt doing a FileCheck. It is just checking for compilation.
// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-amdaie-aie-lowering-pipeline))))" %s

#executable_target_elf = #hal.executable.target<"amd-aie", "elf", {target_arch = "chip-tbd"}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>
#translation = #iree_codegen.translation_info<TransformDialectCodegen codegen_spec = @__transform_main>
#device_target_amd_aie = #hal.device.target<"amd-aie", {executable_targets = [#executable_target_elf], legacy_sync}>
module attributes {hal.device.targets = [#device_target_amd_aie]} {
  hal.executable private @matmul_static_dispatch_0 {
    hal.executable.variant public @elf target(#executable_target_elf) {
      hal.executable.export public @matmul_static_dispatch_0_matmul_128x512x256_i32 ordinal(0) layout(#pipeline_layout) attributes {translation_info = #translation} {
      ^bb0(%arg0: !hal.device):
        %c64 = arith.constant 64 : index
        %c16 = arith.constant 16 : index
        %c1 = arith.constant 1 : index
        hal.return %c64, %c16, %c1 : index, index, index
      }
      builtin.module {
        func.func @matmul_static_dispatch_0_matmul_8x8x16_i32() {
          %c0 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c64 = arith.constant 64 : index
          %c1 = arith.constant 1 : index
          %c2 = arith.constant 2 : index
          %c256 = arith.constant 256 : index
          %c4 = arith.constant 4 : index
          %c0_i32 = arith.constant 0 : i32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<8x16xi32>
          memref.assume_alignment %0, 64 : memref<8x16xi32>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x8xi32>
          memref.assume_alignment %1, 64 : memref<16x8xi32>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<8x8xi32>
          memref.assume_alignment %2, 64 : memref<8x8xi32>
          scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c1, %c1) step (%c1, %c1) {
            %3 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg0)
            %4 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg1)
            %subview = memref.subview %0[%3, 0] [8, 16] [1, 1] : memref<8x16xi32> to memref<8x16xi32, strided<[16, 1], offset: ?>>
            %subview_0 = memref.subview %1[0, %4] [16, 8] [1, 1] : memref<16x8xi32> to memref<16x8xi32, strided<[8, 1], offset: ?>>
            %subview_1 = memref.subview %2[%3, %4] [8, 8] [1, 1] : memref<8x8xi32> to memref<8x8xi32, strided<[8, 1], offset: ?>>
            // Erwei: Allocated a memref at L2 to collect results from the 4 tiles from L1
            %alloc_results = memref.alloc() : memref<8x8xi32, 1>
            %alloc = memref.alloc() : memref<8x16xi32, 1>
            memref.copy %subview, %alloc : memref<8x16xi32, strided<[16, 1], offset: ?>> to memref<8x16xi32, 1>
            %alloc_2 = memref.alloc() : memref<16x8xi32, 1>
            memref.copy %subview_0, %alloc_2 : memref<16x8xi32, strided<[8, 1], offset: ?>> to memref<16x8xi32, 1>
            scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
              %5 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg2)
              %6 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg3)
              %subview_3 = memref.subview %alloc[%5, 0] [4, 16] [1, 1] : memref<8x16xi32, 1> to memref<4x16xi32, strided<[16, 1], offset: ?>, 1>
              %subview_4 = memref.subview %alloc_2[0, %6] [16, 4] [1, 1] : memref<16x8xi32, 1> to memref<16x4xi32, strided<[8, 1], offset: ?>, 1>
              // Erwei: Modified this subview to point to alloc_results (L2) instead of L3
              // %subview_5 = memref.subview %subview_1[%5, %6] [4, 4] [1, 1] : memref<8x8xi32, strided<[8, 1], offset: ?>> to memref<4x4xi32, strided<[8, 1], offset: ?>>
              %subview_5 = memref.subview %alloc_results[%arg2, %arg3] [4, 4] [1, 1] : memref<8x8xi32, 1> to memref<4x4xi32, strided<[8, 1], offset: ?>, 1>
              %alloc_6 = memref.alloc() : memref<4x4xi32, 2>
              linalg.fill ins(%c0_i32 : i32) outs(%alloc_6 : memref<4x4xi32, 2>)
              scf.for %arg4 = %c0 to %c16 step %c4 {
                %subview_7 = memref.subview %subview_3[0, %arg4] [4, 4] [1, 1] : memref<4x16xi32, strided<[16, 1], offset: ?>, 1> to memref<4x4xi32, strided<[16, 1], offset: ?>, 1>
                %subview_8 = memref.subview %subview_4[%arg4, 0] [4, 4] [1, 1] : memref<16x4xi32, strided<[8, 1], offset: ?>, 1> to memref<4x4xi32, strided<[8, 1], offset: ?>, 1>
                %alloc_9 = memref.alloc() : memref<4x4xi32, 2>
                memref.copy %subview_7, %alloc_9 : memref<4x4xi32, strided<[16, 1], offset: ?>, 1> to memref<4x4xi32, 2>
                %alloc_10 = memref.alloc() : memref<4x4xi32, 2>
                memref.copy %subview_8, %alloc_10 : memref<4x4xi32, strided<[8, 1], offset: ?>, 1> to memref<4x4xi32, 2>
                linalg.matmul ins(%alloc_9, %alloc_10 : memref<4x4xi32, 2>, memref<4x4xi32, 2>) outs(%alloc_6 : memref<4x4xi32, 2>)
                memref.dealloc %alloc_9 : memref<4x4xi32, 2>
                memref.dealloc %alloc_10 : memref<4x4xi32, 2>
              }
              // Erwei: Copy from L1 to L2
              // memref.copy %alloc_6, %subview_5 : memref<4x4xi32, 2> to memref<4x4xi32, strided<[8, 1], offset: ?>>
              memref.copy %alloc_6, %subview_5 : memref<4x4xi32, 2> to memref<4x4xi32, strided<[8, 1], offset: ?>, 1>
              memref.dealloc %alloc_6 : memref<4x4xi32, 2>
              scf.yield
            }
            memref.dealloc %alloc : memref<8x16xi32, 1>
            memref.dealloc %alloc_2 : memref<16x8xi32, 1>
            // Erwei: Copy from L2 to L3
            memref.copy %alloc_results, %subview_1 : memref<8x8xi32, 1> to memref<8x8xi32, strided<[8, 1], offset: ?>>
            memref.dealloc %alloc_results : memref<8x8xi32, 1>
            scf.yield
          }
          return
        }
      }
    }
  }
  func.func @matmul_static(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c131072 = arith.constant 131072 : index
    %c524288 = arith.constant 524288 : index
    %c262144 = arith.constant 262144 : index
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c268435488_i32 = arith.constant 268435488 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c128, %c256]) type(%c268435488_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<128x256xi32> in !stream.resource<external>{%c131072}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c256, %c512]) type(%c268435488_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<256x512xi32> in !stream.resource<external>{%c524288}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<external>{%c262144} => !stream.timepoint
    %2 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg2: !stream.resource<external>{%c131072}, %1 as %arg3: !stream.resource<external>{%c524288}, %result as %arg4: !stream.resource<external>{%c262144}) {
      stream.cmd.dispatch @matmul_static_dispatch_0::@elf::@matmul_static_dispatch_0_matmul_128x512x256_i32 {
        ro %arg2[%c0 for %c131072] : !stream.resource<external>{%c131072},
        ro %arg3[%c0 for %c524288] : !stream.resource<external>{%c524288},
        wo %arg4[%c0 for %c262144] : !stream.resource<external>{%c262144}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]}
    } => !stream.timepoint
    %3 = stream.timepoint.await %2 => %result : !stream.resource<external>{%c262144}
    %4 = stream.tensor.export %3 : tensor<128x512xi32> in !stream.resource<external>{%c262144} -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
}
