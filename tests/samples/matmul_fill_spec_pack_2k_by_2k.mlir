//  IREE_DIR=/scratch/iree-to-air
//  IREE_BUILD_DIR=${IREE_BUILD_DIR:-${IREE_DIR}/iree-build}
//  IREE_AMD_AIE_DIR=${IREE_AMD_AIE_DIR:-${IREE_DIR}/iree-amd-aie}
//  IREE_OPT=${IREE_OPT:-${IREE_BUILD_DIR}/tools/iree-opt}
//  SAMPLES_DIR=${SAMPLES_DIR:-${PWD}}
//  
//  DEBUG_FLAGS= # "--mlir-print-ir-after-all --mlir-print-ir-before-all --mlir-disable-threading"
//  
//  ${IREE_OPT} ${SAMPLES_DIR}/matmul_fill_spec_pack_2k_by_2k.mlir \
//    --iree-hal-target-backends=amd-aie \
//    --iree-abi-transformation-pipeline \
//    --iree-flow-transformation-pipeline \
//    --iree-stream-transformation-pipeline \
//    --iree-hal-configuration-pipeline | \
//  ${IREE_OPT} \
//    --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-amdaie-decompose-pack-unpack-to-air)))' | \
//  ${IREE_OPT} \
//    --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(air-par-to-herd{depth=1}, air-par-to-launch{has-air-segment=true}, air-copy-to-dma, canonicalize, cse))))' ${DEBUG_FLAGS} | \
//  ${IREE_OPT} \
//    --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(air-dependency, air-dependency-schedule-opt, air-specialize-dma-broadcast, air-dma-to-channel, canonicalize, cse, air-dependency-canonicalize, canonicalize, cse, air-label-scf-for-to-ping-pong))))' ${DEBUG_FLAGS} | \
//  ${IREE_OPT} \
//    --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(air-ping-pong-transform{keep-memref-dealloc=true}, air-dealias-memref, canonicalize, cse, air-isolate-async-dma-loop-nests, air-specialize-channel-wrap-and-stride, canonicalize, cse))))' ${DEBUG_FLAGS} | \
//  ${IREE_OPT} \
//      --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(air-collapse-herd), canonicalize, cse, air-place-herds{num-rows=4 num-cols=1 row-anchor=2 col-anchor=0}, canonicalize, cse, func.func(air-renumber-dma, convert-linalg-to-loops)))))' ${DEBUG_FLAGS} | \
//  ${IREE_OPT} \
//      --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(air-to-aie{row-offset=2 col-offset=0 device=ipu emit-while-loop=true}, canonicalize, air-to-std, func.func(affine-loop-opt{affine-opt-tile-sizes=4,4}), func.func(air-unroll-outer-affine-loops{depth=2}), affine-expand-index-ops, airrt-to-ipu, canonicalize))))' ${DEBUG_FLAGS}


#executable_target_elf = #hal.executable.target<"amd-aie", "elf", {target_arch = "chip-tbd"}>
#map = affine_map<(d0) -> (d0 * 8)>
#map1 = affine_map<(d0) -> (d0 * 16)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>
#translation = #iree_codegen.translation_info<TransformDialectCodegen codegen_spec = @__transform_main>
#device_target_amd_aie = #hal.device.target<"amd-aie", {executable_targets = [#executable_target_elf], legacy_sync}>
module attributes {hal.device.targets = [#device_target_amd_aie]} {
  hal.executable private @matmul_static_dispatch_0 {
    hal.executable.variant public @elf target(#executable_target_elf) {
      hal.executable.export public @matmul_static_dispatch_0_matmul_2048x512x2048_i32 ordinal(0) layout(#pipeline_layout) attributes {translation_info = #translation} {
      ^bb0(%arg0: !hal.device):
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        hal.return %c2, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @matmul_static_dispatch_0_matmul_2048x512x2048_i32() {
          %c512 = arith.constant 512 : index
          %c32 = arith.constant 32 : index
          %c2048 = arith.constant 2048 : index
          %c0 = arith.constant 0 : index
          %c64 = arith.constant 64 : index
          %c0_i32 = arith.constant 0 : i32
        //   %alloc = memref.alloc() : memref<2048x2048xi32>
          %arg0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<2048x512xi32>
          memref.assume_alignment %arg0, 64 : memref<2048x512xi32>
          %arg1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<512x2048xi32>
          memref.assume_alignment %arg1, 64 : memref<512x2048xi32>
          %argout = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<2048x2048xi32>
          memref.assume_alignment %argout, 64 : memref<2048x2048xi32>
          scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c2048, %c2048) step (%c64, %c64) {
            %subview = memref.subview %arg0[%arg2, 0] [64, 512] [1, 1] : memref<2048x512xi32> to memref<64x512xi32, strided<[512, 1], offset: ?>>
            %subview_0 = memref.subview %arg1[0, %arg3] [512, 64] [1, 1] : memref<512x2048xi32> to memref<512x64xi32, strided<[2048, 1], offset: ?>>
            %subview_1 = memref.subview %argout[%arg2, %arg3] [64, 64] [1, 1] : memref<2048x2048xi32> to memref<64x64xi32, strided<[2048, 1], offset: ?>>
            %alloc_2 = memref.alloc() : memref<1x1x64x64xi32, 1>
            %alloc_3 = memref.alloc() : memref<1x1x64x512xi32, 1>
            %alloc_4 = memref.alloc() : memref<1x1x512x64xi32, 1>
            // memref.copy %subview, %alloc_3 : memref<64x512xi32, strided<[512, 1], offset: ?>> to memref<64x512xi32, 1>
            iree_linalg_ext.pack %subview inner_dims_pos = [0, 1] inner_tiles = [64, 512] into %alloc_3 : (memref<64x512xi32, strided<[512, 1], offset: ?>> memref<1x1x64x512xi32, 1>)
            // memref.copy %subview_0, %alloc_4 : memref<512x64xi32, strided<[2048, 1], offset: ?>> to memref<512x64xi32, 1>
            iree_linalg_ext.pack %subview_0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [512, 64] into %alloc_4 : (memref<512x64xi32, strided<[2048, 1], offset: ?>> memref<1x1x512x64xi32, 1>)
            scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c64, %c64) step (%c32, %c32) {
              %subview_5 = memref.subview %alloc_2[0, 0, %arg4, %arg5] [1, 1, 32, 32] [1, 1, 1, 1] : memref<1x1x64x64xi32, 1> to memref<1x1x32x32xi32, strided<[4096, 4096, 64, 1], offset: ?>, 1>
              %alloc_6 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2>
              linalg.fill ins(%c0_i32 : i32) outs(%alloc_6 : memref<1x1x4x8x4x8xi32, 2>)
              scf.for %arg6 = %c0 to %c512 step %c32 {
                // %subview_7 = memref.subview %alloc_3[%arg4, %arg6] [32, 32] [1, 1] : memref<1x1x64x512xi32, 1> to memref<32x32xi32, strided<[512, 1], offset: ?>, 1>
                %subview_7 = memref.subview %alloc_3[0, 0, %arg4, %arg6] [1, 1, 32, 32] [1, 1, 1, 1] : memref<1x1x64x512xi32, 1> to memref<1x1x32x32xi32, strided<[32768, 32768, 512, 1], offset: ?>, 1>
                // %subview_8 = memref.subview %alloc_4[%arg6, %arg5] [32, 32] [1, 1] : memref<512x64xi32, 1> to memref<32x32xi32, strided<[64, 1], offset: ?>, 1>
                %subview_8 = memref.subview %alloc_4[0, 0, %arg6, %arg5] [1, 1, 32, 32] [1, 1, 1, 1] : memref<1x1x512x64xi32, 1> to memref<1x1x32x32xi32, strided<[32768, 32768, 64, 1], offset: ?>, 1>
                %alloc_9 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2>
                %alloc_10 = memref.alloc() : memref<1x1x4x4x8x8xi32, 2>
                // memref.copy %subview_7, %alloc_9 : memref<32x32xi32, strided<[512, 1], offset: ?>, 1> to memref<32x32xi32, 2>
                iree_linalg_ext.pack %subview_7 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %alloc_9 : (memref<1x1x32x32xi32, strided<[32768, 32768, 512, 1], offset: ?>, 1> memref<1x1x4x8x4x8xi32, 2>)
                // memref.copy %subview_8, %alloc_10 : memref<32x32xi32, strided<[64, 1], offset: ?>, 1> to memref<32x32xi32, 2>
                iree_linalg_ext.pack %subview_8 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 8] into %alloc_10 : (memref<1x1x32x32xi32, strided<[32768, 32768, 64, 1], offset: ?>, 1> memref<1x1x4x4x8x8xi32, 2>)
                // linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%alloc_9, %alloc_10 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%alloc_6 : memref<32x32xi32, 2>)
                linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_9, %alloc_10 : memref<1x1x4x8x4x8xi32, 2>, memref<1x1x4x4x8x8xi32, 2>) outs(%alloc_6 : memref<1x1x4x8x4x8xi32, 2>) {
                ^bb0(%in: i32, %in_10: i32, %out: i32):
                  %5 = arith.muli %in, %in_10 : i32
                  %6 = arith.addi %out, %5 : i32
                  linalg.yield %6 : i32
                }
                memref.dealloc %alloc_9 : memref<1x1x4x8x4x8xi32, 2>
                memref.dealloc %alloc_10 : memref<1x1x4x4x8x8xi32, 2>
              }
            //   memref.copy %alloc_6, %subview_5 : memref<32x32xi32, 2> to memref<32x32xi32, strided<[64, 1], offset: ?>, 1>
              iree_linalg_ext.unpack %alloc_6 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %subview_5 : (memref<1x1x4x8x4x8xi32, 2> memref<1x1x32x32xi32, strided<[4096, 4096, 64, 1], offset: ?>, 1>)
              memref.dealloc %alloc_6 : memref<1x1x4x8x4x8xi32, 2>
              scf.reduce 
            }
            memref.dealloc %alloc_3 : memref<1x1x64x512xi32, 1>
            memref.dealloc %alloc_4 : memref<1x1x512x64xi32, 1>
            // memref.copy %alloc_2, %subview_1 : memref<64x64xi32, 1> to memref<64x64xi32, strided<[2048, 1], offset: ?>>
            iree_linalg_ext.unpack %alloc_2 inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %subview_1 : (memref<1x1x64x64xi32, 1> memref<64x64xi32, strided<[2048, 1], offset: ?>>)
            memref.dealloc %alloc_2 : memref<1x1x64x64xi32, 1>
            scf.reduce 
          }
          return // %alloc : memref<2048x2048xi32>
        }
      }
    }
  }
  func.func @matmul_static(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c512 = arith.constant 512 : index
    %c2048 = arith.constant 2048 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c268435488_i32 = arith.constant 268435488 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c8, %c16]) type(%c268435488_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<8x16xi32> in !stream.resource<external>{%c512}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c16, %c32]) type(%c268435488_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<16x32xi32> in !stream.resource<external>{%c2048}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<external>{%c1024} => !stream.timepoint
    %2 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg2: !stream.resource<external>{%c512}, %1 as %arg3: !stream.resource<external>{%c2048}, %result as %arg4: !stream.resource<external>{%c1024}) {
      stream.cmd.dispatch @matmul_static_dispatch_0::@elf::@matmul_static_dispatch_0_matmul_2048x512x2048_i32 {
        ro %arg2[%c0 for %c512] : !stream.resource<external>{%c512},
        ro %arg3[%c0 for %c2048] : !stream.resource<external>{%c2048},
        wo %arg4[%c0 for %c1024] : !stream.resource<external>{%c1024}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]}
    } => !stream.timepoint
    %3 = stream.timepoint.await %2 => %result : !stream.resource<external>{%c1024}
    %4 = stream.tensor.export %3 : tensor<8x32xi32> in !stream.resource<external>{%c1024} -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
}
