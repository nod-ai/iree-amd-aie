// This test is used to check the IR generated from AMDAIE passes to be the same as
// what generated from transform dialect (matmul_fill_spec_pack_peel.mlir).

// RUN: iree-opt %s --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-amdaie-lower-workgroup-count, cse, builtin.module(func.func(iree-amdaie-tile-and-fuse{tiling-level=0}, iree-amdaie-tile-and-fuse{use-scf-for tiling-level=1}, iree-amdaie-cleanup, canonicalize, cse, iree-amdaie-pack-and-transpose{pack-level=1}, iree-amdaie-bufferize-to-allocation{memory-space=1}, iree-amdaie-tile-and-fuse{tiling-level=2}, iree-amdaie-cleanup, canonicalize, cse, iree-amdaie-pack-and-transpose{pack-level=2}, iree-amdaie-bufferize-to-allocation{memory-space=2}, iree-hoist-statically-bound-allocations, iree-amdaie-peel-for-loop, iree-amdaie-cleanup, canonicalize, cse)))), iree-eliminate-empty-tensors, iree-codegen-iree-comprehensive-bufferize, canonicalize, cse, canonicalize)"

#config = #iree_codegen.lowering_config<tile_sizes = [[16, 64], [0, 0, 64], [1, 1]]>
#translation = #iree_codegen.translation_info<CPUDefault>
hal.executable private @matmul_pack_example_1 {
  hal.executable.variant public @elf target(<"amd-aie", "elf", {target_arch = "chip-tbd"}>) {
    hal.executable.export public @matmul_example_dispatch_0_matmul_16x2048x2048_i8xi8xi32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_example_dispatch_0_matmul_16x256x256_i8xi8xi32() {
        %c0 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x256xi8>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xi8>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x256xi32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [16, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x256xi8>> -> tensor<16x256xi8>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xi8>> -> tensor<256x256xi8>
        %5 = tensor.empty() : tensor<16x256xi32>
        %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<16x256xi32>) -> tensor<16x256xi32>
        %7 = linalg.matmul {lowering_config = #config} ins(%3, %4 : tensor<16x256xi8>, tensor<256x256xi8>) outs(%6 : tensor<16x256xi32>) -> tensor<16x256xi32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [16, 256], strides = [1, 1] : tensor<16x256xi32> -> !flow.dispatch.tensor<writeonly:tensor<16x256xi32>>
        return
      }
    }
  }
}
