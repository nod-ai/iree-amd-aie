// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-amdaie-pack-and-transpose{pack-level=1})))' --split-input-file %s | FileCheck --check-prefix=CHECK-1 %s

hal.executable private @matmul_pack_example_1 {
  hal.executable.variant public @elf target(<"amd-aie", "elf", {target_arch = "chip-tbd"}>) {
    hal.executable.export public @matmul_example_dispatch_0_matmul_16x2048x2048_i8xi8xi32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {translation_info = #iree_codegen.translation_info<TransformDialectCodegen codegen_spec = @__transform_main>} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_example_dispatch_0_matmul_16x256x256_i8xi8xi32() {
        %c0 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x256xi8>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xi8>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x256xi32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [16, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x256xi8>> -> tensor<16x256xi8>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xi8>> -> tensor<256x256xi8>
        %5 = tensor.empty() : tensor<16x256xi32>
        %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<16x256xi32>) -> tensor<16x256xi32>
        // CHECK-1: tensor.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %{{.*}} : tensor<16x256xi8> -> tensor<1x4x16x64xi8>
        // CHECK-1: tensor.pack %{{.*}} outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %{{.*}} : tensor<256x256xi8> -> tensor<4x4x64x64xi8>
        // CHECK-1: tensor.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %{{.*}} : tensor<16x256xi32> -> tensor<1x4x16x64xi32>
        %7 = linalg.matmul ins(%3, %4 : tensor<16x256xi8>, tensor<256x256xi8>) outs(%6 : tensor<16x256xi32>) -> tensor<16x256xi32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [16, 256], strides = [1, 1] : tensor<16x256xi32> -> !flow.dispatch.tensor<writeonly:tensor<16x256xi32>>
        return
      }
    }
  }
}
