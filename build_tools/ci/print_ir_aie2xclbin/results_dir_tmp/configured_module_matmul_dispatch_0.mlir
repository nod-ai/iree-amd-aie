hal.executable public @matmul_dispatch_0 {
  hal.executable.variant public @amdaie_xclbin_fb target(<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "none"}>) {
    hal.executable.export public @matmul_dispatch_0_matmul_64x64x64_i32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_dispatch_0_matmul_64x64x64_i32() {
        %c0_i32 = arith.constant 0 : i32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x64xi32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x64xi32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x64xi32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [64, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x64xi32>> -> tensor<64x64xi32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [64, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x64xi32>> -> tensor<64x64xi32>
        %5 = tensor.empty() : tensor<64x64xi32>
        %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<64x64xi32>) -> tensor<64x64xi32>
        %7 = linalg.matmul ins(%3, %4 : tensor<64x64xi32>, tensor<64x64xi32>) outs(%6 : tensor<64x64xi32>) -> tensor<64x64xi32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [64, 64], strides = [1, 1] : tensor<64x64xi32> -> !flow.dispatch.tensor<writeonly:tensor<64x64xi32>>
        return
      }
    }
  }
}
