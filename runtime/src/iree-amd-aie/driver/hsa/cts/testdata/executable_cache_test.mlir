// bootstrapped from https://github.com/nod-ai/iree-amd-aie/blob/9c4c167baf89a279888fba8db75907845946077c/tests/samples/matmul_pack_peel_objectfifo_e2e.mlir

hal.executable.source public @amdaie_fb {
  hal.executable.export public @matmul_i32_dispatch_0_matmul_128x128x256_i32 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) {
  ^bb0(%arg0: !hal.device):
    %x, %y, %z = flow.dispatch.workgroup_count_from_slice
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @matmul_i32_dispatch_0_matmul_128x128x256_i32() {
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<128x256xi32>>
      %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<256x128xi32>>
      %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<128x128xi32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x256xi32>> -> tensor<128x256xi32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x128xi32>> -> tensor<256x128xi32>
      %5 = tensor.empty() : tensor<128x128xi32>
      %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<128x128xi32>) -> tensor<128x128xi32>
      %7 = linalg.matmul ins(%3, %4 : tensor<128x256xi32>, tensor<256x128xi32>) outs(%6 : tensor<128x128xi32>) -> tensor<128x128xi32>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xi32> -> !flow.dispatch.tensor<writeonly:tensor<128x128xi32>>
      return
    }
  }
}
