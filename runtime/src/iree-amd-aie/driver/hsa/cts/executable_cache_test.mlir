// bootstrapped from https://github.com/nod-ai/iree-amd-aie/blob/9c4c167baf89a279888fba8db75907845946077c/tests/samples/matmul_pack_peel_objectfifo_e2e.mlir

hal.executable.source public @amdaie_fb {
  hal.executable.export public @matmul_f32_dispatch_0_matmul_32x32x32_f32 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) {
  ^bb0(%arg0: !hal.device):
    %x, %y, %z = flow.dispatch.workgroup_count_from_slice
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @matmul_f32_dispatch_0_matmul_32x32x32_f32() {
      %c0_f32 = arith.constant 0.0 : f32
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<32x32xf32>>
      %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<32x32xf32>>
      %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<32x32xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x32xf32>> -> tensor<32x32xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x32xf32>> -> tensor<32x32xf32>
      %5 = tensor.empty() : tensor<32x32xf32>
      %6 = linalg.fill ins(%c0_f32 : f32) outs(%5 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %7 = linalg.matmul ins(%3, %4 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%6 : tensor<32x32xf32>) -> tensor<32x32xf32>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [32, 32], strides = [1, 1] : tensor<32x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x32xf32>>
      return
    }
  }
}
