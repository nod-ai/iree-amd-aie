#pipeline_layout = #hal.pipeline.layout<
  bindings = [
    #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
    #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
    #hal.pipeline.binding<storage_buffer, Indirect>
  ],
  flags = Indirect
>
!t_lhs = tensor<512x4096xbf16>
!fdt_lhs = !flow.dispatch.tensor<readonly:tensor<4096x512xbf16>>
!t_rhs = tensor<4096x512xbf16>
!fdt_rhs = !flow.dispatch.tensor<readonly:tensor<4096x512xbf16>>
!t_res = tensor<512x512xf32>
!fdt_res = !flow.dispatch.tensor<writeonly:tensor<512x512xf32>>
hal.executable.source public @amdaie_fb {
  hal.executable.export public @mm_512_512_4096_bf16_f32 ordinal(0) layout(#pipeline_layout) {
  ^bb0(%arg0: !hal.device):
    %x, %y, %z = flow.dispatch.workgroup_count_from_slice
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @mm_512_512_4096_bf16_f32() {
      %c0_f32 = arith.constant 0.0 : f32
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !fdt_lhs
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !fdt_rhs
      %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !fdt_res
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 4096], strides = [1, 1] : !fdt_lhs -> !t_lhs
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4096, 512], strides = [1, 1] : !fdt_rhs -> !t_rhs
      %5 = tensor.empty() : !t_res
      %6 = linalg.fill ins(%c0_f32 : f32) outs(%5 : !t_res) -> !t_res
      %7 = linalg.matmul ins(%3, %4 : !t_lhs, !t_rhs) outs(%6 : !t_res) -> !t_res
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [512, 512], strides = [1, 1] : !t_res -> !fdt_res
      return
    }
  }
}

