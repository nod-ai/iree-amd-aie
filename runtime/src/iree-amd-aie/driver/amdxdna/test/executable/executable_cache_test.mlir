#pipeline_layout = #hal.pipeline.layout<
  bindings = [
    #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
    #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
    #hal.pipeline.binding<storage_buffer, Indirect>
  ],
  flags = Indirect
>
!t_lhs = tensor<8x4xf32>
!fdt_lhs = !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x4xf32>>
!t_rhs = tensor<4x8xf32>
!fdt_rhs = !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x8xf32>>
!t_res = tensor<8x8xf32>
!fdt_res = !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x8xf32>>
hal.executable.source public @amdaie_fb {
  hal.executable.export public @mm_8x4_4x8_f32 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @mm_8x4_4x8_f32() {
      %c0_f32 = arith.constant 0.0 : f32
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !fdt_lhs
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !fdt_rhs
      %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !fdt_res
      %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 4], strides = [1, 1] : !fdt_lhs -> !t_lhs
      %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4, 8], strides = [1, 1] : !fdt_rhs -> !t_rhs
      %5 = tensor.empty() : !t_res
      %6 = linalg.fill ins(%c0_f32 : f32) outs(%5 : !t_res) -> !t_res
      %7 = linalg.matmul ins(%3, %4 : !t_lhs, !t_rhs) outs(%6 : !t_res) -> !t_res
      iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [8, 8], strides = [1, 1] : !t_res -> !fdt_res
      return
    }
  }
}
