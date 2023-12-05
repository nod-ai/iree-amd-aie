module attributes {hal.device.targets = [#hal.device.target<"amd-aie", {executable_targets = [#hal.executable.target<"amd-aie", "elf", {target_arch = "chip-tbd"}>], legacy_sync}>]} {
  hal.executable private @matmul_static_dispatch_0 {
    hal.executable.variant public @elf target(<"amd-aie", "elf", {target_arch = "chip-tbd"}>) {
      hal.executable.export public @matmul_static_dispatch_0_matmul_8x8x16_i32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
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
          %7 = linalg.matmul ins(%3, %4 : tensor<8x16xi32>, tensor<16x8xi32>) outs(%6 : tensor<8x8xi32>) -> tensor<8x8xi32>
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [8, 8], strides = [1, 1] : tensor<8x8xi32> -> !flow.dispatch.tensor<writeonly:tensor<8x8xi32>>
          return
        }
      }
    }
  }
  func.func @matmul_static(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @matmul_static(%input0: tensor<8x16xi32>, %input1: tensor<16x8xi32>) -> (%output0: tensor<8x8xi32>)"}} {
    %c512 = arith.constant 512 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c268435488_i32 = arith.constant 268435488 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%c8, %c16]) type(%c268435488_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<8x16xi32> in !stream.resource<external>{%c512}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input1") shape([%c16, %c8]) type(%c268435488_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<16x8xi32> in !stream.resource<external>{%c512}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<external>{%c256} => !stream.timepoint
    %2 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg2: !stream.resource<external>{%c512}, %1 as %arg3: !stream.resource<external>{%c512}, %result as %arg4: !stream.resource<external>{%c256}) {
      stream.cmd.dispatch @matmul_static_dispatch_0::@elf::@matmul_static_dispatch_0_matmul_8x8x16_i32 {
        ro %arg2[%c0 for %c512] : !stream.resource<external>{%c512},
        ro %arg3[%c0 for %c512] : !stream.resource<external>{%c512},
        wo %arg4[%c0 for %c256] : !stream.resource<external>{%c256}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]}
    } => !stream.timepoint
    %3 = stream.timepoint.await %2 => %result : !stream.resource<external>{%c256}
    %4 = stream.tensor.export %3 : tensor<8x8xi32> in !stream.resource<external>{%c256} -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
}