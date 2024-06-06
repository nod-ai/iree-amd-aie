// RUN: not iree-compile --compile-mode=hal-executable --iree-hal-target-backends=amd-aie-direct %s 2>&1 | FileCheck %s

// CHECK: Generating:{{.*}}aie_cdo_elfs.bin
module attributes {hal.device.targets = [#hal.device.target<"amd-aie-direct", [#hal.executable.target<"amd-aie-direct", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "none"}>]>]} {
  hal.executable private @dummy1 {
    hal.executable.variant public @amdaie_xclbin_fb target(<"amd-aie-direct", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "none"}>) {
      hal.executable.export public @dummy2 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>]} {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        aie.device(npu1) {
          %tile_0_0 = aie.tile(0, 0)
          %tile_0_2 = aie.tile(0, 2)
          aie.objectfifo @in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x64xi32>>
          aie.objectfifo @out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64x64xi32>>
          aie.objectfifo.link [@in] -> [@out]()
          %core_0_2 = aie.core(%tile_0_2) {
            %c0 = arith.constant 0 : index
            %0 = memref.alloc() : memref<10xf32>
            %1 = memref.load %0[%c0] : memref<10xf32>
            memref.store %1, %0[%c0] : memref<10xf32>
            aie.end
          }
          func.func @sequence(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>, %arg2: memref<4096xi32>) {
            aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0]) {id = 0 : i64, metadata = @out} : memref<4096xi32>
            aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 64, 64, 1][1, 1, 64]) {id = 1 : i64, metadata = @in} : memref<4096xi32>
            aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
            return
          }
        }
      }
    }
  }
  util.func public @dummy3(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = ""}} {
    // this is all gibberish just to hit serializeExecutable
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %element_type_i8 = hal.element_type<i8> : i32
    %dense_row_major = hal.encoding_type<dense_row_major> : i32
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%c1, %c1]) type(%element_type_i8) encoding(%dense_row_major)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<1024x512xi8> in !stream.resource<external>{%c1}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<external>{%c1} => !stream.timepoint

    %2 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg2: !stream.resource<external>{%c1}) {
      stream.cmd.dispatch @dummy1::@amdaie_xclbin_fb::@dummy2 {
        ro %arg2[%c0 for %c1] : !stream.resource<external>{%c1}
      }
    } => !stream.timepoint
    %3 = stream.timepoint.await %2 => %result : !stream.resource<external>{%c1}
    %4 = stream.tensor.export %3 : tensor<1024x1024xi32> in !stream.resource<external>{%c1} -> !hal.buffer_view
    util.return %4 : !hal.buffer_view
  }
}