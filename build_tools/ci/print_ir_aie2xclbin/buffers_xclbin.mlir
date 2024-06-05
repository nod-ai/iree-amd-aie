// RUN: iree-compile --compile-mode=hal-executable --iree-hal-target-backends=amd-aie-direct %s --iree-hal-dump-executable-files-to %T
// RUN: FileCheck %s --input-file=%T/module_dummy1_amdaie_xclbin_fb/kernels.json

// CHECK: {
// CHECK:   "ps-kernels": {
// CHECK:     "kernels": [
// CHECK:       {
// CHECK:         "arguments": [
// CHECK:           {
// CHECK:             "address-qualifier": "SCALAR",
// CHECK:             "name": "opcode",
// CHECK:             "offset": "0x00",
// CHECK:             "type": "uint64_t"
// CHECK:           },
// CHECK:           {
// CHECK:             "address-qualifier": "GLOBAL",
// CHECK:             "memory-connection": "SRAM",
// CHECK:             "name": "instr",
// CHECK:             "offset": "0x08",
// CHECK:             "type": "char *"
// CHECK:           },
// CHECK:           {
// CHECK:             "address-qualifier": "SCALAR",
// CHECK:             "name": "ninstr",
// CHECK:             "offset": "0x10",
// CHECK:             "type": "uint32_t"
// CHECK:           },
// CHECK:           {
// CHECK:             "address-qualifier": "GLOBAL",
// CHECK:             "memory-connection": "HOST",
// CHECK:             "name": "bo0",
// CHECK:             "offset": "0x14",
// CHECK:             "type": "void*"
// CHECK:           },
// CHECK:           {
// CHECK:             "address-qualifier": "GLOBAL",
// CHECK:             "memory-connection": "HOST",
// CHECK:             "name": "bo1",
// CHECK:             "offset": "0x1c",
// CHECK:             "type": "void*"
// CHECK:           },
// CHECK:           {
// CHECK:             "address-qualifier": "GLOBAL",
// CHECK:             "memory-connection": "HOST",
// CHECK:             "name": "bo2",
// CHECK:             "offset": "0x24",
// CHECK:             "type": "void*"
// CHECK:           },
// CHECK:           {
// CHECK:             "address-qualifier": "GLOBAL",
// CHECK:             "memory-connection": "HOST",
// CHECK:             "name": "bo3",
// CHECK:             "offset": "0x2c",
// CHECK:             "type": "void*"
// CHECK:           },
// CHECK:           {
// CHECK:             "address-qualifier": "GLOBAL",
// CHECK:             "memory-connection": "HOST",
// CHECK:             "name": "bo4",
// CHECK:             "offset": "0x34",
// CHECK:             "type": "void*"
// CHECK:           }
// CHECK:         ],
// CHECK:         "extended-data": {
// CHECK:           "dpu_kernel_id": "0x101",
// CHECK:           "functional": "0",
// CHECK:           "subtype": "DPU"
// CHECK:         },
// CHECK:         "instances": [
// CHECK:           {
// CHECK:             "name": "FOO"
// CHECK:           }
// CHECK:         ],
// CHECK:         "name": "dummy2",
// CHECK:         "type": "dpu"
// CHECK:       }
// CHECK:     ]
// CHECK:   }
// CHECK: }



module attributes {hal.device.targets = [#hal.device.target<"amd-aie-direct", [#hal.executable.target<"amd-aie-direct", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "none"}>]>]} {
  hal.executable private @dummy1 {
    hal.executable.variant public @amdaie_xclbin_fb target(<"amd-aie-direct", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "none"}>) {
      hal.executable.export public @dummy2 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>]} {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        aie.device(npu1_4col) {
          memref.global "public" @in0 : memref<1024xi32>
          memref.global "public" @out0 : memref<1024xi32>
          memref.global "public" @in1 : memref<1024xi32>
          memref.global "public" @out1 : memref<1024xi32>
          memref.global "public" @in2 : memref<1024xi32>
          memref.global "public" @out2 : memref<1024xi32>
          %02 = aie.tile(0, 2)
          %12 = aie.tile(1, 2)
          %22 = aie.tile(2, 2)

          aie.core(%12) {
            aie.end
          }
          aie.shim_dma_allocation @in0(MM2S, 0, 0)
          aie.shim_dma_allocation @out0(S2MM, 0, 0)
          aie.shim_dma_allocation @in1(MM2S, 1, 0)
          aie.shim_dma_allocation @out1(S2MM, 1, 0)
          aie.shim_dma_allocation @in2(MM2S, 2, 0)
          aie.shim_dma_allocation @out2(S2MM, 2, 0)

          func.func @sequence(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>, %arg2: memref<1024xi32>, %arg3: memref<1024xi32>, %arg4: memref<1024xi32>, %arg5: memref<1024xi32>) {
            aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0]) {id = 0 : i64, metadata = @in0} : memref<1024xi32>
            aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0]) {id = 1 : i64, metadata = @out0} : memref<1024xi32>
            aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
            aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0]) {id = 2 : i64, metadata = @in1} : memref<1024xi32>
            aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0]) {id = 3 : i64, metadata = @out1} : memref<1024xi32>
            aiex.npu.sync {channel = 1 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
            aiex.npu.dma_memcpy_nd(0, 0, %arg4[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0]) {id = 2 : i64, metadata = @in2} : memref<1024xi32>
            aiex.npu.dma_memcpy_nd(0, 0, %arg5[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0]) {id = 3 : i64, metadata = @out2} : memref<1024xi32>
            aiex.npu.sync {channel = 0 : i32, column = 2 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
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