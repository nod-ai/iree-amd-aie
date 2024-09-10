// RUN: iree-opt --split-input-file --iree-amdaie-link-executables %s | FileCheck %s

#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [<storage_buffer, ReadOnly>, <storage_buffer, ReadOnly>, <storage_buffer>]>
#device_target_amd_aie = #hal.device.target<"amd-aie", [#executable_target_amdaie_xclbin_fb]>
module attributes {hal.device.targets = [#device_target_amd_aie]} {
  hal.executable private @two_mm_dispatch_0 {
    hal.executable.variant public @amdaie_xclbin_fb target(#executable_target_amdaie_xclbin_fb) {
      hal.executable.export public @two_mm_dispatch_0_matmul_512x512x512_f32 ordinal(0) layout(#pipeline_layout) attributes {workgroup_size = [1 : index, 1 : index, 1 : index]} {
      ^bb0(%arg0: !hal.device):
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        aie.device(npu1_4col) {
            func.func @two_mm_dispatch_0_matmul_512x512x512_f32() {
                return
            }
        } {sym_name = "segment_0"}
      }
    }
  }
   hal.executable private @two_mm_dispatch_1 {
    hal.executable.variant public @amdaie_xclbin_fb target(#executable_target_amdaie_xclbin_fb) {
      hal.executable.export public @two_mm_dispatch_1_matmul_512x256x512_f32 ordinal(0) layout(#pipeline_layout) attributes {workgroup_size = [1 : index, 1 : index, 1 : index]} {
      ^bb0(%arg0: !hal.device):
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        aie.device(npu1_4col) {
            func.func @two_mm_dispatch_1_matmul_512x256x512_f32() {
                return
            }
        } {sym_name = "segment_1"}
      }
    }
  }
util.func public @two_mm(%arg0: !hal.buffer, %arg1: !hal.buffer, %arg2: !hal.buffer) attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @two_mm(%input0: tensor<512x512xf32>, %input1: tensor<512x512xf32>, %input2: tensor<512x256xf32>)"}} {
    %c-1_i64 = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c524288 = arith.constant 524288 : index
    %c1048576 = arith.constant 1048576 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %cmd = hal.command_buffer.create device(%device_0 : !hal.device) mode("OneShot") categories("Transfer|Dispatch") affinity(%c-1_i64) : !hal.command_buffer
    %exe = hal.executable.lookup device(%device_0 : !hal.device) executable(@two_mm_dispatch_0) : !hal.executable
    %ordinal = hal.executable.export.ordinal target(@two_mm_dispatch_0::@amdaie_xclbin_fb::@two_mm_dispatch_0_matmul_512x512x512_f32) : index
    hal.command_buffer.dispatch<%cmd : !hal.command_buffer>
          target(%exe : !hal.executable)[%ordinal]
          workgroups([%c1, %c1, %c1])
          bindings([
            (%arg0: !hal.buffer)[%c0, %c1048576],
            (%arg1: !hal.buffer)[%c0, %c1048576],
            (%arg2: !hal.buffer)[%c0, %c1048576]
          ])
          flags(None)
    %exe_1 = hal.executable.lookup device(%device_0 : !hal.device) executable(@two_mm_dispatch_1) : !hal.executable
    %ordinal_1 = hal.executable.export.ordinal target(@two_mm_dispatch_1::@amdaie_xclbin_fb::@two_mm_dispatch_1_matmul_512x256x512_f32) : index
    hal.command_buffer.dispatch<%cmd : !hal.command_buffer>
          target(%exe : !hal.executable)[%ordinal]
          workgroups([%c1, %c1, %c1])
          bindings([
            (%arg0: !hal.buffer)[%c0, %c524288],
            (%arg1: !hal.buffer)[%c0, %c524288],
            (%arg2: !hal.buffer)[%c0, %c1048576]
          ])
          flags(None)
    util.return
  }
}

//  CHECK-NOT: hal.executable private @two_mm_dispatch_0
//  CHECK-NOT: hal.executable private @two_mm_dispatch_1

//      CHECK: hal.executable private @aie_link_executables_linked_amd_aie {
// CHECK-NEXT:   hal.executable.variant public @amdaie_xclbin_fb target(#executable_target_amdaie_xclbin_fb) {
//      CHECK:     hal.executable.export public @two_mm_dispatch_0_matmul_512x512x512_f32 ordinal(0)
// CHECK-SAME:       {workgroup_size = [1 : index, 1 : index, 1 : index]}
//      CHECK:       hal.return %c1, %c1, %c1
//      CHECK:     hal.executable.export public @two_mm_dispatch_1_matmul_512x256x512_f32 ordinal(1)
// CHECK-SAME:       {workgroup_size = [1 : index, 1 : index, 1 : index]}
//      CHECK:     builtin.module {
//      CHECK:       aie.device
//      CHECK:         func.func @two_mm_dispatch_0_matmul_512x512x512_f32()
//      CHECK:       aie.device
//      CHECK:         func.func @two_mm_dispatch_1_matmul_512x256x512_f32()
//      CHECK:     }
//      CHECK:   }
//      CHECK: }
//
