//===- npu_instgen.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// RUN: iree-compile --compile-mode=hal-executable --iree-hal-target-backends=amd-aie-direct %s --iree-hal-dump-executable-files-to %T
// RUN: FileCheck %s --input-file=%T/module_dummy1_amdaie_xclbin_fb/module_dummy1_amdaie_xclbin_fb.npu.txt

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
          func.func @test0(%arg0: memref<16xf32>, %arg1: memref<16xf32>) {

            // TXN header
            // CHECK: 06030100
            // CHECK: 00000105
            // CHECK: 00000003
            // CHECK: 00000068

            %c16_i64 = arith.constant 16 : i64
            %c1_i64 = arith.constant 1 : i64
            %c0_i64 = arith.constant 0 : i64
            %c64_i64 = arith.constant 64 : i64
            %c0_i32 = arith.constant 0 : i32
            %c1_i32 = arith.constant 1 : i32
            // CHECK: 00000001
            // CHECK: 00000000
            // CHECK: 0601D0C0
            // CHECK: 00000030
            // CHECK: 00000001
            // CHECK: 00000002
            // CHECK: 00000000
            // CHECK: 00600005
            // CHECK: 80800007
            // CHECK: 00000009
            // CHECK: 2CD0000C
            // CHECK: 2E107041
            aiex.npu.writebd { bd_id = 6 : i32,
                               buffer_length = 1 : i32,
                               buffer_offset = 2 : i32,
                               enable_packet = 0 : i32,
                               out_of_order_id = 0 : i32,
                               packet_id = 0 : i32,
                               packet_type = 0 : i32,
                               column = 3 : i32,
                               row = 0 : i32,
                               d0_stride = 5 : i32,
                               d0_size = 6 : i32,
                               d1_stride = 7 : i32,
                               d1_size = 8 : i32,
                               d2_stride = 9 : i32,
                               ddr_id = 10 : i32,
                               iteration_current = 11 : i32,
                               iteration_stride = 12 : i32,
                               iteration_size = 13 : i32,
                               lock_acq_enable = 1 : i32,
                               lock_acq_id = 1 : i32,
                               lock_acq_val = 2 : i32,
                               lock_rel_id = 3 : i32,
                               lock_rel_val = 4 : i32,
                               next_bd = 5 : i32,
                               use_next_bd = 1 : i32,
                               valid_bd = 1 : i32}
            // CHECK: 00000000
            // CHECK: 00000000
            // CHECK: 06400DEF
            // CHECK: 00000000
            // CHECK: 00000042
            aiex.npu.write32 { column = 3 : i32, row = 4 : i32, address = 0xabc00def : ui32, value = 0x42 : ui32 }

            // CHECK: 00030401
            // CHECK: 05010200
            aiex.npu.sync { column = 3 : i32, row = 4 : i32, direction = 1 : i32, channel = 5 : i32, column_num = 1 : i32, row_num = 2 : i32 }
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